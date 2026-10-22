"""Export a local diagnostic manifest from compiler JSON provenance and SPTG headers.

No model extras, public metadata pickles, kernels or devices are loaded. This
validates the consistency of a local compiler report; reports are not signed
attestations, and source/reload agreement does not certify driving behavior.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from openpilot.selfdrive.navigation.experimental_catalog import (
  CONTRACT_SHA256, DEFAULT_CATALOG, SOURCE_ONNX_SHA256, TINYGRAD_REF, ExperimentalCatalogError,
  _dtype, _header, _json_file, _local_file, _model_contract, _require, _shape, _warp_contract,
  load_experimental_catalog, load_experimental_model,
)


def _digest(path: Path) -> str:
  with path.open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def _record(path: Path) -> dict[str, str]:
  path = path.resolve(strict=True)
  _require(path.is_file(), 'Provenance inputs must be regular local files')
  return {'path': str(path), 'sha256': _digest(path)}


def _runtime_checkout(root: str | Path) -> None:
  root = Path(root).resolve(strict=True)
  def git(*args):
    try:
      result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
      raise ExperimentalCatalogError('Cannot inspect the local tinygrad checkout') from error
    _require(result.returncode == 0, 'Cannot verify the local tinygrad checkout')
    return result.stdout.strip()
  _require(git('rev-parse', 'HEAD') == TINYGRAD_REF, 'Local tinygrad checkout revision mismatch')
  _require(not git('status', '--porcelain', '--untracked-files=all'), 'Local tinygrad checkout must be clean')


def _source_file(anchor: Path, record: dict) -> None:
  source, source_sha = _local_file(anchor, record, 'Source ONNX')
  _require(source_sha == SOURCE_ONNX_SHA256 and source.is_file(), 'Source ONNX is not the pinned public model')


def _output_comparison(values: Any, contract: dict, *, exact: bool = False) -> None:
  expected = {tensor['name']: tensor for tensor in contract['outputs']}
  _require(isinstance(values, dict) and set(values) == set(expected), 'Verification output names mismatch')
  for name, tensor in expected.items():
    value = values[name]
    _require(_shape(value.get('shape')) == _shape(tensor['shape']) and _dtype(value.get('dtype')) == _dtype(tensor['dtype']),
             'Verification output tensor contract mismatch')
    _require(value.get('passed') is True and value.get('source_finite') is True and value.get('reloaded_finite') is True,
             'Verification contains failed or nonfinite outputs')
    delta = value.get('max_abs_difference')
    _require(type(delta) in (int, float) and math.isfinite(delta) and delta >= 0 and (not exact or delta == 0),
             'Invalid verification difference or source repeatability')
    before, after = value.get('source_sha256'), value.get('reloaded_sha256')
    _require(all(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) for digest in (before, after)),
             'Verification requires output byte digests')
    # Finite float32 +0/-0 compare numerically equal but have different bytes.
    # Reload tolerances permit that distinction; exact source repeatability and
    # integer state buffers still require matching bytes when the error is zero.
    signed_zero_possible = not exact and _dtype(tensor['dtype']) == 'float32' and delta == 0
    _require((before == after) == (delta == 0) or signed_zero_possible,
             'Verification byte hashes and numerical differences disagree')


def _verification_report(report: dict, artifact_sha256: str, compiler_sha256: str, contract: dict, header: dict) -> None:
  _require(report.get('status') == 'passed_source_reload_consistency' and report.get('artifact_promoted') is True,
           'Compilation must have passed verification and promoted its artifact')
  _require(report.get('artifact_sha256') == artifact_sha256 and report.get('source_onnx_sha256') == SOURCE_ONNX_SHA256
           and report.get('compiler_sha256') == compiler_sha256, 'Verification artifact/source/compiler digest mismatch')
  runtime = report.get('tinygrad', {})
  _require(runtime.get('revision') == TINYGRAD_REF and runtime.get('dirty') is False and runtime.get('git_status') == '',
           'Verification requires a clean pinned tinygrad runtime')
  backend = report.get('backend', {})
  _require(backend.get('kind') == 'physical_device' and backend.get('emulated') is False,
           'AMD export requires physical-device verification; CPU/emulated reports are insufficient')
  targets = backend.get('targets')
  actual = [call['target'] for call in header['calls'] if call['kind'] == 'program']
  _require(isinstance(targets, list) and bool(targets)
           and {json.dumps(target, sort_keys=True) for target in targets} == {json.dumps(target, sort_keys=True) for target in actual},
           'Verification backend targets do not match artifact kernels')
  for target in targets:
    _require(target.get('device') == 'AMD' and target.get('renderer') == 'LLVM' and target.get('arch') == 'gfx1200'
             and target.get('interface') == 'USB', 'Unsupported verified AMD target')
  _require(report.get('environment', {}).get('NO_HIPCC') in (None, '', '0'), 'No-op compilation cannot be exported')
  verification = report.get('verification', {})
  _require(verification.get('independent_buffers') is True and verification.get('primary_output_changes_with_seed') is True
           and verification.get('equal_nan') is False, 'Independent finite verification evidence is required')
  for key in ('rtol', 'atol'):
    tolerance = verification.get(key)
    _require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 <= tolerance <= 1e-3,
             'Verification tolerances exceed the supported contract')
  seeds, runs = verification.get('seeds'), verification.get('runs')
  _require(isinstance(seeds, list) and len(seeds) >= 2 and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds),
           'Verification requires at least two distinct deterministic seeds')
  _require(isinstance(runs, list) and [run.get('seed') for run in runs] == seeds, 'Verification seed/run mismatch')
  _output_comparison(runs[0].get('source_repeatability'), contract, exact=True)
  for run in runs:
    _output_comparison(run.get('outputs'), contract)
  _require(runs[0]['outputs']['outputs']['source_sha256'] != runs[-1]['outputs']['outputs']['source_sha256'],
           'Primary prediction must change across verification seeds')


def validate_exported_verification(proof: dict, artifact_sha256: str, contract: dict, header: dict) -> None:
  """Recheck exported provenance on every load; no report claim enables actuation."""
  _require(isinstance(proof, dict) and proof.get('scope') == 'source_reload_consistency_only', 'Unsupported compiler verification scope')
  anchor = Path.cwd() / 'local-manifest.json'
  report_path, _ = _local_file(anchor, proof.get('report'), 'Verification report')
  _, compiler_sha = _local_file(anchor, proof.get('compiler'), 'Compiler source')
  _source_file(anchor, proof.get('source_onnx', {}))
  _runtime_checkout(proof.get('tinygrad_root', ''))
  _verification_report(_json_file(report_path), artifact_sha256, compiler_sha, contract, header)


def export_experimental_model(artifact_path: str | Path, warp_path: str | Path, verification_path: str | Path,
                              source_onnx_path: str | Path, compiler_path: str | Path, tinygrad_root: str | Path,
                              output_manifest: str | Path, *, experimental: bool) -> Path:
  """Export new JSON files after validation; never overwrite an existing manifest.

  Requires a real future AMD artifact and its successful compiler report. The
  checked-in disabled catalog, model binaries, firmware and active selection are
  untouched. The resulting manifest is local to these files/checkouts; moving it
  requires relocating and re-exporting its explicit paths.
  """
  try:
    catalog = copy.deepcopy(load_experimental_catalog(DEFAULT_CATALOG, experimental=experimental))
    entry = catalog['entries'][0]
    contract = entry['contract']
    _require(hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == CONTRACT_SHA256,
             'Pinned public contract changed')
    artifact, warp = Path(artifact_path).resolve(strict=True), Path(warp_path).resolve(strict=True)
    artifact_record, warp_record = _record(artifact), _record(warp)
    header = _header(artifact, entry['target'])
    _warp_contract(_header(warp, entry['target']))
    proof = {'scope': 'source_reload_consistency_only', 'report': _record(Path(verification_path)),
             'compiler': _record(Path(compiler_path)), 'source_onnx': _record(Path(source_onnx_path)),
             'tinygrad_root': str(Path(tinygrad_root).resolve(strict=True))}
    validate_exported_verification(proof, artifact_record['sha256'], contract, header)
    named = {row['name']: row for row in header['inputs'] if row['name'] is not None}
    metadata = {'schema_version': 1, 'source_onnx_sha256': SOURCE_ONNX_SHA256, 'tinygrad_ref': TINYGRAD_REF,
                'target': entry['target'], 'output_slices': contract['output_slices'], 'compiler_verification': proof,
                'hardware_driving_validation': False, 'selector_generation_verified': False,
                'limitations': ['Local report and static byte bindings were validated; reports are not authenticated attestations.',
                                'Source/reload consistency does not prove model quality, vehicle safety or firmware compatibility.',
                                'The exporter never reads pickle extras; diagnostic adapters must use their bounded literal-slice decoder.']}
    for kind in ('input', 'output'):
      metadata[f'{kind}_specs'] = {row['name']: {'shape': row['shape'], 'dtype': _dtype(row['dtype']),
                                              'device': named[row['name']]['device'] if kind == 'input' else 'AMD'}
                                 for row in contract[f'{kind}s']}
    _model_contract(header, metadata, contract)
    output = Path(output_manifest).resolve()
    sidecar = output.with_name(output.name + '.metadata.json')
    inputs = {artifact, warp, Path(verification_path).resolve(), Path(source_onnx_path).resolve(), Path(compiler_path).resolve(), DEFAULT_CATALOG}
    _require(output not in inputs and sidecar not in inputs and not output.exists() and not sidecar.exists(), 'Export must use new local output paths')
    output.parent.mkdir(parents=True, exist_ok=True)
    published_sidecar, published_manifest = False, False
    try:
      with tempfile.TemporaryDirectory(dir=output.parent, prefix='.nav-model-export-') as directory:
        temporary_sidecar, temporary_manifest = Path(directory) / 'metadata.json', Path(directory) / 'manifest.json'
        temporary_sidecar.write_text(json.dumps(metadata, indent=2) + '\n')
        entry.update(enabled=True, compiled_artifact=artifact_record, warp_artifact=warp_record,
                     compiler_metadata={'path': str(sidecar), 'sha256': _digest(temporary_sidecar)})
        entry['limitations'] = metadata['limitations'] + ['Generation 0 remains an unverified placeholder; normal activation is prohibited.']
        temporary_manifest.write_text(json.dumps(catalog, indent=2) + '\n')
        os.link(temporary_sidecar, sidecar)  # Atomic no-overwrite publication.
        published_sidecar = True
        inspected = load_experimental_model(temporary_manifest, experimental=True, backend='AMD')
        _require(inspected.compiler_verification_validated, 'Exported compiler provenance was not validated')
        os.link(temporary_manifest, output)
        published_manifest = True
      return output
    except BaseException:
      if published_sidecar and not published_manifest:
        sidecar.unlink(missing_ok=True)
      raise
  except (OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
    raise ExperimentalCatalogError(f'Cannot export local experimental model: {error}') from error


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  for flag in ('artifact', 'warp-artifact', 'verification', 'source-onnx', 'compiler', 'tinygrad-root', 'output-manifest'):
    parser.add_argument('--' + flag, type=Path, required=True)
  parser.add_argument('--experimental', action='store_true', help='Explicitly acknowledge this diagnostic-only export')
  args = parser.parse_args()
  try:
    output = export_experimental_model(args.artifact, args.warp_artifact, args.verification, args.source_onnx,
                                       args.compiler, args.tinygrad_root, args.output_manifest, experimental=args.experimental)
  except ExperimentalCatalogError as error:
    parser.exit(2, f'{error}\n')
  print(f'Exported local diagnostic manifest: {output}')


if __name__ == '__main__':
  main()
