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
import tempfile
from typing import Any

from openpilot.common.git_inspection import GitInspectionError, read_only_git
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID, RETIRED_REASON, is_retired
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


# Compiler evidence is permanently bound to TINYGRAD_REF. This separate execution
# revision uses the same native kernel-binary ABI with the upstream loader's
# ProgramInfo/ParamArg adapter. Only the exact reviewed corrected artifact and its
# camera warps are admitted; no arbitrary newer checkout inherits approval.
EXECUTION_TINYGRAD_REF = '92b742f711c7d216fd3ed7a46a30a959e3c43bd4'
_EXECUTION_ARTIFACTS = frozenset({
  '4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c',
  'f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b',
  '56cc9690e0e9fb719323d2c6646c1305e8bc6bbdfdc807724a605a0d0a37554b',
})


def _runtime_checkout(root: str | Path, *, artifact_sha256: str | None = None) -> str:
  root = Path(root).resolve(strict=True)
  def git(*args):
    try:
      return read_only_git(root, *args, timeout=10)
    except GitInspectionError as error:
      raise ExperimentalCatalogError(str(error)) from error
  revision = git('rev-parse', 'HEAD')
  compatible = revision == TINYGRAD_REF or (revision == EXECUTION_TINYGRAD_REF and artifact_sha256 in _EXECUTION_ARTIFACTS)
  _require(compatible, 'Local tinygrad checkout revision mismatch for the exact reviewed native artifact')
  _require(not git('status', '--porcelain', '--untracked-files=all'), 'Local tinygrad checkout must be clean')
  return revision


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


def validate_exported_verification(proof: dict, artifact_sha256: str, contract: dict, header: dict, *, anchor: Path | None = None,
                                  artifact_path: Path | None = None, model_id: str = MODEL_ID) -> None:
  """Recheck exported provenance on every load; no report claim enables actuation."""
  _require(not is_retired(model_id, artifact_sha256), RETIRED_REASON)
  _require(isinstance(proof, dict) and proof.get('scope') == 'source_reload_consistency_only', 'Unsupported compiler verification scope')
  anchor = anchor if anchor is not None else Path.cwd() / 'local-manifest.json'
  report_path, _ = _local_file(anchor, proof.get('report'), 'Verification report')
  _, compiler_sha = _local_file(anchor, proof.get('compiler'), 'Compiler source')
  source = proof.get('source_onnx', {})
  if source.get('bytes_retained') is False:
    _require(source == {'sha256': SOURCE_ONNX_SHA256, 'bytes_retained': False}, 'Unretained source must retain the pinned ONNX digest')
  else:
    _source_file(anchor, source)
  runtime = Path(proof.get('tinygrad_root', ''))
  _runtime_checkout(runtime if runtime.is_absolute() else anchor.parent / runtime, artifact_sha256=artifact_sha256)
  report = _json_file(report_path)
  _verification_report(report, artifact_sha256, compiler_sha, contract, header)
  from openpilot.selfdrive.navigation import corrected_bigv2 as corrected
  if model_id == corrected.MODEL_ID or artifact_sha256 == corrected.ARTIFACT_SHA256 or 'map_precision_correction' in report:
    _require(model_id == corrected.MODEL_ID, 'Corrected Bigv2 must retain its distinct model identity')
    _require(proof['report']['sha256'] == corrected.REPORT_SHA256, 'Corrected compiler report differs from the reviewed evidence')
    _, implementation_sha = _local_file(anchor, proof.get('map_precision_implementation'), 'Map precision implementation')
    _require(implementation_sha == corrected.IMPLEMENTATION_SHA256, 'Corrected map implementation source SHA256 mismatch')
    corrected.validate_precision_report(report, contract, _output_comparison)
    _require(artifact_path is not None, 'Corrected model requires actual native artifact byte validation')
    corrected.validate_precision_artifact(artifact_path, header, report['map_precision_correction'])


def build_metadata(header: dict, contract: dict, target: dict, proof: dict, *, camera_profile=None,
                   source_bytes_rechecked=False, correction=None) -> dict:
  """Deterministic sidecar construction; callers separately validate actual files."""
  named = {row['name']: row for row in header['inputs'] if row['name'] is not None}
  metadata = {'schema_version': 1, 'source_onnx_sha256': SOURCE_ONNX_SHA256, 'tinygrad_ref': TINYGRAD_REF,
              'target': target, 'output_slices': contract['output_slices'], 'compiler_verification': proof,
              'hardware_driving_validation': False, 'selector_generation_verified': False,
              'source_onnx_bytes_rechecked': source_bytes_rechecked,
              'limitations': ['Local report and static byte bindings were validated; reports are not authenticated attestations.',
                              'Source/reload consistency does not prove model quality, vehicle safety or firmware compatibility.',
                              'The exporter never reads pickle extras; diagnostic adapters must use their bounded literal-slice decoder.']}
  if camera_profile is not None:
    metadata['camera_profile'] = camera_profile
  if correction is not None:
    metadata['map_precision_correction'] = correction
  for kind in ('input', 'output'):
    metadata[f'{kind}_specs'] = {row['name']: {'shape': row['shape'], 'dtype': _dtype(row['dtype']),
                                            'device': named[row['name']]['device'] if kind == 'input' else 'AMD'}
                               for row in contract[f'{kind}s']}
  return metadata


def export_experimental_model(artifact_path: str | Path, warp_path: str | Path, verification_path: str | Path,
                              source_onnx_path: str | Path | None, compiler_path: str | Path, tinygrad_root: str | Path,
                              output_manifest: str | Path, *, experimental: bool, portable: bool = False,
                              camera_profile: dict | None = None, model_id: str = MODEL_ID,
                              precision_implementation_path: str | Path | None = None) -> Path:
  """Export new JSON files after validation; never overwrite an existing manifest.

  Requires an existing AMD artifact and its successful compiler report. The
  checked-in disabled catalog, model binaries, firmware and active selection are
  untouched. Portable manifests anchor relative provenance paths to the metadata
  file, so an atomic package rename does not invalidate compiler/report bindings.
  Unretained source bytes are recorded explicitly; the complete physical compiler
  report still binds the pinned source digest and actual compiled artifact.
  """
  try:
    _require(not is_retired(model_id), RETIRED_REASON)
    catalog = copy.deepcopy(load_experimental_catalog(DEFAULT_CATALOG, experimental=experimental))
    entry = catalog['entries'][0]
    from openpilot.selfdrive.navigation import corrected_bigv2 as corrected
    _require(model_id == corrected.MODEL_ID, 'Unknown existing navigation model identity')
    entry['id'] = model_id
    if model_id == corrected.MODEL_ID:
      entry['selector21_bundle']['display_name'] = corrected.DISPLAY_NAME
      entry['selector21_bundle']['short_name'] = 'NAV-BIG-V2-MAP-FP32'
    contract = entry['contract']
    _require(hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == CONTRACT_SHA256,
             'Pinned public contract changed')
    artifact, warp = Path(artifact_path).resolve(strict=True), Path(warp_path).resolve(strict=True)
    artifact_record, warp_record = _record(artifact), _record(warp)
    _require(not is_retired(artifact_sha256=artifact_record['sha256']), RETIRED_REASON)
    header = _header(artifact, entry['target'])
    _warp_contract(_header(warp, entry['target']))
    output = Path(output_manifest).resolve()
    sidecar = output.with_name(output.name + '.metadata.json')
    proof: dict[str, Any] = {'scope': 'source_reload_consistency_only', 'report': _record(Path(verification_path)),
             'compiler': _record(Path(compiler_path)),
             'source_onnx': (_record(Path(source_onnx_path)) if source_onnx_path is not None else
                             {'sha256': SOURCE_ONNX_SHA256, 'bytes_retained': False}),
             'tinygrad_root': str(Path(tinygrad_root).resolve(strict=True))}
    if model_id == corrected.MODEL_ID:
      _require(precision_implementation_path is not None, 'Corrected compiler implementation source is required')
      proof['map_precision_implementation'] = _record(Path(precision_implementation_path))
    if portable:
      for key in ('report', 'compiler', 'source_onnx', 'map_precision_implementation'):
        if key not in proof:
          continue
        if 'path' in proof[key]:
          proof[key]['path'] = os.path.relpath(proof[key]['path'], output.parent)
      proof['tinygrad_root'] = os.path.relpath(proof['tinygrad_root'], output.parent)
      for record in (artifact_record, warp_record):
        record['path'] = os.path.relpath(record['path'], output.parent)
    validate_exported_verification(proof, artifact_record['sha256'], contract, header, anchor=sidecar,
                                  artifact_path=artifact, model_id=model_id)
    metadata = build_metadata(header, contract, entry['target'], proof, camera_profile=camera_profile,
                              source_bytes_rechecked=source_onnx_path is not None,
                              correction=_json_file(Path(verification_path))['map_precision_correction'] if model_id == corrected.MODEL_ID else None)
    _model_contract(header, metadata, contract)
    inputs = {artifact, warp, Path(verification_path).resolve(), Path(compiler_path).resolve(), DEFAULT_CATALOG}
    if source_onnx_path is not None:
      inputs.add(Path(source_onnx_path).resolve())
    _require(output not in inputs and sidecar not in inputs and not output.exists() and not sidecar.exists(), 'Export must use new local output paths')
    output.parent.mkdir(parents=True, exist_ok=True)
    published_sidecar, published_manifest = False, False
    try:
      # Same-parent temporary files preserve portable relative-path resolution.
      with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix='.nav-model-export-', suffix='.metadata.json') as sidecar_file, \
           tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix='.nav-model-export-', suffix='.manifest.json') as manifest_file:
        temporary_sidecar, temporary_manifest = Path(sidecar_file.name), Path(manifest_file.name)
        sidecar_file.write(json.dumps(metadata, indent=2) + '\n')
        sidecar_file.flush()
        entry.update(enabled=True, compiled_artifact=artifact_record, warp_artifact=warp_record,
                     compiler_metadata={'path': sidecar.name if portable else str(sidecar), 'sha256': _digest(temporary_sidecar)})
        entry['limitations'] = metadata['limitations'] + ['Generation 0 remains an unverified placeholder; normal activation is prohibited.']
        manifest_file.write(json.dumps(catalog, indent=2) + '\n')
        manifest_file.flush()
        os.link(temporary_sidecar, sidecar)  # Atomic no-overwrite publication.
        published_sidecar = True
        inspected = load_experimental_model(temporary_manifest, experimental=True, backend='AMD', model_id=model_id)
        _require(inspected.compiler_verification_validated, 'Exported compiler provenance was not validated')
        os.link(temporary_manifest, output)
        published_manifest = True
      return output
    except BaseException:
      if published_sidecar and not published_manifest:
        sidecar.unlink(missing_ok=True)
      raise
  except (OSError, KeyError, TypeError, AttributeError) as error:
    raise ExperimentalCatalogError(f'Cannot export local experimental model: {error}') from error


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  for flag in ('artifact', 'warp-artifact', 'verification', 'source-onnx', 'compiler', 'tinygrad-root', 'output-manifest'):
    parser.add_argument('--' + flag, type=Path, required=True)
  parser.add_argument('--precision-implementation', type=Path, required=True,
                      help='Exact retained map-precision source associated with the corrected compiler report')
  parser.add_argument('--experimental', action='store_true', help='Explicitly acknowledge this diagnostic-only export')
  args = parser.parse_args()
  try:
    output = export_experimental_model(args.artifact, args.warp_artifact, args.verification, args.source_onnx,
                                       args.compiler, args.tinygrad_root, args.output_manifest, experimental=args.experimental,
                                       precision_implementation_path=args.precision_implementation)
  except ExperimentalCatalogError as error:
    parser.exit(2, f'{error}\n')
  print(f'Exported local diagnostic manifest: {output}')


if __name__ == '__main__':
  main()
