#!/usr/bin/env python3
"""Export the registered Bigv2 kernel code for offline numerical investigation.

Standard library only: no GPU, service, Params, network, pickle or model loading.
Reads the existing artifact once for SHA256 and then copies its bounded JSON
header and program libraries. Weights, runtime inputs and recurrent state are
excluded. This is evidence export, not a numerical verification or activation.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile


REVIEWED_SHA256 = 'feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4'
MAX_HEADER = 8 << 20
MAX_EXPORT = 32 << 20
REGISTRY = Path(__file__).resolve().parents[1] / 'openpilot/selfdrive/navigation/driving_models.json'
RECEIPT = Path('/data/navigation-models/latest-registration.json')
OUTPUT = Path('/data/nav-bigv2/numerical-kernels.tar.gz')


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _json(raw):
  def unique(pairs):
    result = {}
    for key, value in pairs:
      _require(key not in result, f'Duplicate JSON key: {key}')
      result[key] = value
    return result

  def invalid(value):
    raise ValueError(f'Non-finite JSON value: {value}')

  value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
  _require(isinstance(value, dict), 'Expected a JSON object')
  return value


def _read_json(path):
  with Path(path).open('rb') as stream:
    raw = stream.read(MAX_HEADER + 1)
  _require(len(raw) <= MAX_HEADER, 'JSON metadata exceeds export size limit')
  return raw, _json(raw)


def _range(value, size):
  _require(isinstance(value, list) and len(value) == 2 and all(type(n) is int for n in value)
           and 0 <= value[0] <= value[1] <= size, 'Invalid native artifact blob range')
  return value[0], value[1]


def _identity(stat):
  return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def export_kernels(artifact, expected_sha256, output, *, metadata=None):
  """Copy code bytes without interpreting them; never include weights or extras."""
  artifact, output = Path(artifact).resolve(strict=True), Path(output).resolve()
  _require(artifact != output, 'Export destination must not replace the artifact')
  files, index = {}, []
  with artifact.open('rb') as stream:
    before = _identity(os.fstat(stream.fileno()))
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    _require(digest == expected_sha256, 'Registered Bigv2 artifact SHA256 mismatch')
    stream.seek(0)
    _require(stream.read(8) == b'SPTGPLN1', 'Expected native SPTGPLN1 artifact')
    size_bytes = stream.read(8)
    _require(len(size_bytes) == 8, 'Truncated native artifact header length')
    size = struct.unpack('<Q', size_bytes)[0]
    _require(0 < size <= MAX_HEADER, 'Native JSON header exceeds export size limit')
    raw = stream.read(size)
    _require(len(raw) == size, 'Truncated native artifact JSON header')
    container = _json(raw)
    blob_start = (16 + size + 255) // 256 * 256
    blob_size = before[2] - blob_start
    extras_start, extras_end = _range(container.get('extras'), blob_size)  # Bounds only; never read pickle extras.
    parts = container.get('parts')
    _require(isinstance(parts, list) and len(parts) == 1 and isinstance(parts[0], dict) and parts[0].get('key') == 'run',
             'Expected one native run part')
    start, end = _range(parts[0].get('blob'), blob_size)
    header = parts[0].get('header')
    _require(isinstance(header, dict), 'Missing native run header')

    weight_ranges = []

    def check_ranges(value):
      if isinstance(value, dict):
        for name, child in value.items():
          if name in ('lib', 'init', 'data'):
            interval = _range(child, end - start)
            if name != 'lib':
              weight_ranges.append(interval)
          else:
            check_ranges(child)
      elif isinstance(value, list):
        for child in value:
          check_ranges(child)

    check_ranges(header)
    calls = header.get('calls')
    _require(isinstance(calls, list) and 0 < len(calls) <= 10000, 'Invalid native kernel list')
    files['native-header.json'] = raw
    total = len(raw)
    for number, call in enumerate(calls):
      _require(isinstance(call, dict) and call.get('kind') in ('program', 'copy'), 'Unknown native call kind')
      if call['kind'] == 'copy':
        continue
      target = call.get('target', {})
      _require(isinstance(target, dict) and all(target.get(k) == v for k, v in
               {'device': 'AMD', 'arch': 'gfx1200', 'interface': 'USB', 'renderer': 'LLVM'}.items()),
               'Native kernel does not target the registered AMD LLVM gfx1200 USB device')
      first, last = _range(call.get('lib'), end - start)
      _require(0 < last - first <= MAX_EXPORT, 'Invalid or oversized native program library')
      _require(not any(first < right and left < last for left, right in weight_ranges)
               and not (start + first < extras_end and extras_start < start + last),
               'Program library overlaps weights or pickle extras')
      stream.seek(blob_start + start + first)
      code = stream.read(last - first)
      _require(len(code) == last - first, 'Truncated native program library')
      code_sha = hashlib.sha256(code).hexdigest()
      name = f'kernels/{code_sha}.bin'
      if name not in files:
        total += len(code)
        _require(total <= MAX_EXPORT, 'Kernel evidence exceeds 32 MiB export limit')
        files[name] = code
      index.append({'call_index': number, 'file': name, 'sha256': code_sha, 'bytes': len(code)})
    _require(index, 'Native artifact has no program libraries')
    _require(before == _identity(os.fstat(stream.fileno())) == _identity(artifact.stat()), 'Artifact changed during evidence export')
  report = {'scope': 'static_kernel_evidence_only', 'artifact_sha256': digest, 'artifact_bytes': before[2],
            'header_sha256': hashlib.sha256(raw).hexdigest(), 'kernels': index,
            'gpu_execution_performed': False, 'weights_included': False, 'runtime_tensors_included': False}
  if metadata is not None:
    report['registered_contract'] = {key: metadata[key] for key in
      ('schema_version', 'source_onnx_sha256', 'tinygrad_ref', 'target', 'input_specs', 'output_specs', 'output_slices') if key in metadata}
  files['export.json'] = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
  _require(sum(map(len, files.values())) <= MAX_EXPORT, 'Kernel evidence exceeds 32 MiB export limit')
  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(prefix='.nav-kernels-', dir=output.parent) as temporary:
    with tarfile.open(fileobj=temporary, mode='w:gz') as archive:
      for name, content in files.items():
        entry = tarfile.TarInfo(name)
        entry.size, entry.mode = len(content), 0o600
        archive.addfile(entry, io.BytesIO(content))
    temporary.flush()
    os.fsync(temporary.fileno())
    # No overwrite, including a concurrent publication or symlink at the destination.
    os.link(temporary.name, output)
  return report


def export_registered(output=OUTPUT):
  _, registry = _read_json(REGISTRY)
  matches = [c for c in registry['candidates'] if c['id'] == 'big-nav-v2']
  _require(len(matches) == 1 and matches[0]['artifact_sha256'] == REVIEWED_SHA256, 'Branch Bigv2 artifact approval differs from this export target')
  candidate = matches[0]
  receipt_raw, receipt = _read_json(RECEIPT)
  _require(hashlib.sha256(receipt_raw).hexdigest() == candidate['registration_receipt_sha256'], 'Registration receipt SHA256 mismatch')
  manifest_path = Path(receipt['manifest_path']).resolve(strict=True)
  _, manifest = _read_json(manifest_path)
  entries = [e for e in manifest['entries'] if e['id'] == 'big-nav-v2']
  _require(len(entries) == 1, 'Expected one registered Bigv2 entry')
  entry = entries[0]
  artifact_record, metadata_record = entry['compiled_artifact'], entry['compiler_metadata']
  _require(artifact_record['sha256'] == REVIEWED_SHA256 and metadata_record['sha256'] == candidate['metadata_sha256'],
           'Registered artifact or metadata does not match branch identity')
  artifact = (manifest_path.parent / artifact_record['path']).resolve(strict=True)
  metadata_path = (manifest_path.parent / metadata_record['path']).resolve(strict=True)
  raw, metadata = _read_json(metadata_path)
  _require(hashlib.sha256(raw).hexdigest() == candidate['metadata_sha256'], 'Registered metadata SHA256 mismatch')
  _require(Path(output).resolve() not in (REGISTRY, RECEIPT, manifest_path, metadata_path), 'Export must not replace source metadata')
  return export_kernels(artifact, REVIEWED_SHA256, output, metadata=metadata)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=OUTPUT, help='New private archive path; existing files are never replaced')
  args = parser.parse_args()
  try:
    report = export_registered(args.output)
  except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
    parser.exit(1, f'Kernel evidence export refused: {error}\n')
  print(f"Saved {args.output}: {len(report['kernels'])} program references; no weights, vehicle data or GPU execution.")


if __name__ == '__main__':
  main()
