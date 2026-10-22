"""Data-only admission of local models for non-actuating diagnostics.

This separate wrapper deliberately has no top-level ``bundles`` key. It is not
a production Selector catalog, never writes Params, downloads files or opens a
device, and cannot authorize normal model selection. SPTG JSON headers are read
without loading the pickle extras, reconstructing kernels or running inference.
Static agreement with a compiler sidecar does not establish numerical parity.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any


SOURCE_ONNX_SHA256 = '1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e'
TINYGRAD_REF = '69fd4d34d4430036378aba2b43f9d339cdd6ea73'
CONTRACT_SHA256 = '3f4593574d7ab6a94865624a22a05f3c194cbd83a6e9619c2ae2abf496a98c84'
DEFAULT_CATALOG = Path(__file__).with_name('experimental_models.json')
SCOPE = 'non_actuating_diagnostic_only'
MAX_JSON_BYTES = 8 << 20
MAX_HEADER_BYTES = 64 << 20
DTYPES = {'FLOAT': 'float32', 'UINT8': 'uint8', 'float32': 'float32', '<f4': 'float32',
          'dtypes.f32': 'float32', 'dtypes.float32': 'float32', 'uint8': 'uint8', '|u1': 'uint8',
          'dtypes.u8': 'uint8', 'dtypes.uint8': 'uint8'}


class ExperimentalCatalogError(ValueError):
  pass


@dataclass(frozen=True)
class ExperimentalModelEntry:
  model_id: str
  artifact_path: Path
  warp_path: Path
  metadata: dict[str, Any]
  input_specs: dict[str, tuple[tuple[int, ...], str, str]]
  selector21_bundle: dict[str, Any]
  artifact_sha256: str
  warp_sha256: str
  metadata_sha256: str
  warp_frame_size: int
  target: dict[str, str]
  source_onnx_sha256: str = SOURCE_ONNX_SHA256
  normal_activation_allowed: bool = False
  selector_generation_verified: bool = False
  compiler_verification_validated: bool = False


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ExperimentalCatalogError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    _require(key not in result, f'Duplicate JSON key: {key}')
    result[key] = value
  return result


def _json(data: bytes) -> dict[str, Any]:
  try:
    value = json.loads(data, object_pairs_hook=_unique_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(ExperimentalCatalogError(f'Non-finite JSON: {value}')))
  except (ValueError, UnicodeError, RecursionError) as error:
    raise ExperimentalCatalogError(f'Invalid JSON: {error}') from error
  _require(isinstance(value, dict), 'JSON root must be an object')
  return value


def _json_file(path: Path) -> dict[str, Any]:
  with path.open('rb') as file:
    data = file.read(MAX_JSON_BYTES + 1)
  _require(len(data) <= MAX_JSON_BYTES, 'JSON file exceeds the bounded metadata size')
  return _json(data)


def load_experimental_catalog(manifest_path: str | Path = DEFAULT_CATALOG, *, experimental: bool) -> dict[str, Any]:
  """Inspect a wrapper; explicit experimental context is required even if disabled."""
  _require(experimental is True, 'Explicit experimental=True context is required')
  catalog = _json_file(Path(manifest_path))
  _require(catalog.get('schema_version') == 1 and catalog.get('scope') == SCOPE, 'Unsupported experimental catalog scope/version')
  _require(catalog.get('normal_activation_allowed') is False, 'Normal activation is prohibited')
  _require('bundles' not in catalog, 'Experimental wrappers must not expose production bundles')
  entries = catalog.get('entries', [])
  _require(isinstance(entries, list) and bool(entries), 'Catalog requires entries')
  _require(all(isinstance(entry, dict) and isinstance(entry.get('id'), str) for entry in entries), 'Invalid catalog entry')
  _require(len({entry['id'] for entry in entries}) == len(entries), 'Duplicate experimental model ID')
  return catalog


def _local_file(manifest: Path, record: Any, label: str) -> tuple[Path, str]:
  _require(isinstance(record, dict), f'{label} requires a local path and SHA256')
  name, expected = record.get('path'), record.get('sha256')
  _require(isinstance(name, str) and bool(name) and '://' not in name and '\x00' not in name,
           f'{label} must name a local file, never a URL')
  _require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None, f'{label} requires a real SHA256')
  path = Path(name)
  path = (path if path.is_absolute() else manifest.parent / path).resolve(strict=True)
  _require(path.is_file(), f'{label} is not a regular file')
  before, digest = path.stat(), hashlib.sha256()
  with path.open('rb') as file:
    while chunk := file.read(1 << 20):
      digest.update(chunk)
  after = path.stat()
  _require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
           f'{label} changed during hashing')
  _require(digest.hexdigest() == expected, f'{label} SHA256 mismatch')
  return path, expected


def _dtype(value: Any) -> str:
  _require(isinstance(value, str) and value in DTYPES, f'Unsupported dtype: {value}')
  return DTYPES[value]


def _shape(value: Any) -> tuple[int, ...]:
  _require(isinstance(value, list) and bool(value) and all(type(dim) is int and dim > 0 for dim in value), 'Invalid tensor shape')
  return tuple(value)


def _range(value: Any, blob_size: int) -> None:
  _require(isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value)
           and 0 <= value[0] <= value[1] <= blob_size, 'Invalid SPTG blob range')


def _header(path: Path, target: dict[str, str]) -> dict[str, Any]:
  with path.open('rb') as file:
    _require(file.read(8) == b'SPTGPLN1', 'Compiled artifact must use SPTGPLN1, not ONNX or legacy pickle')
    raw_size = file.read(8)
    _require(len(raw_size) == 8, 'Truncated SPTG header length')
    size = struct.unpack('<Q', raw_size)[0]
    _require(0 < size <= MAX_HEADER_BYTES, 'SPTG JSON header exceeds the bounded size')
    raw = file.read(size)
    _require(len(raw) == size, 'Truncated SPTG JSON header')
  container = _json(raw)
  blob_size = path.stat().st_size - (16 + size + -(16 + size) % 256)
  _require(blob_size >= 0, 'Truncated SPTG alignment padding')
  _range(container.get('extras'), blob_size)  # Validate bounds only; do not read or unpickle extras.
  parts = container.get('parts', [])
  _require(isinstance(parts, list) and len(parts) == 1 and parts[0].get('key') == 'run', 'Expected one native run part')
  _range(parts[0].get('blob'), blob_size)
  header = parts[0].get('header')
  _require(isinstance(header, dict), 'Missing native SPTG header')
  for field in ('buffers', 'inputs'):
    buffers = header.get(field, [])
    _require(isinstance(buffers, list) and all(isinstance(buffer, dict) and buffer.get('device') in ('AMD', 'NPY')
                                             for buffer in buffers),
             'SPTG internal buffers must use only the approved AMD or NPY devices')
  part_size = parts[0]['blob'][1] - parts[0]['blob'][0]

  def ranges(value: Any) -> None:
    if isinstance(value, dict):
      for key, child in value.items():
        if key in ('lib', 'init', 'data'):
          _range(child, part_size)
        else:
          ranges(child)
    elif isinstance(value, list):
      for child in value:
        ranges(child)
  ranges(header)
  calls = header.get('calls')
  _require(isinstance(calls, list) and bool(calls), 'SPTG requires captured calls')
  programs = [call for call in calls if call.get('kind') == 'program']
  _require(bool(programs) and all(call.get('kind') in ('program', 'copy') for call in calls), 'Unsupported SPTG call kind')
  for call in programs:
    actual = call.get('target', {})
    _require(actual.get('device') == target['backend'] and actual.get('arch') == target['arch']
             and actual.get('interface') == target['interface'] and actual.get('renderer') == 'LLVM',
             'SPTG kernel target does not match AMD/architecture/interface/LLVM')
  return header


def _check_tensor(actual: dict, expected: dict, *, flattened: bool = False) -> None:
  _require(_dtype(actual.get('dtype')) == _dtype(expected.get('dtype')), 'Tensor dtype does not match contract')
  if flattened:
    _require(type(actual.get('numel')) is int and actual['numel'] == math.prod(_shape(expected.get('shape'))),
             'Tensor element count does not match contract')
  else:
    _require(_shape(actual.get('shape')) == _shape(expected.get('shape')), 'Tensor shape does not match contract')
    if 'data' in actual:
      size = math.prod(_shape(expected['shape'])) * (4 if _dtype(expected['dtype']) == 'float32' else 1)
      _require(actual['data'][1] - actual['data'][0] == size, 'Captured array byte count does not match tensor shape/dtype')


def _model_contract(header: dict, metadata: dict, contract: dict) -> dict[str, tuple[tuple[int, ...], str, str]]:
  specs: dict[str, tuple[tuple[int, ...], str, str]] = {}
  for kind in ('input', 'output'):
    expected = {entry['name']: entry for entry in contract[f'{kind}s']}
    provided = metadata.get(f'{kind}_specs', {})
    _require(isinstance(provided, dict) and list(provided) == list(expected), f'{kind} names/order do not match public contract')
    for name, tensor in expected.items():
      _check_tensor(provided[name], tensor)
      device = provided[name].get('device')
      _require(device in ('AMD', 'NPY') if kind == 'input' else device == 'AMD', 'Invalid tensor device')
      if kind == 'input':
        specs[name] = (_shape(tensor['shape']), _dtype(tensor['dtype']), device)
  _require(metadata.get('output_slices') == contract['output_slices'], 'Output slice contract mismatch')
  inputs = header.get('inputs', [])
  _require(isinstance(inputs, list), 'Missing captured inputs')
  slots = [entry.get('slot') for entry in inputs]
  _require(all(type(slot) is int and slot >= 0 for slot in slots) and len(set(slots)) == len(slots), 'Invalid or duplicate native input slot')
  names = header.get('input_names', [])
  _require(isinstance(names, list) and len(names) == len(specs) and set(names) == set(specs), 'Native input binding names mismatch')
  named = {entry['name']: entry for entry in inputs if entry.get('name') is not None}
  _require(len(named) == len(specs) and set(named) == set(specs), 'Captured input names do not match contract')
  outputs = sorted((entry for entry in inputs if entry.get('name') is None), key=lambda entry: entry['slot'])
  _require(len(inputs) == len(specs) + len(outputs) and len(outputs) == len(metadata['output_specs']), 'Captured output count mismatch')
  for name, entry in named.items():
    _require(entry['slot'] == names.index(name), 'Native named input slot mismatch')
    _check_tensor(entry, metadata['input_specs'][name], flattened=True)
    _require(entry.get('device') == specs[name][2], 'Captured input device mismatch')
  for index, (entry, tensor) in enumerate(zip(outputs, metadata['output_specs'].values(), strict=True)):
    _require(entry['slot'] == len(names) + index, 'Native output slot order mismatch')
    _check_tensor(entry, tensor, flattened=True)
    _require(entry.get('device') == 'AMD', 'Captured output device mismatch')
  _require(header.get('returns') == {'kind': 'NoneType', 'tensors': []}, 'Expected native output-buffer model')
  check = header.get('check', {})
  _require(isinstance(check, dict) and check.get('output_container') == 'output_buffers', 'Missing native captured output schema')
  captured = check.get('inputs', [])
  expected_paths: dict[tuple[str, ...], dict] = {(name,): spec for name, spec in metadata['input_specs'].items()}
  expected_paths |= {('output_buffers', name): spec for name, spec in metadata['output_specs'].items()}
  _require(isinstance(captured, list) and len(captured) == len(expected_paths), 'Captured tensor metadata count mismatch')
  seen: set[tuple[str, ...]] = set()
  for item in captured:
    _require(isinstance(item, list) and len(item) == 3 and isinstance(item[0], list), 'Invalid captured tensor metadata')
    path, tensor, device = tuple(item[0]), item[1], item[2]
    _require(path in expected_paths and path not in seen, 'Unexpected or duplicate captured tensor path')
    seen.add(path)
    _check_tensor(tensor, expected_paths[path])
    _require(device == expected_paths[path]['device'], 'Captured tensor metadata device mismatch')
  _require(len(check.get('outputs', [])) == len(outputs), 'Captured prediction count mismatch')
  for tensor, expected in zip(check['outputs'], metadata['output_specs'].values(), strict=True):
    _check_tensor(tensor, expected)
  return specs


def _warp_contract(header: dict) -> int:
  inputs = header.get('inputs', [])
  named = {entry.get('name'): entry for entry in inputs}
  _require(len(inputs) == 2 and set(named) == {'M_inv', 'input_frame'}, 'Warp requires native M_inv/input_frame inputs')
  _check_tensor(named['M_inv'], {'shape': [2, 3, 3], 'dtype': 'float32'}, flattened=True)
  frame = named['input_frame']
  _require(_dtype(frame.get('dtype')) == 'uint8' and type(frame.get('numel')) is int
           and frame['numel'] > 0 and frame['numel'] % 2 == 0, 'Warp frame buffer must contain two uint8 frames')
  _require(all(entry.get('device') == 'AMD' for entry in inputs), 'Warp inputs must target AMD')
  returns = header.get('returns', {})
  _require(returns.get('kind') == 'Tensor' and len(returns.get('tensors', [])) == 1, 'Warp must return one image tensor')
  _check_tensor(returns['tensors'][0], {'shape': [2, 6, 128, 256], 'dtype': 'uint8'})
  return frame['numel'] // 2


def load_experimental_model(manifest_path: str | Path, *, experimental: bool, backend: str,
                            expected_arch: str = 'gfx1200', model_id: str = 'big-nav-v2') -> ExperimentalModelEntry:
  """Validate local static contracts for parsed-tensor diagnostics, never actuation.

  The checked-in template is intentionally disabled. A compiler-produced local
  manifest must supply real model/warp/JSON-sidecar paths and their real digests.
  Generation 0 remains unverified and may not authorize model action conversion.
  This verifies admission data only; the caller must additionally enforce its
  file-only/non-actuating session and validate compiler parity before inference.
  """
  try:
    manifest = Path(manifest_path).resolve(strict=True)
    catalog = load_experimental_catalog(manifest, experimental=experimental)
    entries = [entry for entry in catalog['entries'] if entry['id'] == model_id]
    _require(len(entries) == 1, f'Unknown experimental model: {model_id}')
    entry = entries[0]
    _require(entry.get('enabled') is True, 'Experimental model template is disabled until local artifacts are verified')
    _require(entry.get('normal_activation_allowed') is False and entry.get('selector_generation_verified') is False,
             'Normal activation/generation semantics must remain explicitly unverified')
    target = entry.get('target')
    _require(backend == 'AMD' and expected_arch == 'gfx1200'
             and target == {'backend': backend, 'arch': expected_arch, 'interface': 'USB'}, 'Only explicit AMD gfx1200 USB diagnostics are supported')
    _require(entry.get('source_onnx', {}).get('sha256') == SOURCE_ONNX_SHA256, 'Unapproved source ONNX digest')
    _require(entry.get('runtime', {}).get('tinygrad_ref') == TINYGRAD_REF, 'Pinned tinygrad revision mismatch')
    contract = entry.get('contract')
    _require(hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == CONTRACT_SHA256,
             'Public Bigv2 contract has changed')
    bundle = entry.get('selector21_bundle', {})
    _require(bundle.get('minimum_selector_version') == 21 and bundle.get('runner') == 'tinygrad'
             and bundle.get('generation') == 0 and bundle.get('models') == [], 'Invalid isolated Selector 21 bundle template')
    artifact, artifact_hash = _local_file(manifest, entry.get('compiled_artifact'), 'Compiled artifact')
    warp, warp_hash = _local_file(manifest, entry.get('warp_artifact'), 'Warp artifact')
    metadata_path, metadata_hash = _local_file(manifest, entry.get('compiler_metadata'), 'Compiler metadata')
    _require(artifact_hash != SOURCE_ONNX_SHA256 and artifact != warp, 'Source ONNX, compiled model and warp must remain distinct')
    metadata = _json_file(metadata_path)
    _require(metadata.get('schema_version') == 1 and metadata.get('source_onnx_sha256') == SOURCE_ONNX_SHA256
             and metadata.get('tinygrad_ref') == TINYGRAD_REF and metadata.get('target') == target, 'Compiler metadata provenance/target mismatch')
    model_header = _header(artifact, target)
    specs = _model_contract(model_header, metadata, contract)
    frame_size = _warp_contract(_header(warp, target))
    verified = False
    if 'compiler_verification' in metadata:
      from openpilot.selfdrive.navigation.export_experimental_model import validate_exported_verification
      validate_exported_verification(metadata['compiler_verification'], artifact_hash, contract, model_header)
      verified = True
    # This schema-compatible object is returned for diagnostic inspection only.
    # It is never saved to a production catalog or passed to an active manager.
    diagnostic_bundle = bundle | {'models': [{'type': 'chunked', 'artifact': {
      'file_name': artifact.name, 'download_uri': {'url': artifact.as_uri(), 'sha256': artifact_hash}}}]}
    return ExperimentalModelEntry(model_id, artifact, warp, metadata, specs, diagnostic_bundle,
                                  artifact_hash, warp_hash, metadata_hash, frame_size, target,
                                  compiler_verification_validated=verified)
  except (OSError, KeyError, TypeError, AttributeError, struct.error) as error:
    raise ExperimentalCatalogError(f'Invalid local experimental manifest/artifact: {error}') from error
