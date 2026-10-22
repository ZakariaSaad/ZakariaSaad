"""Read-only Bigv2 software/artifact compatibility; no model or device execution.

Run with installed Python: PYTHONDONTWRITEBYTECODE=1 python3 -B -m
openpilot.selfdrive.navigation.software_compatibility. Only an explicitly named
--output is written. Reports distinguish byte/provenance checks from reviewed
regressions and runtime properties that static inspection cannot establish.
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

from openpilot.common.git_inspection import read_only_git
from openpilot.selfdrive.navigation.experimental_catalog import (
  CONTRACT_SHA256, MAX_HEADER_BYTES, SOURCE_ONNX_SHA256, TINYGRAD_REF, _header, _json, _json_file, _local_file,
  _model_contract, _range, _require, _warp_contract, load_experimental_catalog,
)
from openpilot.selfdrive.navigation.export_experimental_model import validate_exported_verification

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = Path(__file__).with_name('driving_models.json')
DEFAULT_RECEIPT = Path('/data/navigation-models/latest-registration.json')
DEFAULT_PARAMS = Path('/data/params/d')
DEFAULT_MODELS = Path('/data/media/0/models')
# Primitive native buffer encodings supported by the pinned loader, never eval.
_ITEMSIZE = {'bool': 1, 'u8': 1, 'uint8': 1, 'i8': 1, 'int8': 1,
             'u16': 2, 'uint16': 2, 'i16': 2, 'int16': 2, 'f16': 2, 'float16': 2, 'bf16': 2, 'bfloat16': 2,
             'u32': 4, 'uint32': 4, 'i32': 4, 'int32': 4, 'f32': 4, 'float32': 4,
             'u64': 8, 'uint64': 8, 'i64': 8, 'int64': 8, 'f64': 8, 'float64': 8,
             'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2, 'half': 2, 'int': 4, 'uint': 4, 'float': 4, 'long': 8, 'ulong': 8, 'double': 8}
_RUNTIME_LIMITS = [
  'This command does not execute model kernels or measure inference latency, memory residency or dropped frames.',
  'The earlier physical AMD compiler report proves source/reload consistency for its recorded seeds, not live camera inference.',
  'Live camera geometry, route freshness and timestamp epochs are checked by the production inference path, not observed here.',
  'Live camera-to-prediction operation at 20 Hz and actual big-model failure recovery remain unmeasured on this device.',
]


def _digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as source:
    while data := source.read(1 << 20):
      digest.update(data)
  return digest.hexdigest()


def _itemsize(dtype: Any) -> int:
  _require(isinstance(dtype, str) and dtype.startswith('dtypes.') and dtype[7:] in _ITEMSIZE,
           f'Unsupported native buffer dtype: {dtype}')
  return _ITEMSIZE[dtype[7:]]


def validate_native_layout(header: dict) -> dict:
  """Inspect the exact native-loader argument views and kernel declarations.

  Binary kernels and pickle extras are never read as executable objects. This
  proves range/type/slot consistency, not generated machine-code correctness.
  """
  buffers, inputs, calls = header['buffers'], header['inputs'], header['calls']
  slots = {row['slot']: row for row in inputs}
  _require(len(slots) == len(inputs), 'Duplicate native input slot')
  for row in buffers:
    size = _itemsize(row.get('dtype'))
    _require(type(row.get('nbytes')) is int and row['nbytes'] > 0 and row['nbytes'] % size == 0,
             'Native buffer size/dtype alignment mismatch')
    if 'init' in row:
      _require(row['init'][1] - row['init'][0] == row['nbytes'], 'Native buffer initialization size mismatch')
  for row in inputs:
    _require(type(row.get('numel')) is int and row['numel'] > 0, 'Native input extent must be positive')
    _itemsize(row.get('dtype'))

  def argument(record: Any) -> int:
    _require(isinstance(record, dict), 'Native argument must be an object')
    if set(record) == {'input'}:
      _require(type(record['input']) is int and record['input'] in slots, 'Native argument references an unknown input slot')
      row = slots[record['input']]
      return row['numel'] * _itemsize(row['dtype'])
    _require(set(record) == {'buffer', 'offset', 'nbytes', 'dtype'}, 'Unsupported native argument view')
    index, offset, extent = record['buffer'], record['offset'], record['nbytes']
    size = _itemsize(record['dtype'])
    _require(type(index) is int and 0 <= index < len(buffers), 'Native argument references an unknown buffer')
    _require(type(offset) is int and type(extent) is int and offset >= 0 and extent > 0
             and offset % size == 0 and extent % size == 0 and offset + extent <= buffers[index]['nbytes'],
             'Native argument view exceeds its buffer or dtype alignment')
    return extent

  programs = 0
  for call in calls:
    args = call.get('args')
    _require(isinstance(args, list) and bool(args), 'Native call requires argument bindings')
    extents = [argument(row) for row in args]
    if call['kind'] == 'copy':
      _require(len(args) == 2 and extents[0] == extents[1], 'Native copy buffer extents mismatch')
      continue
    _require(call['kind'] == 'program', 'Unknown native call kind')
    programs += 1
    _require(isinstance(call.get('name'), str) and bool(call['name']), 'Native kernel name is missing')
    _require(call['lib'][1] > call['lib'][0], 'Native kernel binary is empty')
    for key in ('global_size', 'local_size'):
      value = call.get(key)
      _require(isinstance(value, list) and len(value) == 3 and all(type(dim) is int and dim > 0 for dim in value),
               'Invalid native kernel launch dimensions')
    globals_ = call.get('globals')
    # serialize_jit preserves every outer call argument, while ProgramInfo.globals
    # selects the arguments used by the compiled kernel. ELF signature slots are
    # compact indices into that selection (helpers._program_uop), not args.
    _require(isinstance(globals_, list) and all(type(index) is int and 0 <= index < len(args) for index in globals_)
             and len(globals_) == len(set(globals_)), 'Native kernel global argument mapping mismatch')
    for key in ('outs', 'ins'):
      indices = call.get(key)
      _require(isinstance(indices, list) and len(indices) == len(set(indices))
               and all(type(index) is int and 0 <= index < len(args) for index in indices), 'Native kernel access index mismatch')
    signature = call.get('signature')
    _require(isinstance(signature, list) and len(signature) == len(globals_), 'Native kernel signature count mismatch')
    seen = set()
    for entry in signature:
      _require(isinstance(entry, list) and len(entry) == 4, 'Invalid native kernel signature entry')
      name, slot, dtype, shape = entry
      _require((name is None or isinstance(name, str)) and type(slot) is int and 0 <= slot < len(globals_) and slot not in seen,
               'Native kernel signature slot mismatch')
      seen.add(slot)
      _require(isinstance(shape, list) and bool(shape) and all(type(dim) is int and dim > 0 for dim in shape),
               'Invalid native kernel signature shape')
      _require(math.prod(shape) * _itemsize(dtype) <= extents[globals_[slot]], 'Native kernel signature exceeds its argument view')
  for returned in header.get('returns', {}).get('tensors', []):
    argument({key: value for key, value in returned.items() if key != 'shape'})
  return {'programs': programs, 'copies': len(calls) - programs, 'internal_buffers': len(buffers),
          'internal_buffer_bytes': sum(row['nbytes'] for row in buffers), 'input_output_slots': len(inputs)}


def _registered_entry(receipt: dict, manifest: Path, registry: dict) -> tuple[dict, dict, Path, Path, Path]:
  _require(receipt.get('prepared') is True and receipt.get('registered') is True, 'Receipt does not confirm successful registration')
  _require(receipt.get('model_id') == 'big-nav-v2' and receipt.get('source_onnx_sha256') == SOURCE_ONNX_SHA256,
           'Receipt source/checkpoint identity mismatch')
  _require(receipt.get('status') == 'registered_disabled' and receipt.get('activation_allowed') is False,
           'Registration receipt must not claim driving activation')
  _require(Path(receipt.get('manifest_path', '')).resolve(strict=True) == manifest, 'Receipt manifest path mismatch')
  catalog = load_experimental_catalog(manifest, experimental=True)
  entries = [row for row in catalog['entries'] if row['id'] == receipt['model_id']]
  _require(len(entries) == 1 and entries[0].get('enabled') is True, 'Registered Bigv2 manifest entry is missing or disabled')
  entry = entries[0]
  _require(entry.get('source_onnx', {}).get('sha256') == SOURCE_ONNX_SHA256
           and entry.get('runtime', {}).get('tinygrad_ref') == TINYGRAD_REF, 'Registered source/runtime identity mismatch')
  contract = entry.get('contract')
  _require(hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == CONTRACT_SHA256,
           'Registered public Bigv2 contract mismatch')
  paths = []
  for record, key in (('compiled_artifact', 'artifact_sha256'), ('warp_artifact', 'warp_sha256'), ('compiler_metadata', 'metadata_sha256')):
    _require(entry.get(record, {}).get('sha256') == receipt.get(key) == registry.get(key), f'Receipt/branch-reviewed {key} mismatch')
    path, _ = _local_file(manifest, entry[record], record)
    paths.append(path)
  _require(len(set(paths)) == 3, 'Model, warp and metadata must be separate files')
  _require(entry.get('target') == {'backend': 'AMD', 'arch': 'gfx1200', 'interface': 'USB'}, 'Unsupported registered target')
  metadata = _json_file(paths[2])
  _require(metadata.get('schema_version') == 1 and metadata.get('source_onnx_sha256') == SOURCE_ONNX_SHA256
           and metadata.get('tinygrad_ref') == TINYGRAD_REF and metadata.get('target') == entry['target'],
           'Registered metadata provenance mismatch')
  return entry, metadata, paths[0], paths[1], paths[2]


def _camera_profile(profile: Any, receipt: dict, warp_frame_size: int, root: Path) -> None:
  """Check the registered OS04 saved-capture identity without importing NumPy."""
  expected_camera = {'width': 1344, 'height': 760, 'stride': 1408, 'y_height': 768, 'uv_height': 384,
                     'copied_frame_bytes': 1622016, 'allocated_frame_bytes': 2428928}
  _require(isinstance(profile, dict) and profile == receipt.get('camera_profile'), 'Receipt/manifest camera profile mismatch')
  _require(profile.get('artifact_sha256') == receipt['warp_sha256']
           and profile.get('tinygrad_ref') == TINYGRAD_REF and profile.get('camera') == expected_camera,
           'Registered camera profile identity/geometry mismatch')
  for key in ('width', 'height', 'stride', 'y_height', 'uv_height'):
    _require(type(profile.get(key)) is int and profile[key] == expected_camera[key], 'Registered camera dimensions/layout mismatch')
  _require(profile.get('frame_size') == warp_frame_size == 1622016 and profile.get('allocation_size') == 2428928,
           'Registered camera copied/allocation extent mismatch')
  _require(profile.get('target') == {'backend': 'AMD', 'arch': 'gfx1200', 'interface': 'USB', 'renderer': 'LLVM'}
           and profile.get('output_shape') == [2, 6, 128, 256] and profile.get('output_dtype') == 'uint8',
           'Registered camera model tensor/target mismatch')
  capture = profile.get('captured_reference', {})
  _require(profile.get('scope') == 'branch_pinned_warp_static_and_saved_capture' and profile.get('captured_check_passed') is True
           and capture == {'passed': True, 'output_bytes': 393216, 'differing_bytes': 0,
                           'output_sha256': '124d4a811290f1c4bc1bbf0d922a80ca870a72e46cfea909f6431c97f46d72b1'},
           'Registered camera saved-capture evidence mismatch')
  bindings = profile.get('source_bindings')
  _require(isinstance(bindings, dict) and len(bindings) == 3, 'Registered camera source bindings are missing')
  expected_bindings = {
    'openpilot/sunnypilot/modeld_v2/compile_warp.py': 'a182425b51918b35de8b7317461c4386e465bc5a4c9064a39e4ff64e29ca0282',
    'openpilot/system/camerad/cameras/nv12_info.py': 'b3f39bbfdd75c43d396b01f746d460068aa8e98862217d46bf3cd426cd034e1a',
    'tinygrad:examples/openpilot/compile_warp.py': '97a6814fde7f16e1ce265c6b6388cd5799c8e2f8969ac78cfc4fd5a929b1921d',
  }
  _require(bindings == expected_bindings, 'Registered camera recipe identity mismatch')
  for name, expected in bindings.items():
    path = root / ('tinygrad_repo/' + name[9:] if name.startswith('tinygrad:') else name)
    _require(_digest(path) == expected, 'Installed camera source binding changed: ' + name)


def _fallback_part_key(value: Any) -> str | tuple[int, int]:
  """Mirror write_bundle/read_parts for ordinary compiler entrypoint keys.

  JSON arrays represent camera-resolution tuples, not string entrypoints. Keep
  integer dimensions strict: bools and floats would otherwise alias integer
  tuples during Python dictionary lookup and duplicate detection.
  """
  if isinstance(value, str):
    _require(value in ('run', 'run_policy'), f'Unsupported ordinary fallback entrypoint: {value!r}')
    return value
  _require(isinstance(value, list) and len(value) == 2 and all(type(size) is int and size > 0 for size in value),
           f'Invalid ordinary fallback camera-resolution key: {value!r}')
  return value[0], value[1]


def _fallback_headers(path: Path) -> list[dict]:
  """Bounded data-only QCOM inspection of native and legacy combined bundles.

  Native compile_modeld_v2 artifacts contain run. Legacy compile_modeld bundles
  contain run_policy (including vision/multiple policies) and resolution-keyed
  warp JITs. Metadata pickle extras and compiled kernels are never executed.
  """
  with path.open('rb') as source:
    _require(source.read(8) == b'SPTGPLN1', 'Configured ordinary fallback is not installed in native SPTG format')
    size_bytes = source.read(8)
    _require(len(size_bytes) == 8, 'Truncated ordinary fallback header length')
    size = struct.unpack('<Q', size_bytes)[0]
    _require(0 < size <= MAX_HEADER_BYTES, 'Ordinary fallback header exceeds bounded size')
    raw = source.read(size)
    _require(len(raw) == size, 'Truncated ordinary fallback JSON header')
  document = _json(raw)
  blob_size = path.stat().st_size - (16 + size + -(16 + size) % 256)
  _require(blob_size >= 0, 'Truncated ordinary fallback alignment padding')
  _range(document.get('extras'), blob_size)
  parts = document.get('parts')
  _require(isinstance(parts, list) and bool(parts), 'Ordinary fallback has no native model parts')
  assert isinstance(parts, list)
  keys, headers = set(), []
  for part in parts:
    _require(isinstance(part, dict), 'Invalid ordinary fallback part declaration')
    key = _fallback_part_key(part.get('key'))
    _require(key not in keys, f'Duplicate ordinary fallback part: {key!r}')
    keys.add(key)
    _range(part.get('blob'), blob_size)
    part_size = part['blob'][1] - part['blob'][0]
    header = part.get('header')
    _require(isinstance(header, dict), 'Missing ordinary fallback native header')
    def ranges(value, extent=part_size):
      if isinstance(value, dict):
        for key, child in value.items():
          if key in ('lib', 'init', 'data'):
            _range(child, extent)
          else:
            ranges(child, extent)
      elif isinstance(value, list):
        for child in value:
          ranges(child, extent)
    ranges(header)
    for row in header.get('buffers', []) + header.get('inputs', []):
      _require(row.get('device') in ('QCOM', 'NPY'), 'Ordinary small fallback must use QCOM/NPY buffers')
    names = header.get('input_names')
    _require(isinstance(names, list) and bool(names) and len(names) == len(set(names))
             and all(isinstance(name, str) for name in names) and not {'nav_map', 'navigation'} & set(names),
             'Ordinary fallback unexpectedly requires navigation inputs')
    calls = header.get('calls')
    _require(isinstance(calls, list) and bool(calls), 'Ordinary fallback has no captured calls')
    for call in calls:
      _require(call.get('kind') in ('program', 'copy'), 'Unknown ordinary fallback call kind')
      if call['kind'] == 'program':
        _require(call.get('target', {}).get('device') == 'QCOM', 'Ordinary fallback kernel must target QCOM')
    validate_native_layout(header)
    headers.append(header)
  # Legacy vision_policy, vision_multi_policy and supercombo graphs are already
  # combined inside run_policy; there is no separate run_vision entrypoint.
  # A tuple-only bundle is not an ordinary QCOM model: LegacyModelAdapter.run
  # needs both its camera warp and run_policy. Native run uses frame_fusion's
  # external warp and must not be confused with the legacy layout.
  native = keys == {'run'}
  legacy = 'run_policy' in keys and len(keys) > 1 and all(key == 'run_policy' or isinstance(key, tuple) for key in keys)
  _require(native or legacy, 'Ordinary fallback requires native run or run_policy plus camera-resolution warp parts')
  return headers


def _fallback(params_dir: Path, model_root: Path) -> dict:
  bundle = _json_file(params_dir / 'ModelManager_ActiveBundle')
  _require(bundle.get('minimumSelectorVersion') == 21 and bundle.get('runner') == 'tinygrad', 'Ordinary small fallback requires Selector21 tinygrad')
  models = bundle.get('models')
  _require(isinstance(models, list) and bool(models), 'Configured ordinary fallback is empty')
  assert isinstance(models, list)
  artifact = models[0].get('artifact', {})
  name = artifact.get('fileName')
  _require(isinstance(name, str) and name and Path(name).name == name, 'Invalid ordinary fallback artifact name')
  path = model_root / name
  expected = artifact.get('downloadUri', {}).get('sha256')
  _require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
           'Configured ordinary fallback requires its catalog SHA256')
  _require(_digest(path) == expected, 'Configured ordinary fallback artifact SHA256 mismatch')
  headers = _fallback_headers(path)
  return {'artifact_path': str(path), 'artifact_sha256': expected, 'native_format': True, 'qcom_native_parts': len(headers),
          'scope': 'Configured small slot, exact catalog digest, QCOM kernel/buffer layout; inference and pickle metadata not loaded'}


def validate_software_compatibility(receipt_path: str | Path = DEFAULT_RECEIPT, *, manifest_path: str | Path | None = None,
                                    registry_path: str | Path = REGISTRY, checkout: str | Path = ROOT,
                                    params_dir: str | Path = DEFAULT_PARAMS, model_root: str | Path = DEFAULT_MODELS) -> dict:
  """Read installed files using the checkout owner's identity; never write Params."""
  checks: list[dict] = []
  report: dict[str, Any] = {'schema_version': 1, 'scope': 'software_and_artifact_static_compatibility',
                          'passed': False, 'status': 'software_compatibility_failed', 'checks': checks, 'runtime_compatibility': 'unverified',
                          'activation_allowed': False, 'gpu_execution_performed': False, 'model_recompiled': False,
                          'models_downloaded': False, 'normal_service_modified': False, 'params_modified': False,
                          'runtime_limitations': list(_RUNTIME_LIMITS)}
  context: dict[str, Any] = {}

  def check(name, function):
    try:
      detail = function()
      checks.append({'name': name, 'status': 'passed', 'details': detail})
      return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
      checks.append({'name': name, 'status': 'failed', 'reason': str(error)})
      return False

  root, params, models_root = Path(checkout).resolve(), Path(params_dir), Path(model_root)
  def owner_and_source():
    _require(os.geteuid() != 0 and os.geteuid() == root.stat().st_uid,
             'Run this file-only validator as the repository owner (comma), without sudo')
    source = read_only_git(root, 'rev-parse', 'HEAD')
    report['source_sha'] = source
    report['source_dirty'] = bool(read_only_git(root, 'status', '--porcelain', '--untracked-files=no'))
    return {'source_sha': source, 'source_dirty': report['source_dirty'], 'uid': os.geteuid()}
  if not check('owner_and_source', owner_and_source):
    return report

  def receipt_and_manifest():
    receipt = _json_file(Path(receipt_path))
    registry = _json_file(Path(registry_path))
    candidates = [entry for entry in registry.get('candidates', []) if entry.get('id') == 'big-nav-v2']
    _require(registry.get('schema_version') == 1 and len(candidates) == 1, 'Branch-reviewed Bigv2 registry entry is missing')
    approved = candidates[0]
    manifest = Path(manifest_path if manifest_path is not None else receipt.get('manifest_path', '')).resolve(strict=True)
    entry, metadata, artifact, warp, metadata_path = _registered_entry(receipt, manifest, approved)
    context.update(receipt=receipt, approved=approved, manifest=manifest, entry=entry, metadata=metadata,
                   artifact=artifact, warp=warp, metadata_path=metadata_path)
    report.update(model_id='big-nav-v2', manifest_path=str(manifest), artifact_sha256=receipt['artifact_sha256'],
                  warp_sha256=receipt['warp_sha256'], metadata_sha256=receipt['metadata_sha256'],
                  source_onnx_sha256=SOURCE_ONNX_SHA256)
    return {'registered': True, 'artifact_bytes': artifact.stat().st_size,
            'receipt_registration_gpu_execution': receipt.get('gpu_execution_performed'),
            'source_bytes_rechecked': False}
  if not check('registered_files_and_hashes', receipt_and_manifest):
    return report

  def native_contract():
    entry, metadata = context['entry'], context['metadata']
    header = _header(context['artifact'], entry['target'])
    specs = _model_contract(header, metadata, entry['contract'])
    layout = validate_native_layout(header)
    states = {name: 'next_' + name for name in specs if name.startswith('state_')}
    _require(len(states) == 5, 'Bigv2 requires all five recurrent state pairs')
    for before, after in states.items():
      a, b = metadata['input_specs'][before], metadata['output_specs'][after]
      _require(a['shape'] == b['shape'] and a['dtype'] == b['dtype'], 'Recurrent state input/output shape or dtype mismatch')
    context['header'] = header
    return layout | {'recurrent_pairs': states, 'action_contract': context['approved']['action_contract'],
                     'execution_hz': 20, 'output_elements': 18580, 'action_slice': [2062, 2066],
                     'kernel_correctness_proven': False}
  native_ok = check('native_tensor_kernel_and_recurrent_contract', native_contract)

  def camera():
    header = _header(context['warp'], context['entry']['target'])
    frame_size = _warp_contract(header)
    layout = validate_native_layout(header)
    _camera_profile(context['metadata'].get('camera_profile'), context['receipt'], frame_size, root)
    return layout | {'camera': context['metadata']['camera_profile']['camera'], 'saved_capture_previously_passed': True,
                     'saved_capture_reexecuted': False, 'live_camera_observed': False}
  check('registered_camera_warp', camera)

  def compiler_proof():
    proof = context['metadata'].get('compiler_verification')
    _require(isinstance(proof, dict), 'Registered compiler verification is missing')
    for key, expected_key in (('compiler', 'compiler_sha256'), ('report', 'compiler_report_sha256')):
      _require(proof.get(key, {}).get('sha256') == context['approved'].get(expected_key), 'Branch-reviewed compiler/report identity mismatch')
    validate_exported_verification(proof, context['receipt']['artifact_sha256'], context['entry']['contract'],
                                    context['header'], anchor=context['metadata_path'])
    report_path, _ = _local_file(context['metadata_path'], proof['report'], 'Compiler report')
    compiler = _json_file(report_path)
    return {'physical_execution_previously_verified': True, 'verification_seeds': compiler['verification']['seeds'],
            'independent_buffers': True, 'all_prediction_and_state_outputs_verified': True,
            'source_reload_verification_scope': 'Recorded synthetic seeds; not live inference timing',
            'compiler_sha256': proof['compiler']['sha256'], 'compiler_report_sha256': proof['report']['sha256'],
            'tinygrad_ref': TINYGRAD_REF}
  if native_ok:
    check('compiler_provenance_and_source_reload_evidence', compiler_proof)
  else:
    checks.append({'name': 'compiler_provenance_and_source_reload_evidence', 'status': 'blocked', 'reason': 'Native contract check failed'})
  check('configured_ordinary_small_fallback', lambda: _fallback(params, models_root))

  def reviewed_integration():
    approved = context['approved']
    _require(not report.get('source_dirty'), 'Installed tracked source differs from the reviewed commit')
    _require(approved.get('software_contract_reviewed') is True and approved.get('software_integration_reviewed') is True
             and approved.get('artifact_evidence_reviewed') is True, 'Branch software/integration/artifact review is incomplete')
    _require(approved.get('sunnypilot_generation') is None
             and approved.get('action_contract') == 'comma-nav-v2-20hz-lateral-acceleration-longitudinal-acceleration',
             'Reviewed native 20 Hz action contract mismatch')
    return {'scope': 'Branch-reviewed automated regressions; not rerun by this file-only command',
            'navigation_tensor_contract': {'map': [1, 1, 256, 256], 'map_dtype': 'float32', 'map_range': [0, 255],
                                           'normalization': 'Inside ONNX graph', 'navigation': [1, 47]},
            'execution_policy': 'single_selected_model', 'fallback_policy': 'initialized_for_failure_recovery',
            'runtime_compatibility': approved.get('runtime_compatibility', 'unverified'),
            'branch_activation_approved': approved.get('activation_approved') is True}
  check('branch_reviewed_software_integration', reviewed_integration)
  report['passed'] = all(check['status'] == 'passed' for check in checks)
  report['status'] = 'software_compatible_driving_disabled' if report['passed'] else 'software_compatibility_failed'
  return report


def write_compatibility_report(report: dict, output: str | Path, *, receipt_path: str | Path = DEFAULT_RECEIPT,
                               checkout: str | Path = ROOT, params_dir: str | Path = DEFAULT_PARAMS,
                               model_root: str | Path = DEFAULT_MODELS) -> None:
  """Write only a report, never a registered input, checkout file or Params."""
  root, destination = Path(checkout).resolve(strict=True), Path(output)
  _require(os.geteuid() != 0 and os.geteuid() == root.stat().st_uid,
           'Save reports as the repository owner (comma), without sudo')
  _require(not destination.is_symlink(), 'Report output must not be a symlink')
  target = destination.parent.resolve(strict=True) / destination.name
  _require(target.suffix == '.json', 'Compatibility report output must use .json')
  prohibited = [root, Path(params_dir).resolve(), Path(model_root).resolve()]
  protected = {Path(receipt_path).resolve()}
  receipt = _json_file(Path(receipt_path)) if Path(receipt_path).is_file() else {}
  manifest_name = receipt.get('manifest_path')
  if isinstance(manifest_name, str) and manifest_name:
    manifest = Path(manifest_name).resolve()
    prohibited.append(manifest.parent)
    if manifest.is_file():
      catalog = _json_file(manifest)
      for entry in catalog.get('entries', []):
        for name in ('compiled_artifact', 'warp_artifact', 'compiler_metadata'):
          record = entry.get(name, {})
          if isinstance(record, dict) and isinstance(record.get('path'), str):
            recorded = Path(record['path'])
            protected.add((recorded if recorded.is_absolute() else manifest.parent / recorded).resolve())
  _require(target not in protected and not any(target == path or path in target.parents for path in prohibited),
           'Report output collides with a registered input, model, checkout or Params')
  if target.exists():
    _require(target.is_file() and _json_file(target).get('scope') == 'software_and_artifact_static_compatibility',
             'Existing output is not a software compatibility report')
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix='.software-compatibility-', suffix='.json', delete=False) as stream:
      temporary = Path(stream.name)
      stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, target)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
  parser.add_argument('--manifest', type=Path, help='Optional explicit manifest; must equal the registered receipt path')
  parser.add_argument('--params-dir', type=Path, default=DEFAULT_PARAMS)
  parser.add_argument('--model-root', type=Path, default=DEFAULT_MODELS)
  parser.add_argument('--output', type=Path, help='Only this explicitly requested JSON report is written')
  args = parser.parse_args()
  report = validate_software_compatibility(args.receipt, manifest_path=args.manifest, params_dir=args.params_dir, model_root=args.model_root)
  text = json.dumps(report, indent=2, allow_nan=False) + '\n'
  print(text, end='', flush=True)
  if args.output:
    try:
      write_compatibility_report(report, args.output, receipt_path=args.receipt, params_dir=args.params_dir, model_root=args.model_root)
    except (OSError, ValueError) as error:
      parser.exit(2, 'Cannot save compatibility report: ' + str(error) + '\n')
  raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
  main()
