"""Data-only validation of the existing, branch-pinned Chestnut camera warp.

This never loads pickle extras, kernels, devices or camera streams. It binds a
known warp's full bytes to a reviewed camera geometry and checks its saved NV12
capture against a NumPy implementation of the pinned packing/warp recipe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import numpy as np


from openpilot.common.basedir import BASEDIR
from openpilot.common.git_inspection import GitInspectionError, read_only_git
from openpilot.selfdrive.navigation.experimental_catalog import (
  ExperimentalCatalogError, MAX_HEADER_BYTES, TINYGRAD_REF, _check_tensor, _dtype, _header, _require, _warp_contract,
)
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

TARGET = {'backend': 'AMD', 'arch': 'gfx1200', 'interface': 'USB'}
_PINNED_WARPS = {
  (1344, 760): 'f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b',
  (1928, 1208): '56cc9690e0e9fb719323d2c6646c1305e8bc6bbdfdc807724a605a0d0a37554b',
}
_REVIEWED_SOURCES = {
  'openpilot/sunnypilot/modeld_v2/compile_warp.py': 'a182425b51918b35de8b7317461c4386e465bc5a4c9064a39e4ff64e29ca0282',
  'openpilot/system/camerad/cameras/nv12_info.py': 'b3f39bbfdd75c43d396b01f746d460068aa8e98862217d46bf3cd426cd034e1a',
}
_RECIPE_SHA256 = '97a6814fde7f16e1ce265c6b6388cd5799c8e2f8969ac78cfc4fd5a929b1921d'
_CAPTURE_OUTPUT_SHA256 = {
  (1344, 760): '124d4a811290f1c4bc1bbf0d922a80ca870a72e46cfea909f6431c97f46d72b1',
  (1928, 1208): '00acdfa35426fd043be2e09de36a862d406d60331a9c1d11833ecea89ebdfb33',
}


def _sha256(path: Path) -> str:
  with path.open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def camera_layout(width: int, height: int) -> dict:
  _require(type(width) is int and type(height) is int and (width, height) in _PINNED_WARPS,
           'No reviewed existing Chestnut warp for this camera geometry')
  stride, y_height, uv_height, allocation = get_nv12_info(width, height)
  return {'width': width, 'height': height, 'stride': stride, 'y_height': y_height, 'uv_height': uv_height,
          'copied_frame_bytes': stride * (y_height + uv_height), 'allocated_frame_bytes': allocation}


def installed_warp_path(width: int, height: int) -> Path:
  camera_layout(width, height)
  return Path(BASEDIR) / f'openpilot/sunnypilot/modeld_v2/models/big_driving_warp_{width}x{height}_tinygrad.pkl'


def validate_camera_profile(profile: dict, warp_sha256: str, frame_size: int) -> None:
  """Check persisted profile coherence without reading files or opening devices.

  The model manifest loader independently hashes its warp. Actual camera width
  and height must additionally match this profile at production startup.
  """
  _require(isinstance(profile, dict), 'Camera profile must be an object')
  width, height = profile.get('width'), profile.get('height')
  if type(width) is not int or type(height) is not int:
    raise ExperimentalCatalogError('Camera profile dimensions must be integers')
  layout = camera_layout(width, height)
  _require(warp_sha256 == _PINNED_WARPS[width, height] and profile.get('artifact_sha256') == warp_sha256,
           'Camera profile does not match the reviewed warp identity')
  _require(profile.get('tinygrad_ref') == TINYGRAD_REF, 'Camera profile runtime mismatch')
  _require(type(frame_size) is int and frame_size == layout['copied_frame_bytes'], 'Camera profile frame extent mismatch')
  expected = {'stride': layout['stride'], 'y_height': layout['y_height'], 'uv_height': layout['uv_height'],
              'frame_size': layout['copied_frame_bytes'], 'allocation_size': layout['allocated_frame_bytes']}
  _require(all(type(profile.get(key)) is int and profile[key] == value for key, value in expected.items()),
           'Camera profile layout mismatch')
  _require(profile.get('camera') == layout, 'Camera profile geometry metadata mismatch')
  _require(profile.get('target') == TARGET | {'renderer': 'LLVM'}, 'Camera profile target mismatch')
  _require(profile.get('output_shape') == [2, 6, 128, 256] and profile.get('output_dtype') == 'uint8', 'Camera profile output mismatch')
  source_bindings = _REVIEWED_SOURCES | {'tinygrad:examples/openpilot/compile_warp.py': _RECIPE_SHA256}
  _require(profile.get('source_bindings') == source_bindings, 'Camera profile recipe provenance mismatch')
  _require(profile.get('scope') == 'branch_pinned_warp_static_and_saved_capture' and profile.get('captured_check_passed') is True,
           'Camera profile saved-capture validation is missing')
  capture = profile.get('captured_reference', {})
  _require(isinstance(capture, dict) and capture.get('passed') is True and type(capture.get('output_bytes')) is int
           and capture['output_bytes'] == 393216 and type(capture.get('differing_bytes')) is int and capture['differing_bytes'] == 0
           and capture.get('output_sha256') == _CAPTURE_OUTPUT_SHA256[width, height], 'Camera profile saved-capture result mismatch')
  _require(profile.get('embedded_compiler_attestation') is False and profile.get('gpu_execution_performed') is False
           and profile.get('normal_activation_allowed') is False, 'Camera profile cannot claim physical execution or driving approval')


def _saved_arrays(path: Path, header: dict, layout: dict) -> dict:
  """Read only the JSON-described captured arrays, never the pickle extras."""
  import numpy as np

  expected = {'input_frame': {'shape': [2, layout['copied_frame_bytes']], 'dtype': 'uint8'},
              'M_inv': {'shape': [2, 3, 3], 'dtype': 'float32'}}
  check = header.get('check')
  _require(isinstance(check, dict) and check.get('output_container') is None, 'Warp requires its saved direct-output capture')
  inputs = check.get('inputs', [])
  _require(isinstance(inputs, list) and len(inputs) == 2, 'Warp saved capture must contain both named inputs')
  captured = {}
  for row in inputs:
    _require(isinstance(row, list) and len(row) == 3 and isinstance(row[0], list) and len(row[0]) == 1,
             'Malformed warp saved input capture')
    name = row[0][0]
    _require(name in expected and name not in captured and row[2] == 'AMD', 'Warp saved input names/device mismatch')
    _check_tensor(row[1], expected[name])
    captured[name] = row[1]
  outputs = check.get('outputs', [])
  _require(isinstance(outputs, list) and len(outputs) == 1, 'Warp saved capture requires one packed image output')
  _check_tensor(outputs[0], {'shape': [2, 6, 128, 256], 'dtype': 'uint8'})
  captured['output'] = outputs[0]
  with path.open('rb') as source:
    source.seek(8)
    size = struct.unpack('<Q', source.read(8))[0]
    _require(0 < size <= MAX_HEADER_BYTES, 'Warp saved capture header exceeds the bounded size')
    container = json.loads(source.read(size))
    blob = 16 + size + -(16 + size) % 256 + container['parts'][0]['blob'][0]
    arrays = {}
    for name, record in captured.items():
      start, end = record['data']
      source.seek(blob + start)
      data = source.read(end - start)
      _require(len(data) == end - start, 'Truncated warp saved capture')
      dtype = '|u1' if _dtype(record['dtype']) == 'uint8' else '<f4'
      arrays[name] = np.frombuffer(data, dtype=dtype).reshape(record['shape'])
  return arrays


def _reference_nv12(images: np.ndarray, matrices: np.ndarray, layout: dict) -> np.ndarray:
  """Independent NumPy form of the pinned nearest-neighbor NV12/YUV packing."""
  import numpy as np

  width, height, stride = layout['width'], layout['height'], layout['stride']

  def perspective(source, matrix, out_width, out_height, source_width, source_height):
    x, y = np.meshgrid(np.arange(out_width, dtype=np.float32), np.arange(out_height, dtype=np.float32))
    with np.errstate(divide='raise', invalid='raise', over='raise'):
      denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
      source_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
      source_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    ix = np.rint(source_x).clip(0, source_width - 1).astype(np.int64)
    iy = np.rint(source_y).clip(0, source_height - 1).astype(np.int64)
    return source[iy, ix]

  _require(bool(np.isfinite(matrices).all()), 'Warp saved matrices must be finite')
  packed = []
  uv_offset = stride * layout['y_height']
  for image, matrix in zip(images, matrices, strict=True):
    uv = image[uv_offset:uv_offset + layout['uv_height'] * stride].reshape(layout['uv_height'], stride)
    uv_matrix = matrix * np.array([[1., 1., .5], [1., 1., .5], [2., 2., 1.]], np.float32)
    y = perspective(image[:height * stride].reshape(height, stride), matrix, 512, 256, width, height)
    u = perspective(uv[:height // 2, :width:2], uv_matrix, 256, 128, width // 2, height // 2)
    v = perspective(uv[:height // 2, 1:width:2], uv_matrix, 256, 128, width // 2, height // 2)
    packed.append(np.stack((y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2], u, v)))
  return np.stack(packed)


def validate_existing_warp(warp_path: str | Path, *, camera_width: int = 1344, camera_height: int = 760,
                           tinygrad_root: str | Path | None = None) -> dict:
  """Validate existing warp bytes without compiling or executing a GPU kernel.

  Geometry is associated with an exact reviewed asset identity, not inferred
  merely from a filename or copied-buffer length. The returned record does not
  approve driving or assert a new physical camera/GPU test.
  """
  import numpy as np

  try:
    width, height = camera_width, camera_height
    layout = camera_layout(width, height)
    path = Path(warp_path).resolve(strict=True)
    runtime = Path(tinygrad_root) if tinygrad_root is not None else Path(BASEDIR) / 'tinygrad_repo'
    _require(read_only_git(runtime, 'rev-parse', 'HEAD') == TINYGRAD_REF, 'Warp requires the pinned tinygrad runtime')
    _require(not read_only_git(runtime, 'status', '--porcelain', '--untracked-files=all'), 'Warp requires a clean tinygrad runtime')
    source_bindings = {}
    for name, expected in _REVIEWED_SOURCES.items():
      _require(_sha256(Path(BASEDIR) / name) == expected, 'Reviewed warp compiler/layout source changed: ' + name)
      source_bindings[name] = expected
    _require(_sha256(runtime / 'examples/openpilot/compile_warp.py') == _RECIPE_SHA256, 'Reviewed tinygrad warp recipe changed')
    source_bindings['tinygrad:examples/openpilot/compile_warp.py'] = _RECIPE_SHA256
    before = path.stat()
    digest = _sha256(path)
    _require(digest == _PINNED_WARPS[width, height], 'Warp bytes do not match the reviewed camera-specific branch asset')
    header = _header(path, TARGET)
    copied = _warp_contract(header)
    _require(copied == layout['copied_frame_bytes'], 'Warp copied extent does not match the actual camera layout')
    names = header.get('input_names')
    _require(isinstance(names, list) and len(names) == 2 and set(names) == {'M_inv', 'input_frame'}, 'Warp named-input bindings mismatch')
    slots = [row.get('slot') for row in header['inputs']]
    _require(all(type(slot) is int for slot in slots) and sorted(slots) == [0, 1], 'Warp native input slots mismatch')
    arrays = _saved_arrays(path, header, layout)
    reference = _reference_nv12(arrays['input_frame'], arrays['M_inv'], layout)
    _require(np.array_equal(reference, arrays['output']), 'Saved warp output differs from the camera-layout NumPy reference')
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Warp changed during validation')
    profile = {'scope': 'branch_pinned_warp_static_and_saved_capture', 'path': str(path), 'artifact_sha256': digest,
            'tinygrad_ref': TINYGRAD_REF, 'target': TARGET | {'renderer': 'LLVM'}, 'camera': layout,
            'width': width, 'height': height, 'stride': layout['stride'], 'y_height': layout['y_height'], 'uv_height': layout['uv_height'],
            'frame_size': copied, 'allocation_size': layout['allocated_frame_bytes'],
            'output_shape': [2, 6, 128, 256], 'output_dtype': 'uint8', 'captured_check_passed': True,
            'warp_frame_size': copied, 'model_image': {'shape': [2, 6, 128, 256], 'dtype': 'uint8'},
            'source_bindings': source_bindings, 'embedded_compiler_attestation': False,
            'captured_reference': {'passed': True, 'output_bytes': reference.nbytes, 'differing_bytes': 0,
                                   'output_sha256': hashlib.sha256(reference.tobytes()).hexdigest()},
            'gpu_execution_performed': False, 'normal_activation_allowed': False}
    validate_camera_profile(profile, digest, copied)
    return profile
  except (OSError, KeyError, TypeError, AttributeError, struct.error, FloatingPointError, GitInspectionError) as error:
    raise ExperimentalCatalogError('Cannot validate existing camera warp: ' + str(error)) from error
