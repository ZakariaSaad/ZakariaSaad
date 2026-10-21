#!/usr/bin/env python3
"""Compare trusted CPU-compiled model artifacts on recorded inputs, without messaging/actuation.

No weights are bundled or downloaded. NPZ is loaded with allow_pickle=False.
Navigation-aware graphs cannot be published by the production modeld entrypoints.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def require_cpu_artifact(path):
  """Reject GPU targets before loading any compiled program from a trusted artifact."""
  from openpilot.sunnypilot.modeld_v2.helpers import read_parts

  parts, extras = read_parts(str(path))
  devices = set()
  for header, _ in parts.values():
    devices.update(entry['device'] for entry in header.get('buffers', ()))
    devices.update(entry['device'] for entry in header.get('inputs', ()))
    devices.update(call['target']['device'] for call in header.get('calls', ()) if call['kind'] == 'program')
  for key in ('input_specs', 'output_specs'):
    devices.update(spec[2] for spec in extras.get(key, {}).values())
  if not devices or any(device.split(':', 1)[0] not in ('CPU', 'NPY') for device in devices):
    raise ValueError(f'Offline harness requires CPU/NPY artifact targets, found {sorted(devices)}')
  return parts, extras


def validate_warp_specs(model_extras, warp_header, warp_extras, width, height):
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from openpilot.sunnypilot.modeld_v2.helpers import _dtype
  from tinygrad.dtype import _to_np_dtype

  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  frame_shape, frame_dtype, _ = warp_extras['input_specs']['input_frame']
  transform_shape, transform_dtype, _ = warp_extras['input_specs']['M_inv']
  if tuple(frame_shape) != (2, stride * (y_height + uv_height)) or np.dtype(frame_dtype) != np.uint8:
    raise ValueError('CPU warp input must match the requested camera NV12 copied extent')
  if tuple(transform_shape) != (2, 3, 3) or np.dtype(transform_dtype) != np.float32:
    raise ValueError('CPU warp requires two float32 3x3 transforms')
  image_shape, image_dtype, _ = model_extras['input_specs']['new_img']
  outputs = warp_header['returns']['tensors']
  if (
    len(outputs) != 1
    or tuple(outputs[0]['shape']) != tuple(image_shape)
    or np.dtype(_to_np_dtype(_dtype(outputs[0]['dtype']))) != np.dtype(image_dtype)
  ):
    raise ValueError('CPU warp output must match the driving model image shape/dtype')


def compare_runs(model_factory, frames):
  """Independent recurrent states for neutral and conditioned arms; raw cameras/timing identical."""
  result = {}
  for enabled in (False, True):
    model = model_factory()
    model.warmup()
    outputs = []
    for frame in frames:
      # ModelState updates desire inputs in place. Neither arm may alter the
      # recording or leave auxiliary input mutations for the other arm.
      inputs = {name: np.asarray(value).copy() for name, value in frame['inputs'].items()}
      if enabled:
        inputs.update({name: np.asarray(value).copy() for name, value in frame['navigation'].items()})
      outputs.append({name: np.asarray(value).copy() for name, value in model.run(frame['bufs'], frame['transforms'], inputs).items()})
    result['conditioned' if enabled else 'neutral'] = outputs
  differences = {}
  for before, after in zip(result['neutral'], result['conditioned'], strict=True):
    if before.keys() != after.keys():
      raise ValueError('Output schema changed between replay arms')
    for name in before:
      if before[name].shape != after[name].shape or not np.isfinite(before[name]).all() or not np.isfinite(after[name]).all():
        raise ValueError(f'Invalid replay output: {name}')
      delta = float(np.max(np.abs(after[name] - before[name]))) if before[name].size else 0.0
      differences[name] = max(differences.get(name, 0.0), delta)
  return {'frames': len(frames), 'max_abs_difference': differences}


def load_recording(path, desire_key):
  """Prepared recorded model inputs; nav validity is explicit and defaults to false."""
  with np.load(path, allow_pickle=False) as data:
    arrays = {name: data[name] for name in data.files}
  required = ('img', 'big_img', 'M_img', 'M_big_img', 'desire', 'traffic_convention', 'action_t')
  if any(name not in arrays for name in required):
    raise ValueError(f'Recording requires {required}')
  count = len(arrays['img'])
  if not count or any(len(value) != count for value in arrays.values()):
    raise ValueError('Every recording array must contain the same nonzero frame count')
  if 'nav_valid' in arrays and (arrays['nav_valid'].shape != (count,) or arrays['nav_valid'].dtype != np.bool_):
    raise ValueError('nav_valid must be a boolean array with one value per frame')
  frames = []
  for i in range(count):
    bufs = {}
    transforms = {}
    for name in ('img', 'big_img'):
      if arrays[name].dtype != np.uint8 or arrays[name][i].ndim != 1:
        raise ValueError('Recorded images must be flat uint8 NV12 frames')
      bufs[name] = SimpleNamespace(data=arrays[name][i].tobytes())
      transform = arrays['M_' + name][i]
      if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError('Recorded transforms must be finite 3x3 matrices')
      transforms[name] = transform.astype(np.float32)
    inputs = {desire_key: arrays['desire'][i], 'traffic_convention': arrays['traffic_convention'][i], 'action_t': arrays['action_t'][i]}
    if any(not np.isfinite(value).all() for value in inputs.values()):
      raise ValueError('Recorded auxiliary inputs must be finite')
    nav = {}
    if 'nav_valid' in arrays and bool(arrays['nav_valid'][i]):
      pixels, vector = arrays['nav_map'][i], arrays['navigation'][i]
      if pixels.shape != (256, 256) or pixels.dtype != np.uint8 or vector.shape != (47,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError('Unsupported prepared navigation map/vector contract')
      if np.count_nonzero(vector[:46]) > 1 or np.any((vector[:46] != 0) & (vector[:46] != 1)) or vector[-1] < 0:
        raise ValueError('Navigation must contain at most one command and nonnegative distance')
      nav = {'nav_map': pixels.copy(), 'navigation': vector.copy()}
    frames.append({'bufs': bufs, 'transforms': transforms, 'inputs': inputs, 'navigation': nav})
  return frames


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifact', type=Path, required=True, help='Trusted SPTG artifact compiled for the PC CPU runtime')
  parser.add_argument('--warp-artifact', type=Path, required=True, help='Trusted CPU-compiled SPTG camera warp matching width/height')
  parser.add_argument('--recording', type=Path, required=True, help='Prepared offline NPZ; never live camera/CAN')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--width', type=int, default=1344)
  parser.add_argument('--height', type=int, default=760)
  args = parser.parse_args()
  from openpilot.common.hardware import PC

  if not PC or os.environ.get('NAV_MODEL_OFFLINE') != '1':
    parser.error('Only explicitly opted-in PC offline evaluation is permitted (NAV_MODEL_OFFLINE=1)')
  artifact = args.artifact.resolve(strict=True)
  warp_artifact = args.warp_artifact.resolve(strict=True)
  _, model_extras = require_cpu_artifact(artifact)
  warp_parts, warp_extras = require_cpu_artifact(warp_artifact)
  validate_warp_specs(model_extras, warp_parts['run'][0], warp_extras, args.width, args.height)
  from openpilot.common.params import Params
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.sunnypilot.modeld_v2.modeld import ModelState

  original = os.environ.get('COMBINED_MODEL_PKL')
  try:
    os.environ['COMBINED_MODEL_PKL'] = str(artifact)
    with OpenpilotPrefix():
      params = Params()
      params.put_bool('NavigationModelInputsEnabled', True, block=True)

      # No publisher, manager, controller, safety process or CAN interface is started.
      def factory():
        return ModelState(args.width, args.height, chestnut=False, warp_path=str(warp_artifact))

      prototype = factory()
      frames = load_recording(args.recording, prototype.desire_key)
      del prototype
      summary = compare_runs(factory, frames)
      summary['scope'] = 'offline recorded inference only; differences do not establish driving safety'
      args.output.write_text(json.dumps(summary, indent=2) + '\n')
  finally:
    if original is None:
      os.environ.pop('COMBINED_MODEL_PKL', None)
    else:
      os.environ['COMBINED_MODEL_PKL'] = original


if __name__ == '__main__':
  main()
