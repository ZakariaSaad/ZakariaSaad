"""Cinque Terre V3's explicit input/next-state Tinygrad contract.

State is owned by this runner, rather than inferred from legacy feature queues.
Only commit next-state buffers after every output has been validated.
"""
import math
import os
import numpy as np

from tinygrad import Tensor

LIVE_INPUTS = {'new_img', 'desire', 'traffic_convention', 'action_t'}
REQUIRED_OUTPUTS = {'plan', 'lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob',
                    'meta', 'desire_state', 'desire_pred', 'pose', 'road_transform', 'wide_from_device_euler', 'action'}


def validate_specs(input_specs, output_specs, metadata):
  if not LIVE_INPUTS <= input_specs.keys() or 'outputs' not in output_specs:
    raise ValueError('V3 is missing live inputs or outputs')
  for name, spec in {**input_specs, **output_specs}.items():
    shape, dtype, device = spec
    if not shape or any(type(s) is not int or s <= 0 for s in shape) or math.prod(shape) > 100_000_000:
      raise ValueError(f'Invalid V3 shape: {name}')
    if np.dtype(dtype).kind not in 'fiu' or not isinstance(device, str) or not device:
      raise ValueError(f'Invalid V3 dtype/device: {name}')
  if math.prod(input_specs['desire'][0]) != 8 or math.prod(input_specs['traffic_convention'][0]) != 2:
    raise ValueError('Unsupported V3 desire/traffic convention contract')
  if math.prod(input_specs['action_t'][0]) != 2 or len(input_specs['new_img'][0]) != 4 or tuple(input_specs['new_img'][0][:2]) != (2, 6):
    raise ValueError('Unsupported V3 action/image contract')
  if any(np.dtype(input_specs[k][1]) != np.dtype('float32') for k in LIVE_INPUTS - {'new_img'}):
    raise ValueError('V3 live inputs must be float32')
  pairs = {k: f'next_{k}' for k in input_specs if k not in LIVE_INPUTS}
  if not pairs or set(output_specs) != {'outputs', *pairs.values()}:
    raise ValueError('V3 requires a complete, unambiguous recurrent state contract')
  for name, next_name in pairs.items():
    if tuple(input_specs[name][0]) != tuple(output_specs[next_name][0]) or input_specs[name][1:] != output_specs[next_name][1:]:
      raise ValueError(f'V3 state shape/dtype/device mismatch: {name}')
  shape = tuple(output_specs['outputs'][0])
  if len(shape) != 2 or shape[0] != 1 or np.dtype(output_specs['outputs'][1]) != np.dtype('float32'):
    raise ValueError('Unsupported V3 prediction contract')
  slices = metadata['output_slices']
  if not REQUIRED_OUTPUTS <= slices.keys():
    raise ValueError('V3 is missing outputs required by MVL publication')
  for name, sl in slices.items():
    if not isinstance(sl, slice) or sl.step not in (None, 1) or sl.start is None or sl.stop is None or not 0 <= sl.start < sl.stop <= shape[1]:
      raise ValueError(f'Invalid V3 output slice: {name}')
  return pairs


def validate_predictions(outputs):
  """Reject malformed and nonfinite predictions before control/publication."""
  shapes = {'plan': (1, 33, 15), 'plan_stds': (1, 33, 15), 'lane_lines': (1, 4, 33, 2),
            'lane_lines_stds': (1, 4, 33, 2), 'road_edges': (1, 2, 33, 2), 'road_edges_stds': (1, 2, 33, 2),
            'lead': (1, 3, 6, 4), 'lead_stds': (1, 3, 6, 4), 'lead_prob': (1, 3),
            'lane_lines_prob': (1, 8), 'desire_state': (1, 8), 'desire_pred': (1, 4, 8),
            'pose': (1, 6), 'pose_stds': (1, 6), 'road_transform': (1, 6), 'road_transform_stds': (1, 6), 'wide_from_device_euler': (1, 3),
            'wide_from_device_euler_stds': (1, 3)}
  for name, shape in shapes.items():
    if name not in outputs or outputs[name].shape != shape:
      raise ValueError(f'Invalid prediction shape: {name}')
  if 'meta' not in outputs or outputs['meta'].shape != (1, 55):
    raise ValueError('Invalid prediction shape: meta')
  if 'action' in outputs and outputs['action'].shape != (1, 2):
    raise ValueError('Invalid prediction shape: action')
  if any(not np.all(np.isfinite(value)) for value in outputs.values()):
    raise ValueError('Model prediction is not finite')


class StatefulRunner:
  def __init__(self, artifact, parser):
    self.input_specs, self.output_specs = artifact['input_specs'], artifact['output_specs']
    self.state_pairs = validate_specs(self.input_specs, self.output_specs, artifact['metadata'])
    self.output_slices = artifact['metadata']['output_slices']
    self.run_model, self.parser = artifact['run'], parser
    if not callable(self.run_model):
      raise ValueError('V3 runner is not callable')
    self.inputs = {k: Tensor(np.zeros(shape, dtype=dtype), device=dev).realize()
                   for k, (shape, dtype, dev) in self.input_specs.items() if k != 'new_img'}
    self.outputs = {k: Tensor(np.zeros(shape, dtype=dtype), device=dev).realize()
                    for k, (shape, dtype, dev) in self.output_specs.items()}
    self.prev_desire = np.zeros(8, dtype=np.float32)

  def reset(self):
    for value in (*self.inputs.values(), *self.outputs.values()):
      value.assign(0).realize()
    self.prev_desire[:] = 0

  def run(self, new_img, desire, traffic_convention, action_t):
    desire = np.asarray(desire, dtype=np.float32).reshape(8).copy()
    if not np.all(np.isfinite(desire)):
      raise ValueError('Nonfinite V3 desire')
    desire[0] = 0
    pulse = np.where(desire - self.prev_desire > .99, desire, 0).astype(np.float32)
    live = {'desire': pulse, 'traffic_convention': traffic_convention, 'action_t': action_t}
    for name, value in live.items():
      value = np.asarray(value, dtype=self.input_specs[name][1]).reshape(self.input_specs[name][0])
      if not np.all(np.isfinite(value)):
        raise ValueError(f'Nonfinite V3 input: {name}')
      self.inputs[name].assign(Tensor(value, device=self.input_specs[name][2])).realize()
    shape, dtype, device = self.input_specs['new_img']
    if new_img.shape != tuple(shape):
      raise ValueError('Warp output does not match V3 image input')
    self.run_model(output_buffers=self.outputs, new_img=new_img.cast(np.dtype(dtype).name).to(device).realize(), **self.inputs)
    raw = {k: value.numpy() for k, value in self.outputs.items()}
    if any(not np.all(np.isfinite(value)) for value in raw.values()):
      raise ValueError('V3 output or recurrent state is not finite')
    sliced = {k: raw['outputs'][:, sl].copy() for k, sl in self.output_slices.items()}
    outputs = self.parser.parse_outputs(sliced)
    if os.environ.get('SEND_RAW_PRED'):
      outputs['raw_pred'] = raw['outputs'][0].copy()
    validate_predictions(outputs)
    for name, next_name in self.state_pairs.items():
      self.inputs[name].assign(self.outputs[next_name]).realize()
    self.prev_desire[:] = desire
    return outputs


def make_stateful_jit(model_runner):
  """Compile all ONNX next_* outputs; never discard state after first frame."""
  from tinygrad.engine.jit import TinyJit

  @TinyJit
  def run(output_buffers, **inputs):
    result = model_runner(inputs)
    if set(result) != set(output_buffers):
      raise ValueError('V3 ONNX outputs differ from the declared contract')
    # Realize every result before updating output buffers (state may be a view).
    Tensor.realize(*result.values())
    for name, value in result.items():
      output_buffers[name].assign(value.cast(output_buffers[name].dtype)).realize()
  return run


class PackedStatefulRunner:
  """The integrated run_model format introduced by official 64f9b47b."""
  def __init__(self, artifact, camera_size, frame_copy_size, parser):
    self.run_model = artifact['run_model'][camera_size]
    self.model_device = artifact['input_devices']['model']
    self.input_specs = {k: (tuple(shape), np.dtype(dtype.fmt).name, self.model_device)
                        for k, (shape, dtype) in artifact['input_shapes'].items()}
    self.state_pairs = artifact['state_pairs']
    output_specs = {v: self.input_specs[k] for k, v in self.state_pairs.items()}
    output_specs['outputs'] = (tuple(artifact['metadata']['output_shapes']['outputs']), 'float32', self.model_device)
    self.prediction_shape = output_specs['outputs'][0]
    if validate_specs(self.input_specs, output_specs, artifact['metadata']) != self.state_pairs:
      raise ValueError('Packed V3 state pairs do not match inputs')
    self.output_slices, self.parser = artifact['metadata']['output_slices'], parser
    self.frame_copy_size = frame_copy_size
    shapes = {'tfm': (3, 3), 'big_tfm': (3, 3)} | {k: s for k, (s, _, _) in self.input_specs.items() if k in LIVE_INPUTS - {'new_img'}}
    sizes = [math.prod(s) for s in shapes.values()]
    nbytes = sum(sizes) * 4
    self.packed = np.zeros(nbytes + 2 * frame_copy_size, dtype=np.uint8)
    self.numpy_inputs = {k: v.reshape(shapes[k]) for k, v in zip(shapes, np.split(self.packed[:nbytes].view('float32'),
                                                                                              np.cumsum(sizes[:-1])), strict=True)}
    self.frames = self.packed[nbytes:].reshape(2, frame_copy_size)
    self.inputs = {k: Tensor(np.zeros(shape, dtype=dtype), device=dev).realize()
                   for k, (shape, dtype, dev) in self.input_specs.items() if k in self.state_pairs}
    self.inputs['packed_npy_inputs'] = Tensor(self.packed, device='NPY').realize()
    self.prev_desire = np.zeros(8, dtype=np.float32)

  def reset(self):
    self.packed[:] = 0
    for name in self.state_pairs:
      self.inputs[name].assign(0).realize()
    self.prev_desire[:] = 0

  def run_frames(self, bufs, transforms, inputs):
    desire = np.asarray(inputs['desire'], dtype=np.float32).reshape(8).copy()
    if not np.all(np.isfinite(desire)):
      raise ValueError('Nonfinite V3 desire')
    desire[0] = 0
    self.numpy_inputs['desire'][:] = np.where(desire - self.prev_desire > .99, desire, 0).reshape(self.numpy_inputs['desire'].shape)
    for i, key in enumerate(('img', 'big_img')):
      data = bufs[key].data if hasattr(bufs[key], 'data') else bufs[key]
      np.copyto(self.frames[i], np.frombuffer(data, dtype=np.uint8, count=self.frame_copy_size))
      self.numpy_inputs['big_tfm' if i else 'tfm'][:] = transforms[key]
    for key in ('traffic_convention', 'action_t'):
      self.numpy_inputs[key][:] = inputs[key]
    if any(not np.all(np.isfinite(v)) for v in self.numpy_inputs.values()):
      raise ValueError('Nonfinite packed V3 input')
    raw, = self.run_model(**self.inputs)
    raw = raw.numpy()
    if not np.all(np.isfinite(raw)) or any(not np.all(np.isfinite(self.inputs[k].numpy())) for k in self.state_pairs):
      raise ValueError('Packed V3 prediction or state is not finite')
    if raw.shape != self.prediction_shape:
      raise ValueError('Packed V3 prediction shape mismatch')
    outputs = self.parser.parse_outputs({k: raw[:, sl].copy() for k, sl in self.output_slices.items()})
    if os.environ.get('SEND_RAW_PRED'):
      outputs['raw_pred'] = raw[0].copy()
    validate_predictions(outputs)
    self.prev_desire[:] = desire
    return outputs
