"""File-only numerical replay at the native navigation model's consumed-tensor boundary.

Unlike model_replay's 60-frame CI sample, this streams every explicitly listed
frame with one recurrent state lifetime. FrameReader/encode-index camera helpers
follow process_replay's NV12 layout, but decoded HEVC is not an exact capture of
the prewarped image consumed in a previous drive. No messaging or processes run.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

STATE_NAMES = ('state_img_q', 'state_desire_q', 'state_feat_q', 'state_map_features_q', 'state_navigation_features_q')
MAX_NPZ_BYTES = 64 << 20
MAX_CAPTURE_FRAMES = 8


def private_file(path):
  return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb')


def sha256(path):
  with Path(path).open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def arrays_from_file(path):
  """Reject pickle/object arrays and bound decompression before reading tensors."""
  with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    if len(names) > 32 or len(set(names)) != len(names) or sum(item.file_size for item in archive.infolist()) > MAX_NPZ_BYTES:
      raise ValueError('Tensor archive exceeds bounds or contains duplicate entries')
  with np.load(path, allow_pickle=False) as archive:
    arrays = {name: archive[name].copy() for name in archive.files}
  if any(value.dtype.hasobject for value in arrays.values()):
    raise ValueError('Object tensors are forbidden')
  return arrays


def tensor_stats(array):
  finite = np.isfinite(array)
  values = array[finite]
  return {'shape': list(array.shape), 'dtype': str(array.dtype), 'size': array.size,
          'finite': int(finite.sum()), 'nan': int(np.isnan(array).sum()),
          'posinf': int(np.isposinf(array).sum()), 'neginf': int(np.isneginf(array).sum()),
          'min': float(values.min()) if values.size else None, 'max': float(values.max()) if values.size else None}


def check_tensors(arrays, specs):
  if set(arrays) != set(specs):
    raise ValueError(f'Tensor names differ: missing={sorted(set(specs) - set(arrays))}, extra={sorted(set(arrays) - set(specs))}')
  for name, (shape, dtype) in specs.items():
    if arrays[name].shape != tuple(shape) or arrays[name].dtype != np.dtype(dtype):
      raise ValueError(f'{name}: expected {tuple(shape)} {np.dtype(dtype)}, got {arrays[name].shape} {arrays[name].dtype}')


def input_problems(arrays):
  problems = [f'{name}: non-finite' for name, value in arrays.items() if not np.isfinite(value).all()]
  if 'nav_map' in arrays and (np.any(arrays['nav_map'] < 0) or np.any(arrays['nav_map'] > 255)):
    problems.append('nav_map: raw pixels outside 0..255')
  if 'navigation' in arrays:
    vector = arrays['navigation'].reshape(-1)
    if vector.size != 47 or np.any((vector[:46] != 0) & (vector[:46] != 1)) or np.count_nonzero(vector[:46]) > 1 or vector[-1] < 0:
      problems.append('navigation: invalid command/distance encoding')
  # The official maneuver distance is distance/100, with no upper bound of one.
  return problems


def checkpoint(path, state, *, source_sha256, artifact_sha256, next_frame_id):
  """Write a PRE-inference snapshot, never a next_* alias observed after running."""
  with private_file(path) as destination:
    np.savez(destination, **state, _phase=np.array('pre_inference'), _source_sha256=np.array(source_sha256),
             _artifact_sha256=np.array(artifact_sha256), _next_frame_id=np.array(next_frame_id, dtype=np.uint64))


def load_checkpoint(path, runner, first_frame_id):
  arrays = arrays_from_file(path)
  expected = {'_phase': 'pre_inference', '_source_sha256': runner.source_sha256, '_artifact_sha256': runner.artifact_sha256,
              '_next_frame_id': first_frame_id}
  for key, value in expected.items():
    if key not in arrays or arrays[key].shape != () or arrays.pop(key).item() != value:
      raise ValueError(f'Checkpoint {key} does not match this replay')
  check_tensors(arrays, runner.state_specs)
  return arrays


def iter_frames(manifest):
  """JSONL contains file, sha256, frame_id, log_mono_time; segment is informational."""
  manifest = Path(manifest).resolve(strict=True)
  previous_clock, previous_frame = -1, -1
  with manifest.open() as source:
    for line in source:
      if len(line) > 16384:
        raise ValueError('Frame index line exceeds bounds')
      record = json.loads(line)
      for key in ('frame_id', 'log_mono_time'):
        if type(record.get(key)) is not int or not 0 <= record[key] < (1 << 64):
          raise ValueError(f'Frame {key} must be an unsigned integer')
      if record['log_mono_time'] <= previous_clock or record['frame_id'] <= previous_frame:
        raise ValueError('Frames must be strictly ordered; split restarts into separate replays')
      previous_clock, previous_frame = record['log_mono_time'], record['frame_id']
      path = (manifest.parent / record['file']).resolve(strict=True)
      if sha256(path) != record.get('sha256'):
        raise ValueError('Frame tensor SHA256 mismatch')
      yield record, arrays_from_file(path)


def camera_frames(rlogs, video, camera='narrowRoadCameraState'):
  """Decode actual files with fork schemas and FrameReader, matched by encode index.

  This is an optional preparation helper, not a replacement for missing consumed
  map pixels, prewarp bytes or recurrent state. Use one video's matching segment.
  """
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.tools.lib.framereader import FrameReader
  from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

  meta = meta_from_camera_state(camera)
  if meta is None:
    raise ValueError('Unsupported camera stream')
  messages = sorted((msg for path in rlogs for msg in LogReader(str(path)) if msg.which() in (camera, meta.encode_index)),
                    key=lambda msg: msg.logMonoTime)
  segments = {getattr(msg, meta.encode_index).segmentNum for msg in messages if msg.which() == meta.encode_index}
  if len(segments) != 1:
    raise ValueError('Supply exactly one encoded segment for each video file')
  encoded = {getattr(msg, meta.encode_index).frameId: getattr(msg, meta.encode_index).segmentId
             for msg in messages if msg.which() == meta.encode_index}
  first, last = min(encoded), max(encoded)
  boundary_skips = sum(msg.which() == camera and not first <= getattr(msg, camera).frameId <= last for msg in messages)
  reader = FrameReader(str(video), pix_fmt='nv12', cache_size=2)
  stride, y_height, uv_height, _ = get_nv12_info(reader.w, reader.h)
  for msg in messages:
    if msg.which() != camera:
      continue
    state = getattr(msg, camera)
    if state.frameId not in encoded:
      if not first <= state.frameId <= last:
        continue  # Log/video boundaries may differ by one frame; never invent a matching image.
      raise ValueError(f'No encoded frame index for camera frame {state.frameId}')
    image = reader.get(encoded[state.frameId])
    padded = np.zeros((y_height + uv_height, stride), np.uint8)
    padded[:reader.h, :reader.w] = image[:reader.h * reader.w].reshape(reader.h, reader.w)
    padded[y_height:y_height + reader.h // 2, :reader.w] = image[reader.h * reader.w:].reshape(reader.h // 2, reader.w)
    yield {'frame_id': state.frameId, 'log_mono_time': msg.logMonoTime,
           'camera_sof_ns': state.timestampSof, 'camera_eof_ns': state.timestampEof,
           'boundary_camera_frames_not_encoded': boundary_skips}, padded.reshape(-1)


class NativeRunner:
  """The production SPTG loader and native input_view state/output alias contract.

  Inputs here are already warped model tensors, so this does not run frame_fusion
  or reconstruct host timing/desire pulses. Explicit hashes identify trusted local
  executable artifacts; no pickle cache is loaded and no artifacts are downloaded.
  """
  def __init__(self, artifact, expected_sha256, *, source_sha256, backend='CPU'):
    from openpilot.sunnypilot.modeld_v2.helpers import load_model, read_parts
    from openpilot.sunnypilot.modeld_v2.model_adapters import diagnostic_output_slices, input_view
    from tinygrad import Tensor

    self.artifact_sha256 = sha256(artifact)
    if self.artifact_sha256 != expected_sha256:
      raise ValueError('Compiled artifact SHA256 mismatch')
    self.source_sha256 = source_sha256
    parts, extras = read_parts(str(artifact))
    if set(parts) != {'run'} or set(extras.get('input_specs', {})) == set():
      raise ValueError('Replay requires a native, prewarped single-run model')
    targets = [call['target'] for header, _ in parts.values() for call in header['calls'] if call['kind'] == 'program']
    if not targets or any(target['device'] != backend for target in targets):
      raise ValueError('Compiled backend does not match requested execution backend')
    self.targets = list({json.dumps(target, sort_keys=True): target for target in targets}.values())
    if backend == 'AMD' and any(target.get('arch') != 'gfx1200' or target.get('renderer') != 'LLVM' or
                                target.get('interface') != 'USB' for target in targets):
      raise ValueError('Unsupported AMD artifact target')
    devices = {spec[2].split(':')[0] for specs in ('input_specs', 'output_specs') for spec in extras[specs].values()}
    if not devices <= {backend, 'NPY'}:
      raise ValueError('Native tensor devices disagree with the artifact backend')
    if backend == 'AMD':
      from tinygrad import Device
      from tinygrad.runtime.ops_amd import AMDDevice, USBIface
      device = Device['AMD']
      if not isinstance(device, AMDDevice) or device.arch != 'gfx1200' or device.is_usb is not True or type(device.iface) is not USBIface:
        raise ValueError('Offline AMD execution requires physical gfx1200 Chestnut USB hardware')
    self.backend = backend
    self.jits = load_model(str(artifact), cache=False, verify=True)
    self.output_slices = diagnostic_output_slices(self.jits['metadata']['metadata']['output_slices'],
                                                 int(np.prod(self.jits['output_specs']['outputs'][0])))
    self.input_specs = {name: (tuple(shape), np.dtype(dtype)) for name, (shape, dtype, _) in self.jits['input_specs'].items()}
    self.state_specs = {name: spec for name, spec in self.input_specs.items() if f'next_{name}' in self.jits['output_specs']}
    if set(self.state_specs) != set(STATE_NAMES):
      raise ValueError('Native replay requires the five Bigv2 recurrent buffers')
    self.host_specs = {name: spec for name, spec in self.input_specs.items() if name not in self.state_specs}
    self.inputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                   for name, (shape, dtype, device) in self.jits['input_specs'].items()}
    self.outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                    for name, (shape, dtype, device) in self.jits['output_specs'].items()}
    for name in self.state_specs:
      tensor = self.inputs[name]
      self.outputs[f'next_{name}'] = input_view(tensor._buffer(), tensor.shape, tensor.dtype, 0)

  def set_state(self, values):
    check_tensors(values, self.state_specs)
    self._copy(values)

  def _copy(self, values):
    from tinygrad import Tensor
    for name, array in values.items():
      tensor = self.inputs[name]
      if tensor.device == 'NPY':
        tensor._buffer().as_memoryview(allow_zero_copy=True)[:] = array.tobytes()
      else:
        tensor.assign(Tensor(array.copy(), device=tensor.device)).realize()

  def snapshot(self):
    return {name: self.inputs[name].numpy().copy() for name in self.state_specs}

  def run(self, values):
    check_tensors(values, self.host_specs)
    self._copy(values)
    self.jits['run'](**self.inputs, output_buffers=self.outputs)
    return {name: tensor.numpy().copy() for name, tensor in self.outputs.items()}


class OnnxReference:
  """ONNX Runtime CPU, independent of tinygrad compiler and native kernel loading."""
  backend = 'onnxruntime:CPUExecutionProvider'

  def __init__(self, path, expected_sha256):
    import onnx
    import onnxruntime as ort
    if sha256(path) != expected_sha256:
      raise ValueError('Reference ONNX SHA256 mismatch')
    # Contiguous is a tinygrad storage/layout boundary, not a numerical operator.
    # ORT owns its tensor layouts. Replace only this known custom operation in
    # memory; the source file and its pinned hash remain unchanged. No other
    # custom operation is guessed, executed or silently substituted.
    model = onnx.load(str(path), load_external_data=False)
    adapted = 0

    def visit(nodes):
      nonlocal adapted
      for node in nodes:
        if node.domain not in ('', 'ai.onnx'):
          if node.domain != 'org.tinygrad' or node.op_type != 'Contiguous' or len(node.input) != 1 or len(node.output) != 1 or node.attribute:
            raise ValueError(f'Unsupported reference custom operation: {node.domain}::{node.op_type}')
          node.domain, node.op_type = '', 'Identity'
          adapted += 1
        for attribute in node.attribute:
          if attribute.type == onnx.AttributeProto.GRAPH:
            visit(attribute.g.node)
          elif attribute.type == onnx.AttributeProto.GRAPHS:
            for graph in attribute.graphs:
              visit(graph.node)

    visit(model.graph.node)
    for function in model.functions:
      visit(function.node)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
      raise ValueError('Reference requires self-contained, hash-verified ONNX weights')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    self.adaptations = {'org.tinygrad::Contiguous_to_ONNX_Identity': adapted, 'graph_optimizations': 'disabled',
                        'scope': 'reference memory layout only; source file unchanged', 'onnxruntime_version': ort.__version__,
                        'internal_precision': 'provider-owned; CPU results do not establish AMD intermediate precision or equivalence'}
    self.session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])

  def run(self, values, state):
    supplied = {name: value.copy() for name, value in (values | state).items()}
    names = [value.name for value in self.session.get_outputs()]
    return dict(zip(names, self.session.run(names, supplied), strict=True))


def replay(runner, frames, output, *, initial_zero=False, checkpoint_path=None, reference=None,
           capture_start=None, capture_count=1, rtol=1e-3, atol=1e-3):
  if initial_zero == (checkpoint_path is not None):
    raise ValueError('Choose exactly one explicit initial state: zero or pre-inference checkpoint')
  if not 1 <= capture_count <= MAX_CAPTURE_FRAMES or (capture_start is not None and capture_start < 0):
    raise ValueError('Offline capture must select at most eight frames')
  output = Path(output).resolve()
  if any((ancestor / '.git').exists() for ancestor in (output, *output.parents)):
    raise ValueError('Private replay data must be written outside Git working trees')
  output.mkdir(parents=True, exist_ok=False, mode=0o700)
  count = 0
  summary = {'backend': runner.backend, 'reference_backend': reference.backend if reference else None,
             'compiled_targets': getattr(runner, 'targets', None),
             'reference_adaptations': getattr(reference, 'adaptations', None),
             'source_sha256': runner.source_sha256, 'artifact_sha256': runner.artifact_sha256,
             'source_binding': 'source digest supplied by operator; reference ONNX bytes checked if reference enabled',
             'initial_state': 'explicit_zero' if initial_zero else 'pre_inference_checkpoint',
             'scope': 'offline consumed-tensor numerical replay; no claim of original-drive reproduction without exact inputs/state',
             'vehicle_io': False, 'frames': 0, 'result': 'empty_sequence'}
  with (output / 'events.jsonl').open('x') as journal:
    for index, (record, values) in enumerate(frames):
      check_tensors(values, runner.host_specs)
      if not count:
        initial = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in runner.state_specs.items()} if initial_zero else \
                  load_checkpoint(checkpoint_path, runner, record['frame_id'])
        runner.set_state(initial)
      # Every comparison uses this exact incoming primary state. References do not
      # evolve their own divergent histories, and segment labels never reset state.
      state = runner.snapshot()
      event = {'frame_id': record['frame_id'], 'log_mono_time': record['log_mono_time'], 'index': index,
               'host_inputs': {name: tensor_stats(value) for name, value in values.items()},
               'incoming_state': {name: tensor_stats(value) for name, value in state.items()}}
      problems = input_problems(values)
      state_problems = [name for name, value in state.items() if not np.isfinite(value).all()]
      if problems or state_problems:
        event.update(result='invalid_incoming_state' if state_problems else 'invalid_host_inputs',
                     input_problems=problems, state_problems=state_problems)
      else:
        capture = capture_start is not None and capture_start <= index < capture_start + capture_count
        if capture:
          checkpoint(output / f'{index:08d}.pre.npz', state, source_sha256=runner.source_sha256,
                     artifact_sha256=runner.artifact_sha256, next_frame_id=record['frame_id'])
          with private_file(output / f'{index:08d}.inputs.npz') as destination:
            np.savez(destination, **values)
        started = time.perf_counter()
        try:
          outputs = runner.run(values)
        except Exception as error:
          event.update(result='execution_error', error=f'{type(error).__name__}: {error}')
          journal.write(json.dumps(event, allow_nan=False) + '\n')
          summary.update(frames=count + 1, result=event['result'], last_frame_id=record['frame_id'])
          break
        event['native_seconds_including_transfer_and_readback'] = time.perf_counter() - started
        event['outputs'] = {name: tensor_stats(value) for name, value in outputs.items()}
        event['result'] = 'nonfinite_output' if any(not np.isfinite(value).all() for value in outputs.values()) else 'finite'
        if event['result'] == 'nonfinite_output' and 'outputs' in outputs:
          primary = outputs['outputs'].reshape(-1)
          event['invalid_primary_sections'] = {name: tensor_stats(primary[selection])
                                              for name, selection in getattr(runner, 'output_slices', {}).items()
                                              if not np.isfinite(primary[selection]).all()}
        if reference is not None:
          started = time.perf_counter()
          expected = reference.run(values, state)
          event['reference_seconds'] = time.perf_counter() - started
          if outputs.keys() != expected.keys():
            raise ValueError('Reference output names differ')
          comparisons = {}
          for name, actual in outputs.items():
            before = expected[name]
            if before.shape != actual.shape or before.dtype != actual.dtype:
              raise ValueError(f'Reference output contract differs: {name}')
            both_finite = bool(np.isfinite(before).all() and np.isfinite(actual).all())
            integer = actual.dtype.kind not in 'fc'
            comparisons[name] = {'reference': tensor_stats(before),
                                 'match': both_finite and bool(np.allclose(actual, before, rtol=0 if integer else rtol,
                                                                          atol=0 if integer else atol, equal_nan=False)),
                                 'max_abs_difference': float(np.max(np.abs(actual.astype(np.float64) - before.astype(np.float64))))
                                 if both_finite and actual.size else None}
          event['comparisons'] = comparisons
          if event['result'] == 'finite' and any(not value['match'] for value in comparisons.values()):
            event['result'] = 'reference_divergence'
        if capture:
          with private_file(output / f'{index:08d}.outputs.npz') as destination:
            np.savez(destination, **outputs)
      count += 1
      journal.write(json.dumps(event, allow_nan=False) + '\n')
      journal.flush()
      summary.update(frames=count, result=event['result'], last_frame_id=record['frame_id'])
      if event['result'] != 'finite':
        break
  (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
  return summary


@contextmanager
def isolated_execution():
  """Refuse comma devices and prevent accidental use of vehicle messaging APIs."""
  from openpilot.common.hardware import PC
  if not PC:
    raise RuntimeError('Numerical execution is for an isolated development host; export files only on the comma')
  from openpilot.common.prefix import OpenpilotPrefix
  from tools.navigation_model_diagnostic import no_vehicle_io
  with OpenpilotPrefix(), no_vehicle_io():
    yield


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifact', type=Path, required=True)
  parser.add_argument('--artifact-sha256', required=True)
  parser.add_argument('--source-sha256', required=True)
  parser.add_argument('--backend', choices=('CPU', 'AMD'), default='CPU')
  parser.add_argument('--frames', type=Path, required=True, help='JSONL index of exact consumed tensor NPZ files')
  parser.add_argument('--output', type=Path, required=True)
  initial = parser.add_mutually_exclusive_group(required=True)
  initial.add_argument('--initial-zero', action='store_true', help='Explicit zero state; exact drive reproduction requires its actual model-start boundary')
  initial.add_argument('--checkpoint', type=Path, help='Matching PRE-inference checkpoint produced offline')
  parser.add_argument('--reference-onnx', type=Path)
  parser.add_argument('--capture-start', type=int, help='Opt-in bounded offline pre-state/input/output export at zero-based replay index')
  parser.add_argument('--capture-count', type=int, default=1)
  args = parser.parse_args()
  if os.environ.get('NAV_MODEL_OFFLINE') != '1':
    parser.error('Explicit NAV_MODEL_OFFLINE=1 is required')
  with isolated_execution():
    runner = NativeRunner(args.artifact, args.artifact_sha256, source_sha256=args.source_sha256, backend=args.backend)
    reference = OnnxReference(args.reference_onnx, args.source_sha256) if args.reference_onnx else None
    result = replay(runner, iter_frames(args.frames), args.output, initial_zero=args.initial_zero, checkpoint_path=args.checkpoint,
                    reference=reference, capture_start=args.capture_start, capture_count=args.capture_count)
  print(json.dumps(result, indent=2))
  if result['result'] != 'finite':
    raise SystemExit(1)


if __name__ == '__main__':
  main()
