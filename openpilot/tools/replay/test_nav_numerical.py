"""Offline replay sequencing, exact-state comparisons and native buffer aliases."""
import base64
import json
import importlib.util
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from openpilot.tools.replay.nav_numerical import (
  STATE_NAMES, NativeRunner, OnnxReference, arrays_from_file, camera_frames, checkpoint, input_problems, isolated_execution, iter_frames, replay, sha256,
)


class SequenceRunner:
  backend = 'test:host-numpy'
  source_sha256 = 'a' * 64
  artifact_sha256 = 'b' * 64
  state_specs = dict.fromkeys(STATE_NAMES, ((1,), np.dtype('float32')))
  host_specs = {'navigation': ((1, 47), np.dtype('float32'))}
  output_slices = {'map_features': slice(0, 1)}

  def __init__(self, *, fail_at=None):
    self.calls = self.initializations = 0
    self.fail_at = fail_at
    self.state = {}

  def set_state(self, state):
    self.initializations += 1
    self.state = {name: value.copy() for name, value in state.items()}

  def snapshot(self):
    return {name: value.copy() for name, value in self.state.items()}

  def run(self, values):
    self.calls += 1
    for value in self.state.values():
      value += 1
    output = np.array([self.calls], np.float32)
    if self.calls == self.fail_at:
      output[:] = np.nan
    return {'outputs': output, **{'next_' + name: value.copy() for name, value in self.state.items()}}


def frames(count):
  for index in range(count):
    navigation = np.zeros((1, 47), np.float32)
    navigation[0, 15], navigation[0, -1] = 1, 42.67
    yield {'frame_id': 100 + index, 'log_mono_time': 1000000 + index, 'segment': index // 30}, {'navigation': navigation}


class TestSequence(unittest.TestCase):
  def test_continuous_across_segments_and_reference_uses_primary_prestate(self):
    observed = []

    class Reference:
      backend = 'test:reference'

      def run(self, values, state):
        observed.append(float(state[STATE_NAMES[0]][0]))
        return {'outputs': state[STATE_NAMES[0]] + 1, **{'next_' + name: value + 1 for name, value in state.items()}}

    with tempfile.TemporaryDirectory() as directory:
      runner = SequenceRunner()
      result = replay(runner, frames(94), Path(directory) / 'result', initial_zero=True, reference=Reference())
      self.assertEqual(result['result'], 'finite')
      self.assertEqual(result['frames'], 94)
      self.assertEqual(runner.initializations, 1)
      self.assertEqual(observed, list(range(94)))

  def test_capture_is_before_alias_mutation_and_resumable(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'capture'
      replay(SequenceRunner(), frames(5), output, initial_zero=True, capture_start=3, capture_count=1)
      saved = arrays_from_file(output / '00000003.pre.npz')
      self.assertEqual(saved['_phase'].item(), 'pre_inference')
      self.assertEqual(saved['_next_frame_id'].item(), 103)
      self.assertTrue(all(saved[name][0] == 3 for name in STATE_NAMES))
      self.assertEqual((output / '00000003.pre.npz').stat().st_mode & 0o777, 0o600)
      after = arrays_from_file(output / '00000003.outputs.npz')
      self.assertTrue(all(after['next_' + name][0] == 4 for name in STATE_NAMES))
      resumed = SequenceRunner()
      result = replay(resumed, list(frames(5))[3:], Path(directory) / 'resumed', checkpoint_path=output / '00000003.pre.npz')
      self.assertEqual(result['frames'], 2)
      self.assertTrue(all(value[0] == 5 for value in resumed.state.values()))

  def test_wrong_checkpoint_phase_and_frame_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'checkpoint.npz'
      state = {name: np.zeros(1, np.float32) for name in STATE_NAMES}
      checkpoint(path, state, source_sha256='a' * 64, artifact_sha256='b' * 64, next_frame_id=999)
      with self.assertRaisesRegex(ValueError, '_next_frame_id'):
        replay(SequenceRunner(), frames(1), Path(directory) / 'bad', checkpoint_path=path)
      with (Path(directory) / 'post.npz').open('wb') as destination:
        np.savez(destination, **state, _phase=np.array('post_inference'), allow_pickle=False)
      with self.assertRaisesRegex(ValueError, '_phase'):
        replay(SequenceRunner(), frames(1), Path(directory) / 'postbad', checkpoint_path=Path(directory) / 'post.npz')

  def test_first_invalid_output_stops_and_names_section(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'failure'
      runner = SequenceRunner(fail_at=71)
      result = replay(runner, frames(100), output, initial_zero=True)
      self.assertEqual(result['result'], 'nonfinite_output')
      self.assertEqual(runner.calls, 71)
      event = json.loads((output / 'events.jsonl').read_text().splitlines()[-1])
      self.assertEqual(event['invalid_primary_sections']['map_features']['nan'], 1)
      self.assertTrue(all(value['finite'] == 1 for value in event['incoming_state'].values()))
      self.assertFalse(list(output.glob('*.npz')), 'Capture is opt-in')

  def test_invalid_incoming_state_is_distinct_and_never_executed(self):
    with tempfile.TemporaryDirectory() as directory:
      state = {name: np.zeros(1, np.float32) for name in STATE_NAMES}
      state[STATE_NAMES[2]][0] = np.inf
      path = Path(directory) / 'state.npz'
      checkpoint(path, state, source_sha256='a' * 64, artifact_sha256='b' * 64, next_frame_id=100)
      runner = SequenceRunner()
      result = replay(runner, frames(1), Path(directory) / 'badstate', checkpoint_path=path)
      self.assertEqual(result['result'], 'invalid_incoming_state')
      self.assertEqual(runner.calls, 0)

  def test_bad_input_bounds_dtypes_and_capture_size(self):
    self.assertEqual(input_problems(next(frames(1))[1]), [])  # 42.67 distance is valid, not clipped to one.
    self.assertTrue(input_problems({'nav_map': np.array([256], np.float32)}))
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, 'eight'):
        replay(SequenceRunner(), frames(1), Path(directory) / 'oversized', initial_zero=True, capture_count=9)
      record, values = next(frames(1))
      values['navigation'] = values['navigation'].astype(np.float64)
      with self.assertRaisesRegex(ValueError, 'expected'):
        replay(SequenceRunner(), [(record, values)], Path(directory) / 'dtype', initial_zero=True)

  def test_reference_divergence_stops_without_replacing_primary_outputs(self):
    class Different:
      backend = 'test:divergent'

      def run(self, values, state):
        return {'outputs': np.array([2.], np.float32), **{'next_' + name: value + 1 for name, value in state.items()}}

    with tempfile.TemporaryDirectory() as directory:
      runner = SequenceRunner()
      result = replay(runner, frames(10), Path(directory) / 'different', initial_zero=True, reference=Different())
      self.assertEqual(result['result'], 'reference_divergence')
      self.assertEqual(runner.calls, 1)

  def test_hash_and_clock_checks_before_tensor_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      np.savez(folder / 'frame.npz', navigation=np.zeros((1, 47), np.float32))
      manifest = folder / 'frames.jsonl'
      record = {'file': 'frame.npz', 'sha256': sha256(folder / 'frame.npz'), 'frame_id': 100, 'log_mono_time': 123}
      manifest.write_text(json.dumps(record) + '\n')
      self.assertEqual(len(list(iter_frames(manifest))), 1)
      manifest.write_text(json.dumps(record) + '\n' + json.dumps(record) + '\n')
      with self.assertRaisesRegex(ValueError, 'ordered'):
        list(iter_frames(manifest))
      manifest.write_text(json.dumps(record | {'sha256': '0' * 64}) + '\n')
      with self.assertRaisesRegex(ValueError, 'SHA256'):
        list(iter_frames(manifest))

  def test_comma_device_refused_before_runtime_io(self):
    with patch('openpilot.common.hardware.PC', False):
      with self.assertRaisesRegex(RuntimeError, 'isolated development host'):
        with isolated_execution():
          self.fail('Comma execution must be rejected')

  def test_pc_execution_context_blocks_live_messaging(self):
    from openpilot.cereal import messaging
    from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled
    with isolated_execution():
      with self.assertRaisesRegex(NavigationModelDisabled, 'cannot publish'):
        messaging.PubMaster(['sendcan'])
      with self.assertRaisesRegex(NavigationModelDisabled, 'cannot publish'):
        messaging.SubMaster(['carState'])

  def test_archive_bounds_and_object_data_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'duplicate.npz'
      with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('x.npy', b'first')
        with self.assertWarns(UserWarning):
          archive.writestr('x.npy', b'second')
      with self.assertRaisesRegex(ValueError, 'duplicate'):
        arrays_from_file(path)
      np.savez(Path(directory) / 'object.npz', value=np.array([object()]))
      with self.assertRaisesRegex(ValueError, 'Object arrays'):
        arrays_from_file(Path(directory) / 'object.npz')

  def test_private_replay_output_cannot_be_inside_checkout(self):
    with tempfile.TemporaryDirectory() as directory:
      (Path(directory) / '.git').mkdir()
      with self.assertRaisesRegex(ValueError, 'outside Git'):
        replay(SequenceRunner(), frames(1), Path(directory) / 'private', initial_zero=True)


class TestNativeAliases(unittest.TestCase):
  def test_real_sptg_roundtrip_keeps_continuous_native_state(self):
    from tinygrad import Tensor, TinyJit
    from openpilot.sunnypilot.modeld_v2.helpers import write_bundle
    from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit

    specs = dict.fromkeys(STATE_NAMES, ((1,), 'float32', 'CPU'))
    inputs = {name: Tensor(np.zeros(shape, dtype), device=device).realize() for name, (shape, dtype, device) in specs.items()}
    inputs['navigation'] = Tensor(np.zeros((1, 47), np.float32), device='CPU').realize()
    output_specs = {'outputs': ((1, 1), 'float32', 'CPU')} | {'next_' + name: spec for name, spec in specs.items()}
    outputs = {name: Tensor(np.zeros(shape, dtype), device=device).realize() for name, (shape, dtype, device) in output_specs.items()}

    @TinyJit
    def source(output_buffers, **values):
      predictions = {'outputs': (values['state_feat_q'] + values['navigation'].sum()).reshape(1, 1)}
      predictions.update({'next_' + name: values[name] + 1 for name in STATE_NAMES})
      Tensor.realize(*(output_buffers[name].assign(value) for name, value in predictions.items()))

    with record_kernel_linear() as kernels:
      for _ in range(3):
        source(output_buffers=outputs, **inputs)
    self.assertEqual(len(kernels), 1)
    check, _ = capture_frame(source, inputs | {'output_buffers': outputs}, 'output_buffers')
    part = serialize_jit(source, kernels[0], check)
    extras = {'input_specs': specs | {'navigation': ((1, 47), 'float32', 'CPU')}, 'output_specs': output_specs,
              'metadata': {'metadata': {'output_slices': base64.b64encode(pickle.dumps({'map_features': slice(0, 1)})).decode()}}}
    with tempfile.TemporaryDirectory() as directory:
      artifact = Path(directory) / 'actual-native.sptg'
      write_bundle(str(artifact), {'run': part}, extras)
      runner = NativeRunner(artifact, sha256(artifact), source_sha256='a' * 64)
      result = replay(runner, frames(67), Path(directory) / 'actual-results', initial_zero=True)
      self.assertEqual(result['frames'], 67)
      self.assertEqual(result['result'], 'finite')
      self.assertTrue(all(value[0] == 67 for value in runner.snapshot().values()))

  def test_actual_npy_native_views_overwrite_state_but_not_pre_snapshot(self):
    specs = dict.fromkeys(STATE_NAMES, ((1,), 'float32', 'NPY'))
    jits = {'input_specs': specs | {'navigation': ((1, 47), 'float32', 'NPY')},
            'output_specs': {'outputs': ((1, 1), 'float32', 'NPY')} |
                            {'next_' + name: spec for name, spec in specs.items()},
            'metadata': {'metadata': {'output_slices': base64.b64encode(pickle.dumps({'map_features': slice(0, 1)})).decode()}}}

    def graph(output_buffers, **inputs):
      for name in STATE_NAMES:
        before = inputs[name].numpy().copy()
        buffer = output_buffers['next_' + name]._buffer().as_memoryview(allow_zero_copy=True)
        buffer[:] = (before + 1).tobytes()
      output_buffers['outputs']._buffer().as_memoryview(allow_zero_copy=True)[:] = np.array([[1.]], np.float32).tobytes()

    jits['run'] = graph
    parts = {'run': ({'calls': [{'kind': 'program', 'target': {'device': 'CPU'}}]}, b'')}
    with tempfile.TemporaryDirectory() as directory:
      artifact = Path(directory) / 'fixture.sptg'
      artifact.write_bytes(b'trusted fixture')
      with patch('openpilot.sunnypilot.modeld_v2.helpers.read_parts', return_value=(parts, jits)), \
           patch('openpilot.sunnypilot.modeld_v2.helpers.load_model', return_value=jits) as loader:
        runner = NativeRunner(artifact, sha256(artifact), source_sha256='a' * 64)
        before = runner.snapshot()
        runner.run({'navigation': np.zeros((1, 47), np.float32)})
        self.assertTrue(all(value[0] == 0 for value in before.values()))
        self.assertTrue(all(value[0] == 1 for value in runner.snapshot().values()))
        loader.assert_called_once_with(str(artifact), cache=False, verify=True)


class TestCameraDecode(unittest.TestCase):
  @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is needed for the real FrameReader fixture')
  def test_actual_hevc_decoder_joins_encode_indices_and_pads_nv12(self):
    from openpilot.cereal import messaging
    from openpilot.tools.lib.logreader import save_log
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      video = folder / 'camera.hevc'
      subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=128x64:rate=20:duration=0.1',
                      '-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1:log-level=error', '-f', 'hevc', str(video)],
                     check=True, capture_output=True)
      messages = []
      for index in range(3):
        camera = messaging.new_message('narrowRoadCameraState')
        camera.logMonoTime = 1000 + index * 100
        camera.narrowRoadCameraState.frameId = 7000 + index
        camera.narrowRoadCameraState.timestampSof = 900 + index * 100
        camera.narrowRoadCameraState.timestampEof = 950 + index * 100
        messages.append(camera)
        if index < 2:
          encoded = messaging.new_message('narrowRoadEncodeIdx')
          encoded.logMonoTime = camera.logMonoTime + 1
          encoded.narrowRoadEncodeIdx.frameId = 7000 + index
          encoded.narrowRoadEncodeIdx.segmentId = index
          encoded.narrowRoadEncodeIdx.segmentNum = 6
          messages.append(encoded)
      logfile = folder / 'rlog.zst'
      save_log(str(logfile), [message.as_reader() for message in messages])
      decoded = list(camera_frames([logfile], video))
      self.assertEqual([item[0]['frame_id'] for item in decoded], [7000, 7001])
      self.assertEqual(decoded[0][0]['boundary_camera_frames_not_encoded'], 1)
      self.assertEqual(decoded[0][1].shape, (128 * 96,))
      self.assertEqual(decoded[0][1].dtype, np.uint8)
      messages[1].narrowRoadEncodeIdx.segmentNum = 7
      save_log(str(logfile), [message.as_reader() for message in messages])
      with self.assertRaisesRegex(ValueError, 'exactly one'):
        list(camera_frames([logfile], video))


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX CPU reference packages')
class TestOnnxReference(unittest.TestCase):
  def make_model(self, folder, *, custom='Contiguous'):
    import onnx
    from onnx import helper, TensorProto
    names = ['navigation', *STATE_NAMES]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in names]
    outputs = [helper.make_tensor_value_info('outputs', TensorProto.FLOAT, [1])] + [
      helper.make_tensor_value_info('next_' + name, TensorProto.FLOAT, [1]) for name in STATE_NAMES]
    nodes = [helper.make_node(custom, ['navigation'], ['outputs'], domain='org.tinygrad')] + [
      helper.make_node('Identity', [name], ['next_' + name]) for name in STATE_NAMES]
    model = helper.make_model(helper.make_graph(nodes, 'reference-layout', inputs, outputs),
                              opset_imports=[helper.make_opsetid('', 18), helper.make_opsetid('org.tinygrad', 1)])
    model.ir_version = 10
    path = Path(folder) / 'reference.onnx'
    onnx.save(model, str(path))
    return path

  def test_contiguous_layout_identity_preserves_values_dtypes_shapes_and_source(self):
    with tempfile.TemporaryDirectory() as directory:
      path = self.make_model(directory)
      digest = sha256(path)
      reference = OnnxReference(path, digest)
      states = {name: np.array([index + .5], np.float32) for index, name in enumerate(STATE_NAMES)}
      inputs = {'navigation': np.array([42.67], np.float32)}
      result = reference.run(inputs, states)
      np.testing.assert_array_equal(result['outputs'], inputs['navigation'])
      for name, array in states.items():
        np.testing.assert_array_equal(result['next_' + name], array)
        self.assertEqual(result['next_' + name].dtype, array.dtype)
      self.assertEqual(sha256(path), digest)
      self.assertEqual(reference.adaptations['org.tinygrad::Contiguous_to_ONNX_Identity'], 1)
      self.assertEqual(reference.adaptations['graph_optimizations'], 'disabled')
      self.assertEqual(reference.session.get_providers(), ['CPUExecutionProvider'])

  def test_unknown_custom_operator_and_wrong_source_hash_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = self.make_model(directory, custom='UnreviewedOperation')
      with self.assertRaisesRegex(ValueError, 'SHA256'):
        OnnxReference(path, '0' * 64)
      with self.assertRaisesRegex(ValueError, 'Unsupported reference custom operation'):
        OnnxReference(path, sha256(path))


if __name__ == '__main__':
  unittest.main()
