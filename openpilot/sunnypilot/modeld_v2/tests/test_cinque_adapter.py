import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from openpilot.selfdrive.modeld.helpers import dump_oob
from openpilot.sunnypilot.modeld_v2.cinque import CinqueModelState, load_fallback
from openpilot.sunnypilot.modeld_v2.modeld import ModelState
from openpilot.sunnypilot.modeld_v2.cinque_guard import MODEL_COMMIT, ONNX_SHA256
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.stateful import PackedStatefulRunner
from openpilot.sunnypilot.modeld_v2.tests.test_stateful import artifact
from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref
from openpilot.cereal import custom


class TestCinqueAdapter(unittest.TestCase):
  def test_selection_rejects_missing_or_unverified_fallback(self):
    with patch('openpilot.sunnypilot.modeld_v2.cinque.get_active_bundle', return_value=None):
      with self.assertRaisesRegex(ValueError, 'legacy Tinygrad fallback'):
        load_fallback(1344, 760)
    bundle = custom.ModelManagerSP.ModelBundle(runner='tinygrad')
    with patch('openpilot.sunnypilot.modeld_v2.cinque.get_active_bundle', return_value=bundle):
      with self.assertRaisesRegex(ValueError, 'verified checksums'):
        load_fallback(1344, 760)

  def test_legacy_warmup_resets_device_history(self):
    model = ModelState.__new__(ModelState)
    model.is_run_model = False
    model._vision_input_names, model._road_key, model._wide_key = ['img', 'big_img'], 'img', 'big_img'
    model.frame_copy_size, model.frame_buf_params = 24, {'img': (4, 4, 2, 24)}
    model.numpy_inputs = {'desire': np.ones(8, dtype='float32')}
    model.input_queues = {'img_q': Tensor(np.ones((2, 2), dtype='uint8'), device='CPU').realize(),
                          'feat_q': Tensor(np.ones((2, 2), dtype='float32'), device='CPU').realize()}
    model.prev_desire = np.ones(8, dtype='float32')
    model.full_frames, model._blob_cache = {'img': object()}, {'blob': object()}
    model.run = lambda *args: None
    model.warmup()
    for value in model.input_queues.values():
      np.testing.assert_array_equal(value.numpy(), np.zeros((2, 2)))
    np.testing.assert_array_equal(model.numpy_inputs['desire'], np.zeros(8))
    self.assertFalse(model.full_frames)
    self.assertFalse(model._blob_cache)

  def test_separately_installed_artifact_startup_and_publication(self):
    data = artifact()
    data['input_specs']['new_img'] = ((2, 6, 4, 4), 'uint8', 'CPU')
    # Capture the serialized callable, as a real separately compiled model.
    from openpilot.sunnypilot.modeld_v2.stateful import StatefulRunner
    runner = StatefulRunner(data, Parser())
    image = Tensor(np.zeros((2, 6, 4, 4), dtype='uint8'), device='CPU').realize()
    for _ in range(3):
      runner.run(image, np.zeros(8), np.array([[1., 0.]]), np.zeros((1, 2)))
    data['run'] = runner.run_model
    data['tinygrad_ref'] = get_tinygrad_ref()
    with tempfile.TemporaryDirectory() as root:
      pkl = Path(root) / 'test.pkl'
      with pkl.open('wb') as f:
        dump_oob(data, f)
      manifest = {'model': 'Cinque Terre V3', 'model_commit': MODEL_COMMIT, 'source_onnx_sha256': ONNX_SHA256,
                  'sha256': hashlib.sha256(pkl.read_bytes()).hexdigest(), 'file_name': pkl.name, 'tinygrad_ref': get_tinygrad_ref()}
      with patch('openpilot.sunnypilot.modeld_v2.cinque.Paths.model_root', return_value=root), \
           patch('openpilot.sunnypilot.modeld_v2.cinque.get_nv12_info', return_value=(8, 8, 4, 96)), \
           patch('openpilot.sunnypilot.modeld_v2.cinque.ModelStateBase.__init__', lambda self: setattr(self, 'lat_delay', .2)):
        model = CinqueModelState(8, 8, manifest)
        model.warmup()
        frames = dict.fromkeys(model.vision_input_names, np.zeros(96, dtype='uint8'))
        transforms = dict.fromkeys(model.vision_input_names, np.eye(3, dtype='float32'))
        inputs = {'desire': np.zeros(8), 'traffic_convention': np.array([[1., 0.]]), 'action_t': np.zeros((1, 2))}
        output = model.run(frames, transforms, inputs)
        self.assertEqual(output['action'][0, 0], 0)
        model.check_publication(output)
        self.assertEqual(model.run(frames, transforms, inputs)['action'][0, 0], 1)

  def test_official_packed_format_recurrence_and_reset(self):
    @TinyJit
    def run_model(packed_npy_inputs, state_test):
      # Exercise the official JIT's internal assignment of named state.
      raw = Tensor.zeros(1, 2062, device='CPU').cat(state_test.contiguous(), state_test.contiguous(),
                                                Tensor.zeros(1, 2, device='CPU'), dim=1).contiguous().realize()
      state_test.assign(state_test + 1).realize()
      return raw,
    data = artifact()
    packed = {'metadata': data['metadata'] | {'output_shapes': {'outputs': (1, 2066)}},
              'input_shapes': {k: (shape, dtypes.uint8 if dtype == 'uint8' else dtypes.float32)
                               for k, (shape, dtype, _) in data['input_specs'].items()},
              'state_pairs': {'state_test': 'next_state_test'}, 'input_devices': {'model': 'CPU'}, 'run_model': {(4, 4): run_model}}
    runner = PackedStatefulRunner(packed, (4, 4), 24, Parser())
    frames = dict.fromkeys(('img', 'big_img'), np.zeros(24, dtype='uint8'))
    transforms = dict.fromkeys(('img', 'big_img'), np.eye(3, dtype='float32'))
    inputs = {'desire': np.zeros(8), 'traffic_convention': np.array([[1., 0.]]), 'action_t': np.zeros((1, 2))}
    for frame in range(5):
      self.assertEqual(runner.run_frames(frames, transforms, inputs)['action'][0, 0], frame)
    runner.reset()
    self.assertEqual(runner.run_frames(frames, transforms, inputs)['action'][0, 0], 0)
