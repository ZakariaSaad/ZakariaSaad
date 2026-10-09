import copy
import tempfile
import unittest
import numpy as np
from tinygrad import Tensor

from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.stateful import StatefulRunner, make_stateful_jit, validate_specs

SLICES = {'lane_lines': slice(0, 528), 'lane_lines_prob': slice(528, 536), 'road_edges': slice(536, 800),
          'meta': slice(800, 855), 'desire_pred': slice(855, 887), 'pose': slice(887, 899), 'wide_from_device_euler': slice(899, 905),
          'road_transform': slice(905, 917), 'plan': slice(917, 1907), 'lead': slice(1907, 2051), 'lead_prob': slice(2051, 2054),
          'desire_state': slice(2054, 2062), 'action': slice(2062, 2066)}


class SyntheticModel:
  def __call__(self, inputs):
    state = inputs['state_test']
    raw = Tensor.zeros(1, 2062, device=state.device).cat(state, state, Tensor.zeros(1, 2, device=state.device), dim=1)
    return {'outputs': raw, 'next_state_test': state + 1}


def artifact():
  specs = {'new_img': ((2, 6, 2, 2), 'uint8', 'CPU'), 'desire': ((8,), 'float32', 'CPU'),
           'traffic_convention': ((1, 2), 'float32', 'CPU'), 'action_t': ((1, 2), 'float32', 'CPU'),
           'state_test': ((1, 1), 'float32', 'CPU')}
  return {'input_specs': specs, 'output_specs': {'outputs': ((1, 2066), 'float32', 'CPU'),
                                                'next_state_test': specs['state_test']},
          'metadata': {'output_slices': SLICES}, 'run': make_stateful_jit(SyntheticModel())}


class TestStatefulModel(unittest.TestCase):
  def setUp(self):
    self.runner = StatefulRunner(artifact(), Parser())
    self.image = Tensor(np.zeros((2, 6, 2, 2), dtype='uint8'), device='CPU').realize()

  def run_frame(self, desire=None):
    return self.runner.run(self.image, np.zeros(8) if desire is None else desire, np.array([[1., 0.]]), np.zeros((1, 2)))

  def test_state_advances_across_capture_and_replay(self):
    for frame in range(7):
      self.assertEqual(self.run_frame()['action'][0, 0], frame)
      np.testing.assert_array_equal(self.runner.inputs['state_test'].numpy(), [[frame + 1]])

  def test_reset_restores_first_frame_and_desire_edge(self):
    desire = np.eye(8, dtype='float32')[3]
    self.run_frame(desire)
    self.run_frame(desire)
    self.assertEqual(self.runner.inputs['desire'].numpy()[3], 0)
    self.runner.reset()
    self.assertEqual(self.run_frame(desire)['action'][0, 0], 0)
    self.assertEqual(self.runner.inputs['desire'].numpy()[3], 1)

  def test_serialized_jit_preserves_recurrence_and_reset(self):
    self.run_frame()
    self.run_frame()
    self.run_frame()
    self.runner.reset()
    saved = artifact()
    saved['run'] = self.runner.run_model
    with tempfile.TemporaryFile() as f:
      dump_oob(saved, f)
      f.seek(0)
      self.runner = StatefulRunner(load_oob(f), Parser())
    self.runner.reset()
    self.assertEqual(self.run_frame()['action'][0, 0], 0)
    self.assertEqual(self.run_frame()['action'][0, 0], 1)

  def test_invalid_next_state_is_not_committed(self):
    def bad_run(output_buffers, **kwargs):
      output_buffers['outputs'].assign(0).realize()
      output_buffers['next_state_test'].assign(float('nan')).realize()
    self.runner.run_model = bad_run
    with self.assertRaisesRegex(ValueError, 'recurrent state'):
      self.run_frame()
    np.testing.assert_array_equal(self.runner.inputs['state_test'].numpy(), [[0]])

  def test_invalid_prediction_is_not_committed(self):
    def bad_run(output_buffers, **kwargs):
      output_buffers['outputs'].assign(float('inf')).realize()
      output_buffers['next_state_test'].assign(1).realize()
    self.runner.run_model = bad_run
    with self.assertRaisesRegex(ValueError, 'not finite'):
      self.run_frame()
    np.testing.assert_array_equal(self.runner.inputs['state_test'].numpy(), [[0]])

  def test_missing_and_mismatched_state_contract_rejected(self):
    for change in ('missing', 'shape', 'slice'):
      with self.subTest(change=change):
        original = artifact()
        data = copy.deepcopy({k: v for k, v in original.items() if k != 'run'})
        if change == 'missing':
          data['output_specs'].pop('next_state_test')
        elif change == 'shape':
          data['output_specs']['next_state_test'] = ((2, 1), 'float32', 'CPU')
        else:
          data['metadata']['output_slices']['action'] = slice(2062, 9999)
        with self.assertRaises(ValueError):
          validate_specs(data['input_specs'], data['output_specs'], data['metadata'])

  def test_nonfinite_desire_rejected(self):
    with self.assertRaisesRegex(ValueError, 'desire'):
      self.run_frame(np.full(8, np.nan))
