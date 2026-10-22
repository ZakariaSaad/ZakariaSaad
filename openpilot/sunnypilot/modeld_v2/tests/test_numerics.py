"""Failure-only numerical evidence preserves the existing rejection boundary.

Inputs and kernels are synthetic; no GPU or vehicle interfaces are opened.
"""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tinygrad.tensor import Tensor

from openpilot.selfdrive.navigation.tests.test_activation import reviewed_fixture
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.numerics import nonfinite_details
from openpilot.sunnypilot.modeld_v2.tests.test_driving_navigation import fixture_data


class SensitiveValue:
  def __array__(self, *_args, **_kwargs):
    raise AssertionError('Diagnostic logging must not convert a device or private object')

  def __repr__(self):
    raise AssertionError('Diagnostic logging must not represent private route data')


class TestNonfiniteDetails(unittest.TestCase):
  def test_counts_and_sections_identify_raw_failure_without_prediction_values(self):
    output = np.array([0., 1., np.nan, np.inf, -np.inf, 5.], np.float32)
    sections = {'plan': slice(0, 3), 'action': slice(3, 6), 'finite': slice(0, 2), 'pad': slice(-1, None)}
    details = nonfinite_details(output, sections, {})
    stats = details['output_stats']
    self.assertEqual(stats['shape'], [6])
    self.assertEqual(stats['dtype'], 'float32')
    self.assertEqual({name: stats[name] for name in ('total', 'finite', 'nan', 'posinf', 'neginf')},
                     {'total': 6, 'finite': 3, 'nan': 1, 'posinf': 1, 'neginf': 1})
    self.assertEqual(details['first_invalid_indices'], [2, 3, 4])
    bad = details['invalid_output_sections']
    self.assertEqual(set(bad), {'plan', 'action'})
    self.assertEqual(bad['plan']['nan'], 1)
    self.assertEqual(bad['action']['posinf'], 1)
    self.assertEqual(bad['action']['neginf'], 1)
    json.dumps(details, allow_nan=False)

  def test_invalid_indices_are_bounded_for_a_fully_invalid_output(self):
    details = nonfinite_details(np.full(18580, np.nan, np.float32), {'hidden_state': slice(0, 18580)}, {})
    self.assertEqual(details['output_stats']['nan'], 18580)
    self.assertLessEqual(len(details['first_invalid_indices']), 8)
    self.assertLess(len(json.dumps(details, allow_nan=False)), 5000)

  def test_cpu_input_stats_and_action_timings_do_not_include_route_tensors(self):
    navigation = np.zeros((1, 47), np.float32)
    navigation[0, -1] = 19.123456
    host = {'tfm': np.eye(3, dtype=np.float32), 'nav_map': np.full((256, 256), 123, np.uint8),
            'navigation': navigation, 'action_t': np.array([.275, np.inf], np.float32),
            'private_route_geometry': SensitiveValue()}
    context = {'frame_id': 321, 'camera_sof_ns': 123_000_000_000, 'run_count': 44,
               'map_reason': 'valid', 'maneuver_reason': 'stale_instruction',
               'latitude': 35.123456789, 'route': SensitiveValue(), 'destination': 'PRIVATE_DESTINATION_MARKER'}
    details = nonfinite_details(np.array([np.nan], np.float32), {'action': slice(0, 1)}, host,
                                context=context, inference_ms=42.)
    stats = details['host_input_stats']
    self.assertEqual(stats['tfm']['finite'], 9)
    self.assertEqual(stats['nav_map']['total'], 65536)
    self.assertEqual(stats['navigation']['finite'], 47)
    self.assertEqual(stats['action_t']['posinf'], 1)
    self.assertAlmostEqual(details['action_t'][0], .275)
    self.assertIsNone(details['action_t'][1])
    self.assertEqual(details['recurrent_state_finiteness'], 'not_observed_no_extra_device_reads')
    self.assertNotIn('private_route_geometry', stats)
    self.assertEqual(details['context']['frame_id'], 321)
    self.assertEqual(details['context']['map_reason'], 'valid')
    serialized = json.dumps(details, allow_nan=False)
    self.assertNotIn('PRIVATE_DESTINATION_MARKER', serialized)
    self.assertNotIn('latitude', serialized)
    self.assertNotIn('35.123456789', serialized)
    self.assertNotIn('19.123', serialized)
    self.assertNotIn('private_route_geometry', serialized)

  def test_host_arrays_and_primary_output_are_not_modified(self):
    output = np.array([np.nan, np.inf, -np.inf, .25], np.float32)
    host = {'action_t': np.array([[.2, .3]], np.float32), 'tfm': np.eye(3, dtype=np.float32),
            'navigation': np.arange(47, dtype=np.float32)}
    before = [output.tobytes(), *[value.tobytes() for value in host.values()]]
    nonfinite_details(output, {'action': slice(0, 4)}, host)
    self.assertEqual(before, [output.tobytes(), *[value.tobytes() for value in host.values()]])

  def test_non_array_inputs_are_not_coerced_or_read_from_devices(self):
    details = nonfinite_details(np.array([np.nan], np.float32), {'action': slice(0, 1)},
                                {'navigation': SensitiveValue(), 'state_feat_q': SensitiveValue()})
    self.assertNotIn('navigation', details['host_input_stats'])
    self.assertNotIn('state_feat_q', details['host_input_stats'])
    json.dumps(details, allow_nan=False)

  def test_float16_and_nonfinite_timing_produce_strict_json(self):
    details = nonfinite_details(np.array([1., np.nan, np.inf], np.float16), {'action': slice(0, 3)}, {}, inference_ms=float('nan'))
    self.assertEqual(details['output_stats']['dtype'], 'float16')
    self.assertEqual(details['output_stats']['finite'], 1)
    json.dumps(details, allow_nan=False)


class TestModelStateNonfiniteBoundary(unittest.TestCase):
  def run_model(self, invalid, *, logger_error=None):
    jits, metadata = fixture_data()

    def trusted_graph(host, output_buffers, **_states):
      output = output_buffers['outputs']
      data = np.frombuffer(output._buffer().as_memoryview(allow_zero_copy=True), dtype=np.float32)
      data[:] = 0
      if invalid:
        data[2062] = np.nan

    jits['run'] = trusted_graph
    error_state = SimpleNamespace(host=SimpleNamespace(view=lambda **_kwargs: [0]))
    with reviewed_fixture(jits['input_specs'], metadata) as (_, permit, _), \
         patch.object(modeld, 'load_model', return_value=jits), \
         patch.object(modeld, 'Device', {'AMD': SimpleNamespace(error_state=error_state)}), \
         patch.object(modeld.cloudlog, 'event', side_effect=logger_error) as event:
      state = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
      state.parser = Mock()
      state.parser.parse_outputs.return_value = {'verified_test_result': True}
      states = {id(tensor) for tensor in (*state.adapter.input_queues.values(), *state.adapter.outputs.values())
                if tensor is not state.adapter.outputs['outputs']}
      original_numpy = Tensor.numpy

      def no_state_readback(tensor, *args, **kwargs):
        if id(tensor) in states:
          raise AssertionError('Failure evidence must not read recurrent state tensors')
        return original_numpy(tensor, *args, **kwargs)

      with patch.object(Tensor, 'numpy', no_state_readback):
        frames = {'img': bytes(1622016), 'big_img': bytes(1622016)}
        transforms = {'img': np.eye(3, dtype=np.float32), 'big_img': np.eye(3, dtype=np.float32)}
        inputs = {'desire': np.zeros(8, np.float32), 'action_t': np.array([.2, .4], np.float32),
                  'nav_map': np.zeros((256, 256), np.uint8), 'navigation': np.zeros(47, np.float32)}
        if invalid:
          with self.assertRaisesRegex(RuntimeError, '^model output not finite$'):
            state.run(frames, transforms, inputs)
          state.parser.parse_outputs.assert_not_called()
          raw = state.adapter.outputs['outputs'].numpy()
          self.assertTrue(np.isnan(raw.flat[2062]), 'Do not replace a failed output with a fabricated finite prediction')
        else:
          self.assertEqual(state.run(frames, transforms, inputs), {'verified_test_result': True})
          state.parser.parse_outputs.assert_called_once()
      return event.call_args_list

  def test_primary_failure_is_logged_before_unchanged_rejection(self):
    calls = self.run_model(True)
    self.assertTrue(calls)
    self.assertEqual(calls[-1].args[0], 'model_output_nonfinite')

  def test_logging_failure_cannot_mask_the_original_inference_failure(self):
    self.run_model(True, logger_error=OSError('Log storage unavailable'))

  def test_healthy_inference_does_not_emit_failure_evidence(self):
    self.assertEqual(self.run_model(False), [])
