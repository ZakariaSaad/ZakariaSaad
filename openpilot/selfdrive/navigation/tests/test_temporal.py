from dataclasses import replace
import os
import unittest
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation import model_inputs
from openpilot.selfdrive.navigation.temporal import NavigationContext, NavigationTransitions, context_reason


def valid_context():
  return NavigationContext(camera_ns=100_000_000_000, gps_ns=99_100_000_000, instruction_ns=99_100_000_000,
                           map_source_ns=99_100_000_000, map_render_ns=100_000_000_000,
                           generation=12, destination_key='a'*64, instruction_generation=12, instruction_destination_key='a'*64,
                           map_generation=12, map_destination_key='a'*64, frame_id=40, metadata_frame_id=40,
                           gps_valid=True, instruction_valid=True, map_valid=True)


class TestTemporal(unittest.TestCase):
  def test_physical_sources_and_exact_route_frame_association(self):
    context = valid_context()
    self.assertEqual(context_reason(context), 'valid')  # 1Hz GPS is valid despite old physical sample.
    cases = [
      ({'permission_valid': False}, 'permission_revoked'),
      ({'gps_valid': False}, 'missing_gps'),
      ({'instruction_valid': False}, 'invalid_producer'),
      ({'map_valid': False}, 'invalid_producer'),
      ({'generation': 0}, 'missing_route_identity'),
      ({'instruction_generation': 11}, 'instruction_route_mismatch'),
      ({'map_generation': 11}, 'map_route_mismatch'),
      ({'map_destination_key': 'b'*64}, 'map_route_mismatch'),
      ({'metadata_frame_id': 41}, 'map_frame_mismatch'),
      ({'instruction_ns': 0}, 'missing_source_timestamp'),
      ({'gps_ns': 97_000_000_000}, 'source_time_out_of_range'),
      ({'instruction_ns': 100_300_000_000}, 'source_time_out_of_range'),
      ({'map_render_ns': 99_400_000_000}, 'stale_or_future_render'),
      ({'map_render_ns': 100_300_000_000}, 'stale_or_future_render'),
      ({'map_age': 0.5}, 'stale_delivery'),
      ({'metadata_age': float('nan')}, 'stale_delivery'),
      ({'instruction_age': 2}, 'stale_delivery'),
    ]
    for changes, expected in cases:
      with self.subTest(changes=changes):
        self.assertEqual(context_reason(replace(context, **changes)), expected)

  def test_reroute_loss_recovery_and_clock_reset(self):
    transitions = NavigationTransitions()
    first = transitions.update((1, 'a'), True, 100)
    self.assertTrue(first.infer and first.reset)
    self.assertFalse(transitions.update((1, 'a'), True, 150).reset)
    route = transitions.update((2, 'a'), True, 200)
    self.assertTrue(route.infer and route.reset)
    self.assertEqual(route.reason, 'route_changed')
    lost = transitions.update((2, 'a'), False, 250, 'missing_gps')
    self.assertFalse(lost.infer)
    self.assertTrue(lost.reset)
    self.assertFalse(transitions.update((2, 'a'), False, 300, 'missing_gps').reset)
    recovered = transitions.update((2, 'a'), True, 350)
    self.assertTrue(recovered.infer and recovered.reset)
    self.assertEqual(recovered.reason, 'navigation_reacquired')
    backwards = transitions.update((2, 'a'), True, 100)
    self.assertFalse(backwards.infer)
    self.assertTrue(backwards.reset)
    self.assertEqual(backwards.reason, 'camera_clock_discontinuity')

  def test_camera_gap_clears_history_before_next_valid_sample(self):
    transitions = NavigationTransitions()
    transitions.update((1, 'a'), True, 1_000_000_000)
    self.assertFalse(transitions.update((1, 'a'), True, 1_050_000_000).reset)
    delayed = transitions.update((1, 'a'), True, 1_150_000_000)
    self.assertTrue(delayed.infer and delayed.reset)
    self.assertEqual(delayed.reason, 'camera_frame_gap')
    self.assertFalse(transitions.update((1, 'a'), True, 1_200_000_000).reset)

  def test_unaligned_camera_and_python_clock_domains_fail_closed(self):
    context = valid_context()
    self.assertEqual(context_reason(replace(context, camera_ns=context.camera_ns + 10_000_000_000)), 'source_time_out_of_range')


class TestScopedDiagnosticPermit(OpenpilotTestCase):
  def test_session_token_revocation_and_default_hardware_denial(self):
    params = Params()
    specs = {'nav_map': ((1, 1, 256, 256), 'float32', 'CPU'), 'navigation': ((1, 47), 'float32', 'CPU')}
    with patch.dict(os.environ, {'NAV_MODEL_DIAGNOSTIC': '1'}), patch.object(model_inputs, 'PC', False):
      with self.assertRaises(model_inputs.NavigationModelDisabled):
        with model_inputs.diagnostic_session('CPU'):
          pass
      params.put_bool('NavigationModelInputsEnabled', True, block=True)
      with model_inputs.diagnostic_session('CPU'):
        first = model_inputs.diagnostic_session_token()
        self.assertIsNotNone(first)
        self.assertTrue(model_inputs.validate_model_inputs(specs, params, diagnostic=True))
        with self.assertRaises(model_inputs.NavigationModelDisabled):
          model_inputs.validate_model_inputs(specs, params)  # Normal AGNOS inference stays denied.
        params.put_bool('NavigationModelInputsEnabled', False, block=True)
        self.assertFalse(model_inputs.diagnostic_session_active())
        with self.assertRaises(model_inputs.NavigationModelDisabled):
          model_inputs.validate_model_inputs(specs, params, diagnostic=True)
        params.put_bool('NavigationModelInputsEnabled', True, block=True)
      self.assertIsNone(model_inputs.diagnostic_session_token())
      with model_inputs.diagnostic_session('CPU'):
        self.assertIsNot(first, model_inputs.diagnostic_session_token())
        with self.assertRaises(model_inputs.NavigationModelDisabled):
          model_inputs.validate_model_inputs({'new_img': ((2, 6, 128, 256), 'uint8', 'AMD')}, params, diagnostic=True)
        os.environ['NAV_MODEL_DIAGNOSTIC'] = '0'
        self.assertFalse(model_inputs.diagnostic_session_active())
