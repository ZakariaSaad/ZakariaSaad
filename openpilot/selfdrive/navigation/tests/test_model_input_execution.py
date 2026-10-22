"""Approved-model tensor defaults exercise the real reader without a model/GPU.

Authorization is isolated here; artifact/permission validation has separate
production-path tests. These tests never select a model or emit predictions.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.navigation import model_inputs
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub
from openpilot.selfdrive.navigation.tests.test_model_input_admission import Client, Messages


class TestNavigationExecutionInputs(unittest.TestCase):
  def setUp(self):
    self.now = 100.
    self.params = ParamsStub(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
    self.client = Client(99_100_000_000)
    self.reader = model_inputs.NavigationInputs(self.params, self.client, clock=lambda: self.now, driving_permit=object())
    self.auth = self.enterContext(patch('openpilot.selfdrive.navigation.activation.validate_runtime_permit'))
    self.sm = Messages(self.now)

  def read(self):
    result = self.reader.read_for_model(self.sm, 10., int(self.now * 1e9))
    self.assertEqual(set(result), {'nav_map', 'navigation'})
    self.assertEqual(result['nav_map'].shape, (256, 256))
    self.assertEqual(result['nav_map'].dtype, np.uint8)
    self.assertEqual(result['navigation'].shape, (47,))
    self.assertEqual(result['navigation'].dtype, np.float32)
    self.assertFalse(self.reader.reset_required)
    self.assertIsNone(self.reader.reset_reason)
    self.assertIsNone(self.reader.transitions.identity, 'Driving inputs must not use diagnostic recurrent-reset transitions')
    return result

  def test_fresh_map_and_fix_relative_maneuver_have_native_tensor_values(self):
    result = self.read()
    self.assertTrue((result['nav_map'] == 128).all())
    self.assertEqual(result['navigation'][15], 1.)
    self.assertAlmostEqual(result['navigation'][46], 1.99, places=5)

  def test_missing_or_invalid_instruction_preserves_fresh_map(self):
    self.sm.valid['navInstruction'] = False
    result = self.read()
    self.assertTrue(result['nav_map'].any())
    self.assertFalse(result['navigation'].any())
    self.assertEqual(self.reader.maneuver_reason, 'missing_instruction')

  def test_missing_or_expired_map_preserves_fresh_maneuver(self):
    self.client.frame = None
    result = self.read()
    self.assertFalse(result['nav_map'].any())
    self.assertEqual(result['navigation'][15], 1.)
    self.client.frame = SimpleNamespace(data=bytes([128]) * 256 * 256, width=256, height=256)
    self.assertTrue(self.read()['nav_map'].any())
    self.now += .5  # Match the official strict <0.5-second map expiry.
    result = self.read()
    self.assertFalse(result['nav_map'].any())
    self.assertEqual(result['navigation'][15], 1.)

  def test_stale_instruction_neutralizes_only_maneuver(self):
    self.sm.recv_time['navInstruction'] = self.now - 2.
    result = self.read()
    self.assertTrue(result['nav_map'].any())
    self.assertFalse(result['navigation'].any())
    self.assertEqual(self.reader.maneuver_reason, 'stale_instruction')

  def test_missing_live_gps_does_not_override_fresh_source_tagged_inputs(self):
    # Upstream modeld trusts producer validity. This branch additionally checks
    # tagged physical source timestamps; the GPS subscription is not a model switch.
    self.sm.valid['gpsLocation'] = False
    self.sm.valid['gpsLocationExternal'] = False
    result = self.read()
    self.assertTrue(result['nav_map'].any())
    self.assertEqual(result['navigation'][15], 1.)

  def test_unknown_empty_and_missing_upcoming_maneuvers_leave_fresh_map(self):
    for maneuvers in ([], [SimpleNamespace(type='unknown', modifier='left', distance=20.)],
                      [SimpleNamespace(type='turn', modifier='left', distance=float('nan'))],
                      [SimpleNamespace(type='turn', modifier='left', distance=-20.)]):
      with self.subTest(maneuvers=maneuvers):
        self.sm.data['navInstruction'].allManeuvers = maneuvers
        result = self.read()
        self.assertTrue(result['nav_map'].any())
        self.assertFalse(result['navigation'].any())

  def test_corrupt_map_preserves_independently_valid_maneuver(self):
    self.client.frame = SimpleNamespace(data=b'bad', width=256, height=256)
    result = self.read()
    self.assertFalse(result['nav_map'].any())
    self.assertEqual(result['navigation'][15], 1.)
    self.assertEqual(self.reader.map_reason, 'invalid_map_data')

  def test_destination_clear_and_display_disable_supply_zero_defaults(self):
    self.assertTrue(self.read()['nav_map'].any())
    for key, value in (('NavDestination', ''), ('NavigationEnabled', False)):
      with self.subTest(parameter=key):
        old = self.params.values[key]
        self.params.values[key] = value
        result = self.read()
        self.assertFalse(result['nav_map'].any())
        self.assertFalse(result['navigation'].any())
        self.params.values[key] = old

  def test_new_route_does_not_mix_old_map_and_maneuver_or_wait_to_run(self):
    self.assertTrue(self.read()['nav_map'].any())
    self.now += .05
    self.sm = Messages(self.now)
    self.sm.data['navInstruction'].routeGeneration = 2
    result = self.read()
    self.assertFalse(result['nav_map'].any(), 'The old route map must not condition the new maneuver')
    self.assertEqual(result['navigation'][15], 1., 'The valid new instruction need not wait for a map')
    metadata = self.sm.data['navModelStateSP']
    metadata.routeGeneration = 2
    metadata.frameId = self.client.frame_id = 2
    metadata.locationMonoTime = self.client.timestamp_sof = int((self.now - .1) * 1e9)
    metadata.renderMonoTime = self.client.timestamp_eof = int(self.now * 1e9)
    self.client.frame = SimpleNamespace(data=bytes([64]) * 256 * 256, width=256, height=256)
    result = self.read()
    self.assertTrue((result['nav_map'] == 64).all())
    self.assertEqual(result['navigation'][15], 1.)

  def test_new_map_route_neutralizes_old_maneuver_without_discarding_map(self):
    self.sm.data['navModelStateSP'].routeGeneration = 2
    result = self.read()
    self.assertTrue(result['nav_map'].any())
    self.assertFalse(result['navigation'].any())
    self.assertEqual(self.reader.maneuver_reason, 'instruction_route_mismatch')

  def test_new_route_tag_rejects_old_map_even_when_its_maneuver_is_unavailable(self):
    self.assertTrue(self.read()['nav_map'].any())
    self.now += .05
    self.sm = Messages(self.now)
    self.sm.valid['navInstruction'] = False
    self.sm.data['navInstruction'].routeGeneration = 2
    result = self.read()
    self.assertFalse(result['nav_map'].any())
    self.assertFalse(result['navigation'].any())
    self.assertEqual(self.reader.maneuver_reason, 'missing_instruction')

  def test_permission_revocation_raises_instead_of_silently_passing_neutral_inputs(self):
    self.auth.side_effect = model_inputs.NavigationModelDisabled('permission revoked')
    with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'permission revoked'):
      self.reader.read_for_model(self.sm, 10., int(self.now * 1e9))
    self.assertIsNone(self.reader._map)
    self.reader.driving_permit = None
    with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'runtime permit'):
      self.reader.read_for_model(self.sm, 10., int(self.now * 1e9))


if __name__ == '__main__':
  unittest.main()
