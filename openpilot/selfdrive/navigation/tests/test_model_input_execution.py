"""Production inputs compared with the exact official a03 inline modeld block.

Only native artifact admission is mocked. Both paths receive independent copies
of identical IPC frames/messages; no GPU or vehicle publication is performed.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.navigation import model_inputs
from openpilot.selfdrive.navigation.helpers import navigation_input, valhalla_type
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub
from openpilot.selfdrive.navigation.tests.test_model_input_admission import Client, Messages
from openpilot.selfdrive.navigation.tests.official_reference import OfficialNavigationInputs, load_reference


def frame(value=128, *, width=256, height=256):
  return SimpleNamespace(data=np.full((height, width), value, np.uint8).tobytes(), width=width, height=height)


class TestNavigationExecutionInputs(unittest.TestCase):
  def setUp(self):
    self.now = 100.
    self.params = ParamsStub(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
    self.client, self.reference_client = Client(99_100_000_000), Client(99_100_000_000)
    self.reader = model_inputs.NavigationInputs(self.params, self.client, clock=lambda: self.now, driving_permit=object())
    self.reference = OfficialNavigationInputs()
    self.auth = self.enterContext(patch('openpilot.selfdrive.navigation.activation.validate_runtime_permit'))
    self.sm = Messages(self.now)

  def read(self, *, speed=10., camera_ns=None):
    expected = self.reference.read(copy.deepcopy(self.sm), self.reference_client, self.now, speed)
    actual = self.reader.read_for_model(self.sm, speed, int(self.now * 1e9) if camera_ns is None else camera_ns)
    defaults = {'nav_map': np.zeros((256, 256), np.uint8), 'navigation': np.zeros(47, np.float32)}
    for name, zero in defaults.items():
      np.testing.assert_array_equal(actual[name], expected.get(name, zero), err_msg=name)
      self.assertEqual(actual[name].dtype, expected.get(name, zero).dtype)
    self.assertFalse(self.reader.reset_required)
    self.assertIsNone(self.reader.transitions.identity)
    return actual

  def receive(self, value=128):
    self.client.frame, self.reference_client.frame = frame(value), frame(value)

  def test_fresh_and_republished_instruction_distance_uses_receipt_age(self):
    self.assertAlmostEqual(self.read()['navigation'][-1], 1.99, places=5)
    self.now += .5
    self.assertAlmostEqual(self.read()['navigation'][-1], 1.94, places=5)
    # Republishing the same physical fix starts the official receipt-age clock.
    self.sm.recv_time['navInstruction'] = self.now
    self.assertAlmostEqual(self.read()['navigation'][-1], 2., places=5)

  def test_timeout_boundaries_and_independent_missing_data(self):
    self.read()
    for elapsed in (.499, .5, 1.899, 1.9, 2.):
      with self.subTest(elapsed=elapsed):
        self.now = 100. + elapsed
        self.read()
    self.sm.valid['navInstruction'] = False
    self.receive(32)
    result = self.read()
    self.assertTrue(result['nav_map'].any())
    self.assertFalse(result['navigation'].any())
    self.sm.valid['navInstruction'] = True
    self.sm.recv_time['navInstruction'] = self.now
    self.now += .5
    result = self.read()
    self.assertFalse(result['nav_map'].any())
    self.assertTrue(result['navigation'].any())

  def test_reroutes_and_asynchronous_metadata_do_not_gate_frames_or_maneuvers(self):
    self.read()
    self.now += .05
    self.sm.data['navInstruction'].routeGeneration = 200
    self.sm.data['navInstruction'].destinationKey = 'new-route'
    self.sm.data['navInstruction'].gpsSourceMonoTime = 0
    self.read()  # official continues using its held map with the latest instruction
    self.receive(64)
    self.sm.valid['navModelStateSP'] = False
    self.sm.data['navModelStateSP'].frameId = 999999
    self.sm.data['navModelStateSP'].routeGeneration = 0
    self.client.frame_id = 2
    self.client.timestamp_sof = self.client.timestamp_eof = 0
    self.read()  # map pixels are consumed independently of side-channel metadata

  def test_display_destination_gps_and_camera_epochs_are_not_consumer_policy(self):
    self.read()
    self.params.values.update(NavigationEnabled=False, NavDestination=None)
    self.sm.valid['gpsLocation'] = self.sm.valid['gpsLocationExternal'] = False
    self.sm.logMonoTime = dict.fromkeys(self.sm.logMonoTime, 0)
    self.sm.data['navInstruction'].gpsSourceMonoTime = 1
    self.now += .05
    self.read(camera_ns=int((self.now + 86_400.) * 1e9))
    self.sm.recv_time['navInstruction'] = self.now + .1
    self.read(camera_ns=0)  # official has no additional negative receipt-age gate

  def test_unknown_merge_arrival_empty_and_nonfinite_guidance_matches_official(self):
    cases = [[], [('unknown', 'unknown', 100.)], [('unknown', 'left', 100.)],
             [('merge', 'right', 100.), ('turn', 'right', 200.)], [('depart', '', 10.), ('arrive', '', 50.)],
             [('turn', 'left', float('nan')), ('turn', 'right', 150.)],
             [('turn', 'left', -20.)], [('fork', 'unsupported', 100.)]]
    for commands in cases:
      with self.subTest(commands=commands):
        self.sm.data['navInstruction'].allManeuvers = [SimpleNamespace(type=k, modifier=m, distance=d) for k, m, d in commands]
        self.read()

  def test_startup_and_asynchronous_connection_attempts_match_official(self):
    class Disconnected(Client):
      def __init__(self):
        super().__init__(0)
        self.attempts = 0
      def is_connected(self):
        return False
      def connect(self, blocking):
        assert blocking is False
        self.attempts += 1
        return self.attempts == 3
    self.client = Disconnected()
    self.reference_client = Disconnected()
    self.reader._client = self.client
    for _ in range(4):
      self.read()
      self.now += .05
    self.assertEqual(self.client.attempts, self.reference_client.attempts)

  def test_corrupt_ipc_payload_has_the_same_exception_boundary(self):
    self.client.frame = self.reference_client.frame = SimpleNamespace(data=b'bad', width=256, height=256)
    with self.assertRaises(ValueError):
      self.reference.read(self.sm, self.reference_client, self.now, 10.)
    with self.assertRaises(ValueError):
      self.reader.read_for_model(self.sm, 10., int(self.now * 1e9))

  def test_artifact_revocation_is_not_replaced_with_zero_predictions(self):
    self.auth.side_effect = model_inputs.NavigationModelDisabled('artifact changed')
    with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'artifact changed'):
      self.reader.read_for_model(self.sm, 10., int(self.now * 1e9))
    self.reader.driving_permit = None
    with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'runtime permit'):
      self.reader.read_for_model(self.sm, 10., int(self.now * 1e9))


class TestOfficialManeuverEncoding(unittest.TestCase):
  def test_command_and_normalization_cross_product(self):
    official = load_reference('helpers')
    kinds = ['depart', 'arrive', 'merge', 'on ramp', 'off ramp', 'fork', 'roundabout', 'rotary',
             'roundabout turn', 'exit roundabout', 'exit rotary', 'notification', 'turn', 'continue',
             'new name', 'end of road', 'unrecognized-future-type']
    modifiers = ['', 'straight', 'slight right', 'right', 'sharp right', 'uturn', 'sharp left', 'left', 'slight left', 'unknown']
    for kind in kinds:
      for modifier in modifiers:
        with self.subTest(kind=kind, modifier=modifier):
          self.assertEqual(valhalla_type(kind, modifier), official.valhalla_type(kind, modifier))
          maneuvers = [SimpleNamespace(type=kind, modifier=modifier, distance=100.),
                       SimpleNamespace(type='turn', modifier='right', distance=200.)]
          for traveled in (-10., 0., 99., 100., 101., 210., float('nan')):
            np.testing.assert_array_equal(navigation_input(maneuvers, traveled), official.navigation_input(maneuvers, traveled))
