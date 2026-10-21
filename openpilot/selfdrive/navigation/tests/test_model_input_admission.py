import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
from openpilot.selfdrive.navigation import model_inputs


class Messages:
  def __init__(self, now):
    self.valid = {'gpsLocation': True, 'gpsLocationExternal': False, 'navInstruction': True}
    self.recv_time = dict.fromkeys(self.valid, now - 0.1)
    self.data = {
      'gpsLocation': SimpleNamespace(hasFix=True),
      'gpsLocationExternal': SimpleNamespace(hasFix=False),
      'navInstruction': SimpleNamespace(allManeuvers=[SimpleNamespace(type='turn', modifier='left', distance=200.0)]),
    }

  def __getitem__(self, name):
    return self.data[name]


class Client:
  def __init__(self, source):
    self.timestamp_sof = source
    self.frame = SimpleNamespace(data=np.full((256, 256), 128, np.uint8).tobytes(), width=256, height=256)

  def is_connected(self):
    return True

  def recv(self, timeout):
    frame, self.frame = self.frame, None
    return frame


class TestNavigationAdmission(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(patch.dict('os.environ', {'NAV_MODEL_OFFLINE': '1'}))
    self.enterContext(patch.object(model_inputs, 'PC', True))
    self.params = Params()
    self.params.put_bool('NavigationModelInputsEnabled', True, block=True)
    self.params.put_bool('NavigationEnabled', True, block=True)
    self.params.put('NavDestination', {'latitude': 1.0, 'longitude': 2.0}, block=True)
    self.now = time.monotonic()
    self.camera_ns = int(self.now * 1e9)
    self.sm = Messages(self.now)
    self.client = Client(self.camera_ns - 900_000_000)
    self.reader = model_inputs.NavigationInputs(self.params, self.client, clock=lambda: self.now)

  def test_valid_frame_and_real_maneuver_encoding(self):
    inputs = self.reader.read(self.sm, 10.0, self.camera_ns)
    self.assertEqual(inputs['nav_map'].shape, (256, 256))
    self.assertEqual(inputs['nav_map'].dtype, np.uint8)
    self.assertEqual(inputs['nav_map'].max(), 128)
    self.assertEqual(inputs['navigation'].dtype, np.float32)
    self.assertEqual(inputs['navigation'][15], 1)
    self.assertAlmostEqual(inputs['navigation'][-1], 1.99, places=5)

  def test_no_gps_even_while_instruction_messages_are_fresh(self):
    self.sm.valid['gpsLocation'] = False
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})

  def test_stale_instruction(self):
    self.sm.recv_time['navInstruction'] = self.now - 2
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})

  def test_map_delivery_stale(self):
    self.assertTrue(self.reader.read(self.sm, 10.0, self.camera_ns))
    self.now += 0.6
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})

  def test_invalid_black_producer_timestamp_clears_valid_cache(self):
    self.assertTrue(self.reader.read(self.sm, 10.0, self.camera_ns))
    self.client.frame = SimpleNamespace(data=bytes(256 * 256), width=256, height=256)
    self.client.timestamp_sof = 0
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})
    self.assertIsNone(self.reader._map)

  def test_old_and_future_source_timestamps(self):
    for source in (self.camera_ns - 3_000_000_000, self.camera_ns + 300_000_000):
      client = Client(source)
      reader = model_inputs.NavigationInputs(self.params, client, clock=lambda: self.now)
      with self.subTest(source=source):
        self.assertEqual(reader.read(self.sm, 10.0, self.camera_ns), {})

  def test_one_hertz_gps_not_confused_with_short_map_delivery_timeout(self):
    self.assertTrue(self.reader.read(self.sm, 10.0, self.camera_ns))  # 0.9-second-old physical GPS, just-received map

  def test_toggle_and_destination_removal_clear_inputs(self):
    for key in ('NavigationEnabled', 'NavigationModelInputsEnabled'):
      self.params.put_bool(key, False, block=True)
      with self.subTest(key=key):
        self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})
      self.params.put_bool(key, True, block=True)
    self.params.remove('NavDestination')
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})

  def test_wrong_map_size_and_truncated_payload_are_contained(self):
    for width, data in ((128, bytes(256 * 256)), (256, b'bad')):
      self.client.frame = SimpleNamespace(width=width, height=256, data=data)
      with self.subTest(width=width):
        self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})

  def test_invalid_speed_does_not_reach_tensor(self):
    for speed in (float('nan'), float('inf'), -1.0):
      with self.subTest(speed=speed):
        self.assertEqual(self.reader.read(self.sm, speed, self.camera_ns), {})

  def test_non_navigation_model_is_not_forced_into_experimental_mode(self):
    self.params.put_bool('NavigationModelInputsEnabled', False, block=True)
    self.assertFalse(model_inputs.validate_model_inputs({'desire': ((8,), 'float32', 'CPU')}, self.params))
