"""Explicit diagnostic association checks; production parity lives in test_model_input_execution.

These opt-in diagnostic annotations never drive production navigation tensors.
"""
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
from openpilot.selfdrive.navigation import model_inputs
from openpilot.selfdrive.navigation.generation import destination_key
from openpilot.selfdrive.navigation.temporal import CameraClock
from openpilot.selfdrive.navigation.tests.test_activation import reviewed_fixture


class Messages:
  def __init__(self, now):
    self.valid = {'gpsLocation': True, 'gpsLocationExternal': False, 'navInstruction': True, 'navModelStateSP': True}
    self.recv_time = dict.fromkeys(self.valid, now - 0.1)
    self.logMonoTime = dict.fromkeys(self.valid, int((now-0.1)*1e9))
    key = destination_key({'latitude': 1.0, 'longitude': 2.0})
    self.data = {
      'gpsLocation': SimpleNamespace(hasFix=True, latitude=1.0, longitude=2.0),
      'gpsLocationExternal': SimpleNamespace(hasFix=False, latitude=1.0, longitude=2.0),
      'navInstruction': SimpleNamespace(allManeuvers=[SimpleNamespace(type='turn', modifier='left', distance=200.0)],
                                       routeGeneration=1, destinationKey=key, gpsSourceMonoTime=int((now-0.1)*1e9)),
      'navModelStateSP': SimpleNamespace(frameId=1, locationMonoTime=int(now*1e9)-900_000_000, renderMonoTime=int(now*1e9),
                                          routeGeneration=1, destinationKey=key),
    }

  def __getitem__(self, name):
    return self.data[name]


class Client:
  def __init__(self, source):
    self.timestamp_sof = source
    self.timestamp_eof = source+900_000_000
    self.frame_id = 1
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
    self.camera_ns += 50_000_000
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

  def test_old_generation_rejected_despite_fresh_delivery(self):
    self.sm.data['navInstruction'].routeGeneration = 2
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})
    self.assertTrue(self.reader.reset_required)

  def test_pending_frame_not_relabelled_by_newer_metadata(self):
    self.sm.data['navModelStateSP'].frameId = 2
    self.assertEqual(self.reader.read(self.sm, 10.0, self.camera_ns), {})
    # The exact old frame can be accepted once its metadata arrives.
    self.sm.data['navModelStateSP'].frameId = 1
    self.assertTrue(self.reader.read(self.sm, 10.0, self.camera_ns+50_000_000))

  def test_holds_verified_frame_while_new_metadata_catches_up(self):
    self.assertTrue(self.reader.read(self.sm, 10.0, self.camera_ns))
    self.client.frame_id = 2
    self.client.timestamp_sof += 50_000_000
    self.client.timestamp_eof += 50_000_000
    self.client.frame = SimpleNamespace(data=bytes(256*256), width=256, height=256)
    self.assertTrue(self.reader.read(self.sm, 10.0, self.camera_ns+50_000_000))
    self.assertEqual(self.reader.context.frame_id, 1)
    self.assertFalse(self.reader.reset_required)

  def test_explicit_live_clock_bridge_reacquires_fresh_sources_after_suspend(self):
    offset = [30_000_000_000]
    clock = CameraClock(lambda: int(self.now * 1e9), lambda: int(self.now * 1e9) + offset[0])
    reader = model_inputs.NavigationInputs(self.params, self.client, clock=lambda: self.now, camera_clock=clock)

    def tick(*, advance=True, fresh_source=True, fresh_map=True):
      if advance:
        self.now += .05
      self.sm = Messages(self.now)
      source = int((self.now - (.01 if fresh_source else .9)) * 1e9)
      self.sm.logMonoTime['gpsLocation'] = source
      self.sm.logMonoTime['navInstruction'] = int(self.now * 1e9)
      self.sm.data['navInstruction'].gpsSourceMonoTime = source
      metadata = self.sm.data['navModelStateSP']
      metadata.frameId = self.client.frame_id = self.client.frame_id + 1
      metadata.locationMonoTime = self.client.timestamp_sof = source if fresh_map else int((self.now - .9) * 1e9)
      metadata.renderMonoTime = self.client.timestamp_eof = int(self.now * 1e9)
      self.client.frame = SimpleNamespace(data=bytes([128]) * 256 * 256, width=256, height=256)
      return reader.read(self.sm, 10., int(self.now * 1e9) + offset[0])

    self.assertEqual(tick(advance=False), {})
    self.assertTrue(reader.reset_required)
    self.assertEqual(reader.reset_reason, 'camera_clock_epoch_changed')
    self.assertEqual(reader.input_reason, 'waiting_for_clock_epoch_sources')
    inputs = tick()
    self.assertTrue(inputs, reader.input_reason)
    self.assertEqual(reader.context.camera_ns, int(self.now * 1e9))
    self.assertEqual(reader._map_source_ns, self.client.timestamp_sof)
    self.assertEqual(reader.context.map_render_ns, self.client.timestamp_eof)
    self.assertTrue(reader.reset_required)
    self.assertTrue(tick())
    self.assertFalse(reader.reset_required)
    offset[0] += 45_000_000_000
    self.assertEqual(tick(), {})
    self.assertTrue(reader.reset_required)
    self.assertEqual(reader.reset_reason, 'camera_clock_epoch_changed')
    # Fresh receipt cannot promote a producer sample from before the new epoch.
    self.assertEqual(tick(fresh_source=False), {})
    self.assertEqual(reader.input_reason, 'waiting_for_clock_epoch_sources')
    self.assertEqual(tick(fresh_map=False), {})
    self.assertEqual(reader.input_reason, 'waiting_for_clock_epoch_map')
    self.assertTrue(tick(), reader.input_reason)
    self.assertTrue(reader.reset_required)

  def test_production_receipt_clock_needs_no_camera_epoch_bridge(self):
    permit = object()
    self.assertIsNone(model_inputs.NavigationInputs(self.params, driving_permit=permit)._camera_clock)
    reader = model_inputs.NavigationInputs(self.params, clock=lambda: self.now, driving_permit=permit)
    self.assertIsNone(reader._camera_clock)
    self.assertIsNone(model_inputs.NavigationInputs(self.params)._camera_clock)

  def test_reviewed_driving_permit_requires_clock_epoch_sources_and_revokes_cached_inputs(self):
    now_ns = [100_000_000_000]
    offset = 30_000_000_000
    bridge = CameraClock(lambda: now_ns[0], lambda: now_ns[0] + offset)
    client = Client(99_100_000_000)
    # This fixture simulates reviewed hashes; no artifact is executed or approved.
    # Disable the offline path so both successful reads and revocation use the
    # production permit validation, alongside the measured clock conversion.
    with reviewed_fixture() as (params, permit, _), patch.object(model_inputs, 'PC', False):
      params.values.update(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
      self.assertFalse(params.get_bool('NavigationModelInputsEnabled'))
      reader = model_inputs.NavigationInputs(params, client, clock=lambda: now_ns[0] / 1e9,
                                             driving_permit=permit, camera_clock=bridge)
      self.assertEqual(reader.read(Messages(100.), 10., now_ns[0] + offset), {})
      self.assertEqual(reader.input_reason, 'waiting_for_clock_epoch_sources')
      self.assertTrue(reader.reset_required)

      now_ns[0] += 50_000_000
      sm = Messages(now_ns[0] / 1e9)
      sm.logMonoTime['navInstruction'] = now_ns[0]
      sm.logMonoTime['gpsLocation'] = sm.data['navInstruction'].gpsSourceMonoTime = now_ns[0] - 10_000_000
      metadata = sm.data['navModelStateSP']
      metadata.locationMonoTime = client.timestamp_sof = now_ns[0] - 10_000_000
      metadata.renderMonoTime = client.timestamp_eof = now_ns[0]
      metadata.frameId = client.frame_id = 2
      client.frame = SimpleNamespace(data=bytes([128]) * 256 * 256, width=256, height=256)
      inputs = reader.read(sm, 10., now_ns[0] + offset)
      self.assertTrue(inputs, reader.input_reason)
      self.assertEqual(reader.context.camera_ns, now_ns[0])
      self.assertEqual(reader.context.map_source_ns, client.timestamp_sof)
      self.assertEqual(inputs['navigation'][15], 1.)

      params.values['ModelManager_ActiveBundleChestnut'] = {'ref': 'ordinary-big'}
      now_ns[0] += 50_000_000
      self.assertEqual(reader.read(sm, 10., now_ns[0] + offset), {})
      self.assertTrue(reader.reset_required)
      self.assertIsNone(reader._map)
      self.assertIsNone(reader.context)
