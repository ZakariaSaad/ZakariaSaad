"""Map frames retain route identity and physical GPS timing without changing VisionIPC semantics."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from openpilot.cereal import custom, messaging
from openpilot.selfdrive.navigation.generation import destination_key
from openpilot.selfdrive.navigation.mapsd import MapsD
from openpilot.selfdrive.navigation.renderer import SIZE


class TestMapGeneration(unittest.TestCase):
  def setUp(self):
    self.destination = {'longitude': 0., 'latitude': .002}
    self.params = MagicMock()
    self.params.get.return_value = dict(self.destination)
    self.mapd = MapsD(self.params)
    self.addCleanup(self.mapd.close)
    self.mapd.renderer = MagicMock()
    self.mapd.renderer.render.return_value = (np.full((SIZE, SIZE), 128, np.uint8), True)
    self.route = messaging.new_message('navRoute', valid=True).navRoute
    self.route.coordinates = [{'longitude': 0, 'latitude': 0}, {'longitude': 0, 'latitude': .002}]
    self.route.routeGeneration = 101
    self.route.destinationKey = destination_key(self.destination)
    self.gps = SimpleNamespace(hasFix=True, longitude=0., latitude=0., bearingDeg=0., speed=10.)
    self.sm = MagicMock()
    self.sm.updated = {'navRoute': True}
    self.sm.valid = {'navRoute': True, 'gpsLocation': False, 'gpsLocationExternal': True}
    self.sm.recv_time = {'gpsLocation': 99.5, 'gpsLocationExternal': 99.5}
    self.sm.logMonoTime = {'gpsLocation': 99_500_000_000, 'gpsLocationExternal': 99_500_000_000, 'navRoute': 99_000_000_000}
    self.sm.__getitem__.side_effect = lambda service: self.route if service == 'navRoute' else self.gps
    self.enterContext(patch('openpilot.selfdrive.navigation.mapsd.time.monotonic', return_value=100.))

  def test_metadata_matches_exact_rendered_frame_identity(self):
    image, source = self.mapd.update(self.sm)
    msg = self.mapd.state_message(17, source, 100_000_000_000)
    self.assertTrue(msg.valid)
    state = msg.navModelStateSP
    self.assertEqual(state.frameId, 17)
    self.assertEqual(state.locationMonoTime, source)
    self.assertEqual(state.renderMonoTime, 100_000_000_000)
    self.assertEqual(state.routeGeneration, 101)
    self.assertEqual(state.destinationKey, self.route.destinationKey)
    self.assertTrue(image.any())

  def test_same_route_resend_preserves_tracker_but_new_generation_resets_it(self):
    self.mapd.update(self.sm)
    first = self.mapd.tracker
    self.mapd.update(self.sm)
    self.assertIs(self.mapd.tracker, first)
    self.route.routeGeneration += 1
    self.mapd.update(self.sm)
    self.assertIsNot(self.mapd.tracker, first)

  def test_live_destination_mismatch_emits_invalid_black_frame(self):
    self.params.get.return_value = {'longitude': 1, 'latitude': 1}
    image, source = self.mapd.update(self.sm)
    self.assertFalse(image.any())
    self.assertEqual(source, 0)
    self.mapd.renderer.render.assert_not_called()
    state = self.mapd.state_message(1, source, 100_000_000_000)
    self.assertFalse(state.valid)
    self.assertEqual((state.navModelStateSP.routeGeneration, state.navModelStateSP.destinationKey), (0, ''))

  def test_destination_change_during_prefetch_discards_old_frame(self):
    self.mapd.renderer.prefetch.side_effect = lambda _: setattr(self.params.get, 'return_value', {'longitude': 2, 'latitude': 3})
    image, source = self.mapd.update(self.sm)
    self.assertFalse(image.any())
    self.assertEqual(source, 0)
    self.assertFalse(self.mapd.state_message(1, source, 100_000_000_000).valid)

  def test_invalid_gps_clears_metadata_after_good_frame(self):
    _, source = self.mapd.update(self.sm)
    self.assertGreater(source, 0)
    self.sm.valid['gpsLocationExternal'] = False
    image, source = self.mapd.update(self.sm)
    self.assertFalse(image.any())
    self.assertEqual(source, 0)
    self.assertFalse(self.mapd.state_message(2, source, 100_000_000_000).valid)

  def test_invalid_route_cannot_supply_model_frame(self):
    self.sm.valid['navRoute'] = False
    image, source = self.mapd.update(self.sm)
    self.assertFalse(image.any())
    self.assertEqual(source, 0)
    self.assertIsNone(self.mapd.tracker)

  def test_official_pose_extrapolates_since_gps_receipt(self):
    self.mapd.update(self.sm)
    longitude, latitude, bearing, _ = self.mapd.renderer.render.call_args.args
    self.assertAlmostEqual(longitude, 0.)
    self.assertAlmostEqual(latitude * 111195.0802335329, 5., delta=.001)  # 10 m/s * 0.5 seconds
    self.assertAlmostEqual(bearing, 0.)

  def test_legacy_untagged_route_remains_viewable_but_model_metadata_invalid(self):
    self.mapd.params = None
    self.route.routeGeneration = 0
    self.route.destinationKey = ''
    image, source = self.mapd.update(self.sm)
    self.assertTrue(image.any())
    self.assertGreater(source, 0)
    self.assertFalse(self.mapd.state_message(1, source, 100_000_000_000).valid)

  def test_metadata_rejects_mismatched_source_or_missing_render_timestamp(self):
    _, source = self.mapd.update(self.sm)
    self.assertFalse(self.mapd.state_message(1, source - 1, 100_000_000_000).valid)
    self.assertFalse(self.mapd.state_message(1, source, 0).valid)
    self.assertFalse(self.mapd.state_message(1, source, source - 1).valid)

  def test_reordered_route_cannot_roll_back_same_destination_generation(self):
    self.mapd.update(self.sm)
    self.route.routeGeneration = 102
    self.sm.logMonoTime['navRoute'] += 1
    self.mapd.update(self.sm)
    current = self.mapd.tracker
    self.route.routeGeneration = 101
    self.sm.logMonoTime['navRoute'] += 1  # even a fresh replay of an old identity is rejected
    self.mapd.update(self.sm)
    self.assertEqual(self.mapd.route_generation, 102)
    self.assertIs(self.mapd.tracker, current)

  def test_same_generation_geometry_is_immutable(self):
    self.mapd.update(self.sm)
    current = self.mapd.tracker
    self.route.coordinates[1].longitude = .001
    self.sm.logMonoTime['navRoute'] += 1
    self.mapd.update(self.sm)
    self.assertIs(self.mapd.tracker, current)
    self.assertEqual(self.mapd.route[-1, 0], 0.)

  def test_clear_tombstone_blocks_delayed_old_route_until_new_generation(self):
    self.mapd.update(self.sm)
    self.sm.valid['navRoute'] = False
    self.sm.logMonoTime['navRoute'] += 1
    self.mapd.update(self.sm)
    self.sm.valid['navRoute'] = True
    self.sm.logMonoTime['navRoute'] += 1
    self.assertEqual(self.mapd.update(self.sm)[1], 0)
    self.assertIsNone(self.mapd.tracker)
    self.route.routeGeneration = 102  # a restarted navd allocates a new route identity
    self.sm.logMonoTime['navRoute'] += 1
    self.assertGreater(self.mapd.update(self.sm)[1], 0)

  def test_older_clear_does_not_remove_a_newer_route(self):
    self.mapd.update(self.sm)
    current = self.mapd.tracker
    self.sm.valid['navRoute'] = False
    self.sm.logMonoTime['navRoute'] -= 1
    self.assertGreater(self.mapd.update(self.sm)[1], 0)
    self.assertIs(self.mapd.tracker, current)

  def test_live_destination_rechecked_between_render_and_metadata_send(self):
    _, source = self.mapd.update(self.sm)
    self.params.get.return_value = {'longitude': 1, 'latitude': 1}
    self.assertFalse(self.mapd.state_message(1, source, 100_000_000_000).valid)

  def test_fresh_receipt_does_not_revalidate_old_or_foreign_clock_gps(self):
    for source in (97_000_000_000, 110_000_000_000):
      with self.subTest(source=source):
        self.sm.logMonoTime['gpsLocationExternal'] = source
        self.assertEqual(self.mapd.update(self.sm)[1], 0)

  def test_reserved_custom_struct_id_and_wire_roundtrip(self):
    self.assertEqual(custom.NavModelStateSP.schema.node.id, 0xcb9fd56c7057593a)
    _, source = self.mapd.update(self.sm)
    msg = self.mapd.state_message(0xffffffff, source, 100_000_000_000)
    with messaging.log.Event.from_bytes(msg.to_bytes()) as decoded:
      self.assertEqual(decoded.which(), 'navModelStateSP')
      self.assertEqual(decoded.navModelStateSP.frameId, 0xffffffff)
      self.assertEqual(decoded.navModelStateSP.routeGeneration, 101)
      self.assertEqual(decoded.navModelStateSP.locationMonoTime, source)


def _mapd_test_launcher(proc: str, name: str) -> None:
  # Build mocks in the child so the test also works with multiprocessing's spawn start method.
  from openpilot.system.manager.process import launcher

  renderer = MagicMock()
  renderer.render.return_value = (np.full((SIZE, SIZE), 128, np.uint8), True)
  with patch('openpilot.selfdrive.navigation.mapsd.MapRenderer', return_value=renderer), \
       patch('openpilot.selfdrive.navigation.mapsd.PMTiles'), patch('openpilot.selfdrive.navigation.mapsd.RemoteTiles'):
    launcher(proc, name)


class TestMapMetadataWireIntegration(unittest.TestCase):
  def test_native_mapsd_frame_and_metadata_have_same_physical_snapshot(self):
    import time
    from msgq.visionipc import VisionIpcClient
    from openpilot.cereal.visionipc import VisionStreamType
    from openpilot.common.params import Params
    from openpilot.common.prefix import OpenpilotPrefix
    from openpilot.system.manager.process import PythonProcess

    with OpenpilotPrefix():
      destination = {'longitude': 0., 'latitude': .002}
      params = Params()
      params.put('NavDestination', destination, block=True)
      process = PythonProcess('mapsd', 'openpilot.selfdrive.navigation.mapsd', lambda *_: True)
      process.launcher = _mapd_test_launcher
      client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
      pm = messaging.PubMaster(['navRoute', 'gpsLocationExternal'])
      sm = messaging.SubMaster(['navModelStateSP'])
      route = messaging.new_message('navRoute', valid=True)
      route.navRoute.coordinates = [{'longitude': 0, 'latitude': 0}, {'longitude': 0, 'latitude': .002}]
      route.navRoute.routeGeneration = 1001
      route.navRoute.destinationKey = destination_key(destination)
      route_payload = route.to_bytes()
      process.start()
      try:
        deadline = time.monotonic() + 10
        matched = False
        while time.monotonic() < deadline:
          gps = messaging.new_message('gpsLocationExternal', valid=True)
          gps.gpsLocationExternal.hasFix = True
          pm.send('gpsLocationExternal', gps)
          pm.send('navRoute', route_payload)
          sm.update(10)
          if client.is_connected() or client.connect(False):
            frame = client.recv(timeout_ms=10)
            if frame is not None and sm.valid['navModelStateSP']:
              state = sm['navModelStateSP']
              if client.frame_id == state.frameId:
                self.assertEqual(client.timestamp_sof, state.locationMonoTime)
                self.assertEqual(client.timestamp_eof, state.renderMonoTime)
                self.assertEqual(state.routeGeneration, 1001)
                self.assertEqual(state.destinationKey, route.navRoute.destinationKey)
                self.assertTrue(np.frombuffer(frame.data, dtype=np.uint8).any())
                matched = True
                break
          time.sleep(.01)
        self.assertTrue(matched, 'No matching valid mapsd VisionIPC frame and native metadata message')
      finally:
        self.assertEqual(process.stop(), 0)
