"""Pixel/pose differential checks against the pinned official Bigv2 map path.

Synthetic in-memory PMTiles exercise real decoding/rasterization. No tile service,
GPU, recorded route or vehicle process is used by the differential cases.
"""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from openpilot.cereal import custom, messaging
from openpilot.selfdrive.navigation.generation import destination_key
from openpilot.selfdrive.navigation.mapsd import MapsD
from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import SIZE, MapRenderer
from openpilot.selfdrive.navigation.tests.official_reference import load_reference
from openpilot.selfdrive.navigation.tests.test_navigation import local_archive, make_archive, road_tile
from openpilot.selfdrive.navigation.vector_tile import decode_roads


class TestOfficialMapParity(unittest.TestCase):
  def setUp(self):
    self.stack = self.enterContext(ExitStack())
    helpers = load_reference('helpers')
    route = load_reference('route')
    renderer = load_reference('renderer', namespace={'PMTiles': PMTiles, 'cumulative_distance': route.cumulative_distance,
                                                   'nearest_segment': route.nearest_segment, 'decode_roads': decode_roads,
                                                   'Road': object})
    official = load_reference('mapsd', namespace={'latest_gps_source': helpers.latest_gps_source, 'RouteTracker': route.RouteTracker,
                                                'MapRenderer': renderer.MapRenderer, 'PMTiles': PMTiles, 'RemoteTiles': MagicMock(),
                                                'SIZE': SIZE, 'cloudlog': MagicMock()})
    self.params = MagicMock()
    self.local, self.official = MapsD(self.params), official.MapsD()
    self.addCleanup(self.local.close)
    self.addCleanup(self.official.close)
    # Use independent real readers and renderers on the same synthetic vector bytes.
    data = make_archive(road_tile('bridge'), leaf=True)
    self.local.renderer = MapRenderer(self.stack.enter_context(local_archive(data)))
    self.official.renderer = renderer.MapRenderer(self.stack.enter_context(local_archive(data)))
    self.now = 100.
    self.enterContext(patch('openpilot.selfdrive.navigation.mapsd.time.monotonic', side_effect=lambda: self.now))
    self.route = messaging.new_message('navRoute', valid=True).navRoute
    self.route.coordinates = [{'longitude': 0., 'latitude': -.002}, {'longitude': 0., 'latitude': .002}]
    self.route.routeGeneration = 101
    self.route.destinationKey = 'diagnostic-only'
    self.gps = SimpleNamespace(hasFix=True, longitude=0., latitude=0., bearingDeg=10., speed=10.)
    self.sm = MagicMock()
    self.sm.updated = {'navRoute': True}
    self.sm.valid = {'navRoute': True, 'gpsLocation': False, 'gpsLocationExternal': True}
    self.sm.recv_time = {'gpsLocation': 99.5, 'gpsLocationExternal': 99.5}
    self.sm.logMonoTime = {'gpsLocation': 99_500_000_000, 'gpsLocationExternal': 99_500_000_000, 'navRoute': 99_000_000_000}
    self.sm.__getitem__.side_effect = lambda service: self.route if service == 'navRoute' else self.gps

  def assert_frame_equal(self):
    local, timestamp = self.local.update(self.sm)
    official, official_timestamp = self.official.update(self.sm)
    np.testing.assert_array_equal(local, official)
    self.assertEqual(timestamp, official_timestamp)
    if self.local.tracker is not None:
      self.assertEqual(self.local.tracker.progress, self.official.tracker.progress)
    self.params.get.assert_not_called()
    return local, timestamp

  def test_actual_pixels_pose_and_receipt_extrapolation(self):
    image, source = self.assert_frame_equal()
    self.assertTrue(image.any())
    self.assertEqual(source, 99_500_000_000)
    for speed, latitude, elapsed in ((12., .0001, .1), (5., .0002, .4), (-2., .0003, .2)):
      self.gps.speed, self.gps.latitude = speed, latitude
      self.now += elapsed
      self.sm.recv_time['gpsLocationExternal'] = self.now - .25
      self.sm.logMonoTime['gpsLocationExternal'] += 1
      self.assert_frame_equal()

  def test_route_republication_and_diagnostic_identity_do_not_reset_progress(self):
    self.assert_frame_equal()
    local_tracker, official_tracker = self.local.tracker, self.official.tracker
    self.route.routeGeneration += 1
    self.route.destinationKey = 'different-diagnostic-destination'
    self.sm.logMonoTime['navRoute'] -= 1
    self.sm.valid['navRoute'] = False
    self.assert_frame_equal()
    self.assertIs(self.local.tracker, local_tracker)
    self.assertIs(self.official.tracker, official_tracker)

  def test_reroute_reordering_clear_and_startup_match_geometry_semantics(self):
    self.sm.updated['navRoute'] = False
    image, source = self.assert_frame_equal()
    self.assertFalse(image.any() or source)
    self.sm.updated['navRoute'] = True
    self.assert_frame_equal()
    previous = self.local.tracker
    self.route.coordinates = [{'longitude': 0., 'latitude': -.002}, {'longitude': .0001, 'latitude': .002}]
    self.route.routeGeneration = 1
    self.sm.logMonoTime['navRoute'] = 1
    self.assert_frame_equal()
    self.assertIsNot(self.local.tracker, previous)
    self.route.coordinates = []
    image, source = self.assert_frame_equal()
    self.assertFalse(image.any() or source)

  def test_receipt_freshness_and_has_fix_match_official_not_source_timestamp_policy(self):
    for source_ns in (1, 97_000_000_000, 110_000_000_000):
      self.sm.logMonoTime['gpsLocationExternal'] = source_ns
      image, source = self.assert_frame_equal()
      self.assertTrue(image.any())
      self.assertEqual(source, source_ns)
    for recv_time in (97.9, 100.1):
      self.sm.recv_time['gpsLocationExternal'] = recv_time
      image, source = self.assert_frame_equal()
      self.assertFalse(image.any() or source)
    self.sm.recv_time['gpsLocationExternal'] = 99.5
    self.gps.hasFix = False
    image, source = self.assert_frame_equal()
    self.assertFalse(image.any() or source)

  def test_prefetch_latency_uses_official_pre_fetch_receipt_snapshot(self):
    frames = []
    for maps in (self.local, self.official):
      original = maps.renderer.prefetch
      def delayed(route, original=original):
        original(route)
        self.now += 3.
      self.now = 100.
      with patch.object(maps.renderer, 'prefetch', side_effect=delayed):
        frames.append(maps.update(self.sm))
    np.testing.assert_array_equal(frames[0][0], frames[1][0])
    self.assertEqual(frames[0][1], frames[1][1])
    self.assertTrue(frames[0][0].any())

  def test_metadata_is_diagnostic_only_and_retains_wire_identity(self):
    image, source = self.assert_frame_equal()
    self.assertTrue(image.any())
    self.route.routeGeneration = 0
    self.route.destinationKey = ''
    self.assert_frame_equal()
    msg = self.local.state_message(17, source, 100_000_000_000)
    self.assertTrue(msg.valid)
    self.assertEqual(custom.NavModelStateSP.schema.node.id, 0xcb9fd56c7057593a)
    with messaging.log.Event.from_bytes(msg.to_bytes()) as decoded:
      self.assertEqual(decoded.navModelStateSP.frameId, 17)
      self.assertEqual(decoded.navModelStateSP.routeGeneration, 0)
      self.assertEqual(decoded.navModelStateSP.locationMonoTime, source)
    self.assertFalse(self.local.state_message(18, 0, 100_000_000_001).valid)


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
