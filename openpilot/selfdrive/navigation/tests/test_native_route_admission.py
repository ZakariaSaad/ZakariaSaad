"""Native navd/mapsd transport compared with the exact official model consumer.

Real Msgq, VisionIPC, cereal, PMTiles, rasterization and route tracking execute in
an isolated OpenpilotPrefix. HTTP responses and map data are synthetic; neither
model inference nor vehicle control runs.
"""
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from msgq.visionipc import VisionIpcClient, VisionIpcServer

from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.navigation import model_inputs
from openpilot.selfdrive.navigation.mapsd import MapsD
from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.selfdrive.navigation.renderer import SIZE, MapRenderer
from openpilot.selfdrive.navigation.tests.official_reference import OfficialNavigationInputs
from openpilot.selfdrive.navigation.tests.test_maneuver_contract import METER, step
from openpilot.selfdrive.navigation.tests.test_navigation import local_archive, make_archive


class NativeRouteHarness:
  def setUp(self):
    self.enterContext(OpenpilotPrefix())
    self.params = Params()
    self.params.put_bool('NavigationEnabled', True, block=True)
    self.destination = {'longitude': 100 * METER, 'latitude': 100 * METER}
    self.params.put('NavDestination', self.destination, block=True)
    self.pm = messaging.PubMaster(['gpsLocationExternal', 'navRoute', 'navInstruction', 'navModelStateSP'])
    self.gps_sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])
    self.map_sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navRoute'])
    self.reader_sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navInstruction', 'navModelStateSP'])
    self.enterContext(patch('openpilot.selfdrive.navigation.navd.Api'))
    self.enterContext(patch('openpilot.selfdrive.navigation.activation.validate_runtime_permit'))
    self.engine = RouteEngine(self.gps_sm, self.pm, self.params)
    self.response = MagicMock(status_code=200)
    self.set_route()
    self.enterContext(patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=self.response))
    self.maps = self.new_maps()
    self.server = VisionIpcServer('mapsd')
    self.server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
    self.server.start_listener()
    self.client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
    self.reference_client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
    for client in (self.client, self.reference_client):
      for _ in range(100):
        if client.connect(False):
          break
        time.sleep(.001)
      self.assertTrue(client.is_connected())
    self.now = 100.
    self.enterContext(patch('time.monotonic', side_effect=lambda: self.now))
    self.reader = model_inputs.NavigationInputs(self.params, self.client, clock=lambda: self.now, driving_permit=object())
    self.reference = OfficialNavigationInputs()
    self.frame_id = 0
    self.rendered = None

  def new_maps(self):
    maps = MapsD(self.params)
    self.addCleanup(maps.close)
    maps.renderer = MapRenderer(self.enterContext(local_archive(make_archive())))
    return maps

  def set_route(self, modifier='left', x=100):
    self.response.json.return_value = {'routes': [{'legs': [{'steps': [
      step((0, 0), (0, 100), 'depart'), step((0, 100), (x, 100), 'turn', modifier)]}]}]}

  def read_inputs(self, speed=0.):
    expected = self.reference.read(self.reader_sm, self.reference_client, self.now, speed)
    actual = self.reader.read_for_model(self.reader_sm, speed, int((self.now + 5000.) * 1e9))
    defaults = {'nav_map': np.zeros((SIZE, SIZE), np.uint8), 'navigation': np.zeros(47, np.float32)}
    for name, zero in defaults.items():
      np.testing.assert_array_equal(actual[name], expected.get(name, zero), err_msg=name)
      self.assertEqual(actual[name].dtype, expected.get(name, zero).dtype)
    self.assertFalse(self.reader.reset_required)
    self.assertIsNone(self.reader.transitions.identity)
    return actual

  def tick(self, *, fix=True, update_engine=True, render=True):
    self.now += .05
    gps = messaging.new_message('gpsLocationExternal', valid=True)
    gps.gpsLocationExternal.hasFix = fix
    self.pm.send('gpsLocationExternal', gps)
    if update_engine:
      self.engine.update()
    self.map_sm.update(0)
    if render:
      image, source = self.maps.update(self.map_sm)
      self.rendered = image.copy()
      self.server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), self.frame_id, source, int(self.now * 1e9))
      self.pm.send('navModelStateSP', self.maps.state_message(self.frame_id, source, int(self.now * 1e9)))
      self.frame_id = (self.frame_id + 1) & 0xffffffff
    self.reader_sm.update(0)
    return self.read_inputs()


class TestNativeRouteAdmission(NativeRouteHarness, unittest.TestCase):
  def test_real_map_pixels_maneuvers_asynchronous_reroute_and_gps_loss(self):
    startup = self.read_inputs()
    self.assertFalse(startup['nav_map'].any() or startup['navigation'].any())
    accepted = self.tick()
    np.testing.assert_array_equal(accepted['nav_map'], self.rendered)
    self.assertEqual(int(accepted['nav_map'].max()), 255)
    self.assertFalse(np.array_equal(accepted['nav_map'], accepted['nav_map'][::-1]))
    self.assertFalse(np.array_equal(accepted['nav_map'], accepted['nav_map'][:, ::-1]))
    self.assertEqual(accepted['navigation'][15], 1)

    # Official modeld consumes independently delivered guidance and map pixels.
    # A settings/destination change does not neutralize a still-fresh frame.
    changed = {'longitude': 200 * METER, 'latitude': 100 * METER}
    self.params.put('NavDestination', changed, block=True)
    held = self.read_inputs()
    np.testing.assert_array_equal(held['nav_map'], accepted['nav_map'])
    self.set_route('right', 200)
    self.engine.calculate_route((changed['longitude'], changed['latitude']))
    self.engine.send_instruction()
    self.now += .001
    self.reader_sm.update(0)
    guidance_first = self.read_inputs()
    np.testing.assert_array_equal(guidance_first['nav_map'], accepted['nav_map'])
    self.assertEqual(guidance_first['navigation'][10], 1)
    self.engine.send_route()
    self.tick()
    no_fix = self.tick(fix=False)
    self.assertFalse(no_fix['nav_map'].any())
    self.assertEqual(no_fix['navigation'][10], 1)  # navd retains its last position, as upstream
    self.assertTrue(self.tick()['nav_map'].any())
