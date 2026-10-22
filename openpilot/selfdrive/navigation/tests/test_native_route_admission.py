"""Native navd/mapsd transport into the offline reader, without inference or production endpoints."""
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
from openpilot.selfdrive.navigation.renderer import SIZE
from openpilot.selfdrive.navigation.tests.test_maneuver_contract import METER, step


class TestNativeRouteAdmission(unittest.TestCase):
  def test_native_producers_reader_route_change_and_gps_loss(self):
    with OpenpilotPrefix(), patch.dict('os.environ', {'NAV_MODEL_OFFLINE': '1'}), patch.object(model_inputs, 'PC', True):
      params = Params()
      params.put_bool('NavigationEnabled', True, block=True)
      params.put_bool('NavigationModelInputsEnabled', True, block=True)
      destination = {'longitude': 100 * METER, 'latitude': 100 * METER}
      params.put('NavDestination', destination, block=True)
      pm = messaging.PubMaster(['gpsLocationExternal', 'navRoute', 'navInstruction', 'navModelStateSP'])
      gps_sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])
      map_sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navRoute'])
      reader_sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navInstruction', 'navModelStateSP'])
      with patch('openpilot.selfdrive.navigation.navd.Api'):
        engine = RouteEngine(gps_sm, pm, params)
      response = MagicMock()
      response.status_code = 200
      response.json.return_value = {'routes': [{'legs': [{'steps': [
        step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left')]}]}]}
      maps = MapsD(params)
      self.addCleanup(maps.close)
      maps.renderer = MagicMock()
      maps.renderer.render.return_value = (np.full((SIZE, SIZE), 128, np.uint8), True)
      server = VisionIpcServer('mapsd')
      server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
      server.start_listener()
      client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
      reader = model_inputs.NavigationInputs(params, client)
      frame_id = 0

      def tick(fix=True):
        nonlocal frame_id
        gps = messaging.new_message('gpsLocationExternal', valid=True)
        gps.gpsLocationExternal.hasFix = fix
        pm.send('gpsLocationExternal', gps)
        with patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
          engine.update()
        map_sm.update(0)
        image, source = maps.update(map_sm)
        render_ns = time.monotonic_ns()
        server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), frame_id, source, render_ns)
        pm.send('navModelStateSP', maps.state_message(frame_id, source, render_ns))
        frame_id += 1
        reader_sm.update(0)
        return reader.read(reader_sm, 0., time.monotonic_ns())

      def receive_valid():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
          inputs = tick()
          if inputs:
            return inputs
          time.sleep(.01)
        self.fail(f'No matching native map/instruction pair accepted: {reader.reason}')

      accepted = receive_valid()
      self.assertEqual(accepted['nav_map'].shape, (256, 256))
      self.assertEqual(accepted['nav_map'].dtype, np.uint8)
      self.assertEqual(accepted['navigation'][15], 1)
      self.assertTrue(reader.reset_required)
      old_context = reader.context
      old_generation = engine.route_generation

      # Changing the live destination cannot admit the old, still-fresh native messages/frame.
      changed = {'longitude': 200 * METER, 'latitude': 100 * METER}
      params.put('NavDestination', changed, block=True)
      self.assertEqual(reader.read(reader_sm, 0., time.monotonic_ns()), {})
      self.assertTrue(reader.reset_required)

      # Compute the new route/instruction but deliberately deliver an old-generation map frame.
      response.json.return_value = {'routes': [{'legs': [{'steps': [
        step((0, 0), (0, 100), 'depart'), step((0, 100), (200, 100), 'turn', 'right')]}]}]}
      with patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
        engine.calculate_route((changed['longitude'], changed['latitude']))
      self.assertGreater(engine.route_generation, old_generation)
      engine.send_instruction()
      render_ns = time.monotonic_ns()
      server.send(VisionStreamType.VISION_STREAM_MAP, accepted['nav_map'].tobytes(), frame_id, old_context.map_source_ns, render_ns)
      pm.send('navModelStateSP', maps.state_message(frame_id, old_context.map_source_ns, render_ns))
      frame_id += 1
      reader_sm.update(0)
      self.assertEqual(reader.read(reader_sm, 0., time.monotonic_ns()), {})
      self.assertEqual(reader.reason, 'route_changed')
      self.assertTrue(reader.reset_required)

      engine.send_route()
      reacquired = receive_valid()
      self.assertEqual(reacquired['navigation'][10], 1)
      self.assertTrue(reader.reset_required)
      self.assertEqual(reader.route_generation, engine.route_generation)

      self.assertEqual(tick(fix=False), {})
      self.assertTrue(reader.reset_required)
      self.assertIsNone(reader._map)
      self.assertEqual(engine.route_generation, reader.route_generation)
