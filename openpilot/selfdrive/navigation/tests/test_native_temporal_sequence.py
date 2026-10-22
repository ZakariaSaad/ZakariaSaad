"""Long synthetic timing sequences over native Msgq/VisionIPC, without model inference.

The production producers/reader execute unchanged. HTTP routing and map tiles are
synthetic fixtures, and the explicit virtual clock advances at the checkpoint's
20 Hz. This checks transport/reset decisions beyond 128 samples, not learned
convergence, real map rendering, hardware clock alignment or vehicle behavior.
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
from openpilot.selfdrive.navigation.renderer import SIZE
from openpilot.selfdrive.navigation.tests.test_maneuver_contract import METER, step


class Clock:
  def __init__(self):
    self.value_ns = time.monotonic_ns()

  def now(self):
    return self.value_ns / 1e9

  def ns(self):
    return self.value_ns

  def advance(self):
    self.value_ns += 50_000_000


class TestNativeTemporalSequence(unittest.TestCase):
  def test_more_than_128_frames_each_side_of_reroute_loss_and_producer_restart(self):
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
      response = MagicMock(status_code=200)
      response.json.return_value = {'routes': [{'legs': [{'steps': [
        step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left')]}]}]}

      def new_engine():
        with patch('openpilot.selfdrive.navigation.navd.Api'):
          return RouteEngine(gps_sm, pm, params)

      def new_maps(pixel):
        maps = MapsD(params)
        self.addCleanup(maps.close)
        maps.renderer = MagicMock()
        maps.renderer.render.return_value = (np.full((SIZE, SIZE), pixel, np.uint8), True)
        return maps

      engine, maps = new_engine(), new_maps(128)
      server = VisionIpcServer('mapsd')
      server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
      server.start_listener()
      client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
      for _ in range(100):
        if client.connect(False):
          break
        time.sleep(.001)
      self.assertTrue(client.is_connected())
      clock = Clock()
      reader = model_inputs.NavigationInputs(params, client, clock=clock.now)
      frame_id = 0xffffff80  # cross UInt32 rollover during the first history window
      decisions = []

      def read():
        result = reader.read(reader_sm, 0., clock.ns())
        decisions.append((bool(result), reader.reset_required, reader.reason))
        return result

      def tick(*, fix=True, update_engine=True):
        nonlocal frame_id
        clock.advance()
        gps = messaging.new_message('gpsLocationExternal', valid=True)
        gps.gpsLocationExternal.hasFix = fix
        pm.send('gpsLocationExternal', gps)
        if update_engine:
          engine.update()
        else:
          gps_sm.update(0)
          engine.send_instruction()
        map_sm.update(0)
        image, source_ns = maps.update(map_sm)
        server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), frame_id, source_ns, clock.ns())
        pm.send('navModelStateSP', maps.state_message(frame_id, source_ns, clock.ns()))
        frame_id = (frame_id + 1) & 0xffffffff
        reader_sm.update(0)
        return read()

      with patch('time.monotonic', side_effect=clock.now), patch('time.monotonic_ns', side_effect=clock.ns), \
           patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
        first = tick()
        self.assertTrue(first, reader.reason)
        self.assertTrue(reader.reset_required)
        first_generation = reader.route_generation
        old_route = messaging.new_message('navRoute', valid=True)
        old_route.navRoute.coordinates = [{'longitude': lon, 'latitude': lat} for lon, lat in maps.route.tolist()]
        old_route.navRoute.routeGeneration = first_generation
        old_route.navRoute.destinationKey = maps.route_destination_key
        old_route_bytes = old_route.to_bytes()
        for frame in range(140):
          inputs = tick()
          self.assertTrue(inputs, (frame, reader.reason))
          self.assertFalse(reader.reset_required, (frame, reader.reason))
          self.assertEqual(reader.route_generation, first_generation)
          self.assertEqual(inputs['nav_map'][0, 0], 128)
          self.assertEqual(inputs['navigation'][15], 1)

        def pending_route(*_args, **_kwargs):
          reader_sm.update(0)
          self.assertEqual(read(), {})
          self.assertTrue(reader.reset_required)
          return response

        clock.advance()
        with patch('openpilot.selfdrive.navigation.navd.requests.get', side_effect=pending_route):
          engine.calculate_route((destination['longitude'], destination['latitude']))
        self.assertGreater(engine.route_generation, first_generation)
        engine.send_route()
        maps.renderer.render.return_value = (np.full((SIZE, SIZE), 192, np.uint8), True)
        self.assertTrue(tick())
        self.assertTrue(reader.reset_required)
        new_generation = reader.route_generation
        for frame in range(140):
          inputs = tick()
          self.assertTrue(inputs, (frame, reader.reason))
          self.assertFalse(reader.reset_required, (frame, reader.reason))
          self.assertEqual(reader.route_generation, new_generation)
          self.assertEqual(inputs['nav_map'][0, 0], 192)

        # Deliver an original route event after the new route has 128+ frames.
        pm.send('navRoute', old_route_bytes)
        self.assertTrue(tick(update_engine=False))
        self.assertEqual(maps.route_generation, new_generation)
        self.assertFalse(reader.reset_required)
        self.assertEqual(tick(fix=False), {})
        self.assertTrue(reader.reset_required)
        self.assertTrue(tick())
        self.assertTrue(reader.reset_required)

        # A restarted mapsd first has no route; a normal resend restores it.
        maps.close()
        maps = new_maps(240)
        frame_id = 0
        self.assertEqual(tick(update_engine=False), {})
        self.assertTrue(reader.reset_required)
        engine.send_route()
        self.assertTrue(tick(update_engine=False))
        self.assertTrue(reader.reset_required)
        self.assertEqual(reader.context.frame_id, 1)

        # A restarted navd computes a new identity even for unchanged geometry.
        engine = new_engine()
        self.assertTrue(tick())
        self.assertGreater(reader.route_generation, new_generation)
        self.assertTrue(reader.reset_required)
        for _ in range(11):
          self.assertTrue(tick())
          self.assertFalse(reader.reset_required)
        self.assertEqual(len(decisions), 300)
