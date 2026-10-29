"""Continuous native producer/consumer sequence with exact official tensor parity.

This is transport/timing coverage, not learned-state convergence or GPU testing.
There is no model-history threshold, priming policy or live messaging namespace.
"""
import unittest

import numpy as np

from openpilot.cereal import messaging
from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.selfdrive.navigation.tests.test_native_route_admission import NativeRouteHarness


class TestNativeTemporalSequence(NativeRouteHarness, unittest.TestCase):
  def test_rollover_reroute_republication_gps_loss_and_restart_match_official(self):
    self.frame_id = 0xfffffff0
    initial = self.tick()
    old_route = messaging.new_message('navRoute', valid=True)
    old_route.navRoute.coordinates = [{'longitude': lon, 'latitude': lat} for lon, lat in self.maps.route.tolist()]
    old_route.navRoute.routeGeneration = 99  # diagnostic metadata does not order official geometry
    old_route.navRoute.destinationKey = 'old-diagnostic-id'
    old_route_bytes = old_route.to_bytes()
    for _ in range(20):
      current = self.tick()
      np.testing.assert_array_equal(current['nav_map'], initial['nav_map'])
      self.assertEqual(current['navigation'][15], 1)
    self.assertEqual(self.frame_id, 5)  # real UInt32 VisionIPC frame rollover

    self.set_route('right', 200)
    self.engine.calculate_route((self.destination['longitude'], self.destination['latitude']))
    self.engine.send_instruction()
    self.now += .001
    self.reader_sm.update(0)
    self.assertEqual(self.read_inputs()['navigation'][10], 1)
    self.engine.send_route()
    for _ in range(10):
      self.tick(update_engine=False)
    new_route = self.maps.route.copy()

    # A later publication of older geometry is consumed as upstream does; no
    # custom sequence floor overrules the incoming route/geometry.
    self.pm.send('navRoute', old_route_bytes)
    self.tick(update_engine=False)
    self.assertFalse(np.array_equal(self.maps.route, new_route))
    self.engine.send_route()
    self.tick(update_engine=False)
    np.testing.assert_array_equal(self.maps.route, new_route)
    self.assertFalse(self.tick(fix=False)['nav_map'].any())
    self.assertTrue(self.tick()['nav_map'].any())

    # A restarted map producer has no route until the periodic route resend.
    # Its black frame is still received, independent of source metadata.
    self.maps.close()
    self.maps = self.new_maps()
    self.frame_id = 0
    self.assertFalse(self.tick(update_engine=False)['nav_map'].any())
    self.engine.send_route()
    self.assertTrue(self.tick(update_engine=False)['nav_map'].any())
    self.engine = RouteEngine(self.gps_sm, self.pm, self.params)
    for _ in range(10):
      self.assertTrue(self.tick()['nav_map'].any())

    # Receipt timeouts independently zero the corresponding model inputs.
    self.now += .6
    expired_map = self.read_inputs()
    self.assertFalse(expired_map['nav_map'].any())
    self.assertTrue(expired_map['navigation'].any())
    self.now += 2.
    expired_both = self.read_inputs()
    self.assertFalse(expired_both['nav_map'].any() or expired_both['navigation'].any())
