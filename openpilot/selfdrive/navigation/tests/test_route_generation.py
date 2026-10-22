"""Route identities prevent an old destination or an old daemon session from conditioning a model."""
import unittest
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.navigation.generation import UINT64_MAX, destination_key, gps_source, next_generation
from openpilot.selfdrive.navigation.tests.test_maneuver_contract import engine_for, instruction, step


class TestDestinationIdentity(unittest.TestCase):
  def test_identity_ignores_display_metadata_and_numeric_spelling(self):
    reference = destination_key({'latitude': 32, 'longitude': -117})
    self.assertEqual(len(reference), 64)
    self.assertEqual(reference, destination_key({'latitude': 32., 'longitude': -117., 'place_name': 'New label'}))
    self.assertEqual(destination_key({'latitude': -0., 'longitude': 0}), destination_key({'latitude': 0, 'longitude': 0.}))
    self.assertNotEqual(reference, destination_key({'latitude': 32.0000001, 'longitude': -117}))

  @parameterized.expand([None, {}, [], {'latitude': True, 'longitude': 0}, {'latitude': 0, 'longitude': float('nan')},
                         {'latitude': 0, 'longitude': float('inf')}, {'latitude': 91, 'longitude': 0},
                         {'latitude': 0, 'longitude': 181}, {'latitude': '0', 'longitude': 0},
                         {'latitude': 10**500, 'longitude': 0}])
  def test_invalid_destination_has_no_identity(self, destination):
    self.assertEqual(destination_key(destination), '')

  def test_route_ids_advance_and_separate_normal_restarts(self):
    first = next_generation(clock=lambda: 100)
    second = next_generation(first, clock=lambda: 100)
    restarted = next_generation(clock=lambda: 102)
    self.assertEqual((first, second, restarted), (100, 101, 102))
    self.assertEqual(next_generation(clock=lambda: 0), 1)
    with self.assertRaises(OverflowError):
      next_generation(UINT64_MAX, clock=lambda: 1)

  def test_physical_gps_time_checked_independently_of_delivery(self):
    sm = MagicMock()
    sm.valid = {'gpsLocation': True, 'gpsLocationExternal': False}
    sm.recv_time = dict.fromkeys(sm.valid, 100.)
    sm.__getitem__.return_value = SimpleNamespace(hasFix=True, latitude=0., longitude=0.)
    for source_ns, expected in ((99_000_000_000, 'gpsLocation'), (98_000_000_000, None),
                                (100_300_000_000, None), (110_000_000_000, None), (0, None)):
      sm.logMonoTime = dict.fromkeys(sm.valid, source_ns)
      with self.subTest(source_ns=source_ns):
        self.assertEqual(gps_source(sm, now=100.), expected)
    sm.logMonoTime = dict.fromkeys(sm.valid, 1)
    self.assertEqual(gps_source(sm, now=100., check_source_time=False), 'gpsLocation')  # explicit legacy viewer policy


class TestRoutePublicationGeneration(unittest.TestCase):
  def setUp(self):
    self.steps = [step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left')]
    self.engine, self.pm = engine_for(self.steps)
    self.destination = {'longitude': self.steps[-1]['geometry']['coordinates'][-1][0],
                        'latitude': self.steps[-1]['geometry']['coordinates'][-1][1]}

  def test_route_and_instruction_share_full_identity(self):
    msg = instruction(self.engine, self.pm)
    self.engine.send_route()
    route = self.pm.send.call_args.args[1].navRoute
    self.assertGreater(route.routeGeneration, 0)
    self.assertEqual(route.routeGeneration, msg.routeGeneration)
    self.assertEqual(route.destinationKey, msg.destinationKey)
    self.assertEqual(msg.destinationKey, destination_key(self.destination))
    self.assertGreater(msg.gpsSourceMonoTime, 0)

  def test_future_maneuvers_preserve_current_step_remaining_distance_and_time(self):
    # The arrival step has zero length. Its maneuver metadata must not replace
    # the active 100m/10s step when computing the UI's remaining totals.
    steps = self.steps + [step((100, 100), (100, 100), 'arrive')]
    engine, pm = engine_for(steps, position=(0, 25))
    msg = instruction(engine, pm)
    self.assertAlmostEqual(msg.maneuverDistance, 75., delta=.01)
    self.assertAlmostEqual(msg.distanceRemaining, 175., delta=.01)
    self.assertAlmostEqual(msg.timeRemaining, 17.5, delta=.001)
    self.assertEqual([maneuver.type for maneuver in msg.allManeuvers], ['turn', 'arrive'])

  def test_identical_reroute_gets_new_generation(self):
    first = self.engine.route_generation
    response = MagicMock()
    response.json.return_value = {'routes': [{'legs': [{'steps': self.steps}]}]}
    with patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
      self.engine.calculate_route((self.destination['longitude'], self.destination['latitude']))
    self.assertGreater(self.engine.route_generation, first)

  def test_same_destination_request_revokes_old_model_snapshot_before_http(self):
    old_generation = self.engine.route_generation
    response = MagicMock()
    response.json.return_value = {'routes': [{'legs': [{'steps': self.steps}]}]}

    def inspect_pending(*_args, **_kwargs):
      pending = self.pm.send.call_args.args[1]
      self.assertEqual(pending.which(), 'navInstruction')
      self.assertTrue(pending.valid)  # retain UI, revoke only model provenance
      self.assertEqual(pending.navInstruction.routeGeneration, old_generation)
      self.assertEqual(pending.navInstruction.gpsSourceMonoTime, 0)
      self.assertFalse(list(pending.navInstruction.allManeuvers))
      return response

    with patch('openpilot.selfdrive.navigation.navd.requests.get', side_effect=inspect_pending):
      self.engine.calculate_route((self.destination['longitude'], self.destination['latitude']))
    self.assertGreater(self.engine.route_generation, old_generation)
    self.assertGreater(instruction(self.engine, self.pm).gpsSourceMonoTime, 0)

  def test_destination_change_clears_old_route_before_hold_and_without_gps(self):
    self.engine.recompute_hold = 10
    self.engine.params.get.side_effect = lambda key: {'longitude': 1, 'latitude': 1} if key == 'NavDestination' else None
    self.engine.recompute_route(gps_ok=False)
    self.assertIsNone(self.engine.route)
    self.assertEqual(self.engine.route_generation, 0)
    self.assertEqual(self.engine.route_destination_key, '')
    self.assertEqual(self.engine.recompute_hold, 0)
    instruction(self.engine, self.pm)
    self.assertFalse(self.pm.send.call_args.args[1].valid)

  def test_destination_removal_invalidates_even_without_last_position(self):
    self.engine.last_position = None
    self.engine.params.get.side_effect = lambda key: None
    self.engine.recompute_route(gps_ok=False)
    self.assertIsNone(self.engine.route)
    self.assertEqual(self.engine.route_generation, 0)

  def test_destination_change_during_http_discards_finished_old_response(self):
    values = {'NavDestination': dict(self.destination)}
    self.engine.params.get.side_effect = values.get
    response = MagicMock()

    def changed_response():
      values['NavDestination'] = {'longitude': 2, 'latitude': 3}
      return {'routes': [{'legs': [{'steps': self.steps}]}]}

    response.json.side_effect = changed_response
    with patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
      self.engine.calculate_route((self.destination['longitude'], self.destination['latitude']))
    self.assertIsNone(self.engine.route)
    self.assertEqual(self.engine.route_generation, 0)
    self.assertEqual(self.engine.recompute_hold, 0)

  def test_clear_route_removes_wire_identity(self):
    self.engine.clear_route()
    self.engine.send_route()
    route = self.pm.send.call_args.args[1].navRoute
    self.assertEqual((route.routeGeneration, route.destinationKey, len(route.coordinates)), (0, '', 0))

  def test_bad_gps_coordinates_keep_ui_but_remove_model_snapshot(self):
    self.engine.sm.__getitem__.return_value.latitude = np.nan
    msg = instruction(self.engine, self.pm)
    self.assertTrue(self.pm.send.call_args.args[1].valid)
    self.assertEqual(msg.gpsSourceMonoTime, 0)
    self.assertFalse(list(msg.allManeuvers))

  def test_model_distance_uses_the_exact_source_named_by_its_timestamp(self):
    now = time.monotonic()
    internal_ns = int((now - .1) * 1e9)
    self.engine.sm.valid = {'gpsLocation': True, 'gpsLocationExternal': True}
    self.engine.sm.recv_time = dict.fromkeys(self.engine.sm.valid, now)
    self.engine.sm.logMonoTime = {'gpsLocation': internal_ns, 'gpsLocationExternal': int((now + .3) * 1e9)}
    internal = SimpleNamespace(hasFix=True, longitude=0., latitude=25 / 111195.0802335329, bearingDeg=0.)
    external = SimpleNamespace(hasFix=True, longitude=1., latitude=1., bearingDeg=0.)
    self.engine.sm.__getitem__.side_effect = {'gpsLocation': internal, 'gpsLocationExternal': external}.get
    self.engine.last_position = (1., 1.)  # legacy display selected the newer, foreign-clock external fix
    msg = instruction(self.engine, self.pm)
    self.assertEqual(msg.gpsSourceMonoTime, internal_ns)
    self.assertAlmostEqual(msg.allManeuvers[0].distance, 75., delta=.01)

  def test_invalid_model_gps_does_not_advance_tracker_from_retained_display_position(self):
    self.engine.last_position = (1., 1.)
    self.engine.sm.valid = dict.fromkeys(self.engine.sm.valid, False)
    instruction(self.engine, self.pm)
    self.assertEqual(self.engine.route_tracker.progress, -np.inf)
    self.engine.sm.valid['gpsLocation'] = True
    self.assertAlmostEqual(instruction(self.engine, self.pm).allManeuvers[0].distance, 75., delta=.01)

  def test_gps_loss_cannot_advance_or_remove_route_from_retained_display_position(self):
    self.engine.last_position = (1000 / 111195.0802335329, 100 / 111195.0802335329)
    self.engine.sm.valid = dict.fromkeys(self.engine.sm.valid, False)
    first_step = self.engine.step_idx
    instruction(self.engine, self.pm)
    self.assertEqual(self.engine.step_idx, first_step)
    self.engine.step_idx = len(self.engine.route) - 1
    msg = instruction(self.engine, self.pm)
    self.assertTrue(self.pm.send.call_args.args[1].valid)
    self.assertEqual(msg.gpsSourceMonoTime, 0)
    self.engine.params.remove.assert_not_called()
    self.assertIsNotNone(self.engine.route)

  def test_native_cereal_roundtrip_preserves_new_fields(self):
    self.engine.send_instruction()
    message = self.pm.send.call_args.args[1]
    with messaging.log.Event.from_bytes(message.to_bytes()) as decoded:
      self.assertEqual(decoded.navInstruction.routeGeneration, self.engine.route_generation)
      self.assertEqual(decoded.navInstruction.destinationKey, self.engine.route_destination_key)
      self.assertEqual(decoded.navInstruction.gpsSourceMonoTime, self.engine.sm.logMonoTime['gpsLocation'])

  @parameterized.expand(['empty_steps', 'bad_geometry', 'nonfinite_distance', 'missing_duration', 'invalid_banners'])
  def test_malformed_response_clears_all_old_identity_atomically(self, problem):
    route = [dict(s) for s in self.steps]
    if problem == 'empty_steps':
      route = []
    elif problem == 'bad_geometry':
      route[0]['geometry'] = {'coordinates': [[0, float('nan')], [1, 1]]}
    elif problem == 'nonfinite_distance':
      route[0]['distance'] = float('inf')
    elif problem == 'missing_duration':
      route[0].pop('duration')
    else:
      route[0]['bannerInstructions'] = None
    response = MagicMock()
    response.json.return_value = {'routes': [{'legs': [{'steps': route}]}]}
    with patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
      self.engine.calculate_route((self.destination['longitude'], self.destination['latitude']))
    self.assertIsNone(self.engine.route)
    self.assertIsNone(self.engine.route_geometry)
    self.assertIsNone(self.engine.route_tracker)
    self.assertEqual(self.engine.route_generation, 0)
    self.assertEqual(self.engine.route_destination_key, '')
