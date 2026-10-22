"""Synthetic route-to-model contracts using native cereal messages, without model inference."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.navigation.helpers import latest_gps_source, navigation_input
from openpilot.selfdrive.navigation.navd import RouteEngine

METER = 1 / 111195.0802335329  # match route.py's mean-Earth metric at the equator


def step(start, end, kind, modifier='', reported_distance=None):
  length = float(np.linalg.norm(np.asarray(end) - start))
  return {'geometry': {'coordinates': [[x * METER, y * METER] for x, y in (start, end)]},
          'distance': length if reported_distance is None else reported_distance, 'duration': length / 10,
          'bannerInstructions': [], 'maneuver': {'type': kind, 'modifier': modifier}}


def engine_for(steps, position=(0, 25)):
  sm, pm, params = MagicMock(), MagicMock(), MagicMock()
  destination = {'longitude': steps[-1]['geometry']['coordinates'][-1][0], 'latitude': steps[-1]['geometry']['coordinates'][-1][1]}
  params.get.side_effect = lambda key: destination if key == 'NavDestination' else None
  sm.valid = {'gpsLocation': True, 'gpsLocationExternal': False}
  sm.recv_time = dict.fromkeys(sm.valid, time.monotonic())
  sm.logMonoTime = dict.fromkeys(sm.valid, time.monotonic_ns())
  sm.__getitem__.return_value = SimpleNamespace(hasFix=True, longitude=position[0] * METER, latitude=position[1] * METER,
                                               bearingDeg=0, speed=0)
  with patch('openpilot.selfdrive.navigation.navd.Api'):
    engine = RouteEngine(sm, pm, params)
  engine.last_position = (position[0] * METER, position[1] * METER)
  response = MagicMock()
  response.json.return_value = {'routes': [{'legs': [{'steps': steps}]}]}
  with patch('openpilot.selfdrive.navigation.navd.requests.get', return_value=response):
    engine.calculate_route(tuple(steps[-1]['geometry']['coordinates'][-1]))
  return engine, pm


def instruction(engine, pm):
  engine.send_instruction()
  return pm.send.call_args.args[1].navInstruction


class TestManeuverContract(unittest.TestCase):
  @parameterized.expand([
    ('turn', 'left', 15), ('turn', 'right', 10), ('off ramp', 'right', 20), ('off ramp', 'left', 21),
    ('roundabout', 'right', 26), ('rotary', 'left', 26), ('exit roundabout', 'right', 27), ('on ramp', 'slight right', 18),
    ('fork', 'left', 24), ('turn', 'uturn', 13),
  ])
  def test_real_publication_and_47_input(self, kind, modifier, command):
    steps = [step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), kind, modifier),
             step((100, 100), (100, 100), 'arrive')]
    engine, pm = engine_for(steps)
    msg = instruction(engine, pm)
    self.assertEqual(len(msg.allManeuvers), 2)
    np.testing.assert_allclose([m.distance for m in msg.allManeuvers], [75, 175], atol=.01)
    tensor = navigation_input(msg.allManeuvers, 15)
    self.assertEqual((tensor.dtype, tensor.shape), (np.dtype(np.float32), (47,)))
    self.assertEqual(tensor[command], 1)
    self.assertEqual(np.count_nonzero(tensor[:-1]), 1)
    self.assertAlmostEqual(tensor[-1], .6, delta=1e-4)

  def test_consecutive_turns_advance_without_step_distance_shift(self):
    steps = [step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left'),
             step((100, 100), (100, 200), 'turn', 'right'), step((100, 200), (100, 200), 'arrive')]
    engine, pm = engine_for(steps)
    self.assertEqual(navigation_input(instruction(engine, pm).allManeuvers, 0)[15], 1)
    # Respect the tracker's 100 m search window: left is now passed but right remains ahead.
    engine.last_position = (0, 95 * METER)
    instruction(engine, pm)
    engine.last_position = (5 * METER, 100 * METER)
    msg = instruction(engine, pm)
    self.assertEqual([(m.type, m.modifier) for m in msg.allManeuvers], [('turn', 'right'), ('arrive', '')])
    vector = navigation_input(msg.allManeuvers, 0)
    self.assertEqual(vector[10], 1)
    self.assertAlmostEqual(vector[-1], .95, delta=1e-4)

  def test_geometry_not_mapbox_reported_step_distance(self):
    steps = [step((0, 0), (0, 100), 'depart', reported_distance=900), step((0, 100), (100, 100), 'turn', 'right')]
    engine, pm = engine_for(steps)
    msg = instruction(engine, pm)
    self.assertAlmostEqual(msg.maneuverDistance, 875, delta=.01)  # existing UI distance unchanged
    self.assertAlmostEqual(msg.allManeuvers[0].distance, 75, delta=.01)  # model uses geometric distance

  @parameterized.expand(['absent', 'missing_type', 'invalid_type', 'invalid_modifier'])
  def test_incomplete_future_metadata_fails_closed_without_losing_ui(self, missing):
    steps = [step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'right')]
    if missing == 'absent':
      steps[1].pop('maneuver')
    elif missing == 'missing_type':
      steps[1]['maneuver'].pop('type')
    elif missing == 'invalid_type':
      steps[1]['maneuver']['type'] = None
    else:
      steps[1]['maneuver']['modifier'] = []
    engine, pm = engine_for(steps)
    msg = instruction(engine, pm)
    self.assertTrue(pm.send.call_args.args[1].valid)
    self.assertFalse(list(msg.allManeuvers))
    self.assertFalse(navigation_input(msg.allManeuvers, 0).any())

  def test_missing_route_clears_maneuver_list(self):
    engine, pm = engine_for([step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left')])
    self.assertTrue(instruction(engine, pm).allManeuvers)
    engine.clear_route()
    msg = instruction(engine, pm)
    self.assertFalse(pm.send.call_args.args[1].valid)
    self.assertFalse(list(msg.allManeuvers))
    self.assertIsNone(engine.route_tracker)
    self.assertIsNone(engine.maneuver_distances)

  def test_empty_and_passed_maneuvers_zero_tensor(self):
    self.assertFalse(navigation_input([], 0).any())
    msg = messaging.new_message('navInstruction', valid=True)
    msg.navInstruction.allManeuvers = [{'type': 'turn', 'modifier': 'left', 'distance': 10}]
    self.assertFalse(navigation_input(msg.navInstruction.allManeuvers, 11).any())

  def test_depart_arrive_and_merges_do_not_invent_turn_command(self):
    msg = messaging.new_message('navInstruction', valid=True)
    msg.navInstruction.allManeuvers = [{'type': kind, 'modifier': 'right', 'distance': 100} for kind in ('depart', 'merge', 'arrive')]
    self.assertFalse(navigation_input(msg.navInstruction.allManeuvers, 0).any())

  def test_stale_gps_requires_consumer_freshness_gate(self):
    sm = MagicMock()
    sm.valid = {'gpsLocation': True, 'gpsLocationExternal': True}
    sm.recv_time = dict.fromkeys(sm.valid, time.monotonic() - 3)
    sm.__getitem__.return_value = SimpleNamespace(hasFix=True)
    self.assertIsNone(latest_gps_source(sm))
    # Encoding has no freshness context: the experimental consumer must reject stale physical source data.
    msg = messaging.new_message('navInstruction', valid=True)
    msg.navInstruction.allManeuvers = [{'type': 'turn', 'modifier': 'left', 'distance': 100}]
    self.assertEqual(navigation_input(msg.navInstruction.allManeuvers, 0)[15], 1)

  def test_unknown_maneuver_never_fabricates_straight(self):
    msg = messaging.new_message('navInstruction', valid=True)
    msg.navInstruction.allManeuvers = [{'type': 'unknown future type', 'modifier': 'unknown', 'distance': 100}]
    self.assertFalse(navigation_input(msg.navInstruction.allManeuvers, 0).any())

  def test_display_can_retain_route_during_gps_loss(self):
    steps = [step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left')]
    engine, pm = engine_for(steps)
    engine.sm.valid = {'gpsLocation': False, 'gpsLocationExternal': False}
    engine.sm.recv_time = dict.fromkeys(engine.sm.valid, time.monotonic() - 30)
    engine.sm.frame = 1
    engine.params.get.side_effect = lambda key: {'latitude': 100 * METER, 'longitude': 100 * METER} if key == 'NavDestination' else None
    engine.update()
    nav = [call.args[1] for call in pm.send.call_args_list if call.args[0] == 'navInstruction'][-1]
    # The UI can retain the route through a tunnel; no maneuver/model GPS snapshot remains usable.
    self.assertTrue(nav.valid)
    self.assertEqual(nav.navInstruction.gpsSourceMonoTime, 0)
    self.assertFalse(list(nav.navInstruction.allManeuvers))
