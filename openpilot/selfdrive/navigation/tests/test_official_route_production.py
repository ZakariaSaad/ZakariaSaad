"""Run actual producer methods against the pinned official definitions.

Networking is replaced with identical deterministic Mapbox responses. Route
tracking, instruction publication and native cereal serialization are real.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from openpilot.cereal import messaging
from openpilot.selfdrive.navigation import navd
from openpilot.selfdrive.navigation.tests.official_reference import load_reference, source
from openpilot.selfdrive.navigation.tests.test_maneuver_contract import METER, step


class Params:
  def __init__(self, destination):
    self.values = {'NavDestination': destination, 'LanguageSetting': 'main_en'}
  def get(self, name):
    return self.values.get(name)
  def remove(self, name):
    self.values.pop(name, None)


class Messages:
  def __init__(self):
    self.frame = 1
    self.valid = {'gpsLocation': True, 'gpsLocationExternal': False}
    self.recv_time = {'gpsLocation': 99.9, 'gpsLocationExternal': 99.8}
    self.logMonoTime = {'gpsLocation': 99_900_000_000, 'gpsLocationExternal': 99_800_000_000}
    self.data = {key: SimpleNamespace(hasFix=True, longitude=0., latitude=25 * METER, bearingDeg=0.) for key in self.valid}
  def update(self, _timeout):
    pass
  def __getitem__(self, name):
    return self.data[name]


class Publications:
  def __init__(self):
    self.messages = []
  def send(self, service, event):
    self.messages.append((service, event.valid, getattr(event, service).to_dict()))


def pair():
  route = load_reference('route')
  helpers = load_reference('helpers')
  upstream = load_reference('navd', namespace={
    'messaging': messaging, 'Params': Params, 'API_HOST': 'https://fixture.invalid', 'Api': Mock(),
    'cloudlog': Mock(), 'latest_gps_source': helpers.latest_gps_source,
    'RouteTracker': route.RouteTracker, 'along_path': route.along_path, 'local_meters': route.local_meters,
    'requests': requests,
  })
  steps = [step((0, 0), (0, 100), 'depart'), step((0, 100), (100, 100), 'turn', 'left'),
           step((100, 100), (100, 200), 'off ramp', 'right'), step((100, 200), (100, 200), 'arrive')]
  destination = {'longitude': 100 * METER, 'latitude': 200 * METER}
  engines = []
  with patch.dict('os.environ', {'MAPBOX_TOKEN': 'inert-test-token'}):
    for cls in (navd.RouteEngine, upstream.RouteEngine):
      engine = cls(Messages(), Publications(), Params(copy.deepcopy(destination)))
      engines.append(engine)
  return engines, steps


def response(steps):
  value = Mock(status_code=200)
  value.json.return_value = {'routes': [{'legs': [{'steps': copy.deepcopy(steps)}]}]}
  return value


def equal(engines):
  actual, expected = engines
  assert actual.pm.messages == expected.pm.messages
  for key in ('last_position', 'last_bearing', 'nav_destination', 'step_idx', 'route', 'recompute_hold', 'reroute_counter'):
    assert getattr(actual, key) == getattr(expected, key), key
  if actual.route_geometry is None or expected.route_geometry is None:
    assert actual.route_geometry is expected.route_geometry
  else:
    for left, right in zip(actual.route_geometry, expected.route_geometry, strict=True):
      np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(actual.maneuver_distances, expected.maneuver_distances)
    assert actual.route_tracker.progress == expected.route_tracker.progress
  assert actual.params.values == expected.params.values


def update(engines, steps):
  with patch('time.monotonic', return_value=100.), patch.object(navd.requests, 'get', return_value=response(steps)):
    for engine in engines:
      engine.update()
  equal(engines)


def test_actual_startup_guidance_repulication_and_consecutive_maneuvers():
  engines, steps = pair()
  update(engines, steps)
  for frame, position in ((2, (0, 95)), (3, (5, 100)), (10, (50, 100)), (11, (100, 105))):
    for engine in engines:
      engine.sm.frame = frame
      gps = engine.sm['gpsLocation']
      gps.longitude, gps.latitude = position[0] * METER, position[1] * METER
    update(engines, steps)


def test_gps_loss_and_new_destination_preserve_official_held_route_behavior():
  engines, steps = pair()
  update(engines, steps)
  for engine in engines:
    engine.sm.valid = dict.fromkeys(engine.sm.valid, False)
    engine.params.values['NavDestination'] = {'longitude': 1., 'latitude': 1.}
  update(engines, steps)
  assert engines[0].route is not None
  assert engines[0].pm.messages[-1][1]
  assert engines[0].pm.messages[-1][2]['allManeuvers']


def test_route_recompute_hold_and_clear_match_actual_upstream_producer():
  engines, steps = pair()
  update(engines, steps)
  for engine in engines:
    engine.recompute_hold = 5
    engine.params.values['NavDestination'] = {'longitude': 1., 'latitude': 1.}
  update(engines, steps)
  assert engines[0].recompute_hold == 4
  for engine in engines:
    engine.params.values['NavDestination'] = None
  update(engines, steps)
  assert engines[0].route is None
  assert not engines[0].pm.messages[-2][1]  # invalid instruction then empty route


def test_destination_change_during_http_is_not_reinterpreted_by_local_policy():
  engines, steps = pair()
  for engine in engines:
    engine.last_position = (0., 25 * METER)
    result = response(steps)
    def changed_response(engine=engine):
      engine.params.values['NavDestination'] = {'longitude': 2., 'latitude': 3.}
      return {'routes': [{'legs': [{'steps': copy.deepcopy(steps)}]}]}
    result.json.side_effect = changed_response
    with patch.object(navd.requests, 'get', return_value=result):
      engine.calculate_route((100 * METER, 200 * METER))
    engine.send_instruction()
    engine.send_route()
  equal(engines)
  assert engines[0].route is not None


def test_no_pre_http_neutralization_and_empty_result_match_upstream():
  engines, steps = pair()
  update(engines, steps)
  for engine in engines:
    before = copy.deepcopy(engine.pm.messages)
    result = response(steps)
    result.json.return_value = {'routes': []}
    def pending(_url, engine=engine, before=before, result=result, **_kwargs):
      assert engine.pm.messages == before
      return result
    with patch.object(navd.requests, 'get', side_effect=pending):
      engine.calculate_route(engine.nav_destination)
    engine.send_instruction()
    engine.send_route()
  equal(engines)


def test_source_selection_uses_receipt_freshness_without_custom_capture_age():
  engines, steps = pair()
  for engine in engines:
    engine.sm.valid['gpsLocationExternal'] = True
    engine.sm.logMonoTime['gpsLocationExternal'] = 200_000_000_000
    engine.sm.recv_time['gpsLocationExternal'] = 99.95
    engine.sm['gpsLocationExternal'].longitude = 10 * METER
  update(engines, steps)
  assert engines[0].last_position == (10 * METER, 25 * METER)


def test_producer_and_encoder_definitions_are_exact_official_source():
  local = Path(navd.__file__)
  def bodies(text):
    tree = ast.parse(text)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return ast.dump(tree, include_attributes=False)
  assert bodies(local.read_text()) == bodies(source('navd'))
  assert local.with_name('helpers.py').read_text() == source('helpers')
