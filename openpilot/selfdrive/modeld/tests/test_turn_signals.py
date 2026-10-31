"""Signal freshness and standard modeld wiring without messaging or vehicle I/O."""
import ast
import inspect
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.carstate import CarState
from openpilot.cereal import log, messaging
from openpilot.selfdrive.modeld import modeld
from openpilot.selfdrive.modeld.turn_signals import fresh_turn_signal_active


@pytest.fixture
def sm():
  with patch.object(messaging, 'sub_sock', return_value=Mock()), patch.object(messaging, 'Poller', return_value=Mock()):
    return messaging.SubMaster(['carState'])


def receive(sm, left=False, right=False, *, stamp=100., received=None, valid=True):
  event = messaging.new_message('carState', valid=valid, logMonoTime=int(stamp * 1e9))
  event.carState.leftBlinker, event.carState.rightBlinker = left, right
  event.carState.vEgo = 1.
  sm.update_msgs(stamp if received is None else received, [event.as_reader()])


@pytest.mark.parametrize(('left', 'right', 'active'), [(False, False, False), (True, False, True),
                                                     (False, True, True), (True, True, True)])
def test_sustained_signal_and_hazards(sm, left, right, active):
  receive(sm, left, right)
  assert fresh_turn_signal_active(sm, now=100.) is active


@pytest.mark.parametrize('condition', ['missing', 'invalid', 'dead', 'stale', 'old_redelivery', 'processing_delay',
                                      'future', 'zero_timestamp', 'simulation_stale'])
def test_missing_or_untrustworthy_signal_returns_pr39182(sm, condition):
  if condition != 'missing':
    receive(sm, True, True, valid=condition != 'invalid')
  now = 100.
  if condition == 'dead':
    sm.alive['carState'] = False
  elif condition in ('stale', 'simulation_stale'):
    sm.simulation = condition == 'simulation_stale'
    now = 100.2
    sm.update_msgs(now, [])
  elif condition == 'old_redelivery':
    now = 100.2
    receive(sm, True, True, stamp=100., received=now)
    assert sm.alive['carState']  # A fresh receive must not launder an old event.
  elif condition == 'processing_delay':
    now = 100.2  # No new SubMaster update while inference was running.
    assert sm.alive['carState']
  elif condition == 'future':
    sm.logMonoTime['carState'] += 1
  elif condition == 'zero_timestamp':
    sm.logMonoTime['carState'] = 0
  assert fresh_turn_signal_active(sm, now=now) is False


def test_existing_100ms_freshness_boundary(sm):
  receive(sm, True)
  assert fresh_turn_signal_active(sm, now=100.099999999)
  assert not fresh_turn_signal_active(sm, now=100.1)


def test_use_hondas_existing_sustained_state_without_redebouncing(sm):
  interface = object.__new__(CarState)
  interface.left_blinker_cnt = interface.right_blinker_cnt = 0
  interface.left_blinker_prev = interface.right_blinker_prev = False
  # Honda already holds a short stalk activation. Modeld consumes its published
  # result, including cancellation, instead of inventing a lamp-phase timer.
  for frame in range(251):
    left, right = interface.update_blinker_from_stalk(250, frame == 0, False)
    now = 100. + frame * .01
    receive(sm, left, right, stamp=now)
    assert fresh_turn_signal_active(sm, now=now) is (frame < 249)


@pytest.mark.parametrize(('left', 'right', 'age', 'active'), [(False, False, 0., False), (True, False, 0., True),
                                                          (False, True, 0., True), (True, True, 0., True),
                                                          (True, True, .2, False)])
def test_standard_main_passes_fresh_signal_into_actual_action_helper(sm, left, right, age, active):
  receive(sm, left, right)
  tree = ast.parse(inspect.getsource(modeld.main))
  call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and
              isinstance(node.func, ast.Name) and node.func.id == 'get_action_from_model')
  # Execute the actual production call (including its freshness argument).
  namespace = {'get_action_from_model': modeld.get_action_from_model,
               'fresh_turn_signal_active': fresh_turn_signal_active, 'sm': sm,
               'model_output': {'action': np.array([[.125, -.2, 0., 0.]], np.float32)},
               'prev_action': log.ModelDataV2.Action(desiredCurvature=-.012),
               'lat_action_t': .175, 'long_action_t': .575, 'v_ego': 1.}
  with patch('openpilot.selfdrive.modeld.turn_signals.time.monotonic_ns', return_value=int((100. + age) * 1e9)):
    action = eval(compile(ast.Expression(body=call), 'stock_main_action_call', 'eval'), namespace)
  assert action.desiredCurvature == float(np.float32(.125 if active else .125 / 6.25))
