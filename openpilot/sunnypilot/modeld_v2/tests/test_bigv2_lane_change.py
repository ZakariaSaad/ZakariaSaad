"""Bigv2 shares ordinary lane-change policy, except for road-edge blocking.

Synthetic car states and isolated real Params only; no vehicle or model I/O.
"""
from types import SimpleNamespace

import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import (
  DesireHelper as OrdinaryDesireHelper, LaneChangeState, LaneChangeDirection,
  LANE_CHANGE_SPEED_MIN, LANE_CHANGE_TIME_MAX,
)
from openpilot.sunnypilot.modeld_v2.bigv2_desire_helper import DesireHelper
from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import AutoLaneChangeMode, AUTO_LANE_CHANGE_TIMER


MODES = tuple(AUTO_LANE_CHANGE_TIMER)
AUTO_MODES = tuple(mode for mode in MODES if mode > AutoLaneChangeMode.NUDGE)


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def helpers(mode=AutoLaneChangeMode.NUDGE, bsm_delay=False):
  params = Params()
  params.put('AutoLaneChangeTimer', mode, block=True)
  params.put_bool('AutoLaneChangeBsmDelay', bsm_delay, block=True)
  params.put_bool('LaneTurnDesire', False, block=True)
  return DesireHelper(), OrdinaryDesireHelper()


def carstate(side='left', **changes):
  state = SimpleNamespace(vEgo=22., leftBlinker=side == 'left', rightBlinker=side == 'right',
                          steeringPressed=False, steeringTorque=0., leftBlindspot=False, rightBlindspot=False,
                          brakePressed=False)
  vars(state).update(changes)
  return state


def snapshot(helper):
  return ({key: getattr(helper, key) for key in
           ('lane_change_state', 'lane_change_direction', 'lane_change_timer', 'prev_one_blinker', 'desire')},
          {key: value for key, value in vars(helper.alc).items() if key not in ('DH', 'params')})


def step(pair, state, active=True, probability=.5):
  for helper in pair:
    helper.update(state, active, probability)
  assert snapshot(pair[0]) == snapshot(pair[1])
  return pair[0]


def finish_phase(pair, state, phase, probability=.5):
  for _ in range(100):
    step(pair, state, probability=probability)
    if pair[0].lane_change_state != phase:
      return
  pytest.fail(f'Lane-change phase did not complete: {phase}')


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ('left', 'right'))
def test_all_modes_match_ordinary_and_require_correct_nudge(mode, side):
  pair, state = helpers(mode), carstate(side)
  actual = step(pair, state)
  expected_direction = LaneChangeDirection.left if side == 'left' else LaneChangeDirection.right
  if mode == AutoLaneChangeMode.OFF:
    state.steeringPressed, state.steeringTorque = True, 1. if side == 'left' else -1.
    for _ in range(100):
      step(pair, state)
    assert actual.lane_change_state == LaneChangeState.off
    assert actual.desire == log.Desire.none
    return

  assert actual.lane_change_state == LaneChangeState.preLaneChange
  assert actual.lane_change_direction == expected_direction
  if mode == AutoLaneChangeMode.NUDGE:
    for _ in range(100):
      step(pair, state)
    assert actual.lane_change_state == LaneChangeState.preLaneChange
    state.steeringPressed, state.steeringTorque = True, -1. if side == 'left' else 1.
    step(pair, state)
    assert actual.lane_change_state == LaneChangeState.preLaneChange
    state.steeringTorque *= -1.
    step(pair, state)
  else:
    delay, elapsed = AUTO_LANE_CHANGE_TIMER[mode], 0.
    while elapsed <= delay:
      elapsed += DT_MDL
      step(pair, state)
      assert actual.alc.lane_change_wait_timer == elapsed
      assert actual.lane_change_state == (LaneChangeState.laneChangeStarting if elapsed > delay else LaneChangeState.preLaneChange)
  assert actual.desire == (log.Desire.laneChangeLeft if side == 'left' else log.Desire.laneChangeRight)


@pytest.mark.parametrize('mode', AUTO_MODES)
def test_exact_timer_boundary_is_strictly_greater(mode):
  pair, state = helpers(mode), carstate()
  step(pair, state)
  step(pair, state)
  for helper in pair:
    helper.alc.lane_change_wait_timer = AUTO_LANE_CHANGE_TIMER[mode]
    assert not helper.alc.update_allowed()
  actual = step(pair, state)
  assert actual.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('mode', AUTO_MODES)
@pytest.mark.parametrize('bsm_delay', (False, True))
@pytest.mark.parametrize('side', ('left', 'right'))
def test_actual_blindspot_blocks_and_clear_delay_matches_ordinary(mode, bsm_delay, side):
  pair, state = helpers(mode, bsm_delay), carstate(side)
  setattr(state, side + 'Blindspot', True)
  actual = step(pair, state)
  for _ in range(100):
    step(pair, state)
    assert actual.lane_change_state == LaneChangeState.preLaneChange
  delay = AUTO_LANE_CHANGE_TIMER[mode]
  if bsm_delay:
    assert actual.alc.lane_change_wait_timer == (-1. if mode == AutoLaneChangeMode.NUDGELESS else delay - 1.)
  setattr(state, side + 'Blindspot', False)
  elapsed = actual.alc.lane_change_wait_timer
  for _ in range(100):
    elapsed += DT_MDL
    step(pair, state)
    assert actual.lane_change_state == (LaneChangeState.laneChangeStarting if elapsed > delay else LaneChangeState.preLaneChange)
    if actual.lane_change_state == LaneChangeState.laneChangeStarting:
      break
  assert actual.lane_change_state == LaneChangeState.laneChangeStarting
  assert actual.alc.prev_lane_change


@pytest.mark.parametrize('side', ('left', 'right'))
@pytest.mark.parametrize('mode', (AutoLaneChangeMode.NUDGE, AutoLaneChangeMode.NUDGELESS))
def test_road_edge_bypassed_but_real_blindspot_still_blocks(mode, side):
  actual, ordinary = helpers(mode, bsm_delay=True)
  Params().put_bool('RoadEdgeLaneChangeEnabled', True, block=True)
  state = carstate(side, steeringPressed=True, steeringTorque=1. if side == 'left' else -1.)
  edges = (side == 'left', side == 'right')
  setattr(state, side + 'Blindspot', True)
  for _ in range(100):
    actual.update(state, True, .5, *edges)
    ordinary.update(state, True, .5, *edges)
  assert actual.lane_change_state == ordinary.lane_change_state == LaneChangeState.preLaneChange
  setattr(state, side + 'Blindspot', False)
  actual.update(state, True, .5, *edges)
  ordinary.update(state, True, .5, *edges)
  assert actual.lane_change_state == LaneChangeState.laneChangeStarting
  assert ordinary.lane_change_state == LaneChangeState.preLaneChange
  assert Params().get_bool('RoadEdgeLaneChangeEnabled')


@pytest.mark.parametrize('mode', AUTO_MODES)
def test_brake_latches_auto_inhibition_but_keeps_ordinary_nudge_behavior(mode):
  pair, state = helpers(mode), carstate()
  actual = step(pair, state)
  state.brakePressed = True
  step(pair, state)
  state.brakePressed = False
  for _ in range(100):
    step(pair, state)
  assert actual.lane_change_state == LaneChangeState.preLaneChange
  assert actual.alc.prev_brake_pressed
  assert not actual.alc.auto_lane_change_allowed
  state.steeringPressed, state.steeringTorque = True, 1.
  step(pair, state)
  assert actual.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('mode', AUTO_MODES)
def test_no_repeated_auto_lane_changes_until_signal_cancelled(mode):
  pair, state = helpers(mode), carstate()
  actual = step(pair, state)
  finish_phase(pair, state, LaneChangeState.preLaneChange)
  assert actual.lane_change_state == LaneChangeState.laneChangeStarting
  finish_phase(pair, state, LaneChangeState.laneChangeStarting, probability=.01)
  assert actual.lane_change_state == LaneChangeState.preLaneChange
  for _ in range(100):
    step(pair, state)
    assert actual.lane_change_state == LaneChangeState.preLaneChange
  state.leftBlinker = False
  step(pair, state)
  assert actual.lane_change_state == LaneChangeState.off
  assert actual.alc.lane_change_wait_timer == 0.
  assert not actual.alc.prev_lane_change
  state.leftBlinker = True
  step(pair, state)
  assert actual.lane_change_state == LaneChangeState.preLaneChange
  finish_phase(pair, state, LaneChangeState.preLaneChange)
  assert actual.desire == log.Desire.laneChangeLeft


@pytest.mark.parametrize('reason', ('cancel', 'hazards', 'slow', 'disengage', 'timeout', 'setting_off'))
def test_pending_lane_change_cancellation_and_resets_match_ordinary(reason):
  pair, state = helpers(AutoLaneChangeMode.THREE_SECONDS), carstate(brakePressed=True)
  actual = step(pair, state)
  step(pair, state)
  assert actual.alc.prev_brake_pressed
  if reason == 'cancel':
    state.leftBlinker = False
  elif reason == 'hazards':
    state.rightBlinker = True
  elif reason == 'slow':
    state.vEgo = LANE_CHANGE_SPEED_MIN - .001
  elif reason == 'timeout':
    for helper in pair:
      helper.lane_change_timer = LANE_CHANGE_TIME_MAX + DT_MDL
  elif reason == 'setting_off':
    Params().put('AutoLaneChangeTimer', AutoLaneChangeMode.OFF, block=True)
    for helper in pair:
      helper.alc.param_read_counter = 50
  step(pair, state, active=reason != 'disengage')
  assert actual.lane_change_state == LaneChangeState.off
  assert actual.lane_change_direction == LaneChangeDirection.none
  assert actual.lane_change_timer == actual.alc.lane_change_wait_timer == 0.
  assert not actual.alc.prev_brake_pressed
  assert not actual.alc.prev_lane_change


@pytest.mark.parametrize('speed,eligible', ((LANE_CHANGE_SPEED_MIN - .001, False), (LANE_CHANGE_SPEED_MIN, True)))
def test_lane_change_speed_boundary_and_hazards(speed, eligible):
  pair, state = helpers(), carstate(vEgo=speed)
  actual = step(pair, state)
  assert actual.lane_change_state == (LaneChangeState.preLaneChange if eligible else LaneChangeState.off)
  state.rightBlinker = True
  for _ in range(100):
    step(pair, state)
  assert actual.lane_change_state == LaneChangeState.off
  assert actual.desire == log.Desire.none


def test_saved_params_refresh_at_ordinary_cadence_without_writes():
  pair, state = helpers(), carstate(side=None)
  params = Params()
  params.put_bool('RoadEdgeLaneChangeEnabled', True, block=True)
  actual = step(pair, state)
  params.put('AutoLaneChangeTimer', AutoLaneChangeMode.TWO_SECONDS, block=True)
  params.put_bool('AutoLaneChangeBsmDelay', True, block=True)
  for _ in range(49):
    step(pair, state)
    assert actual.alc.lane_change_set_timer == AutoLaneChangeMode.NUDGE
    assert not actual.alc.lane_change_bsm_delay
  step(pair, state)
  assert actual.alc.lane_change_set_timer == AutoLaneChangeMode.TWO_SECONDS
  assert actual.alc.lane_change_bsm_delay
  assert params.get('AutoLaneChangeTimer') == AutoLaneChangeMode.TWO_SECONDS
  assert params.get_bool('AutoLaneChangeBsmDelay')
  assert params.get_bool('RoadEdgeLaneChangeEnabled')
  assert not params.get_bool('LaneTurnDesire')


@pytest.mark.parametrize('side', ('left', 'right'))
def test_low_speed_turn_conditioning_is_not_added_to_bigv2(side):
  actual, ordinary = helpers()
  params = Params()
  params.put_bool('LaneTurnDesire', True, block=True)
  params.put('LaneTurnValue', 20., block=True)
  state = carstate(side, vEgo=1.)
  actual.update(state, True, .5)
  ordinary.update(state, True, .5)
  assert actual.desire == log.Desire.none
  assert ordinary.desire == (log.Desire.turnLeft if side == 'left' else log.Desire.turnRight)
  assert params.get_bool('LaneTurnDesire')


def test_existing_lane_change_desire_survives_speed_drop_with_turn_setting_saved():
  actual, ordinary = helpers()
  Params().put_bool('LaneTurnDesire', True, block=True)
  Params().put('LaneTurnValue', 20., block=True)
  state = carstate(steeringPressed=True, steeringTorque=1.)
  for helper in (actual, ordinary):
    helper.update(state, True, .5)
    helper.update(state, True, .5)
    assert helper.desire == log.Desire.laneChangeLeft
  state.vEgo = 1.
  actual.update(state, True, .5)
  ordinary.update(state, True, .5)
  assert actual.desire == log.Desire.laneChangeLeft
  assert ordinary.desire == log.Desire.turnLeft
