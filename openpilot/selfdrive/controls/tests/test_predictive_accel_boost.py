"""Predictive boost through the real MVL planner, model publishers and controller.

The baseline SP.update excerpt is unchanged from 689d2b40a2d27237f50f844d7526828d91fc6933,
openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py (MIT, repository
LICENSE.md). The parent planner and legacy AccelBoost remain unchanged; bypassing
only the new selection hook reproduces that baseline without Git/network access.
Synthetic model outputs and radar observations; no model weights, CAN or vehicle I/O.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import custom, log, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_planner import CONTROL_N_T_IDX, LongitudinalPlanSource
from openpilot.selfdrive.controls.tests import test_upstream_longitudinal as helpers
from openpilot.selfdrive.modeld import fill_model_msg as native_publish
from openpilot.selfdrive.modeld.parse_model_outputs import Parser as NativeParser
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.modeld_v2 import fill_model_msg as ordinary_publish
from openpilot.sunnypilot.modeld_v2.constants import Meta, ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser as OrdinaryParser
from openpilot.sunnypilot.selfdrive.controls.lib.accel_boost import AccelBoostSP
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

isolated_params = helpers.isolated_params
PARAM_FRAMES = int(PARAMS_UPDATE_PERIOD / DT_MDL)


def baseline_update():
  body = (Path(__file__).with_name('fixtures') / 'predictive_boost_baseline_sp_update.txt').read_text()
  assert hashlib.sha256(body.encode()).hexdigest() == 'dda380651da4e992e7269b4e54d565d26dacd548813b644d406e8d0f1c72e706'
  namespace = {'messaging': messaging}
  exec(compile(body, 'baseline_689d2b40a_sp_update', 'exec'), namespace)
  return namespace['update']


def model_message(kind='native', *, accel=0., speed=20., brake=.05, gas=.3):
  fixtures = Path(__file__).resolve().parents[2] / 'navigation/tests/fixtures/nav_models.json'
  signature = next(model for model in json.loads(fixtures.read_text())['models']
                   if model['tag'] == 'big-nav-model-integration-1d18c42e')
  output = next(item for item in signature['outputs'] if item['name'] == 'outputs')
  raw = np.zeros(tuple(output['shape']), dtype=np.float32)
  slices = {name: slice(bounds['start'], bounds['stop'], bounds['step']) for name, bounds in signature['output_slices'].items()}
  parser = NativeParser() if kind == 'native' else OrdinaryParser()
  parsed = parser.parse_outputs({name: raw[:, bounds].copy() for name, bounds in slices.items()})
  parsed['plan'][0, :, Plan.POSITION] = np.column_stack((np.asarray(ModelConstants.T_IDXS) * speed, np.zeros((33, 2))))
  parsed['plan'][0, :, Plan.VELOCITY] = [speed, 0., 0.]
  parsed['plan'][0, :, Plan.ORIENTATION_RATE] = [0., 0., .01]
  parsed['meta'][0, Meta.BRAKE_PRESS] = brake
  parsed['meta'][0, Meta.GAS_PRESS] = gas
  action = log.ModelDataV2.Action(desiredAcceleration=accel, desiredCurvature=0., shouldStop=False)
  model = messaging.new_message('modelV2')
  if kind == 'native':
    native_publish.fill_model_msg(model, parsed, action, native_publish.PublishState(), 1, 1, 1, 0., 0, 0., True)
  else:
    ordinary_publish.fill_model_msg(messaging.new_message('drivingModelData'), model, parsed, action,
                                   ordinary_publish.PublishState(), 1, 1, 1, 0., 0, 0., True, Meta)
  model.modelV2.big = kind == 'native'
  return model.modelV2


def advance(control, sm, frames=PARAM_FRAMES + 1):
  for _ in range(frames):
    control.update(sm)


def assert_same_plan(actual, expected):
  for name in ('output_a_target', 'output_should_stop', 'a_cruise', 'fcw'):
    assert getattr(actual, name) == getattr(expected, name), name
  assert actual.mpc.source == expected.mpc.source
  assert vars(actual.accel_boost) == vars(expected.accel_boost)
  for name in ('v_desired_trajectory', 'a_desired_trajectory', 'j_desired_trajectory'):
    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize('kind', ['native', 'ordinary'])
def test_toggle_off_matches_baseline_planner_and_honda_longcontrol(kind):
  Params().put_bool('AccelBoostSP', False, block=True)
  actual, expected = helpers.planner(), helpers.planner()
  original = baseline_update()
  sm = helpers.Messages(model_message(kind))
  cp_sp = custom.CarParamsSP.new_message()
  actual_long, expected_long = LongControl(actual.CP, cp_sp), LongControl(expected.CP, cp_sp)
  saw_learned_boost = saw_braking = saw_stop = False
  for frame in range(360):
    sm['carState'].gasPressed = 60 <= frame < 140 or 160 <= frame < 240
    sm['modelV2'].action.desiredAcceleration = -.4 if 240 <= frame < 260 else -2. if 280 <= frame < 300 else 0.
    sm['modelV2'].action.shouldStop = 280 <= frame < 300
    sm['modelV2'].action.desiredCurvature = .005 if frame % 17 == 0 else 0.
    sm['radarState'].leadOne.present = 260 <= frame < 280
    sm['radarState'].leadOne.dRel, sm['radarState'].leadOne.vLead = 12., 1.
    sm['radarState'].leadOne.modelProb = 1.
    sm['carState'].vEgo = .1 if 280 <= frame < 300 else 20.
    sm['carState'].aEgo = -.1
    enabled = not 300 <= frame < 320
    sm['selfdriveState'].enabled = sm['carControl'].enabled = enabled
    sm['controlsState'].longControlState = 'pid' if enabled else 'off'
    actual.update(sm)
    with patch.object(LongitudinalPlannerSP, 'update', original):
      expected.update(sm)
    assert_same_plan(actual, expected)
    limits = CarInterface.get_pid_accel_limits(actual.CP, cp_sp, sm['carState'].vEgo, sm['carState'].vCruise / 3.6)
    args = (enabled, sm['carState'], actual.output_a_target, actual.output_should_stop, limits)
    got = actual_long.update(*args)
    wanted = expected_long.update(enabled, sm['carState'], expected.output_a_target, expected.output_should_stop, limits)
    assert got == wanted and limits[0] <= got <= limits[1]
    saw_learned_boost |= actual.accel_boost.total_boost > .05
    saw_braking |= actual.output_a_target < 0
    saw_stop |= actual.output_should_stop
  assert saw_learned_boost and saw_braking and saw_stop


def test_stock_longitudinal_keeps_legacy_boost_history_even_when_toggle_changes():
  params = Params()
  params.put_bool('AccelBoostSP', False, block=True)
  actual, expected = helpers.planner(openpilot_long=False), helpers.planner(openpilot_long=False)
  original = baseline_update()
  sm = helpers.Messages(model_message())
  legacy = actual.accel_boost
  for frame in range(250):
    if frame % 50 == 0:
      params.put_bool('AccelBoostSP', bool((frame // 50) % 2), block=True)
    sm['carState'].gasPressed = 40 <= frame < 180
    actual.update(sm)
    with patch.object(LongitudinalPlannerSP, 'update', original):
      expected.update(sm)
    assert actual.accel_boost is legacy and type(legacy) is AccelBoost
    assert_same_plan(actual, expected)
  assert legacy.total_boost > 0.


@pytest.mark.parametrize('kind', ['native', 'ordinary'])
@pytest.mark.parametrize(('personality', 'cap'), [('aggressive', .4), ('standard', .25), ('relaxed', .1)])
def test_model_publishers_supply_predictions_and_new_boost_replaces_legacy_once(kind, personality, cap):
  params = Params()
  params.put_bool('AccelBoostSP', False, block=True)
  control = helpers.planner()
  sm = helpers.Messages(model_message(kind))
  sm['selfdriveState'].personality = personality
  probs = sm['modelV2'].meta.disengagePredictions
  assert len(probs.brakePressProbs) >= 2 and len(probs.gasPressProbs) >= 2
  assert probs.brakePressProbs[1] == pytest.approx(.05)
  assert probs.gasPressProbs[1] == pytest.approx(.3)
  control.update(sm)
  old = control.accel_boost
  old.total_boost = .2
  old.update = Mock(wraps=old.update)
  params.put_bool('AccelBoostSP', True, block=True)
  advance(control, sm)
  assert control.accel_boost is control.accel_boost_sp
  old.update.reset_mock()
  predictive = Mock(wraps=control.accel_boost_sp.update)
  with patch.object(control.accel_boost_sp, 'update', predictive):
    for _ in range(80):
      before = predictive.call_count
      control.update(sm)
      assert predictive.call_count == before + 1
  old.update.assert_not_called()
  assert control.accel_boost.total_boost == pytest.approx(cap)
  assert control.output_a_target == pytest.approx(cap)  # No extra learned .2 added.
  sent = {}
  control.publish(sm, SimpleNamespace(send=lambda name, value: sent.update({name: value})))
  plan = sent['longitudinalPlan'].longitudinalPlan
  assert plan.accelBoost == pytest.approx(cap) and plan.aTarget == pytest.approx(cap)


def test_switches_disengagement_and_short_predictions_reset_predictive_history():
  params = Params()
  params.put_bool('AccelBoostSP', True, block=True)
  control = helpers.planner()
  sm = helpers.Messages(model_message())
  advance(control, sm, 80)
  assert control.accel_boost.total_boost > 0
  sm['selfdriveState'].enabled = False
  sm['controlsState'].longControlState = 'off'
  control.update(sm)
  assert control.accel_boost.boost == control.accel_boost.total_boost == 0.
  sm['selfdriveState'].enabled = True
  sm['controlsState'].longControlState = 'pid'
  control.update(sm)
  assert control.accel_boost.total_boost == pytest.approx(.01)
  for key in ('brakePressProbs', 'gasPressProbs'):
    advance(control, sm, 80)
    setattr(sm['modelV2'].meta.disengagePredictions, key, [])
    control.update(sm)
    assert control.accel_boost.boost == control.accel_boost.total_boost == 0.
    assert control.output_a_target == pytest.approx(sm['modelV2'].action.desiredAcceleration)
    sm['modelV2'] = model_message()
  params.put_bool('AccelBoostSP', False, block=True)
  advance(control, sm)
  assert type(control.accel_boost) is AccelBoost
  assert control.accel_boost_sp.boost == control.accel_boost_sp.total_boost == 0.
  assert control.accel_boost.total_boost == 0.
  control.accel_boost.total_boost = .12
  params.put_bool('AccelBoostSP', True, block=True)
  for _ in range(PARAM_FRAMES + 1):
    control.update(sm)
    if type(control.accel_boost) is AccelBoostSP:
      assert control.accel_boost.total_boost == pytest.approx(.01)
      break
  else:
    pytest.fail('Predictive mode was not selected within its parameter refresh interval')


@pytest.mark.parametrize('gate', ['lead', 'cruise', 'braking', 'stop', 'curve'])
def test_upstream_gates_and_downstream_planner_minimum_remain_effective(gate):
  Params().put_bool('AccelBoostSP', True, block=True)
  control = helpers.planner()
  sm = helpers.Messages(model_message())
  advance(control, sm, 80)
  assert control.accel_boost.total_boost > 0
  if gate == 'lead':
    sm['radarState'].leadOne.present = True
    sm['radarState'].leadOne.dRel, sm['radarState'].leadOne.vLead = 12., 1.
    sm['radarState'].leadOne.modelProb = 1.
  elif gate == 'cruise':
    sm['carState'].vCruise = sm['carState'].vCruiseCluster = 30.
  elif gate == 'braking':
    sm['modelV2'].action.desiredAcceleration = -1.
  elif gate == 'stop':
    sm['modelV2'].action.shouldStop = True
  elif gate == 'curve':
    sm['modelV2'].action.desiredCurvature = 3. / sm['carState'].vEgo**2
  # The existing cruise candidate retains its jerk-limited ramp after lowering
  # the set speed. Let that unchanged ramp cross zero before requiring braking.
  for _ in range(100 if gate == 'cruise' else 20):
    control.update(sm)
    mpc_target = get_accel_from_plan(control.v_desired_trajectory, control.a_desired_trajectory, CONTROL_N_T_IDX,
                                     action_t=control.CP.longitudinalActuatorDelay + DT_MDL)
    boosted = sm['modelV2'].action.desiredAcceleration + control.accel_boost.total_boost
    assert control.output_a_target == pytest.approx(np.clip(min(mpc_target, control.a_cruise, boosted), ACCEL_MIN, ACCEL_MAX))
    assert control.output_a_target <= control.a_cruise
  if gate in ('lead', 'cruise'):
    assert control.output_a_target < 0 and control.mpc.source != LongitudinalPlanSource.e2e
  else:
    assert control.accel_boost.total_boost == 0.
  if gate == 'stop':
    assert control.output_should_stop
