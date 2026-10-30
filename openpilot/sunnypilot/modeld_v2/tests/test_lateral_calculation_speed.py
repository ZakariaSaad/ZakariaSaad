"""PR #39182 curvature calculation boundaries; no model inference or vehicle I/O."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.modeld import modeld as stock
from openpilot.sunnypilot.modeld_v2.tests.test_blendful_runtime import bits, outputs
from openpilot.sunnypilot.modeld_v2.tests.test_light_speed_state import make_model
from openpilot.sunnypilot.modeld_v2.tests.test_official_bigv2_runtime import native_state


def action_path(path):
  if path.startswith('stock'):
    return None, stock.get_action_from_model
  state = native_state() if path == 'bigv2' else make_model() if path == 'light_speed' else make_model(ref='ordinary-test')
  return state, state.get_action_from_model


@pytest.mark.parametrize('path', ['stock_action', 'stock_plan', 'bigv2', 'light_speed',
                                 'ordinary_plan', 'ordinary_planplus', 'legacy', 'unclassified'])
@pytest.mark.parametrize('correction', [0., .0001])
def test_curvature_hold_boundary_preserves_longitudinal_and_existing_generation_gate(path, correction):
  # Real constructors with inert artifact readers exercise initialized constants
  # and native subclass dispatch. Light Speed's v28 catalog generation is 12.
  state, get_action = action_path(path)
  if state is not None:
    if path == 'legacy':
      state.generation = 9
    elif path == 'unclassified':
      state.generation = None
    elif path == 'ordinary_planplus':
      state.PLANPLUS_CONTROL = .7
  assert stock.MIN_LAT_CONTROL_SPEED == .6
  if state is not None:
    assert state.MIN_LAT_CONTROL_SPEED == .6

  kind = 'planplus' if path.endswith('planplus') else 'plan' if path.endswith('plan') else 'action'
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  holds = path not in ('legacy', 'unclassified')
  changed_in_interval = False
  for index, speed in enumerate((0., .3, .30001, .59999, .6, np.nextafter(.6, np.inf), 1., 2.5, 20., .6, .5, 2.)):
    output = outputs(kind, speed, index)
    args = (output, previous, .175, .575, speed)
    callback = Mock(return_value=correction)
    actual = get_action(*args, blendful=callback)
    callback.assert_called_once()
    # Evaluate the same function/input/previous action with the former threshold.
    # This isolates the requested change from smoothing, stop, and Blendful math.
    threshold_owner = stock if state is None or path == 'bigv2' else state
    with patch.object(threshold_owner, 'MIN_LAT_CONTROL_SPEED', .3):
      baseline = get_action(*args, blendful=lambda _: correction)
    assert actual.desiredAcceleration == baseline.desiredAcceleration
    assert actual.shouldStop == baseline.shouldStop
    if holds and speed <= .6:
      assert actual.desiredCurvature == previous.desiredCurvature
      assert actual.desiredCurvature != 0.  # Hold a real command, not disengagement/zeroing.
      if speed > .3:
        changed_in_interval |= actual.desiredCurvature != baseline.desiredCurvature
    else:
      assert bits(actual) == bits(baseline)
    previous = actual
  assert changed_in_interval == holds


@pytest.mark.parametrize('path', ['stock_action', 'bigv2', 'light_speed', 'ordinary_action'])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('smoothing', [0., .15])
def test_direct_action_floor_precedes_blendful_and_smoothing(path, sign, smoothing):
  state, get_action = action_path(path)
  smoothing_owner = stock if state is None or path == 'bigv2' else state
  output = {'action': np.array([[sign * 1.25, -.2, 0., 0.]], np.float32)}
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  correction = .0001
  with patch.object(smoothing_owner, 'LAT_SMOOTH_SECONDS', smoothing):
    for speed in (.5, .6, np.nextafter(.6, np.inf), 1., np.nextafter(2.5, 0.), 2.5, np.nextafter(2.5, np.inf), 3., .6):
      callback = Mock(return_value=correction)
      actual = get_action(output, previous, .175, .575, speed, blendful=callback)
      raw_curvature = output['action'][0, 0] / max(2.5, speed)**2
      callback.assert_called_once_with(raw_curvature)
      expected = (stock.smooth_value(raw_curvature + correction, previous.desiredCurvature, smoothing)
                  if speed > .6 else previous.desiredCurvature)
      assert actual.desiredCurvature == float(np.float32(expected))
      previous = actual


@pytest.mark.parametrize('path', ['stock_plan', 'ordinary_plan', 'ordinary_planplus'])
def test_plan_curvature_is_not_divided_by_direct_action_speed_floor(path):
  from openpilot.sunnypilot.modeld_v2 import modeld

  state, get_action = action_path(path)
  if state is not None:
    state.PLANPLUS_CONTROL = .7
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  kind = 'planplus' if path.endswith('planplus') else 'plan'
  for speed in (np.nextafter(.6, np.inf), 1., 2.49999, 2.5, 3.):
    output = outputs(kind, speed, 0)
    plan = output['plan'][0]
    if state is None:
      raw_curvature = stock.get_curvature_from_plan(plan[:, stock.Plan.T_FROM_CURRENT_EULER][:, 2],
                                                    plan[:, stock.Plan.ORIENTATION_RATE][:, 2],
                                                    stock.ModelConstants.T_IDXS, speed, .175)
    else:
      curvature_plan = plan + (state.PLANPLUS_CONTROL - 1.) * output['planplus'][0] if kind == 'planplus' else plan
      raw_curvature = modeld.get_curvature_from_output(output, curvature_plan, speed, .175, state.mlsim)
    callback = Mock(return_value=0.)
    actual = get_action(output, previous, .175, .575, speed, blendful=callback)
    callback.assert_called_once_with(raw_curvature)
    assert actual.desiredCurvature == float(np.float32(raw_curvature))
