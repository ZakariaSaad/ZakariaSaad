"""Signal-dependent curvature calculations; no model inference or vehicle I/O."""
from copy import copy
from functools import partial
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.modeld import modeld as stock
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.tests.test_blendful_runtime import REFERENCE_HASHES, bits, outputs
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
@pytest.mark.parametrize('turn_signal_active', [False, True])
def test_direct_action_floor_precedes_blendful_and_smoothing(path, sign, smoothing, turn_signal_active):
  state, get_action = action_path(path)
  smoothing_owner = stock if state is None or path == 'bigv2' else state
  output = {'action': np.array([[sign * 1.25, -.2, 0., 0.]], np.float32)}
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  correction = .0001
  threshold, floor = (.3, 1.) if turn_signal_active else (.6, 2.5)
  with patch.object(smoothing_owner, 'LAT_SMOOTH_SECONDS', smoothing):
    for speed in (.3, np.nextafter(.3, np.inf), .5, .6, np.nextafter(.6, np.inf),
                  np.nextafter(1., 0.), 1., np.nextafter(1., np.inf), np.nextafter(2.5, 0.), 2.5,
                  np.nextafter(2.5, np.inf), 3., .3):
      callback = Mock(return_value=correction)
      actual = get_action(output, previous, .175, .575, speed, blendful=callback, turn_signal_active=turn_signal_active)
      raw_curvature = output['action'][0, 0] / max(floor, speed)**2
      callback.assert_called_once_with(raw_curvature)
      expected = (stock.smooth_value(raw_curvature + correction, previous.desiredCurvature, smoothing)
                  if speed > threshold else previous.desiredCurvature)
      assert actual.desiredCurvature == float(np.float32(expected))
      previous = actual


@pytest.mark.parametrize('path', ['stock_plan', 'ordinary_plan', 'ordinary_planplus'])
@pytest.mark.parametrize('turn_signal_active', [False, True])
def test_plan_curvature_is_not_divided_by_direct_action_speed_floor(path, turn_signal_active):
  state, get_action = action_path(path)
  if state is not None:
    state.PLANPLUS_CONTROL = .7
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  kind = 'planplus' if path.endswith('planplus') else 'plan'
  threshold = .3 if turn_signal_active else .6
  for speed in (np.nextafter(threshold, np.inf), 1., 2.49999, 2.5, 3.):
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
    actual = get_action(output, previous, .175, .575, speed, blendful=callback, turn_signal_active=turn_signal_active)
    callback.assert_called_once_with(raw_curvature)
    assert actual.desiredCurvature == float(np.float32(raw_curvature))


def original_action(path, state):
  """Execute the immutable pre-39174/39182 excerpt with its original threshold."""
  name = 'baseline_native' if state is None or path == 'bigv2' else 'baseline_ordinary'
  source = (Path(__file__).with_name('fixtures') / f'blendful_{name}_action.txt').read_text()
  assert hashlib.sha256(source.encode()).hexdigest() == REFERENCE_HASHES[name]
  namespace = dict(vars(stock if name == 'baseline_native' else modeld), MIN_LAT_CONTROL_SPEED=.3)
  exec(compile(source, f'original_{name}_action', 'exec'), namespace)
  reference = namespace['get_action_from_model']
  if name == 'baseline_native':
    return reference
  reference_state = copy(state)
  reference_state.MIN_LAT_CONTROL_SPEED = .3
  return partial(reference, reference_state)


@pytest.mark.parametrize('path', ['stock_action', 'stock_plan', 'bigv2', 'light_speed',
                                 'ordinary_plan', 'ordinary_planplus', 'legacy', 'unclassified'])
def test_active_signal_restores_original_action_values_and_generation_gate(path):
  state, get_action = action_path(path)
  if state is not None:
    if path in ('legacy', 'unclassified'):
      state.generation = 9 if path == 'legacy' else None
    elif path == 'ordinary_planplus':
      state.PLANPLUS_CONTROL = .7
  original = original_action(path, state)
  kind = 'planplus' if path.endswith('planplus') else 'plan' if path.endswith('plan') else 'action'
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  for index, speed in enumerate((0., .3, np.nextafter(.3, np.inf), .5, .6, np.nextafter(.6, np.inf),
                                np.nextafter(1., 0.), 1., np.nextafter(1., np.inf), 2.5, 3., .3)):
    output = outputs(kind, speed, index)
    expected = original(output, previous, .175, .575, speed)
    actual = get_action(output, previous, .175, .575, speed, turn_signal_active=True)
    assert bits(actual) == bits(expected)
    previous = actual


@pytest.mark.parametrize('path', ['stock_action', 'bigv2', 'light_speed', 'ordinary_action'])
@pytest.mark.parametrize('speed', [.5, 1.2, 2.5])
def test_signal_activation_and_cancellation_preserve_history_without_extra_smoothing(path, speed):
  state, get_action = action_path(path)
  output = {'action': np.array([[.00625, -.2, 0., 0.]], np.float32)}
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  if state is not None:
    state.prev_desire[:] = .25
    original_desire = state.prev_desire.copy()
    original_queues = dict(state.adapter.input_queues)
    original_action_t = state.numpy_inputs['action_t'].copy()
  actions = []
  for active in (False, False, True, True, False, False, True, False):
    actual = get_action(output, previous, .175, .575, speed, turn_signal_active=active)
    ordinary = get_action(output, previous, .175, .575, speed)
    threshold, floor = (.3, 1.) if active else (.6, 2.5)
    expected = output['action'][0, 0] / max(floor, speed)**2 if speed > threshold else previous.desiredCurvature
    assert actual.desiredCurvature == float(np.float32(expected))
    assert actual.desiredAcceleration == ordinary.desiredAcceleration
    assert actual.shouldStop == ordinary.shouldStop
    actions.append(actual.desiredCurvature)
    previous = actual
  if speed == .5:
    # Cancelling while below the default threshold holds the last signal-active
    # target; it neither zeros it nor adds a new decay/smoothing policy.
    assert actions[0] != actions[2]
    assert actions[2] == actions[4] == actions[7]
  elif speed == 1.2:
    assert actions[0] != actions[2] and actions[4] == actions[0]
  else:
    assert len(set(actions)) == 1
  if state is not None:
    np.testing.assert_array_equal(state.prev_desire, original_desire)
    np.testing.assert_array_equal(state.numpy_inputs['action_t'], original_action_t)
    assert all(state.adapter.input_queues[name] is queue for name, queue in original_queues.items())
