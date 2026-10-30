"""Compare real action consumers with the pre-port branch and merged upstream.

Reference function bodies are pinned source excerpts with the explicitly
reviewed PR #39182 calculation update applied below. Synthetic model predictions
exercise software consumption; no model
inference, vehicle process or CAN transport runs here.
"""
import ast
from collections.abc import Callable
from functools import partial
import hashlib
import inspect
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.modeld import modeld as stock
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.tests.test_blendful import frame


REFERENCE_HASHES = {
  'baseline_native': 'b93be2b60917a9685450522e16b7d1e980ba5308839f188e785e918a0476a149',
  'baseline_ordinary': '271a50690746eb6c52214864b846d91b7f1487721f9786d0ffdd9bf6c7d591f1',
  'upstream_ordinary': 'aa907fd0647d148f533439566e61bebcdd24d120bc842c38d9dcf23c89c72424',
}


def reference(name):
  source = (Path(__file__).with_name('fixtures') / f'blendful_{name}_action.txt').read_text()
  assert hashlib.sha256(source.encode()).hexdigest() == REFERENCE_HASHES[name]
  # Preserve the historical fixtures; compare Blendful ON/OFF at the same
  # direct-action calculation introduced by PR #39182 (406048c253b2).
  assert source.count('max(1.0, v_ego)') == 1
  source = source.replace('max(1.0, v_ego)', 'max(2.5, v_ego)')
  values = dict(vars(stock if name == 'baseline_native' else modeld), Callable=Callable)
  exec(compile(source, f'blendful_{name}_action', 'exec'), values)
  return values['get_action_from_model']


def model_state(generation=12, lateral=.1, longitudinal=.3):
  # Compare Blendful OFF at the same configured threshold, including the later
  # PR #39182 update. The new calculations have dedicated boundary coverage.
  return SimpleNamespace(constants=ModelConstants(), generation=generation, LAT_SMOOTH_SECONDS=lateral,
                         LONG_SMOOTH_SECONDS=longitudinal, MIN_LAT_CONTROL_SPEED=stock.MIN_LAT_CONTROL_SPEED,
                         PLANPLUS_CONTROL=.7, mlsim=generation is not None and generation >= 11)


def outputs(kind, speed, index):
  prediction = frame(center=.3, k=.002)
  plan = np.zeros((1, 33, 15), np.float32)
  times = np.asarray(ModelConstants.T_IDXS)
  plan[0, :, :3] = prediction['plan'][0]
  plan[0, :, Plan.VELOCITY.start] = speed + np.sin(times) * .2
  plan[0, :, Plan.ACCELERATION.start] = np.cos(times) * .2
  plan[0, :, Plan.T_FROM_CURRENT_EULER.stop - 1] = times * speed * .002
  plan[0, :, Plan.ORIENTATION_RATE.stop - 1] = speed * .002
  prediction['plan'] = plan
  if kind == 'planplus':
    prediction['planplus'] = plan * .1
  elif kind == 'action':
    prediction['action'] = np.array([[(-1. if index % 2 else 1.) * .002 * max(1., speed)**2,
                                     -.1 + index*.01, 0., 0.]], np.float32)
  return prediction


def bits(action):
  # Preserve signed zero and every serialized float bit, as well as stop intent.
  return struct.pack('<ff?', action.desiredCurvature, action.desiredAcceleration, action.shouldStop)


@pytest.mark.parametrize('kind', ['action', 'plan', 'planplus'])
@pytest.mark.parametrize('generation', [None, 9, 12])
@pytest.mark.parametrize('lateral', [0., .15])
def test_off_ordinary_actions_match_starting_branch_exactly(kind, generation, lateral):
  state = model_state(generation, lateral)
  baseline = reference('baseline_ordinary')
  actual_previous = expected_previous = log.ModelDataV2.Action(desiredCurvature=-.001, desiredAcceleration=.3)
  blendful = Blendful()
  for index, speed in enumerate((0., .3, .6, .60001, 1., 2.49999, 2.5, 2.50001, 4., 5., 10., 20., 30.)):
    output = outputs(kind, speed, index)
    callback = Mock(wraps=partial(blendful.update, output, speed, True, enabled=False))
    args = (.075 + index*.025, .575 + index*.01, speed)
    expected = baseline(state, output, expected_previous, *args)
    actual = modeld.ModelState.get_action_from_model(state, output, actual_previous, *args, blendful=callback)
    callback.assert_called_once()
    assert bits(actual) == bits(expected)
    assert blendful.curvature == 0.
    actual_previous, expected_previous = actual, expected


@pytest.mark.parametrize('sign', [-1., 1.])
def test_off_signed_zero_direct_action_preserves_baseline_bits(sign):
  state = model_state(generation=None, lateral=0.)
  output = {'action': np.array([[np.copysign(0., sign), 0.]], np.float32)}
  previous = log.ModelDataV2.Action()
  expected = reference('baseline_ordinary')(state, output, previous, .2, .575, 20.)
  actual = modeld.ModelState.get_action_from_model(state, output, previous, .2, .575, 20., blendful=lambda _: 0.)
  assert bits(actual) == bits(expected)


def test_off_native_navigation_action_matches_starting_branch_exactly():
  state = object.__new__(modeld.NavigationModelState)
  baseline = reference('baseline_native')
  previous = log.ModelDataV2.Action(desiredCurvature=.003, desiredAcceleration=-.8)
  blendful = Blendful()
  for index, speed in enumerate((0., .3, .6, .60001, 1., 2.49999, 2.5, 2.50001, 4., 5., 10., 20., 30.)):
    output = outputs('action', speed, index)
    callback = Mock(wraps=partial(blendful.update, output, speed, True, enabled=False))
    expected = baseline(output, previous, .15, .575, speed)
    actual = state.get_action_from_model(output, previous, .15, .575, speed, blendful=callback)
    callback.assert_called_once()
    assert bits(actual) == bits(expected)
    previous = actual


@pytest.mark.parametrize('kind', ['action', 'plan', 'planplus', 'native'])
def test_on_correction_matches_upstream_once_before_smoothing(kind):
  state = model_state(lateral=0. if kind == 'native' else .15)
  native = object.__new__(modeld.NavigationModelState) if kind == 'native' else None
  upstream = reference('upstream_ordinary')
  ours, theirs = Blendful(), Blendful()
  previous_actual = previous_expected = log.ModelDataV2.Action()
  nonzero = False
  for _ in range(100):
    speed = 20.
    output = outputs('action' if kind == 'native' else kind, speed, 0)
    callback = Mock(wraps=partial(ours.update, output, speed, True, enabled=True))
    expected = upstream(state, output, previous_expected, .2, .575, speed,
                        partial(theirs.update, output, speed, True, enabled=True))
    if native is None:
      actual = modeld.ModelState.get_action_from_model(state, output, previous_actual, .2, .575, speed, blendful=callback)
    else:
      actual = native.get_action_from_model(output, previous_actual, .2, .575, speed, blendful=callback)
    callback.assert_called_once()
    assert bits(actual) == bits(expected)
    assert ours.curvature == theirs.curvature
    nonzero |= ours.curvature != 0.
    previous_actual, previous_expected = actual, expected
  assert nonzero


def test_runtime_off_clears_existing_correction_without_changing_longitudinal_action():
  state = model_state()
  output = outputs('action', 20., 0)
  blendful = Blendful()
  previous = log.ModelDataV2.Action()
  for _ in range(100):
    previous = modeld.ModelState.get_action_from_model(state, output, previous, .2, .575, 20.,
                                                       blendful=partial(blendful.update, output, 20., True))
  assert blendful.curvature != 0.
  expected = reference('baseline_ordinary')(state, output, previous, .2, .575, 20.)
  actual = modeld.ModelState.get_action_from_model(state, output, previous, .2, .575, 20.,
                                                 blendful=partial(blendful.update, output, 20., True, enabled=False))
  assert bits(actual) == bits(expected)
  assert blendful.curvature == blendful.trust == 0.


def test_stock_default_runner_does_not_request_blendful_correction():
  tree = ast.parse(inspect.getsource(stock.main))
  calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'get_action_from_model']
  assert len(calls) == 1
  assert len(calls[0].args) == 5 and not calls[0].keywords
