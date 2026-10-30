"""PR #39174 curvature calculation boundary; no model inference or vehicle I/O."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.modeld import modeld as stock
from openpilot.sunnypilot.modeld_v2.tests.test_blendful_runtime import bits, outputs
from openpilot.sunnypilot.modeld_v2.tests.test_light_speed_state import make_model
from openpilot.sunnypilot.modeld_v2.tests.test_official_bigv2_runtime import native_state


@pytest.mark.parametrize('path', ['stock_action', 'stock_plan', 'bigv2', 'light_speed',
                                 'ordinary_plan', 'ordinary_planplus', 'legacy', 'unclassified'])
@pytest.mark.parametrize('correction', [0., .0001])
def test_curvature_hold_boundary_preserves_longitudinal_and_existing_generation_gate(path, correction):
  # Real constructors with inert artifact readers exercise initialized constants
  # and native subclass dispatch. Light Speed's v28 catalog generation is 12.
  if path.startswith('stock'):
    state, get_action = None, stock.get_action_from_model
  elif path == 'bigv2':
    state = native_state()
    get_action = state.get_action_from_model
  else:
    state = make_model() if path == 'light_speed' else make_model(ref='ordinary-test')
    if path == 'legacy':
      state.generation = 9
    elif path == 'unclassified':
      state.generation = None
    elif path == 'ordinary_planplus':
      state.PLANPLUS_CONTROL = .7
    get_action = state.get_action_from_model
  assert stock.MIN_LAT_CONTROL_SPEED == 1.6
  if state is not None:
    assert state.MIN_LAT_CONTROL_SPEED == 1.6

  kind = 'planplus' if path.endswith('planplus') else 'plan' if path.endswith('plan') else 'action'
  previous = log.ModelDataV2.Action(desiredCurvature=-.012, desiredAcceleration=-.7)
  holds = path not in ('legacy', 'unclassified')
  changed_in_interval = False
  for index, speed in enumerate((0., .3, .30001, 1., 1.59999, 1.6, np.nextafter(1.6, np.inf), 2., 20., 1.6, .8, 2.)):
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
    if holds and speed <= 1.6:
      assert actual.desiredCurvature == previous.desiredCurvature
      assert actual.desiredCurvature != 0.  # Hold a real command, not disengagement/zeroing.
      if speed > .3:
        changed_in_interval |= actual.desiredCurvature != baseline.desiredCurvature
    else:
      assert bits(actual) == bits(baseline)
    previous = actual
  assert changed_in_interval == holds
