import math

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.timing import (
  COMMAND_DELAY_BP, COMMAND_DELAY_V, DELAY_SCHEDULE_BP, DELAY_SCHEDULE_V,
  CommandDelay, command_delay, prediction_delay,
)


def test_prediction_schedule_endpoints_and_interpolation():
  for speed, expected in zip(DELAY_SCHEDULE_BP, DELAY_SCHEDULE_V, strict=True):
    assert prediction_delay(speed) == expected
  assert prediction_delay(-1) == prediction_delay(0) == .15
  assert prediction_delay(100) == .30
  assert prediction_delay(9.5) == pytest.approx(.09)


def test_separate_command_delay_schedule():
  for speed, expected in zip(COMMAND_DELAY_BP, COMMAND_DELAY_V, strict=True):
    assert command_delay(speed) == expected
  assert command_delay(12.5) == pytest.approx(.10)
  assert command_delay(0) == .175
  assert command_delay(40) == .025


def test_fractional_delay_and_warm_history():
  delay = CommandDelay(.01)
  assert delay.update(42, .175) == 42  # Before history is available, hold oldest.
  for value in np.arange(100, 140):
    actual = delay.update(float(value), .175)
  assert actual == pytest.approx(121.5)
  assert delay.update(140, 0) == 140
  assert len(delay.buf) == 32


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_timing_rejected(value):
  with pytest.raises(ValueError):
    prediction_delay(value)
  with pytest.raises(ValueError):
    command_delay(value)
  delay = CommandDelay(.01)
  delay.update(1, .1)
  with pytest.raises(ValueError):
    delay.update(value, .1)
  assert not delay.buf


@pytest.mark.parametrize("dt,max_delay", [(0, .3), (-1, .3), (math.nan, .3), (.01, -1), (.01, math.inf)])
def test_invalid_delay_configuration(dt, max_delay):
  with pytest.raises(ValueError):
    CommandDelay(dt, max_delay)


@pytest.mark.parametrize("delay", [-.01, .31, math.nan, math.inf])
def test_invalid_requested_delay(delay):
  with pytest.raises(ValueError):
    CommandDelay(.01).update(1, delay)
