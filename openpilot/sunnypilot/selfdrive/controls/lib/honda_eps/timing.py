"""NRDR's Insight prediction schedule and SunnyPilot command delay, in seconds."""
from collections import deque
import math

import numpy as np

DELAY_SCHEDULE_BP = (3.5, 7.0, 12.0, 20.0, 30.0)
DELAY_SCHEDULE_V = (0.15, 0.08, 0.10, 0.20, 0.30)
COMMAND_DELAY_BP = (10.0, 15.0)
COMMAND_DELAY_V = (0.175, 0.025)
MAX_COMMAND_DELAY = 0.30


def _finite_speed(speed: float) -> float:
  if not math.isfinite(speed):
    raise ValueError("Nonfinite Honda EPS speed")
  return max(float(speed), 0.0)


def prediction_delay(speed: float) -> float:
  return float(np.interp(_finite_speed(speed), DELAY_SCHEDULE_BP, DELAY_SCHEDULE_V))


def command_delay(speed: float) -> float:
  # Source Insight .15 -> 0 s, plus NRDR's .025 s compensation for source
  # action interpolation absent in these MVL pipelines. Not a second model lag.
  return float(np.interp(_finite_speed(speed), COMMAND_DELAY_BP, COMMAND_DELAY_V))


class CommandDelay:
  """Fractional-sample delay, filled while inactive and retained across reset."""
  def __init__(self, dt: float, max_delay: float = MAX_COMMAND_DELAY):
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_delay) or max_delay < 0:
      raise ValueError("Invalid Honda EPS command delay configuration")
    self.dt = dt
    self.max_delay = max_delay
    self.buf: deque[float] = deque(maxlen=int(math.ceil(max_delay / dt)) + 2)

  def update(self, value: float, delay: float) -> float:
    if not math.isfinite(value) or not math.isfinite(delay) or not 0 <= delay <= self.max_delay:
      self.buf.clear()
      raise ValueError("Invalid Honda EPS delayed command")
    self.buf.append(float(value))
    steps = delay / self.dt
    i = int(steps)
    fraction = steps - i
    n = len(self.buf)
    newer, older = self.buf[max(n - 1 - i, 0)], self.buf[max(n - 2 - i, 0)]
    return newer + fraction * (older - newer)
