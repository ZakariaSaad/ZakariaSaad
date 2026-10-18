"""Controller-local Insight geometry from JamesL787 bfcbbf24/f3a6a124.

The optional yaw fit changes only this controller's desired wheel angle. It does
not change VehicleModel, paramsd learning, model input, or other Honda platforms.
"""
import math
from functools import lru_cache

import numpy as np


def _build_vgr_position_inverse(raw_x, raw_y, raw_units_per_degree=10.0, max_raw_step=20):
  """Preserve the source firmware's interpolate-divisor-then-divide operation."""
  linear_bp, angle_bp = [], []
  for raw0, raw1, divisor0, divisor1 in zip(raw_x, raw_x[1:], raw_y, raw_y[1:], strict=False):
    steps = max(1, math.ceil((raw1 - raw0) / max_raw_step))
    for step in range(steps):
      fraction = step / steps
      raw = raw0 + (raw1 - raw0) * fraction
      divisor = divisor0 + (divisor1 - divisor0) * fraction
      linear_bp.append(raw * (1 << 14) / raw_y[0] / raw_units_per_degree)
      angle_bp.append(raw * (1 << 14) / divisor / raw_units_per_degree)
  linear_bp.append(raw_x[-1] * (1 << 14) / raw_y[0] / raw_units_per_degree)
  angle_bp.append(raw_x[-1] * (1 << 14) / raw_y[-1] / raw_units_per_degree)
  return tuple(linear_bp), tuple(angle_bp)


# NRDR PR #22 uses this exact Clarity table for its reference calibration's R6.
_CLARITY_POSITION_X = (0, 40, 80, 119, 158, 198, 237, 277, 317, 398,
                       604, 820, 1047, 1164, 1210, 1257, 1305, 1352, 1398, 1447,
                       1493, 1540, 1588, 1634, 1989, 2344, 2700, 3056, 3413, 5020)
_CLARITY_POSITION_Y = (16384, 16174, 16173, 16247, 16179, 16220, 16180, 16209, 16231, 16303,
                       16506, 16812, 17177, 17352, 17411, 17473, 17538, 17600, 17644, 17705,
                       17749, 17792, 17843, 17885, 18130, 18303, 18439, 18547, 18645, 18952)
NRDR_CLARITY_VGR_LINEAR_BP, NRDR_CLARITY_VGR_ANGLE_BP = _build_vgr_position_inverse(_CLARITY_POSITION_X, _CLARITY_POSITION_Y)

# Insight primary A table at flash X=0x11374 / Y=0x11338. The adjacent B table
# divides the rate input and must not be used as the position curve.
INSIGHT_POSITION_X = (0, 43, 88, 130, 175, 219, 263, 306, 351, 441,
                      671, 914, 1166, 1297, 1348, 1401, 1454, 1507, 1559, 1613,
                      1664, 1718, 1771, 1823, 2217, 2610, 3005, 3402, 3798, 5515)
INSIGHT_POSITION_Y = (17613, 17613, 18022, 17886, 17971, 17981, 17988, 17964, 18022, 18084,
                      18269, 18631, 19026, 19226, 19313, 19387, 19442, 19523, 19583, 19636,
                      19692, 19745, 19798, 19839, 20113, 20317, 20474, 20593, 20702, 20989)
INSIGHT_VGR_LINEAR_BP, INSIGHT_VGR_ANGLE_BP = _build_vgr_position_inverse(INSIGHT_POSITION_X, INSIGHT_POSITION_Y)
INSIGHT_RACK_RATIO_BP = (4.6, 13.8, 35.8, 55.1, 83.3, 122.0, 174.0, 220.7)
INSIGHT_RACK_RATIO_V = (17.86, 17.86, 17.86, 17.40, 17.28, 17.28, 17.21, 17.17)
INSIGHT_SLIP_FACTOR = -0.00066


class InsightRackMap:
  """Exact James Insight angle/curvature conversion, including its grid inversion.

  The fit comes from route 0000001e, about seven minutes of turns. The source
  reports thin coverage past 50 degrees and an outlying fast small-angle cell.
  """
  def __init__(self, wheelbase):
    self.wheelbase = float(wheelbase)
    if not math.isfinite(self.wheelbase) or self.wheelbase <= 0:
      raise ValueError("rack geometry requires a finite positive wheelbase")
    self.angle_grid = np.unique(np.r_[INSIGHT_VGR_ANGLE_BP, np.linspace(0.0, INSIGHT_VGR_ANGLE_BP[-1], 1001)])
    linear = np.radians(np.interp(self.angle_grid, INSIGHT_VGR_ANGLE_BP, INSIGHT_VGR_LINEAR_BP))
    self.path_grid = linear / np.interp(self.angle_grid, INSIGHT_RACK_RATIO_BP, INSIGHT_RACK_RATIO_V)
    if not np.all(np.diff(self.path_grid) > 0):
      raise ValueError("rack map must be monotonic to invert")

  def angle_from_curvature(self, curvature, v_ego, roll):
    path = self.wheelbase * (-curvature * (1.0 - INSIGHT_SLIP_FACTOR * v_ego ** 2) - 9.81 * INSIGHT_SLIP_FACTOR * roll)
    return math.copysign(float(np.interp(abs(path), self.path_grid, self.angle_grid)), path)

  def curvature_from_angle(self, angle_deg, v_ego, roll):
    path = math.copysign(float(np.interp(abs(angle_deg), self.angle_grid, self.path_grid)), angle_deg)
    return -(path / self.wheelbase + 9.81 * INSIGHT_SLIP_FACTOR * roll) / (1.0 - INSIGHT_SLIP_FACTOR * v_ego ** 2)


@lru_cache(maxsize=4)
def _rack_map(wheelbase):
  return InsightRackMap(wheelbase)


def desired_angle_no_offset(VM, v_ego, roll, desired_curvature, mode="mvl"):
  """Select latched geometry without mutating the shared VehicleModel."""
  if not all(math.isfinite(float(value)) for value in (v_ego, roll, desired_curvature)):
    raise ValueError("geometry requires finite inputs")
  # Honda's speed Kalman filter can briefly undershoot zero while stopping.
  v_ego = max(float(v_ego), 0.0)
  if mode == "mvl":
    angle = math.degrees(VM.get_steer_from_curvature(-desired_curvature, v_ego, roll))
  elif mode == "james":
    angle = _rack_map(float(VM.l)).angle_from_curvature(desired_curvature, v_ego, roll)
  else:
    raise ValueError(f"unsupported Honda EPS geometry: {mode}")
  if not math.isfinite(angle):
    raise ValueError("nonfinite desired steering angle")
  return angle


def curvature_from_angle(VM, v_ego, roll, angle_no_offset, mode="mvl"):
  """Use matching measured-curvature geometry for controller initialization."""
  if not all(math.isfinite(float(value)) for value in (v_ego, roll, angle_no_offset)):
    raise ValueError("geometry requires finite inputs")
  v_ego = max(float(v_ego), 0.0)
  if mode == "mvl":
    curvature = -VM.calc_curvature(math.radians(angle_no_offset), v_ego, roll)
  elif mode == "james":
    curvature = _rack_map(float(VM.l)).curvature_from_angle(angle_no_offset, v_ego, roll)
  else:
    raise ValueError(f"unsupported Honda EPS geometry: {mode}")
  if not math.isfinite(curvature):
    raise ValueError("nonfinite measured steering curvature")
  return curvature
