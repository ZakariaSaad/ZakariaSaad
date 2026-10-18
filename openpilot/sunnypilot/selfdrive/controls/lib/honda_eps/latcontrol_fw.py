"""SunnyPilot shell for the corrected Insight EPS controller.

NRDR's SunnyPilot interface surrounds James's source command-domain core. Convert
its complete 3840-count output once, retaining MVL's 4096-count transport and all
existing CAN/safety limits. Selection and geometry are latched by controlsd.
"""
import math

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.core import (
  HondaEpsFirmwareFeedforward, HondaEpsLateralCore, INSIGHT_I_SCALE, INSIGHT_P_SCALE,
  INSIGHT_TXM_A040, INSIGHT_TXM_A040_CANONICAL, INSIGHT_TXM_EPS_LOAD, validate_calibration,
)
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.driver_override import (
  FirmwareDriverDelivery, eps_modified_steering_pressed,
)
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.geometry import desired_angle_no_offset
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.response import resolve_response_profile
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import is_supported_vehicle
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.timing import CommandDelay, command_delay

GAIN_BP = (0.0, 11.175, 11.176, 22.352)
KP = (0.018, 0.024, 0.048, 0.060)
KI = (0.006, 0.008, 0.016, 0.020)
ANGLE_RATE_LIMIT = 300.0


class LatControlFirmware(LatControl):
  owns_output_filter = True
  uses_firmware_controller = True

  def __init__(self, CP, CP_SP, CI, dt, *, geometry_mode="mvl", response_profile="source"):
    if not is_supported_vehicle(CP, CP_SP):
      raise ValueError("firmware control requires the supported modified Insight EPS family")
    if geometry_mode not in ("mvl", "james"):
      raise ValueError("unsupported Honda EPS geometry mode")
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError("invalid Honda EPS controller timestep")
    validate_calibration()
    validate_calibration(INSIGHT_TXM_A040_CANONICAL)
    super().__init__(CP, CP_SP, CI, dt)
    self.geometry_mode = geometry_mode
    self.response_profile = resolve_response_profile(response_profile)
    self.calibration = INSIGHT_TXM_A040
    self.output_scale = INSIGHT_TXM_A040_CANONICAL.e4_per_output / self.calibration.e4_per_output
    # The original core's +/-1 rail is +/-3840 E4. No extra command range is
    # introduced by SunnyPilot's larger transport normalization.
    self.steer_max = min(self.steer_max, self.output_scale)
    self.core = HondaEpsLateralCore(
      GAIN_BP, KP, GAIN_BP, KI, dt,
      ff=HondaEpsFirmwareFeedforward(dt, cal=INSIGHT_TXM_A040_CANONICAL, load=INSIGHT_TXM_EPS_LOAD),
      p_scale=INSIGHT_P_SCALE, i_scale=INSIGHT_I_SCALE,
    )
    self.core.output_lpf_tau = self.response_profile.output_lpf_tau
    self.cmd_delay = CommandDelay(dt)
    self.prev_angle = 0.0
    self.pressed_duration = 0.0
    self.pressed_prev = False
    self.delivery = FirmwareDriverDelivery(dt, fade_down_s=self.response_profile.override_fade_down_s)

  @property
  def driver_override_active(self):
    return self.delivery.override_active

  def update_model_v2(self, model_v2):
    # MVL's model/lane-change path already shapes desired curvature. The NRDR
    # owner-specific optimized-lane-change bypass is deliberately not enabled.
    pass

  def reset(self):
    super().reset()
    self.core.reset()
    self.pressed_duration = 0.0
    self.pressed_prev = False
    self.delivery.reset()
    # Preserve the source behavior: inactive updates warm command history and
    # the previous desired angle, so engaging does not replay a startup step.

  def _invalidate(self):
    self.reset()
    self.cmd_delay.buf.clear()
    self.prev_angle = 0.0

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature,
             calibrated_pose, curvature_limited, lat_delay):
    try:
      values = (CS.steeringAngleDeg, CS.steeringRateDeg, CS.steeringTorque, CS.vEgo,
                params.roll, params.angleOffsetDeg, desired_curvature, lat_delay, self.prev_angle)
      if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("nonfinite Honda EPS controller input")
      return self._update(active, CS, VM, params, steer_limited_by_safety, desired_curvature,
                          calibrated_pose, curvature_limited, lat_delay)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
      # Raising stops controlsd from publishing an actuator command for invalid
      # data. Clear every history/filter before a subsequent restart or update.
      self._invalidate()
      raise ValueError("invalid Honda EPS controller input or result") from exc

  def _update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature,
              calibrated_pose, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)
    delay = command_delay(CS.vEgo)
    delayed_curvature = self.cmd_delay.update(desired_curvature, delay)
    desired = desired_angle_no_offset(VM, CS.vEgo, params.roll, delayed_curvature, self.geometry_mode)
    if not math.isfinite(desired):
      raise ValueError("nonfinite Honda EPS geometry result")
    if active:
      step = ANGLE_RATE_LIMIT * self.dt
      desired = float(np.clip(desired, self.prev_angle - step, self.prev_angle + step))
    self.prev_angle = desired
    angle = desired + params.angleOffsetDeg
    if not math.isfinite(angle):
      raise ValueError("nonfinite Honda EPS desired angle")
    pid_log.steeringAngleDesiredDeg = angle
    pid_log.angleError = angle - CS.steeringAngleDeg
    if not active:
      self.reset()
      output = 0.0
    else:
      self.pressed_duration, pressed = eps_modified_steering_pressed(
        bool(CS.steeringPressed), float(CS.steeringTorque), self.core.output,
        self.pressed_duration, self.pressed_prev, self.dt,
      )
      self.pressed_prev = pressed
      steer_limited = steer_limited_by_safety or self.delivery.limited
      source_output = self.core.update(
        desired, params.angleOffsetDeg, CS.steeringAngleDeg, CS.vEgo, params.roll,
        pressed, steer_limited, optimized_lane_change=False,
      )
      delivered = self.delivery.update(source_output, bool(CS.steeringPressed), float(CS.steeringTorque))
      output = float(np.clip(delivered * self.output_scale, -self.steer_max, self.steer_max))
      pid_log.p = float(self.core.pid.p * self.output_scale)
      pid_log.i = float(self.core.pid.i * self.output_scale)
      pid_log.f = float(self.core.pid.f * self.output_scale)
      pid_log.saturated = bool(self._check_saturation(
        self.steer_max - abs(output) < 1e-3, CS, steer_limited or self.delivery.limited, curvature_limited))
    ff = self.core.ff
    pid_log.epsFfActive = bool(active)
    pid_log.epsFfWeight = float(self.core.ff_weight)
    pid_log.epsFfFeedforward = float(ff.output * self.output_scale)
    pid_log.epsFfR5 = float(ff.r5)
    pid_log.epsFfLoad = float(ff.load)
    pid_log.epsFfDesiredRate = float(ff.rate)
    pid_log.commandDelay = float(delay)
    pid_log.active = bool(active)
    pid_log.output = output
    if not all(math.isfinite(value) for value in (output, angle, ff.output, ff.r5, ff.load, ff.rate)):
      raise ValueError("nonfinite Honda EPS actuator result")
    return output, angle, pid_log
