# ruff: noqa
"""Unmodified JamesL787 actuator-method excerpts, executed as parity fixtures.

Source: ns-bosch-radar-testing bfcbbf24bc857f9dbf93af52b7bb72852f4540bb
File: opendbc_repo/opendbc/car/honda/carcontroller.py
Original file SHA-256: 8a2ddc5a5b1470111319dcf3d796d91d9862ccf115aa6d64d1119bb0f6c80243
Only imports and class scaffolding below are supplied by this fixture.
"""
import numpy as np
from opendbc.car import DT_CTRL, rate_limit

def get_eps_modified_steering_pressed(
  raw_pressed: bool, steering_torque: float, torque_cmd: float, filter_s: float, previous_pressed: bool
) -> tuple[float, bool]:
  if not raw_pressed:
    return 0.0, False

  torque_product = float(steering_torque) * float(torque_cmd)
  torque_cmd_abs = abs(float(torque_cmd))
  if previous_pressed or torque_cmd_abs < 0.10 or torque_product < 0.0:
    return 1.0, True

  filter_s = min(1.0, filter_s + DT_CTRL)
  return filter_s, filter_s >= 0.28


class JamesDriverDelivery:
  def _filtered_steering_pressed(self, CS, torque_cmd: float) -> bool:
    # Every modified-EPS Honda -- Clarity, Civic and Civic Bosch -- shares this detector:
    # instant latch on opposing or near-zero-command driver torque, 0.28 s on same-direction
    # torque, instant release. Civic Bosch previously had a graded variant of its own with a
    # 1.60 s same-direction trip and a decaying release; it was removed rather than left
    # unrouted, so there is no second override path to fall out of sync.
    self.steering_pressed_filter_s, steering_pressed = get_eps_modified_steering_pressed(
      bool(CS.out.steeringPressed),
      float(getattr(CS.out, "steeringTorque", 0.0)),
      float(torque_cmd),
      self.steering_pressed_filter_s,
      self.steering_pressed_robust_prev,
    )
    self.steering_pressed_robust_prev = steering_pressed
    return steering_pressed

  def _update_steering_torque(self, CC, CS, live):
    torque_cmd = float(CC.actuators.torque) if CC.latActive else 0.0
    steering_pressed = False
    below_min_steer_speed = CS.out.vEgo < live["min_steer_speed"]

    # INVARIANT: every stage below may only reduce the command because the car genuinely cannot
    # deliver it -- below the rack's minimum steer speed, cut for a driver override, or held back
    # by the steer-delta limiter. controlsd turns any difference between CC.actuators.torque and
    # the value returned here into steer_limited_by_safety, which freezes the lateral integrator,
    # so comfort shaping must NOT live here. Smoothing belongs upstream in the lateral controller,
    # where it is part of the command rather than a silent deviation from it.
    if below_min_steer_speed:
      torque_cmd = 0.0

    if CC.latActive:
      # Match the modified-EPS PID's override policy. A raw threshold crossing can chatter
      # around the driver-torque boundary; letting the actuator fade while the PID still sees
      # a different override state creates false safety limiting and integrator freeze churn.
      steering_pressed = self._filtered_steering_pressed(CS, torque_cmd)

      if not self.lat_active_prev:
        self.override_ramp = 0.0

      if steering_pressed:
        fade_down_s = live["override_fade_down_s"]
        if fade_down_s <= 0.0:
          self.override_ramp = live["override_torque_scale"]
        else:
          self.override_ramp = max(live["override_torque_scale"], self.override_ramp - DT_CTRL / fade_down_s)
      else:
        fade_up_s = live["override_fade_up_s"]
        self.override_ramp = 1.0 if fade_up_s <= 0.0 else min(1.0, self.override_ramp + DT_CTRL / fade_up_s)

      torque_cmd *= self.override_ramp

      # The speed-banded torque-output LPF now runs in the lateral controller (LatControlPID, or
      # LatControlHondaEps on the cars that run it) after all modified-EPS shaping. It uses the
      # HondaTorqueOutputLowPassFilter toggle and
      # HondaTorqueOutputLpfTau{LowSpeed,Standard,Highway} taus, so actuators.torque already
      # contains the command that this controller will deliver to the rack.
      #
      # Filtering here made this controller's own shaping indistinguishable from safety limiting.
      # controlsd derives steer_limited_by_safety from
      #   abs(CC.actuators.torque - carOutput.actuatorsOutput.torque) > 1e-2
      # and a first-order lag holds a steady-state gap of tau * slew, so at tau = 0.1 s that
      # tripped for any command moving faster than 0.1 authority/s -- under 6 deg/s of angle-error
      # change at the Clarity's low-speed kp -- and froze the lateral integrator through every
      # curve. Smoothing upstream of actuators.torque leaves that comparison meaning only what it
      # says: genuine safety clipping.
      #
      # There is intentionally no second filter here: keeping this stage unfiltered ensures the
      # controlsd CC-versus-carOutput comparison only reports actual safety limiting.

    else:
      self.override_ramp = 0.0
      self.steering_pressed_filter_s = 0.0
      self.steering_pressed_robust_prev = False

    # Opt-in slew limit on the delivered command. Unlike a low-pass filter this genuinely
    # withholds authority for as long as it is saturated, so reporting it as limiting -- and
    # freezing the integrator against it -- is correct.
    if live["steer_delta_limiter_enabled"]:
      limited_torque = rate_limit(torque_cmd, self.last_torque, -live["steer_delta_down"] * DT_CTRL, live["steer_delta_up"] * DT_CTRL)
    else:
      limited_torque = torque_cmd

    self.last_torque = limited_torque
    self.lat_active_prev = CC.latActive

    lkas_active = CC.latActive and (live["driver_assist_during_override"] or not steering_pressed) and not below_min_steer_speed
    return limited_torque, lkas_active, steering_pressed
