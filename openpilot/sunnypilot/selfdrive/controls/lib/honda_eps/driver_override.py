"""Modified-EPS driver press detector from NRDR and JamesL787.

A same-direction press while assisting can be motor/column reaction torque.
Opposing driver torque and low-command presses retain immediate detection.
"""
import math

OVERRIDE_FADE_UP_SECONDS = 1.5
FIRMWARE_DRIVER_OVERRIDE_THRESHOLD = 2400.0


def apply_firmware_driver_threshold(CS, firmware_config):
  """Use James's raw press threshold only for an admitted firmware drive.

  card applies this before publishing CarState, so controls, events and the
  Honda actuator path see the same threshold. Native MVL fallback is untouched.
  """
  if firmware_config is not None:
    torque = float(CS.steeringTorque)
    if not math.isfinite(torque):
      raise ValueError("nonfinite Honda EPS raw driver torque")
    CS.steeringPressed = abs(torque) > FIRMWARE_DRIVER_OVERRIDE_THRESHOLD
  return CS


def eps_modified_steering_pressed(raw_pressed: bool, steering_torque: float, torque_command: float,
                                 filter_seconds: float, previous_pressed: bool, dt: float) -> tuple[float, bool]:
  if not all(math.isfinite(value) for value in (steering_torque, torque_command, filter_seconds, dt)) or dt <= 0:
    raise ValueError("invalid Honda EPS driver-override input")
  if not raw_pressed:
    return 0.0, False
  if previous_pressed or abs(torque_command) < 0.10 or steering_torque * torque_command < 0.0:
    return 1.0, True
  filter_seconds = min(1.0, filter_seconds + dt)
  return filter_seconds, filter_seconds >= 0.28


class FirmwareDriverDelivery:
  """James's actuator override, before MVL's existing CAN rate limits.

  Commands and the detector's 0.10 threshold stay in the source 3840-count
  normalized domain. The actuator detector runs on the current core command,
  independently of the core's detector on its previous command, as in James.
  The caller must clear the CAN torque request while ``override_active``.
  """
  def __init__(self, dt: float, *, fade_down_s: float = 0.):
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError("invalid Honda EPS delivery timestep")
    if not math.isfinite(fade_down_s) or not 0. <= fade_down_s <= 10.:
      raise ValueError("invalid Honda EPS override fade-down")
    self.dt = dt
    self.fade_down_s = float(fade_down_s)
    self.reset()

  def reset(self):
    self.ramp = 0.0
    self.pressed_seconds = 0.0
    self.override_active = False
    self.limited = False

  def update(self, command: float, raw_pressed: bool, steering_torque: float) -> float:
    self.pressed_seconds, self.override_active = eps_modified_steering_pressed(
      raw_pressed, steering_torque, command, self.pressed_seconds, self.override_active, self.dt,
    )
    if self.override_active:
      self.ramp = 0.0 if self.fade_down_s <= 0.0 else max(0.0, self.ramp - self.dt / self.fade_down_s)
    else:
      self.ramp = min(1.0, self.ramp + self.dt / OVERRIDE_FADE_UP_SECONDS)
    delivered = command * self.ramp
    # Feed this reduction back on the next core tick, as carOutput does for the
    # source actuator fade; otherwise the held I neither freezes nor bleeds.
    self.limited = abs(command - delivered) > 0.01
    return delivered
