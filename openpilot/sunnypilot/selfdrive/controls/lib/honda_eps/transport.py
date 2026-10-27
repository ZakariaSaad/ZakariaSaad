"""Drive-latched Insight transport adapter for sp-steer's Honda interface.

The firmware controller already owns the EPS feedforward and 3840→4096
conversion. sp-steer's learned inverse plant belongs only to the ordinary
controller. Use its interface here to retain the original final rate limit,
without stacking feedforward, learning its response, or changing other Hondas.
"""
import math

from opendbc.car import DT_CTRL, rate_limit
from opendbc.car.honda.values import CarControllerParams
from opendbc.sunnypilot.car.honda.eps_mod import get_eps_mod_torque_lookup

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.driver_override import FIRMWARE_DRIVER_OVERRIDE_THRESHOLD


class FirmwareTransport:
  wire_limit = 1.0
  gain_now = 0.0
  applied_correction = 0.0
  ceiling = 1.0
  learning = False

  def __init__(self, controller_params):
    self.delta_down = controller_params.STEER_DELTA_DOWN * DT_CTRL
    self.delta_up = controller_params.STEER_DELTA_UP * DT_CTRL

  def update(self, request, last, active, steer_active, pressed, speed, desired, actual, angle, rate):
    if not math.isfinite(request):
      raise ValueError("nonfinite Honda EPS transport request")
    return request

  def limit(self, request, last, speed):
    if not all(math.isfinite(value) for value in (request, last, speed)):
      raise ValueError("nonfinite Honda EPS transport input")
    return rate_limit(request, last, -self.delta_down, self.delta_up)

  def reported_torque(self, request, corrected_request, wire, wire_limit):
    return wire

  def learned_values(self):
    return {}


class FirmwareDriverThreshold:
  """Do not update or overwrite the ordinary controller's learned threshold."""
  def update(self, torque, base, active, timestamp):
    if not math.isfinite(torque):
      raise ValueError("nonfinite Honda EPS raw driver torque")

  def effective_threshold(self, base):
    return FIRMWARE_DRIVER_OVERRIDE_THRESHOLD


def uses_supported_transport(CP, CP_SP):
  return CarControllerParams(CP).STEER_MAX == 4096 and get_eps_mod_torque_lookup(CP, CP_SP) is None


def configure_firmware_transport(CI, config):
  """Called before CarParams publication; no owner-parameter reads on control ticks."""
  if config is None:
    return
  if not uses_supported_transport(CI.CP, CI.CP_SP):
    raise ValueError("unsupported Honda EPS transport mapping")
  if CI.CC is not None:
    CI.CC.lat_model = FirmwareTransport(CI.CC.params)
  CI.CS.steer_threshold_learner = FirmwareDriverThreshold()
