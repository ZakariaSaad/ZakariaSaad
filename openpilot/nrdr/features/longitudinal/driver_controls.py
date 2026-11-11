"""NRDR's manual Experimental/profile controls, scoped to the imported path.

The target has no NRDR HUD submode or MVL personality writer. Keep its existing
controls outside Bosch-A non-MVL mode. Inside it, use the donor CruiseHelper
and the donor short-release/SLA-reservation rules without importing unrelated
HUD or stock-CMBS features.
"""
import time

from openpilot.cereal import custom
from opendbc.car.honda.values import HONDA_BOSCH_A
from opendbc.car.structs import car
from openpilot.nrdr.features.longitudinal.bosch_a.gate import vehicle_supported
from openpilot.nrdr.features.longitudinal.policy import longitudinal_personality, mvl_longitudinal_enabled
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper

ButtonType = car.CarState.ButtonEvent.Type


def imported_mode(CP):
  return vehicle_supported(CP) and not mvl_longitudinal_enabled(CP)


def initialize_experimental(CP, params):
  # Donor retains a False marker when leaving openpilot longitudinal.
  if CP.brand == "honda" and CP.carFingerprint in HONDA_BOSCH_A and not CP.openpilotLongitudinalControl:
    if params.get("ExperimentalMode") is not False:
      params.put_bool("ExperimentalMode", False, block=True)


class BoschADriverControls:
  def __init__(self, CP, params):
    self.cruise = CruiseHelper(CP)
    self.cruise.params = params
    self.sla_button_reserved_until = 0.0

  def update(self, selfdrive, CS, *, now=None):
    now = time.monotonic() if now is None else now
    sla_state = selfdrive.sm["longitudinalPlanSP"].speedLimit.assist.state
    if sla_state == custom.LongitudinalPlanSP.SpeedLimit.AssistState.preActive:
      self.sla_button_reserved_until = now + 0.75
    reserved = now <= self.sla_button_reserved_until
    if not reserved:
      self.sla_button_reserved_until = 0.0

    self.cruise.update(CS, selfdrive.events_sp, selfdrive.experimental_mode, distance_button_reserved=reserved)
    released = any(not event.pressed and event.type == ButtonType.gapAdjustCruise for event in CS.buttonEvents)
    switched = self.cruise.experimental_mode_switched
    if released:
      self.cruise.experimental_mode_switched = False
    if not selfdrive.CP.openpilotLongitudinalControl or reserved or self.cruise.distance_button_consumed:
      return False
    selfdrive.personality = longitudinal_personality(selfdrive.personality, True)
    if released and not switched:
      selfdrive.personality = (selfdrive.personality - 1) % 4
      selfdrive.params.put("LongitudinalPersonality", selfdrive.personality)
      return True
    return False
