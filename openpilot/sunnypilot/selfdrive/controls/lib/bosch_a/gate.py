"""Explicit, startup-latched eligibility for the experimental Bosch-A path.

The interface sets BOSCH_A_LONG only for an opted-in supported platform using
openpilot longitudinal. Missing CarParamsSP never enables this integration.
"""
from opendbc.car.honda.values import HONDA_BOSCH_A
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


def startup_selected(CP, CP_SP=None) -> bool:
  """Identify the Honda opt-in before card finalizes controller availability."""
  if CP is None or CP_SP is None:
    return False
  return bool(CP.brand == 'honda' and CP.carFingerprint in HONDA_BOSCH_A and
              int(CP_SP.flags) & HondaFlagsSP.BOSCH_A_LONG)


def vehicle_supported(CP, CP_SP=None) -> bool:
  return bool(startup_selected(CP, CP_SP) and CP.openpilotLongitudinalControl and
              not int(CP_SP.flags) & HondaFlagsSP.STOCK_LONGITUDINAL and
              not any(getattr(CP, name, False) for name in ('passive', 'dashcamOnly', 'notCar')))


def finalize_startup(CP, CP_SP) -> None:
  """Revoke the opt-in when card's final controller/passive decision forbids it.

  get_car applies the parameter before card knows OpenpilotEnabledToggle or
  controller availability. Other brands may reuse this numeric flag bit, so
  only Honda's flag is handled here. Unflagged paths remain unchanged.
  """
  if (CP.brand == 'honda' and CP_SP is not None and int(CP_SP.flags) & HondaFlagsSP.BOSCH_A_LONG and
      not vehicle_supported(CP, CP_SP)):
    CP_SP.flags = int(CP_SP.flags) & ~int(HondaFlagsSP.BOSCH_A_LONG)
    if CP.carFingerprint in HONDA_BOSCH_A:
      CP.radarUnavailable = True


def active_for(CP, CP_SP=None) -> bool:
  return vehicle_supported(CP, CP_SP) and not CP.radarUnavailable


def human_following_enabled(CP, CP_SP=None) -> bool:
  return vehicle_supported(CP, CP_SP)
