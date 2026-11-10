"""Explicit, startup-latched eligibility for the experimental Bosch-A path.

The interface sets BOSCH_A_LONG only for an opted-in supported platform using
openpilot longitudinal. Missing CarParamsSP never enables this integration.
"""
from opendbc.car.honda.values import HONDA_BOSCH_A
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


def vehicle_supported(CP, CP_SP=None) -> bool:
  if CP is None or CP_SP is None:
    return False
  return bool(CP.brand == 'honda' and CP.carFingerprint in HONDA_BOSCH_A and
              CP.openpilotLongitudinalControl and int(CP_SP.flags) & HondaFlagsSP.BOSCH_A_LONG)


def active_for(CP, CP_SP=None) -> bool:
  return vehicle_supported(CP, CP_SP) and not CP.radarUnavailable


def human_following_enabled(CP, CP_SP=None) -> bool:
  return vehicle_supported(CP, CP_SP)
