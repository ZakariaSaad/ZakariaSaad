"""Strict startup eligibility for logging Bosch-A data alongside stock ACC.

This selection never changes control authority, radar availability, timing or
safety configuration. The normal RadarInterface remains the stock path; a
separate observer owns the decoder and publishes only its logging payload.
"""
from opendbc.car import structs
from opendbc.car.honda.values import CAR, HondaSafetyFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


def observation_config_supported(CP, CP_SP) -> bool:
  if CP is None or CP_SP is None:
    return False
  flags = int(CP_SP.flags)
  configs = getattr(CP, 'safetyConfigs', ())
  return bool(CP.brand == 'honda' and CP.carFingerprint == CAR.HONDA_INSIGHT and
              not CP.openpilotLongitudinalControl and CP.pcmCruise and CP_SP.pcmCruiseSpeed and
              flags & HondaFlagsSP.STOCK_LONGITUDINAL and not flags & HondaFlagsSP.BOSCH_A_LONG and
              CP.radarUnavailable and not CP_SP.enableGasInterceptor and
              not any(getattr(CP, name, False) for name in ('passive', 'dashcamOnly', 'notCar')) and
              len(configs) == 1 and configs[0].safetyModel == structs.CarParams.SafetyModel.hondaBosch and
              not int(configs[0].safetyParam) & HondaSafetyFlags.BOSCH_LONG)


def observation_enabled(CP, CP_SP) -> bool:
  return bool(observation_config_supported(CP, CP_SP) and
              int(CP_SP.flags) & HondaFlagsSP.BOSCH_A_RADAR_OBSERVATION)
