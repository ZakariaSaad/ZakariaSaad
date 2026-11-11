"""The donor parameter names consumed by the imported longitudinal feature."""
from enum import StrEnum


class NrdrParamKey(StrEnum):
  LONG_PID_TUNE_SCALE_AGGRESSIVE = 'LongPidTuneScaleAggressive'
  LONG_PID_TUNE_SCALE_STANDARD = 'LongPidTuneScaleStandard'
  LONG_PID_TUNE_SCALE_RELAXED = 'LongPidTuneScaleRelaxed'
  LONG_PID_TUNE_SCALE_ECON = 'LongPidTuneScaleEcon'
  HONDA_LIVE_LEARNING_GAS = 'HondaLiveLearningGas'
  STATIC_FEEDFORWARD_LONG = 'StaticFeedforwardLong'
  HONDA_STOP_ACCEL = 'HondaStopAccel'
  HONDA_STOPPING_DECEL_RATE_LONG = 'HondaStoppingDecelRateLong'
  HONDA_V_EGO_STARTING = 'HondaVEgoStarting'
  HONDA_V_EGO_STOPPING = 'HondaVEgoStopping'
  NRDR_ROEN_ACCELERATION_LIMITS = 'NrdrRoenAccelerationLimits'
  NRDR_CRUISE_MISMATCH_CORRECTION = 'NrdrCruiseMismatchCorrection'
  NRDR_STANDSTILL_GAP_EXTRA = 'NrdrStandstillGapExtra'
