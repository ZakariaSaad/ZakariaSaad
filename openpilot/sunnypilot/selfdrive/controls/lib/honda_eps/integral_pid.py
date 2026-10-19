"""Source Ki scheduling: scale accumulation, retain MVL's PID antiwindup.

Adapted from NRDR integral_pid.py / JamesL787 662933847fcad19625dcfb8a6549331222087155.
"""
from openpilot.common.pid import PIDController


class IntegralScaledPIDController(PIDController):
  integrator_gain_scale = 1.0

  @property
  def k_i(self):
    return super().k_i * self.integrator_gain_scale

  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0.0, freeze_integrator=False,
             integrator_gain_scale=1.0, reset_integrator=False):
    self.integrator_gain_scale = float(integrator_gain_scale)
    if reset_integrator:
      self.i = 0.0
    return super().update(error, error_rate, speed, feedforward, freeze_integrator)
