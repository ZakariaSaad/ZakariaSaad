"""Native-schema wrapper checks around the physical-command-domain controller."""
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car import structs
from opendbc.car.honda.values import HondaFlags
from openpilot.cereal import custom
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware


def car():
  cp = structs.CarParams.new_message()
  cp.brand = "honda"
  cp.carFingerprint = "HONDA_INSIGHT"
  cp.flags = int(HondaFlags.BOSCH)
  cp.lateralTuning.init("pid")
  cp.steerControlType = "torque"
  cp.lateralParams.torqueBP = [0, 4096]
  cp.lateralParams.torqueV = [0, 4096]
  cp.carFw = [{"ecu": "eps", "fwVersion": b"39990-TXM,A040"}]
  cp.steerLimitTimer = .2
  sp = custom.CarParamsSP.new_message(flags=int(HondaFlagsSP.EPS_MODIFIED))
  return cp, sp


class VehicleModel:
  l = 2.7

  def get_steer_from_curvature(self, curvature, speed, roll):
    return curvature


def tick(control, *, active=True, curvature=-.01, speed=15., angle=0., pressed=False, torque=0., limited=False):
  state = SimpleNamespace(vEgo=speed, steeringAngleDeg=angle, steeringRateDeg=0., steeringTorque=torque,
                          steeringPressed=pressed)
  params = SimpleNamespace(roll=0., angleOffsetDeg=0.)
  return control.update(active, state, VehicleModel(), params, limited, curvature, None, False, .2)


def test_native_initialization_uses_corrected_profile_without_mutating_carparams():
  cp, sp = car()
  before = cp.to_dict()
  control = LatControlFirmware(cp, sp, None, .01)
  assert cp.to_dict() == before
  assert control.geometry_mode == "mvl"
  assert control.output_scale == control.steer_max == .9375
  assert control.calibration.e4_per_output == 4096.
  assert control.core.ff.cal.e4_per_output == 3840.


@pytest.mark.parametrize("change", ("stock", "wrong_car", "wrong_eps", "wrong_scale", "torque_tuning"))
def test_unsupported_initialization_rejected(change):
  cp, sp = car()
  if change == "stock":
    sp.flags = 0
  elif change == "wrong_car":
    cp.carFingerprint = "HONDA_CIVIC"
  elif change == "wrong_eps":
    cp.carFw[0].fwVersion = b"39990-TXM,A030"
  elif change == "wrong_scale":
    cp.lateralParams.torqueV = [0, 3840]
  else:
    cp.lateralTuning.init("torque")
  with pytest.raises(ValueError):
    LatControlFirmware(cp, sp, None, .01)


@pytest.mark.parametrize("mode", ("mvl", "james"))
def test_disengagement_warms_history_and_reset_retains_last_target(mode):
  cp, sp = car()
  control = LatControlFirmware(cp, sp, None, .01, geometry_mode=mode)
  for frame in range(50):
    output, _, logged = tick(control, active=False, curvature=-.002 * frame, speed=5.)
    assert output == 0. and not logged.active
  history = tuple(control.cmd_delay.buf)
  previous = control.prev_angle
  assert len(history) > 15 and previous != 0.
  control.reset()
  assert tuple(control.cmd_delay.buf) == history and control.prev_angle == previous
  output, angle, logged = tick(control, speed=5.)
  assert abs(angle - previous) <= 3. + 1e-5
  assert logged.commandDelay == pytest.approx(.175)
  assert abs(output) <= .9375


def test_complete_output_and_logged_terms_convert_once_to_mvl_normalization():
  cp, sp = car()
  control = LatControlFirmware(cp, sp, None, .01)
  for _ in range(200):
    output, _, logged = tick(control)
  assert output * 4096. == pytest.approx(control.core.output * 3840.)
  assert logged.p * 4096. == pytest.approx(control.core.pid.p * 3840., rel=1e-6)
  assert logged.i * 4096. == pytest.approx(control.core.pid.i * 3840., rel=1e-6)
  assert logged.f * 4096. == pytest.approx(control.core.pid.f * 3840., rel=1e-6)
  assert logged.epsFfFeedforward * 4096. == pytest.approx(control.core.ff.output * 3840., rel=1e-6)


def test_output_physical_saturation_never_exceeds_james_source_rail():
  cp, sp = car()
  control = LatControlFirmware(cp, sp, None, .01)
  for _ in range(200):
    output, _, logged = tick(control, curvature=-1000., angle=-500.)
    assert abs(output * 4096.) <= 3840.
  assert output == pytest.approx(.9375)
  assert logged.saturated


def test_active_rate_limit_and_no_second_output_filter():
  cp, sp = car()
  control = LatControlFirmware(cp, sp, None, .01)
  output, angle, _ = tick(control, curvature=-100., speed=30.)
  assert angle == pytest.approx(3.)
  assert output == pytest.approx(control.core.output * .9375 * control.delivery.ramp)
  assert control.owns_output_filter


@pytest.mark.parametrize("field", ("curvature", "speed", "angle", "torque"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_nonfinite_input_clears_every_state_then_fails_closed(field, value):
  cp, sp = car()
  control = LatControlFirmware(cp, sp, None, .01)
  for _ in range(25):
    tick(control)
  with pytest.raises(ValueError):
    tick(control, **{field: value})
  assert control.core.output == control.core.pid.i == control.core.ff.output == control.prev_angle == 0.
  assert not control.cmd_delay.buf
  assert not control.pressed_prev
