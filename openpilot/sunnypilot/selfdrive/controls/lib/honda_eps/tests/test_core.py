"""Numerical/behavior checks for the corrected Insight's firmware law and loop."""
import copy
import math

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps import core
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.driver_override import eps_modified_steering_pressed
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import GAIN_BP, KI, KP

CAL = core.INSIGHT_TXM_A040_CANONICAL


def controller():
  return core.HondaEpsLateralCore(
    GAIN_BP, KP, GAIN_BP, KI, .01,
    ff=core.HondaEpsFirmwareFeedforward(.01, cal=CAL, load=core.INSIGHT_TXM_EPS_LOAD),
    p_scale=core.INSIGHT_P_SCALE, i_scale=core.INSIGHT_I_SCALE,
  )


def tick(control, *, desired=10., angle=10., speed=15., pressed=False, limited=False):
  return control.update(desired, 0., angle, speed, 0., pressed, limited)


def test_corrected_insight_calibration_and_explicit_source_domain():
  transport = core.INSIGHT_TXM_A040
  core.validate_calibration(transport)
  core.validate_calibration(CAL)
  assert transport.e4_per_output == 4096. and CAL.e4_per_output == 3840.
  assert transport.r5_per_key == 18.04 and CAL.r5_per_key is None
  assert CAL.r5_key_bp == (0, 111, 222, 333, 443, 665, 887, 1108, 1663)
  assert CAL.r5_v == (0, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 30000)
  assert CAL.kp_key_bp == (0, 222, 443, 665, 887, 1108, 1330, 1552, 1774)
  assert CAL.envelope_v == (1774,) * 9
  assert core.INSIGHT_TXM_EPS_LOAD == (-4.876, -.1376, -4.452, -397.2, 65.41, -4.29)
  assert core.firmware_r6(1., 0., CAL) == -131.15
  assert core.firmware_r6(1., 150., CAL) == pytest.approx(-131.15 * 1.2156)
  assert core.INSIGHT_P_SCALE == (1.15, 1.25, 1.15)
  assert core.INSIGHT_I_SCALE == (.75, .95, 1.)


@pytest.mark.parametrize("mutation", [
  {"e4_per_output": 0.}, {"e4_per_output": 4095.}, {"r6_per_deg_s": float("nan")},
  {"r6_per_deg_s": -131.}, {"envelope_v": (1600.,) * 9},
  {"envelope_v": (1774.,) * 8}, {"r5_v": (0.,) * 9},
  {"r6_angle_gain": (float("inf"),) * 19}, {"kp_pieces": ()}, {"kp_r5_v": (117.,) * 16},
])
def test_invalid_calibration_rejected(mutation):
  calibration = copy.deepcopy(CAL)
  for key, value in mutation.items():
    setattr(calibration, key, value)
  with pytest.raises(ValueError):
    core.validate_calibration(calibration)


@pytest.mark.parametrize("r5,kp", [(0., 117.), (4000., 148.), (8000., 184.),
                                   (12000., 220.), (16000., 245.), (20000., 257.)])
def test_exact_p_knots_do_not_use_linear_r5_shortcut(r5, kp):
  assert core.firmware_kp(r5, CAL) == pytest.approx(kp)


@pytest.mark.parametrize("angle", (0., 30., 150., 350.))
@pytest.mark.parametrize("rate", (-200., -40., 0., 40., 200.))
@pytest.mark.parametrize("load", (-5000., -500., 0., 500., 5000.))
def test_quadratic_inverse_satisfies_actual_firmware_law(angle, rate, load):
  r5 = core.r5_for_motion(load, rate, 0., angle, CAL)
  # Independent forward lookup: read the command key through its table, then P.
  key = np.interp(abs(r5), CAL.r5_v, CAL.r5_key_bp)
  kp = np.interp(key, core.KP_KEY_BP, core.KP_V)
  damping = core.firmware_r6(rate, angle, CAL)
  firmware_output = (kp * (r5 - damping) + 45. * r5) / 1024.
  assert firmware_output == pytest.approx(load, abs=1e-8)


@pytest.mark.parametrize("source_output", np.linspace(-1., 1., 31))
def test_transport_conversion_preserves_physical_e4_and_firmware_target(source_output):
  transport_output = source_output * 3840. / 4096.
  assert source_output * 3840. == pytest.approx(transport_output * 4096.)
  assert core.r5_from_output(source_output, 25., CAL) == pytest.approx(
    core.r5_from_output(transport_output, 25., core.INSIGHT_TXM_A040))
  r5 = core.r5_from_output(source_output, 25., CAL)
  assert core.output_from_r5(r5, CAL) == pytest.approx(source_output, abs=.001)


@pytest.mark.parametrize("speed", (2., 8., 15., 30.))
def test_feedforward_caps_and_filters_rapid_requested_motion(speed):
  feedforward = core.HondaEpsFirmwareFeedforward(.01, cal=CAL, load=core.INSIGHT_TXM_EPS_LOAD)
  previous = feedforward.update(0., speed, 0.)
  for frame in range(1, 300):
    output = feedforward.update(500. * (-1 if frame % 2 else 1), speed, 0.)
    assert math.isfinite(output)
    assert abs(feedforward.r5) <= 27000.
    assert abs(feedforward.rate) <= 400.
    # FF smoothing cannot move more than alpha times the full target span.
    assert abs(output - previous) <= 2. * .01 / .16
    previous = output


def test_feedforward_join_crawl_gate_and_override_rejoin():
  control = controller()
  for _ in range(75):
    tick(control, desired=30., angle=0.)
  assert control.ff_weight == 0.
  for _ in range(50):
    tick(control, desired=30., angle=30.)
  assert control.ff_weight == pytest.approx(1.)
  tick(control, desired=30., angle=30., pressed=True)
  assert control.ff_weight == 0.
  tick(control, desired=30., angle=30.)
  assert control.ff_weight == pytest.approx(.02)
  control.reset()
  for _ in range(100):
    tick(control, desired=4., angle=4., speed=3.)
  assert control.ff_weight == 0.
  for _ in range(100):
    tick(control, desired=20., angle=20., speed=3.)
  assert control.ff_weight == pytest.approx(.5)


def test_integrator_accumulation_uses_ki_trim_and_freezes_on_press():
  control = controller()
  tick(control, desired=11., angle=10., speed=10.)
  expected_i = np.interp(10., GAIN_BP, KI) * .01 * core.INSIGHT_I_SCALE[0]
  assert control.pid.i == pytest.approx(expected_i)
  held = control.pid.i
  tick(control, desired=11., angle=10., speed=10., pressed=True)
  assert control.pid.i == held
  tick(control, desired=11., angle=10., speed=1.)
  assert control.pid.i == 0.


def test_only_post_override_limited_frames_decay_integrator():
  control = controller()
  control.pid.i = .2
  tick(control, pressed=True, limited=True)
  assert control.pid.i == .2
  for _ in range(50):
    tick(control, limited=True)
  assert control.pid.i == pytest.approx(.2 * math.exp(-1.))
  held = control.pid.i
  control.since_press_s = 5.
  tick(control, limited=True)
  assert control.pid.i == held


def test_reset_clears_controller_state_and_saturation_stays_bounded():
  control = controller()
  for _ in range(1000):
    assert abs(tick(control, desired=500., angle=0.)) <= 1.
  assert abs(control.output) == pytest.approx(1.)
  control.reset()
  assert (control.pid.i, control.ff.output, control.ff_weight, control.output, control.output_lpf.x) == (0.,) * 5
  assert control.ff.prev_angle is None and math.isinf(control.since_press_s)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize("field", ("desired", "angle", "speed"))
def test_nonfinite_feedback_inputs_reset_then_raise(value, field):
  control = controller()
  tick(control)
  with pytest.raises(ValueError):
    tick(control, **{field: value})
  assert control.output == control.pid.i == control.ff.output == 0.


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_feedforward_inputs_reset_then_raise(value):
  feedforward = core.HondaEpsFirmwareFeedforward(.01, cal=CAL, load=core.INSIGHT_TXM_EPS_LOAD)
  feedforward.update(20., 10., 0.)
  with pytest.raises(ValueError):
    feedforward.update(value, 10., 0.)
  assert feedforward.output == feedforward.rate == 0.
  assert feedforward.prev_angle is None


@pytest.mark.parametrize("dt", (.005, .01, .02))
def test_driver_override_debounce_tracks_actual_timestep(dt):
  duration = 0.
  for _ in range(int(.28 / dt) - 1):
    duration, pressed = eps_modified_steering_pressed(True, 500., .5, duration, False, dt)
    assert not pressed
  duration, pressed = eps_modified_steering_pressed(True, 500., .5, duration, False, dt)
  # Binary float sums may cross .28 on the following sample, as in the source.
  if not pressed:
    duration, pressed = eps_modified_steering_pressed(True, 500., .5, duration, False, dt)
  assert pressed and .28 <= duration <= .28 + dt + 1e-12
  assert eps_modified_steering_pressed(True, -500., .5, 0., False, dt) == (1., True)
  assert eps_modified_steering_pressed(True, 500., .01, 0., False, dt) == (1., True)
  assert eps_modified_steering_pressed(False, 500., .5, duration, True, dt) == (0., False)


def test_changed_insight_load_or_gain_constants_rejected(monkeypatch):
  for name, changed in (("INSIGHT_TXM_EPS_LOAD", (-5., -.1376, -4.452, -397.2, 65.41, -4.29)),
                        ("INSIGHT_P_SCALE", (1., 1., 1.))):
    with monkeypatch.context() as patch:
      patch.setattr(core, name, changed)
      with pytest.raises(ValueError):
        core.validate_calibration()


@pytest.mark.parametrize("field", ("r6_angle_bp", "r6_angle_gain"))
def test_missing_r6_axis_or_gains_rejected_consistently(field):
  calibration = copy.deepcopy(CAL)
  setattr(calibration, field, None)
  with pytest.raises(ValueError):
    core.validate_calibration(calibration)
  with pytest.raises(ValueError):
    core.firmware_r6(1., 20., calibration)
