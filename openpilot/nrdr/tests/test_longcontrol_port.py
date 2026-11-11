"""Native-message/PID regression for the scoped NRDR controller port.

The old controller fixture is the exact target d0d58d284 implementation. Only
parameter storage and the optional tune-file location are substituted here;
these tests never send CAN or start control processes.
"""
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.car.structs import car
from openpilot.cereal import custom
from openpilot.nrdr.features.longitudinal import longcontrol as donor_control
from openpilot.nrdr.features.longitudinal.long_tune import LongTune
from openpilot.nrdr.features.longitudinal.stopping_inputs import stopping_inputs
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState


class Snapshot(dict):
  generation = 1


@pytest.fixture
def cached_params(monkeypatch, tmp_path):
  snapshot = Snapshot({
    "LongPidTuneScaleAggressive": 100,
    "LongPidTuneScaleStandard": 100,
    "LongPidTuneScaleRelaxed": 100,
    "LongPidTuneScaleEcon": 100,
    "HondaLiveLearningGas": False,
    "StaticFeedforwardLong": False,
    "HondaStopAccel": -2.0,
    "HondaStoppingDecelRateLong": 0.3,
    "HondaVEgoStarting": 0.5,
    "HondaVEgoStopping": 0.5,
    "NrdrRoenAccelerationLimits": False,
  })
  provider = SimpleNamespace(snapshot=snapshot, generation=1)
  monkeypatch.setattr(donor_control, "get_live_params", lambda *args: provider)
  monkeypatch.setattr(donor_control, "LongTune", lambda: LongTune(path=str(tmp_path / "absent.json")))
  return provider


def configuration(*, candidate=CAR.HONDA_INSIGHT, brand="honda", long=True, mvl=False, radar_unavailable=False, starting=False):
  cp = car.CarParams.new_message()
  cp.brand = brand
  cp.carFingerprint = candidate
  cp.openpilotLongitudinalControl = long
  cp.flags = int(HondaFlags.MVL_LONGITUDINAL) if mvl else 0
  cp.radarUnavailable = radar_unavailable
  cp.stopAccel = -3.5
  cp.deprecated.startingState = starting
  cp.deprecated.startAccel = 0.2
  cp.deprecated.vEgoStarting = 0.5
  cp.deprecated.vEgoStopping = 0.5
  cp.deprecated.stoppingDecelRate = 0.3
  cp.longitudinalTuning.deprecated.kpBP = [0.0]
  cp.longitudinalTuning.deprecated.kpV = [0.3]
  cp.longitudinalTuning.kiBP = [0.0, 10.0]
  cp.longitudinalTuning.kiV = [0.2, 0.1]
  return cp, custom.CarParamsSP.new_message()


def state(speed=0.0, acceleration=0.0, *, brake=False, gas=False, cruise_standstill=False):
  cs = car.CarState.new_message()
  cs.vEgo = speed
  cs.aEgo = acceleration
  cs.standstill = speed == 0.0
  cs.brakePressed = brake
  cs.gasPressed = gas
  cs.cruiseState.standstill = cruise_standstill
  return cs


@pytest.mark.parametrize("options, selected", [
  ({}, True),
  ({"radar_unavailable": True}, True),
  ({"mvl": True}, False),
  ({"long": False}, False),
  ({"candidate": CAR.HONDA_CIVIC_2022}, False),
  ({"candidate": CAR.HONDA_ACCORD}, True),
  ({"candidate": CAR.ACURA_MDX_4G}, False),
  ({"candidate": CAR.HONDA_CIVIC}, False),
  ({"candidate": "OTHER", "brand": "toyota"}, False),
])
def test_controller_scope_matches_shared_bosch_a_and_startup_mvl(cached_params, options, selected):
  control = LongControl(*configuration(**options))
  assert (control.nrdr_controller is not None) is selected


@pytest.mark.parametrize("options", [
  {"mvl": True},
  {"long": False},
  {"candidate": CAR.HONDA_CIVIC_2022},
  {"candidate": CAR.ACURA_MDX_4G},
  {"candidate": CAR.HONDA_CIVIC},
  {"candidate": "OTHER", "brand": "toyota"},
])
def test_inactive_modes_match_frozen_target_through_state_transitions(cached_params, options):
  fixture = Path(__file__).parent / "fixtures/longcontrol_baseline.py.txt"
  baseline = ModuleType("baseline_longcontrol")
  exec(compile(fixture.read_text(), str(fixture), "exec"), baseline.__dict__)
  cp, sp = configuration(**options)
  actual, reference = LongControl(cp, sp), baseline.LongControl(cp, sp)
  assert actual.nrdr_controller is None
  for frame in range(240):
    active = 10 <= frame < 210
    stopping = 80 <= frame < 160
    speed = max(0.0, 8.0 - (frame - 50) * 0.12)
    cs = state(speed, acceleration=math.sin(frame / 7) * 0.3,
               brake=120 <= frame < 130, gas=frame >= 210,
               cruise_standstill=145 <= frame < 155)
    target = -0.8 if stopping else 0.5 * math.sin(frame / 12)
    actual_output = actual.update(active, cs, target, stopping, (-3.5, 2.0), pitch=0.1, drel=3.0, personality=1)
    baseline_output = reference.update(active, cs, target, stopping, (-3.5, 2.0))
    assert actual_output == baseline_output
    assert actual.long_control_state == reference.long_control_state
    assert (actual.pid.p, actual.pid.i, actual.pid.f) == (reference.pid.p, reference.pid.i, reference.pid.f)


def test_selected_controller_consumes_donor_pid_gains(cached_params):
  control = LongControl(*configuration())
  output = control.update(True, state(1.0, acceleration=0.1), 0.5, False, (-3.5, 2.0), personality=1)
  assert control.long_control_state == LongCtrlState.pid
  assert output == pytest.approx(0.5 + 0.3 * 0.4 + 0.19 * 0.01 * 0.4)
  assert control.pid is control.nrdr_controller.pid


@pytest.mark.parametrize("radar", (False, True))
@pytest.mark.parametrize("mvl", (False, True))
def test_real_insight_startup_parameters_reach_pid_without_empty_gain_arrays(cached_params, radar, mvl):
  from opendbc.car.honda.tests.test_bosch_a_integration import parameters

  cp, sp = parameters(enabled=radar, mvl=mvl)
  control = LongControl(cp, sp)
  assert (control.nrdr_controller is None) is mvl
  if not mvl:
    assert list(cp.longitudinalTuning.deprecated.kpBP) == [0.0]
    assert list(cp.longitudinalTuning.deprecated.kpV) == [0.0]
  output = control.update(True, state(5.0, acceleration=0.1), 0.5, False, (-3.5, 2.0), personality=1)
  assert math.isfinite(output)
  assert control.long_control_state == LongCtrlState.pid


def test_stopping_positive_prior_output_uses_exact_donor_ramp(cached_params):
  control = LongControl(*configuration())
  control.nrdr_controller.last_output_accel = 0.09
  output = control.update(True, state(1.0), 0.4, True, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.stopping
  assert output == pytest.approx(-0.003)
  assert control.last_output_accel == control.nrdr_controller.last_output_accel


def test_donor_stopping_restart_and_cancellation(cached_params):
  control = LongControl(*configuration(starting=True))
  control.update(True, state(), -0.2, True, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.stopping
  control.update(True, state(brake=True), 0.4, False, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.stopping
  control.update(True, state(cruise_standstill=True), 0.4, False, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.stopping
  assert control.update(True, state(), 0.4, False, (-3.5, 2.0)) == pytest.approx(0.2)
  assert control.long_control_state == LongCtrlState.starting
  control.update(True, state(0.6), 0.4, False, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.pid
  assert control.update(False, state(0.6, gas=True), 0.4, False, (-3.5, 2.0)) == 0.0
  assert control.long_control_state == LongCtrlState.off
  assert control.pid.i == 0.0


def test_nearby_lead_and_pitch_are_forwarded_into_stopping(cached_params):
  far, close = LongControl(*configuration()), LongControl(*configuration())
  far.nrdr_controller.last_output_accel = close.nrdr_controller.last_output_accel = -0.1
  far_output = far.update(True, state(), -0.2, True, (-3.5, 2.0), pitch=0.0)
  close_output = close.update(True, state(), -0.2, True, (-3.5, 2.0), pitch=0.0, drel=2.0)
  assert far_output == pytest.approx(-0.1009)
  assert close_output == pytest.approx(-0.100225)
  # The donor's proximity law deliberately reduces the ramp near a lead.
  assert close_output > far_output


def test_live_snapshot_refresh_preserves_donor_values(cached_params):
  control = LongControl(*configuration())
  assert control.update(True, state(1.0), -1.0, True, (-3.5, 2.0)) == pytest.approx(-0.003)
  cached_params.snapshot["HondaStoppingDecelRateLong"] = 0.8
  cached_params.snapshot.generation = cached_params.generation = 2
  assert control.update(True, state(1.0), -1.0, True, (-3.5, 2.0)) == pytest.approx(-0.011)


def test_donor_distance_filter_resets_when_reference_disappears(cached_params):
  control = LongControl(*configuration()).nrdr_controller
  assert control._filter_drel(20.0) == 20.0
  assert control._filter_drel(30.0) == 20.0
  assert control._filter_drel(10.0) == 17.0
  assert control._filter_drel(None) == math.inf
  assert control._filter_drel(30.0) == 30.0


@pytest.mark.parametrize("has_lead, trajectory, expected", [
  (True, [7.0, 8.0], 7.0),
  (True, [], None),
  (False, [7.0], None),
])
def test_stopping_inputs_take_published_trajectory_origin(has_lead, trajectory, expected):
  pose = SimpleNamespace(orientation=SimpleNamespace(xyz=np.array([0.0, 0.12, 0.0])))
  plan = SimpleNamespace(hasLead=has_lead, leadTrajectoryX0=trajectory)
  assert stopping_inputs(pose, plan) == (0.12, expected)
  assert stopping_inputs(None, plan) == (None, expected)
