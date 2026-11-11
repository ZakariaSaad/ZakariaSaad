"""Native-solver comparison against the pinned donor planner, plus inactive parity.

The exact donor source is retained as a fixture. Only the unrelated MVL import
is supplied by the host's preserved baseline implementation. The active tests
always select Insight with MVL off. No CAN, model inference or vehicle I/O runs.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import custom, log
from openpilot.common.params import Params
from openpilot.nrdr.features.longitudinal import longitudinal_planner as nrdr_planner
from openpilot.nrdr.features.longitudinal.bosch_a import model_lead_qualification
from openpilot.nrdr.params import LiveParams, PLANNER_GROUPS
from openpilot.nrdr.tests.test_bosch_a_planner_wiring import cp, submaster, NOW_NS, BOOT_OFFSET_NS
from openpilot.selfdrive.controls.lib import longitudinal_planner as port
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import baseline_planner
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.baseline_planner import BaselineLongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


def load_fixture_module(name):
  path = Path(__file__).with_name("fixtures") / name
  module = ModuleType("pinned_" + name.replace(".", "_"))
  module.__file__ = str(path)
  exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
  return module


def load_donor_planner(monkeypatch):
  path = Path(__file__).with_name("fixtures") / "nrdr_longitudinal_planner_622a1479.py.txt"
  source = path.read_text()
  module = ModuleType("pinned_nrdr_planner")
  shim = ModuleType("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.mvl_planner")
  shim.MvlLongitudinalPlanner = BaselineLongitudinalPlanner
  monkeypatch.setitem(sys.modules, shim.__name__, shim)
  econ = ModuleType("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.mvl_econ")
  econ.ECON_T_FOLLOW, econ.ECON_JERK_FACTOR = 2.25, 1.5
  monkeypatch.setitem(sys.modules, econ.__name__, econ)
  donor_mpc = load_fixture_module("nrdr_long_mpc_622a1479.py.txt")
  exec(compile(source, str(path), "exec"), module.__dict__)
  module.LongitudinalMpc = donor_mpc.LongitudinalMpc
  return module.LongitudinalPlanner


@pytest.fixture
def native_environment(tmp_path, monkeypatch):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path / "params"))
  params = Params()
  for key, value in {"HondaVEgoStopping": 0.5, "NrdrCruiseMismatchCorrection": 100.0,
                     "NrdrStandstillGapExtra": 0.0}.items():
    params.put(key, value, block=True)
  params.put_bool("NrdrRoenAccelerationLimits", False, block=True)
  snapshot = LiveParams(PLANNER_GROUPS, params=params, start_worker=False)
  monkeypatch.setattr(nrdr_planner, "get_live_params", lambda _: snapshot)
  monkeypatch.setattr(model_lead_qualification, "_model_lead_clock_pair_ns",
                      lambda: (NOW_NS, NOW_NS + BOOT_OFFSET_NS))
  # Map/remote speed selection is orthogonal to this same-input planner test.
  # The same unchanged host target-selection function is exercised separately.
  monkeypatch.setattr(LongitudinalPlannerSP, "update_targets",
                      lambda self, sm, v, a, cruise: (cruise, a))
  yield params, snapshot
  snapshot.close()


def sequence_message(frame, *, experimental=False, personality=1):
  # Keep the exact clock envelope fixed while moving the synthetic scene.
  sm = submaster(frame % 20, personality=personality)
  ss, cs, md = sm["selfdriveState"], sm["carState"], sm["modelV2"]
  ss.experimentalMode = experimental
  md.action.desiredAcceleration = (-1.8 if 25 <= frame < 45 else 0.7)
  md.action.shouldStop = 45 <= frame < 60
  cs.standstill = 48 <= frame < 60
  cs.vEgo = 0.0 if cs.standstill else 0.2 if 45 <= frame < 48 else 0.8 if 60 <= frame < 65 else 20.0
  cs.aEgo = -0.2 if frame < 60 else 0.3
  cs.gasPressed = 65 <= frame < 70
  cs.leftBlinker = 70 <= frame < 75
  sm["controlsState"].longControlState = "off" if frame in (0, 1, 76, 77) else "pid"
  ss.enabled = frame not in (0, 1, 76, 77)
  cs.vCruise = 100.0
  for lead in (sm["radarState"].leadOne, sm["radarState"].leadTwo):
    lead.dRel = 5.0 if cs.standstill else 12.0 if frame >= 25 else 40.0
    lead.vLead = lead.vLeadK = 0.0 if cs.standstill else 12.0
    lead.aLeadK = -1.0 if frame < 60 else 0.5
    lead.deprecated.fcw = frame < 45
  md.velocity.x = [cs.vEgo] * 33
  md.acceleration.x = [0.5] * 33
  return sm


@pytest.mark.parametrize("radar_unavailable", [False, True])
@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("personality", [0, 1, 2, 3])
def test_native_active_sequence_matches_exact_donor(native_environment, monkeypatch, radar_unavailable,
                                                   experimental, personality):
  params, _ = native_environment
  params.put_bool("AccelBoostSP", True, block=True)
  car = cp("HONDA_INSIGHT", radar_unavailable=radar_unavailable)
  car_sp = custom.CarParamsSP.new_message()
  actual = port.LongitudinalPlanner(car, car_sp, init_v=20.0)
  reference = load_donor_planner(monkeypatch)(car, car_sp, init_v=20.0)
  # Select the host SP adapter for the donor's equivalent active branch.
  reference.use_bosch_a_nrdr = True
  actual.bosch_a_clock_ns = reference.bosch_a_clock_ns = lambda: NOW_NS
  assert actual.bosch_a_wired and reference.bosch_a_wired
  for frame in range(80):
    sm = sequence_message(frame, experimental=experimental, personality=personality)
    actual.update(sm)
    reference.update(sm)
    for name in ("output_a_target", "output_should_stop", "a_cruise", "fcw", "allow_throttle"):
      assert getattr(actual, name) == getattr(reference, name), (frame, name)
    assert actual.mpc.source == reference.mpc.source
    assert actual.mpc.solution_status == reference.mpc.solution_status == 0
    for name in ("params", "x_sol", "u_sol", "lead_xv_0", "lead_xv_1"):
      np.testing.assert_array_equal(getattr(actual.mpc, name), getattr(reference.mpc, name))


@pytest.mark.parametrize("car", [cp("HONDA_INSIGHT", mvl=True), cp("HONDA_INSIGHT", long=False),
                                 cp("HONDA_CIVIC"), cp("HONDA_ACCORD_11G")])
def test_inactive_sequence_matches_preserved_baseline(native_environment, monkeypatch, car):
  car_sp = custom.CarParamsSP.new_message()
  actual = port.LongitudinalPlanner(car, car_sp, init_v=20.0)
  original_mpc = load_fixture_module("baseline_long_mpc_d0d58d284.py.txt").LongitudinalMpc
  with monkeypatch.context() as context:
    context.setattr(baseline_planner, "LongitudinalMpc", original_mpc)
    reference = BaselineLongitudinalPlanner(car, car_sp, init_v=20.0)
  assert not actual.bosch_a_wired
  for frame in range(80):
    sm = sequence_message(frame)
    actual.update(sm)
    reference.update(sm)
    for name in ("output_a_target", "output_should_stop", "a_cruise", "fcw"):
      assert getattr(actual, name) == getattr(reference, name), (frame, name)
    assert actual.mpc.source == reference.mpc.source
    np.testing.assert_array_equal(actual.mpc.x_sol, reference.mpc.x_sol)


def test_active_publish_keeps_native_schema_fields(native_environment):
  actual = port.LongitudinalPlanner(cp("HONDA_INSIGHT"), custom.CarParamsSP.new_message(), init_v=20.0)
  actual.bosch_a_clock_ns = lambda: NOW_NS
  sm = sequence_message(2)
  sm.all_checks = lambda **kwargs: True
  # Host SP telemetry fields are exercised by the real base publication.
  actual.update(sm)
  sent = {}
  class Publisher:
    def send(self, name, message):
      sent[name] = message.to_bytes()
  actual.resolver.distance = 0.0  # Set by the target update hook isolated in this fixture.
  actual.publish(sm, Publisher())
  with log.Event.from_bytes(sent["longitudinalPlan"]) as event:
    plan = event.longitudinalPlan
    assert len(plan.leadTrajectoryX0) == len(actual.mpc.lead_xv_0)
    assert list(plan.leadTrajectoryV1) == pytest.approx(actual.mpc.lead_xv_1[:, 1])
    assert plan.allowThrottle == actual.allow_throttle


def test_donor_fixture_hash():
  path = Path(__file__).with_name("fixtures") / "nrdr_longitudinal_planner_622a1479.py.txt"
  assert hashlib.sha256(path.read_bytes()).hexdigest() == "9a7d5a10fbfc8f5842f2f9537eb8ac1f29ca52fc4cf707b7825e61f1a538d459"


def test_imported_donor_helpers_are_exact():
  root = Path(__file__).resolve().parents[3]
  manifest = json.loads((Path(__file__).with_name("fixtures") / "donor_longitudinal_sha256.json").read_text())
  for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("active_feature", [False, True])
@pytest.mark.parametrize("enabled,active", [(False, False), (True, False), (True, True)])
def test_scc_target_arbitration_preserves_host_and_donor_modes(active_feature, enabled, active):
  planner = object.__new__(LongitudinalPlannerSP)
  planner.use_bosch_a_nrdr = active_feature
  planner.events_sp = None
  planner.scc = SimpleNamespace(
    update=lambda *args: None,
    vision=SimpleNamespace(enabled=enabled, is_active=active, output_v_target=6.0, output_a_target=-1.0),
    map=SimpleNamespace(output_v_target=100.0, output_a_target=0.0),
  )
  planner.resolver = SimpleNamespace(
    update=lambda *args: None, speed_limit_valid=False, speed_limit_last_valid=False,
    speed_limit=0.0, speed_limit_final_last=0.0, distance=0.0,
  )
  planner.sla = SimpleNamespace(update=lambda *args: None, output_v_target=100.0, output_a_target=0.0)
  result = planner.update_targets(sequence_message(2), 20.0, 0.0, 20.0)
  assert result == ((20.0, 0.0) if active_feature and not (enabled and active) else (6.0, -1.0))


def test_live_stopped_gap_applies_only_at_reset(native_environment):
  params, snapshot = native_environment
  planner = port.LongitudinalPlanner(cp("HONDA_INSIGHT"), custom.CarParamsSP.new_message())
  assert planner.nrdr.stopped_gap(False) == 0.0
  params.put("NrdrStandstillGapExtra", 2.0, block=True)
  assert snapshot.poll_once()
  planner.nrdr.refresh()
  assert planner.nrdr.stopped_gap(False) == 0.0
  assert planner.nrdr.stopped_gap(True) == 2.0
  params.put("NrdrStandstillGapExtra", 1.0, block=True)
  assert snapshot.poll_once()
  planner.nrdr.refresh()
  assert planner.nrdr.stopped_gap(False) == 2.0
  assert planner.nrdr.stopped_gap(True) == 1.0


@pytest.mark.parametrize("kind", ["native", "ordinary"])
@pytest.mark.parametrize("experimental", [False, True])
def test_native_and_ordinary_model_publishers_match_donor(native_environment, monkeypatch, kind, experimental):
  from openpilot.selfdrive.controls.tests.test_predictive_accel_boost import model_message
  from openpilot.cereal import messaging
  car = cp("HONDA_INSIGHT")
  actual = port.LongitudinalPlanner(car, custom.CarParamsSP.new_message(), init_v=20.0)
  reference = load_donor_planner(monkeypatch)(car, custom.CarParamsSP.new_message(), init_v=20.0)
  reference.use_bosch_a_nrdr = True
  for frame in range(25):
    sm = sequence_message(frame, experimental=experimental)
    sm["modelV2"] = model_message(kind, accel=-1.5, speed=20.0)
    sm["vehicleParameters"] = messaging.new_message("vehicleParameters").vehicleParameters
    actual.update(sm)
    reference.update(sm)
    assert np.isfinite(actual.output_a_target)
    assert actual.output_a_target == reference.output_a_target
    assert actual.output_should_stop == reference.output_should_stop
    np.testing.assert_array_equal(actual.mpc.x_sol, reference.mpc.x_sol)
