"""Opt-in Bosch-A integration through the real planner and native MPC.

Synthetic model publications and radar observations only. The native solver,
SP planner, boost implementation and all longitudinal helpers run unchanged;
these tests do not publish messages, run inference, or send CAN.
"""
import numpy as np
import pytest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, get_stopped_equivalence_factor
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests import test_upstream_longitudinal as helpers
from openpilot.selfdrive.controls.tests.test_predictive_accel_boost import model_message
from openpilot.sunnypilot.selfdrive.controls.lib.accel_boost import AccelBoostSP
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a import human_following, model_lead_qualification

isolated_params = helpers.isolated_params


@pytest.fixture(autouse=True)
def no_native_solver_failures(monkeypatch):
  # The inherited reset clears solution_status. Check the actual native solve
  # result before that reset can hide an error; successful execution is untouched.
  original_reset = LongitudinalMpc.reset

  def checked_reset(self):
    assert getattr(self, 'solution_status', 0) == 0
    original_reset(self)

  monkeypatch.setattr(LongitudinalMpc, 'reset', checked_reset)


def configured_planner(*, speed=20., enabled=True, openpilot_long=True, radar_available=True):
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.HONDA_INSIGHT, fingerprint, [], False, False, False)
  cp_sp = CarInterface.get_params_sp(cp, CAR.HONDA_INSIGHT, fingerprint, [], False, False, False)
  cp.openpilotLongitudinalControl = openpilot_long
  cp.radarUnavailable = not radar_available
  cp_sp.flags = int(cp_sp.flags) | int(HondaFlagsSP.BOSCH_A_LONG) if enabled else 0
  return LongitudinalPlanner(cp, cp_sp, init_v=speed)


def messages(*, speed=20., accel=0., should_stop=False, distance=80., lead_speed=None, kind='native'):
  sm = helpers.Messages(model_message(kind, accel=accel, speed=speed), speed=speed)
  sm['modelV2'].action.shouldStop = should_stop
  sm['carState'].canValid = True
  lead = sm['radarState'].leadOne
  lead.present = True
  lead.radar = True
  lead.radarTrackId = 1
  lead.dRel = distance
  lead.vLead = speed if lead_speed is None else lead_speed
  lead.vLeadK = lead.vLead
  lead.vRel = lead.vLead - speed
  lead.aLeadK = .4
  lead.aLeadTau = 1.5
  lead.modelProb = .99
  return sm


def assert_native(control):
  assert type(control.mpc.solver).__module__.endswith('acados_ocp_solver_pyx')
  assert control.mpc.solution_status == 0
  assert np.isfinite(control.mpc.x_sol).all()


@pytest.mark.parametrize('kind', ['native', 'ordinary'])
@pytest.mark.parametrize('speed,accel,distance,lead_speed', [
  (20., -2.5, 80., 20.),
  (10.4, -2.4, 34., 10.54),
])
def test_nonstop_model_braking_survives_entire_helper_chain(kind, speed, accel, distance, lead_speed):
  control = configured_planner(speed=speed)
  sm = messages(speed=speed, accel=accel, distance=distance, lead_speed=lead_speed, kind=kind)
  assert not sm['modelV2'].action.shouldStop
  for _ in range(8):
    control.update(sm)
    assert_native(control)
    # A distant/receding radar lead cannot give comfort shaping permission to
    # weaken the negative model candidate selected by the MVL planner.
    assert ACCEL_MIN <= control.output_a_target <= sm['modelV2'].action.desiredAcceleration
    assert not control.output_should_stop


@pytest.mark.parametrize('accel', [-2.5, -.1, .1])
def test_incoming_stop_flag_and_target_survive_helpers(accel):
  control = configured_planner()
  sm = messages(accel=accel, should_stop=True)
  for _ in range(4):
    control.update(sm)
    assert_native(control)
    assert control.output_should_stop
    assert control.output_a_target <= sm['modelV2'].action.desiredAcceleration
  sm['modelV2'].action.shouldStop = False
  sm['modelV2'].action.desiredAcceleration = .5
  control.update(sm)
  assert not control.output_should_stop


def test_predictive_boost_lifecycle_runs_without_cancelling_braking_or_stop():
  Params().put_bool('AccelBoostSP', True, block=True)
  control = configured_planner()
  sm = messages(accel=.2, distance=150.)
  sm['radarState'].leadOne.present = False
  for _ in range(60):
    control.update(sm)
  assert isinstance(control.accel_boost, AccelBoostSP)
  assert control.accel_boost.total_boost > .1

  # A shallow negative model request used to be eligible for positive boost.
  # The learner still updates, but its positive intervention is not applied.
  sm['modelV2'].action.desiredAcceleration = -.03
  control.update(sm)
  assert control.accel_boost.total_boost > 0.
  assert control.output_a_target <= sm['modelV2'].action.desiredAcceleration
  assert_native(control)

  sm['modelV2'].action.shouldStop = True
  old_boost = control.accel_boost.boost
  control.update(sm)
  assert control.accel_boost.boost < old_boost
  assert control.output_should_stop
  assert control.output_a_target <= sm['modelV2'].action.desiredAcceleration

  sm['selfdriveState'].enabled = sm['carControl'].enabled = False
  sm['controlsState'].longControlState = 'off'
  control.update(sm)
  assert control.accel_boost.total_boost == 0.
  assert not control.bosch_a_stop_gap.holding
  assert not control.bosch_a_coast_first.active


def test_stop_hold_releases_on_lead_departure_without_clearing_model_stop():
  control = configured_planner(speed=0.)
  sm = messages(speed=0., accel=.3, distance=4., lead_speed=0.)
  control.update(sm)
  assert control.bosch_a_stop_gap.holding
  assert control.output_should_stop
  assert control.output_a_target <= -.25

  lead = sm['radarState'].leadOne
  lead.vLead = lead.vLeadK = lead.vRel = 2.
  lead.dRel = 6.
  sm['modelV2'].action.shouldStop = True
  control.update(sm)
  assert not control.bosch_a_stop_gap.holding
  assert control.output_should_stop

  sm['modelV2'].action.shouldStop = False
  for _ in range(30):
    control.update(sm)
    assert_native(control)
  assert not control.output_should_stop
  assert 0. < control.output_a_target <= ACCEL_MAX


def test_final_acceleration_respects_native_human_acceleration_cap():
  control = configured_planner(speed=10.)
  sm = messages(speed=10., accel=2., distance=150.)
  sm['carState'].vCruise = sm['carState'].vCruiseCluster = 36.
  sm['radarState'].leadOne.present = False
  for _ in range(20):
    control.update(sm)
    assert_native(control)
    assert np.all(control.mpc.params[:, 1] == 0.)
    assert control.output_a_target <= 0.


@pytest.mark.parametrize('kwargs', [
  {'enabled': False},
  {'enabled': True, 'openpilot_long': False},
  {'enabled': True, 'radar_available': False},
])
def test_ineligible_configuration_does_not_construct_or_run_helpers(kwargs):
  control = configured_planner(**kwargs)
  assert not control.bosch_a_wired
  assert not hasattr(control, 'bosch_a_coast_first')
  control.update(messages(accel=-1.))
  assert_native(control)
  assert control.output_a_target <= -1.


@pytest.mark.parametrize('invalid_timing', ['stale_frame', 'future_publication'])
def test_qualified_model_horizon_runs_through_planner_into_native_mpc(monkeypatch, invalid_timing):
  now_ns, boot_offset_ns = 10_000_000_000, 20_000_000_000
  monkeypatch.setattr(model_lead_qualification, '_model_lead_clock_pair_ns',
                      lambda: (now_ns, now_ns + boot_offset_ns))

  def prepared_inputs():
    control = configured_planner(speed=10.)
    control.bosch_a_clock_ns = lambda: now_ns
    sm = messages(speed=10., accel=-.3, distance=35., lead_speed=10.)
    sm['carState'].vCruise = sm['carState'].vCruiseCluster = 39.6
    sm.logMonoTime.update(modelV2=now_ns - 50_000_000, radarState=now_ns - 20_000_000,
                         carState=now_ns - 10_000_000)
    sm.recv_time = {name: (stamp + 1_000_000) / 1e9 for name, stamp in sm.logMonoTime.items()}
    sm.alive = dict.fromkeys(sm, True)
    sm.freq_ok = dict.fromkeys(sm, True)
    sm['modelV2'].timestampEof = now_ns + boot_offset_ns - 60_000_000
    sm['radarState'].mdMonoTime = sm.logMonoTime['modelV2']
    times = human_following.LEAD_T_IDXS_MODEL
    for lead in sm['modelV2'].leadsV3:
      lead.prob = .99
      lead.t = times.tolist()
      lead.x = (36.52 + 10. * times - .5 * times**2).tolist()
      lead.y = np.zeros(len(times)).tolist()
      lead.v = (10. - times).tolist()

    # Observe the production builder's actual result at the MPC boundary, then
    # invoke the original update and native solve without substituting numerics.
    observed = {}
    original_update = control.mpc.update

    def observe_update(radar_state, *args, **kwargs):
      observed.update(radar_state=radar_state, **kwargs)
      return original_update(radar_state, *args, **kwargs)

    monkeypatch.setattr(control.mpc, 'update', observe_update)
    return control, sm, observed

  qualified, fresh_sm, selected = prepared_inputs()
  fallback, bad_sm, rejected = prepared_inputs()
  if invalid_timing == 'stale_frame':
    bad_sm['modelV2'].timestampEof = now_ns + boot_offset_ns - 151_000_000
  else:
    bad_sm.logMonoTime['modelV2'] = now_ns + 1

  qualified.update(fresh_sm)
  fallback.update(bad_sm)
  assert_native(qualified)
  assert_native(fallback)
  assert fresh_sm.logMonoTime['modelV2'] in qualified.bosch_a_model_lead_history.frames
  assert not fallback.bosch_a_model_lead_history.frames
  assert selected['lead_paths'][0] is not None
  assert rejected['lead_paths'] == (None, None)

  path = selected['lead_paths'][0]
  np.testing.assert_allclose(path[0], [35., 10.])
  np.testing.assert_allclose(qualified.mpc.params[:, 2],
                             path[:, 0] + get_stopped_equivalence_factor(path[:, 1]))
  raw = fallback.mpc.process_lead(rejected['radar_state'].leadOne)
  np.testing.assert_allclose(fallback.mpc.params[:, 2],
                             raw[:, 0] + get_stopped_equivalence_factor(raw[:, 1]))
  assert np.max(np.abs(qualified.mpc.v_solution - fallback.mpc.v_solution)) > .1

  for control, sm, observed in ((qualified, fresh_sm, selected), (fallback, bad_sm, rejected)):
    cap = observed['acceleration_max']
    assert 0. < cap < ACCEL_MAX
    assert np.all(control.mpc.params[:, 1] == cap)
    assert control.output_a_target <= cap
    assert control.output_a_target <= sm['modelV2'].action.desiredAcceleration
