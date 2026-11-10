"""Bosch-A optional inputs through the real six-parameter native acados solver.

Synthetic radar observations and lead horizons only. No mock solver,
model inference, message publication, vehicle interfaces or CAN transmission.
"""
import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.cereal import messaging
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, N, PARAM_DIM, T_IDXS, get_stopped_equivalence_factor,
)


def radar(*, present=True, distance=35., lead_speed=10., probability=.99, fcw=False):
  state = messaging.new_message('radarState').radarState
  state.leadOne.present = present
  state.leadOne.radar = present
  state.leadOne.radarTrackId = 7 if present else -1
  state.leadOne.dRel = distance
  state.leadOne.vLead = state.leadOne.vLeadK = lead_speed
  state.leadOne.vRel = lead_speed - 10.
  state.leadOne.aLeadK = 0.
  state.leadOne.aLeadTau = 1.5
  state.leadOne.modelProb = probability
  state.leadOne.deprecated.fcw = fcw
  return state


class NativeSolveAudit:
  """Record actual native return codes before the planner's error reset clears them."""
  def __init__(self, native):
    self.native = native
    self.statuses = []

  def __getattr__(self, name):
    return getattr(self.native, name)

  def solve(self):
    status = self.native.solve()
    self.statuses.append(status)
    return status


def solve(state, *, speed=10., acceleration=0., frames=8, **options):
  mpc = LongitudinalMpc()
  assert type(mpc.solver).__module__.endswith('acados_ocp_solver_pyx')
  mpc.solver = NativeSolveAudit(mpc.solver)
  assert PARAM_DIM == 6 and mpc.params.shape == (N + 1, 6)
  mpc.set_weights(prev_accel_constraint=False)
  for _ in range(frames):
    mpc.set_cur_state(speed, acceleration)
    mpc.update(state, **options)
    assert mpc.solver.statuses[-1] == 0
    assert mpc.solution_status == 0
    assert np.isfinite(mpc.x_sol).all()
  return mpc


def assert_same_solution(actual, expected):
  np.testing.assert_array_equal(actual.params, expected.params)
  np.testing.assert_allclose(actual.x_sol, expected.x_sol, rtol=0., atol=1e-10)
  np.testing.assert_allclose(actual.u_sol, expected.u_sol, rtol=0., atol=1e-10)
  assert actual.source == expected.source
  assert actual.crash_cnt == expected.crash_cnt


def test_optional_noops_leave_native_solution_unchanged():
  state = radar()
  expected = solve(state)
  actual = solve(state, acceleration_max=ACCEL_MAX, lead_paths=(None, None), radar_fcw_enabled=False)
  assert_same_solution(actual, expected)


@pytest.mark.parametrize('lead_index', [0, 1])
def test_qualified_lead_path_reaches_native_solver_and_changes_trajectory(lead_index):
  state = radar(distance=35.)
  if lead_index == 1:
    state.leadTwo = state.leadOne
    state.leadOne.present = state.leadOne.radar = False
  raw = solve(state)
  # The lead starts at the same range/speed, then decelerates to a stop.
  decel = 1.5
  stop_t = state.leadOne.vLead / decel
  time = np.minimum(T_IDXS, stop_t)
  path = np.column_stack((35. + 10. * time - .5 * decel * time**2,
                          np.maximum(10. - decel * T_IDXS, 0.)))
  paths = [None, None]
  paths[lead_index] = path
  actual = solve(state, lead_paths=paths)
  obstacle = path[:, 0] + get_stopped_equivalence_factor(path[:, 1])
  np.testing.assert_allclose(actual.params[:, 2], obstacle, rtol=0., atol=1e-10)
  assert actual.a_solution[3] < raw.a_solution[3] - .05
  assert np.max(np.abs(actual.v_solution - raw.v_solution)) > .1
  assert np.isfinite(actual.params).all()


@pytest.mark.parametrize('invalid_path', [
  None, [], {}, 'invalid', np.zeros((N, 2)), np.zeros((N + 1, 3)),
  np.full((N + 1, 2), np.nan), np.full((N + 1, 2), np.inf),
  [[10**1000, 0]] * (N + 1),
])
def test_invalid_lead_path_falls_back_to_same_native_radar_solution(invalid_path):
  state = radar()
  expected = solve(state)
  actual = solve(state, lead_paths=(invalid_path, None))
  assert_same_solution(actual, expected)


@pytest.mark.parametrize('invalid_container', [None, [], {}, 1., 'invalid'])
def test_missing_or_malformed_path_container_falls_back(invalid_container):
  state = radar()
  assert_same_solution(solve(state, lead_paths=invalid_container), solve(state))


def test_absent_lead_cannot_be_created_by_optional_path():
  state = radar(present=False)
  fictitious_obstacle = np.column_stack((np.full(N + 1, .1), np.zeros(N + 1)))
  actual = solve(state, lead_paths=(fictitious_obstacle, fictitious_obstacle))
  assert_same_solution(actual, solve(state))


@pytest.mark.parametrize('cap', [0., .25, .8])
def test_acceleration_cap_constrains_native_solution(cap):
  state = radar(present=False)
  uncapped = solve(state)
  capped = solve(state, acceleration_max=cap)
  np.testing.assert_array_equal(capped.params[:, 1], np.full(N + 1, cap))
  # The inherited OCP constrains running stages 0..N-1, not terminal stage N.
  # These are soft constraints; allow numerical slack at the constrained stages.
  # The planner must separately cap the final target it emits.
  assert np.max(capped.a_solution[:-1]) <= cap + .02
  assert np.max(uncapped.a_solution[:-1]) > cap + .2
  np.testing.assert_array_equal(capped.params[:, 0], np.full(N + 1, ACCEL_MIN))


def test_nonnegative_throttle_cap_does_not_remove_braking_solution():
  state = radar(distance=15., lead_speed=5.)
  uncapped = solve(state, speed=15.)
  capped = solve(state, speed=15., acceleration_max=.2)
  assert np.min(uncapped.a_solution[1:]) < -1.
  assert np.min(capped.a_solution[1:]) < -1.
  assert capped.a_solution[3] < 0.
  np.testing.assert_array_equal(capped.params[:, 0], uncapped.params[:, 0])


@pytest.mark.parametrize('invalid_cap', [
  '0.2', 'invalid', [], [.2], np.array([.2]), np.array([.2, .3]),
  True, False, np.nan, np.inf, -np.inf, -.01, ACCEL_MAX + .01, 10**1000,
])
def test_invalid_optional_cap_falls_back_without_exception(invalid_cap):
  state = radar(present=False)
  assert_same_solution(solve(state, acceleration_max=invalid_cap), solve(state))


@pytest.mark.parametrize('strict_carrier,legacy_probability,expected_strict,expected_legacy', [
  (False, .99, 0, 4),
  (True, .99, 4, 4),
  (False, .9, 0, 0),
])
def test_radar_fcw_uses_strict_carrier_and_legacy_uses_probability(
    strict_carrier, legacy_probability, expected_strict, expected_legacy):
  # This deliberately close stationary synthetic lead exercises the crash gate.
  # It does not claim a realistic warning event or validate vehicle FCW timing.
  state = radar(distance=.1, lead_speed=0., probability=legacy_probability, fcw=strict_carrier)
  strict = solve(state, speed=0., frames=4, radar_fcw_enabled=True)
  legacy = solve(state, speed=0., frames=4)
  assert strict.crash_cnt == expected_strict
  assert legacy.crash_cnt == expected_legacy
  np.testing.assert_allclose(strict.x_sol, legacy.x_sol, rtol=0., atol=1e-10)


def test_strict_fcw_counter_resets_when_carrier_is_removed():
  state = radar(distance=.1, lead_speed=0., fcw=True)
  mpc = solve(state, speed=0., frames=4, radar_fcw_enabled=True)
  assert mpc.crash_cnt == 4
  state.leadOne.deprecated.fcw = False
  mpc.update(state, radar_fcw_enabled=True)
  assert mpc.solver.statuses[-1] == 0
  assert mpc.crash_cnt == 0
