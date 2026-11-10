#!/usr/bin/env python3
import time
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a import gate, human_acceleration, human_following
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.bosch_a_coast_first import BoschACoastFirst
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.bosch_a_release_coast import BoschAReleaseCoast
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.bosch_a_smooth import BoschASmooth
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.bosch_a_onset_ramp import BoschAOnsetRamp
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.bosch_a_stop_gap import BoschAStopGap
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.honda_bosch_a_long import HondaBoschALongEnhancement
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.model_lead_qualification import ModelLeadHistory, qualified_model_leads

J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -0.5
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, dt):
  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, ACCEL_MAX)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.accel_boost = AccelBoost()
    self.bosch_a_wired = gate.vehicle_supported(CP, CP_SP)
    if self.bosch_a_wired:
      self.bosch_a_long = HondaBoschALongEnhancement(CP, dt, CP_SP=CP_SP)
      self.bosch_a_coast_first = BoschACoastFirst(CP, dt, CP_SP=CP_SP)
      self.bosch_a_release_coast = BoschAReleaseCoast(CP, dt, CP_SP=CP_SP)
      self.bosch_a_smooth = BoschASmooth(CP, dt, CP_SP=CP_SP)
      self.bosch_a_onset_ramp = BoschAOnsetRamp(CP, dt, CP_SP=CP_SP)
      self.bosch_a_stop_gap = BoschAStopGap(CP, dt, CP_SP=CP_SP)
      self.bosch_a_model_lead_history = ModelLeadHistory()
      self.bosch_a_clock_ns = time.monotonic_ns

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    planner_radar_state = sm['radarState']
    bosch_a_mpc = {}
    if self.bosch_a_wired:
      # Use this branch's acceleration envelope; do not import NRDR's unrelated
      # personality/launch tuning. MPC and the post-processing see the same leads.
      planner_radar_state = self.bosch_a_long.planner_radar_state(planner_radar_state, float(sm['carState'].aEgo), sm['modelV2'])
      if reset_state:
        self.bosch_a_model_lead_history.clear()
      bosch_a_mpc = {
        'acceleration_max': human_acceleration.cap(ACCEL_MAX, v_cruise, v_ego),
        'lead_paths': human_following.upstream_lead_paths(
          qualified_model_leads(sm, self.bosch_a_clock_ns(), self.bosch_a_model_lead_history), planner_radar_state, self.mpc.x0[1]),
        'radar_fcw_enabled': not self.CP.radarUnavailable,
      }
    self.mpc.update(planner_radar_state, personality=sm['selfdriveState'].personality, **bosch_a_mpc)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t = (self.bosch_a_smooth.action_t(self.CP.longitudinalActuatorDelay) if self.bosch_a_wired else
                self.CP.longitudinalActuatorDelay + DT_MDL)
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    self.a_cruise = get_cruise_accel(v_cruise, v_ego, self.a_cruise, self.dt)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    boosted_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, self.a_cruise)
    # Do not let positive boost cancel a braking/stop request in this new path.
    # Keep the existing boost update/reset lifecycle and all unwired behavior.
    if not self.bosch_a_wired or (output_a_target_e2e >= 0.0 and not output_should_stop_e2e):
      output_a_target_e2e = boosted_e2e

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
                  (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e)]

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    if self.bosch_a_wired:
      output_a_target = min(output_a_target, bosch_a_mpc['acceleration_max'])
      if self.bosch_a_long.stock_feel and reset_state:
        # Donor engage initialization, using this branch's own cruise law.
        self.a_cruise = min(self.a_cruise, float(np.clip(v_cruise - v_ego, A_CRUISE_MIN, ACCEL_MAX)))
      lead_one, lead_two = planner_radar_state.leadOne, planner_radar_state.leadTwo
      standstill = bool(sm['carState'].standstill)
      stop_target = output_a_target if self.output_should_stop else None
      output_a_target = self.bosch_a_long.step(
        lead_one=lead_one, lead_two=lead_two, prev=float(a_prev), target=float(output_a_target), v_ego=v_ego,
        stopping=bool(self.output_should_stop), standstill=standstill, reset_state=bool(reset_state))
      output_a_target = self.bosch_a_coast_first.step(
        lead_one=lead_one, lead_two=lead_two, target=float(output_a_target), v_ego=v_ego,
        follow_s=float(self.mpc.params[0, 4]), fcw=bool(self.fcw), reset_state=bool(reset_state or self.output_should_stop))
      output_a_target = self.bosch_a_release_coast.update(
        lead_one=lead_one, target=float(output_a_target), lane_change_state=sm['modelV2'].meta.laneChangeState,
        left_blinker=bool(sm['carState'].leftBlinker), right_blinker=bool(sm['carState'].rightBlinker),
        gas_pressed=bool(sm['carState'].gasPressed), reset_state=bool(reset_state))
      output_a_target = self.bosch_a_smooth.step(float(a_prev), float(output_a_target), reset=bool(reset_state or standstill))
      output_a_target = self.bosch_a_onset_ramp.step(
        lead_one=lead_one, lead_two=lead_two, prev=float(a_prev), target=float(output_a_target), fcw=bool(self.fcw),
        stopping=bool(self.output_should_stop), reset=bool(reset_state or standstill))
      output_a_target, self.output_should_stop = self.bosch_a_stop_gap.step(
        lead_one=lead_one, lead_two=lead_two, target=float(output_a_target), should_stop=bool(self.output_should_stop),
        v_ego=v_ego, standstill=standstill, reset_state=bool(reset_state))
      if stop_target is not None:
        output_a_target = min(output_a_target, stop_target)
      output_a_target = min(output_a_target, bosch_a_mpc['acceleration_max'])
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.accelBoost = float(self.accel_boost.total_boost)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
