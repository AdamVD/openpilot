#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

# 2026-06-09: trimmed toward the MEASURED Odyssey NIDEC deliverable (rail tests, drive 00000027:
# settled aego ~0.65-0.70 @ 14-17 m/s, ~0.45-0.65 @ 24-31 m/s with pcm_off railed at 8; the
# pcm_off->aego curve is flat from ~3 to 8, so the PCM's internal accel schedule is the ceiling).
# Stock vals [1.6, 1.2, 0.8, 0.6] promised ~2x the plant above 20 m/s -> permanent integrator
# pressure + rail-unwind overshoot. Honest plans also calm the follow loop.
A_CRUISE_MAX_VALS = [1.6, 1.0, 0.65, 0.5]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]

# 2026-07-11: opening-gap chase governor ("lazy re-close", FINDINGS_follow_policy_2026-07-09.md rev 7/10).
# Factory follow is ASYMMETRIC: it barely chases an opening gap (+0.016 m/s^2 per m/s of vRel vs our
# plan's +0.064; opening-cell p90 envelope 0.31-0.41 m/s^2) and re-closes lazily, responding firmly only
# when CLOSING. Our symmetric MPC chases both ways -> chase -> overshoot -> re-brake limit cycle: realized
# surges 71.9/hr vs factory 33.5, 65% re-braked within 8s vs 38%. Fix: while a tracked lead is pulling
# AWAY at follow range, cap the DELIVERED accel (aTarget only -- the published plan trajectory stays
# uncapped so shadow/counterfactual analysis keeps seeing the raw ask) at the factory envelope. Closing
# side, cut-ins (closing by definition), lead-brake, and no-lead cruise keep full authority: the cap
# binds positive accel only and never engages without an opening gap. Passing Assist hands authority
# back instantly (explicit driver intent). Secondary win (FINDINGS_sustained_hold_overshoot_2026-07-10.md):
# a 0.35-capped ask feed-forwards pcm_off ~2.7 marginal-at-the-knee instead of 2.5-3.6 across it, so the
# mid-hold TCU kickdown that turns "0.3 held for 6s" into 0.6-0.8 mostly never fires. Highway-fitted:
# inert below 12 m/s (<18 m/s follow is 3-4 min per era in the corpus -- city regime unsampled), full
# factory cap from 18 m/s. PCM gas authority only exists above ~9.6 m/s anyway.
CHASE_GOVERNOR = True         # False = exact prior behavior
CHASE_A_CAP_BP = [12., 18.]   # m/s; ramp from barely-binding to the factory envelope
CHASE_A_CAP_V = [0.75, 0.35]  # m/s^2; 0.35 = factory opening-cell p90 envelope
CHASE_THW_ON = 2.2            # s; engage only at genuine follow range
CHASE_THW_OFF = 2.5           # s; THW release hysteresis
CHASE_VREL_ON = 0.3           # m/s; engage: gap opening (vRel > 0 = lead faster)
CHASE_VREL_OFF = 0.0          # m/s; hold until the gap stops opening
CHASE_RELEASE_T = 1.0         # s; linger after conditions drop (incl. lead departure -- no step resume)
CHASE_RELEASE_RATE = 0.5      # m/s^2 per s; cap ramps back to inert after the linger
CHASE_CAP_INERT = max(A_CRUISE_MAX_VALS)
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    # opening-gap chase governor state (constants above)
    self.chase_active = False
    self.chase_release = 0.0
    self.chase_cap = CHASE_CAP_INERT

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Passing Assist: compute the lane-change boost before accel limits so its accel
    # headroom is still subject to in-turn limiting below (a no-op when disabled).
    self.pla.update(
      left_blinker=sm['carState'].leftBlinker,
      right_blinker=sm['carState'].rightBlinker,
      left_blindspot=sm['carState'].leftBlindspot,
      right_blindspot=sm['carState'].rightBlindspot,
      v_ego=v_ego,
      lead_present=sm['radarState'].leadOne.status,
      t_follow_base=get_T_FOLLOW(sm['selfdriveState'].personality),
    )

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    if self.pla.accel_headroom > 0.0:
      accel_clip[1] = min(accel_clip[1] + self.pla.accel_headroom, max(A_CRUISE_MAX_VALS))
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality, t_follow=self.pla.t_follow)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    # Odyssey NIDEC: friction brake responds in ~0.12s vs the PCM servo's 0.5-0.65s -> shorter horizon
    # for real braking demands (blended inside get_accel_from_plan; gas/mild-decel keep action_t).
    action_t_brake = (0.45 + DT_MDL) if self.CP.brand == 'honda' else None
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping,
                                                                        action_t_brake=action_t_brake)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    if self.is_e2e(sm):
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    # Opening-gap chase governor: cap delivered accel at the factory re-close envelope while a
    # tracked lead pulls away at follow range (see the CHASE_* block above). Latch engages on
    # THW<ON & vRel>ON, holds while THW<OFF & vRel>OFF, then lingers RELEASE_T and ramps the cap
    # out at RELEASE_RATE -- no step on release or lead departure. Cap-only: decel unaffected.
    if CHASE_GOVERNOR:
      lead = sm['radarState'].leadOne
      thw = lead.dRel / max(v_ego, 0.1)
      in_range = bool(lead.status) and v_ego > CHASE_A_CAP_BP[0]
      if self.pla.accel_headroom > 0.0:
        # driver signaled a pass: hand full authority back immediately
        self.chase_active = False
        self.chase_release = 0.0
        self.chase_cap = CHASE_CAP_INERT
      elif in_range and thw < CHASE_THW_ON and lead.vRel > CHASE_VREL_ON:
        self.chase_active = True
        self.chase_release = CHASE_RELEASE_T
      elif self.chase_active:
        if in_range and thw < CHASE_THW_OFF and lead.vRel > CHASE_VREL_OFF:
          self.chase_release = CHASE_RELEASE_T  # gap still opening at follow range: hold the cap
        else:
          self.chase_release -= self.dt
          self.chase_active = self.chase_release > 0.0
      if self.chase_active:
        self.chase_cap = float(np.interp(v_ego, CHASE_A_CAP_BP, CHASE_A_CAP_V))
      else:
        self.chase_cap = min(self.chase_cap + CHASE_RELEASE_RATE * self.dt, CHASE_CAP_INERT)
      output_a_target = min(output_a_target, self.chase_cap)

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
