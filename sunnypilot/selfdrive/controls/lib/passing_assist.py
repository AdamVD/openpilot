"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Passing Assist (longitudinal lane-change assist)
================================================

When the driver signals a lane change at highway speed while following a slower
lead, this temporarily:
  1. reduces the follow time-gap (t_follow)  -> the MPC closes the gap to the lead
  2. raises the acceleration ceiling (kick)  -> lets the gap-closing surge come through
  3. allows a small overspeed above set speed -> finishes the pass

It is bounded by a hard time cap measured from the blinker (no dependence on the
blinker staying on), and is suppressed in real time if the target-side blind spot
is occupied or speed drops below the threshold. With a slower lead this matches
the "eager to pass, but still holds you off if you don't" behavior, because once
the boost relaxes the MPC re-opens the gap on its own.

The whole feature is a no-op when disabled: t_follow falls back to the
personality default and the accel/overspeed deltas are zero.
"""
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

# Direction parameter values
LEFT_ONLY = 0
BOTH = 1

# Direction telemetry values (mirror cereal custom.LongitudinalPlanSP.PassingAssist.Direction)
DIR_NONE = 0
DIR_LEFT = 1
DIR_RIGHT = 2

# Engagement ramp time constants (seconds). Engage briskly, relax gently so the
# gap re-opens smoothly and the cut at the time cap is never abrupt.
ENGAGE_TIME = 0.6
RELEASE_TIME = 1.5

# Speed hysteresis so we don't chatter right at the threshold.
SPEED_HYSTERESIS = 1.0 * CV.MPH_TO_MS

# Safety/limit clamps on the user-tunable parameters.
MIN_T_FOLLOW_FLOOR = 0.9   # never command a follow gap below this, regardless of the param
MAX_ACCEL_HEADROOM = 0.8   # m/s^2, hard cap on the accel kick
MAX_OVERSPEED = 5.0 * CV.MPH_TO_MS  # hard cap on the overspeed allowance


class PassingAssist:
  def __init__(self):
    self.params = Params()
    self.frame = 0

    # tunables (refreshed on the params cadence)
    self.enabled = False
    self.accel_param = 0.0       # m/s^2
    self.duration = 0.0          # s
    self.min_t_follow = 1.45     # s
    self.overspeed_param = 0.0   # m/s
    self.direction_mode = LEFT_ONLY
    self.min_speed = 45.0 * CV.MPH_TO_MS

    # state
    self.qualifying_prev = False
    self.latched = False
    self.boost_timer = 0.0
    self.latched_direction = DIR_NONE
    self.engagement = 0.0

    # outputs (read by the longitudinal planner)
    self.active = False
    self.t_follow: float | None = None
    self.accel_headroom = 0.0
    self.overspeed = 0.0

    self.read_params()

  def read_params(self) -> None:
    # speed params are entered in the user's display unit (mph or km/h)
    speed_to_ms = CV.KPH_TO_MS if self.params.get_bool("IsMetric") else CV.MPH_TO_MS
    self.enabled = self.params.get_bool("LaneChangeAssistEnabled")
    self.accel_param = min(max(self.params.get("LaneChangeAssistAccel", return_default=True), 0.0), MAX_ACCEL_HEADROOM)
    self.duration = max(self.params.get("LaneChangeAssistDuration", return_default=True), 0.0)
    self.min_t_follow = max(self.params.get("LaneChangeAssistMinTFollow", return_default=True), MIN_T_FOLLOW_FLOOR)
    self.overspeed_param = min(self.params.get("LaneChangeAssistOverspeed", return_default=True) * speed_to_ms, MAX_OVERSPEED)
    self.direction_mode = self.params.get("LaneChangeAssistDirection", return_default=True)
    self.min_speed = self.params.get("LaneChangeAssistMinSpeed", return_default=True) * speed_to_ms

  def update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.read_params()

  def _direction_allowed(self, direction: int) -> bool:
    if direction == DIR_LEFT:
      return True
    if direction == DIR_RIGHT:
      return self.direction_mode == BOTH
    return False

  @staticmethod
  def _blindspot_on(direction: int, left_blindspot: bool, right_blindspot: bool) -> bool:
    if direction == DIR_LEFT:
      return left_blindspot
    if direction == DIR_RIGHT:
      return right_blindspot
    return False

  def _reset_outputs(self) -> None:
    self.active = False
    self.t_follow = None
    self.accel_headroom = 0.0
    self.overspeed = 0.0

  def update(self, left_blinker: bool, right_blinker: bool, left_blindspot: bool, right_blindspot: bool,
             v_ego: float, lead_present: bool, t_follow_base: float) -> None:
    self.update_params()
    self.frame += 1

    if not self.enabled:
      self.qualifying_prev = False
      self.latched = False
      self.boost_timer = 0.0
      self.latched_direction = DIR_NONE
      self.engagement = 0.0
      self._reset_outputs()
      return

    single_blinker = left_blinker != right_blinker
    direction = DIR_LEFT if left_blinker else (DIR_RIGHT if right_blinker else DIR_NONE)
    qualifying = single_blinker and self._direction_allowed(direction)

    # Rising edge of a qualifying blinker latches a fresh boost window, but only
    # if there is a lead to close on, we are fast enough, and the target lane is clear.
    # The lead is checked only at the latch instant: once latched, the window
    # carries the boost through lead-loss so the pass completes.
    if qualifying and not self.qualifying_prev:
      speed_ok = v_ego >= self.min_speed
      bsm_clear = not self._blindspot_on(direction, left_blindspot, right_blindspot)
      if speed_ok and lead_present and bsm_clear and self.duration > 0.0:
        self.latched = True
        self.boost_timer = 0.0
        self.latched_direction = direction
    self.qualifying_prev = qualifying

    # Hard time cap from the rising edge: the blinker state no longer matters.
    boost_window = False
    if self.latched:
      self.boost_timer += DT_MDL
      boost_window = self.boost_timer < self.duration
      if not boost_window:
        # window expired -> release the latch so a later blinker can re-trigger
        self.latched = False
        self.latched_direction = DIR_NONE

    # Real-time safety suppression: speed (with hysteresis) and target-side BSM.
    speed_ok = v_ego >= (self.min_speed - SPEED_HYSTERESIS)
    bsm_clear = not self._blindspot_on(self.latched_direction, left_blindspot, right_blindspot)
    boost_active = boost_window and speed_ok and bsm_clear

    # Engagement ramp: brisk up, gentle down.
    if boost_active:
      self.engagement = min(self.engagement + DT_MDL / ENGAGE_TIME, 1.0)
    else:
      self.engagement = max(self.engagement - DT_MDL / RELEASE_TIME, 0.0)

    self.active = boost_active

    if self.engagement <= 0.0:
      self.engagement = 0.0
      self.latched_direction = DIR_NONE if not self.latched else self.latched_direction
      self._reset_outputs()
      return

    # Map engagement onto the three levers. The follow floor never exceeds the
    # personality baseline, so the gap can only shrink, never grow.
    t_follow_floor = min(self.min_t_follow, t_follow_base)
    self.t_follow = t_follow_base + (t_follow_floor - t_follow_base) * self.engagement
    self.accel_headroom = self.accel_param * self.engagement
    self.overspeed = self.overspeed_param * self.engagement
