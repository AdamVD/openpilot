"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Tests for Passing Assist (longitudinal lane-change boost). These exercise the
state machine in isolation: rising-edge latch, the hard time cap from the
blinker (independent of the blinker staying on), real-time BSM/speed
suppression, lead-required-at-trigger (held through lead loss), direction
gating, the follow-gap floor, and that the feature is inert when disabled.
"""
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.passing_assist import (
  PassingAssist, LEFT_ONLY, BOTH, DIR_LEFT, DIR_RIGHT)

DT = 0.05
HWY = 60 * CV.MPH_TO_MS
SLOW = 40 * CV.MPH_TO_MS
BASE = 1.45  # standard personality t_follow


def _make(**overrides):
  params = Params()
  params.put_bool("IsMetric", False)  # pin imperial so mph thresholds are deterministic
  defaults = {
    "LaneChangeAssistEnabled": True,
    "LaneChangeAssistAccel": 0.4,
    "LaneChangeAssistDuration": 7,
    "LaneChangeAssistMinTFollow": 1.1,
    "LaneChangeAssistOverspeed": 2,
    "LaneChangeAssistDirection": LEFT_ONLY,
    "LaneChangeAssistMinSpeed": 45,
  }
  defaults.update(overrides)
  for k, v in defaults.items():
    if isinstance(v, bool):
      params.put_bool(k, v)
    else:
      params.put(k, v)
  return PassingAssist()


def _step(pla, secs, **kw):
  args = dict(left_blinker=False, right_blinker=False, left_blindspot=False,
             right_blindspot=False, v_ego=HWY, lead_present=True, t_follow_base=BASE)
  args.update(kw)
  for _ in range(max(1, round(secs / DT))):
    pla.update(**args)


def _prime(pla, **kw):
  """One update with blinkers off to establish the falling baseline before a rising edge."""
  args = dict(left_blinker=False, right_blinker=False, left_blindspot=False,
             right_blindspot=False, v_ego=HWY, lead_present=True, t_follow_base=BASE)
  args.update(kw)
  pla.update(**args)


class TestPassingAssist:
  def test_disabled_is_inert(self):
    pla = _make(LaneChangeAssistEnabled=False)
    _step(pla, 1.0, left_blinker=True)
    assert pla.t_follow is None
    assert pla.accel_headroom == 0.0
    assert pla.overspeed == 0.0
    assert not pla.active

  def test_engage_left(self):
    pla = _make()
    _prime(pla)
    _step(pla, 1.0, left_blinker=True)
    assert pla.active
    assert pla.t_follow is not None and pla.t_follow < BASE
    assert pla.t_follow >= 1.1 - 1e-9
    assert pla.accel_headroom > 0.0
    assert pla.overspeed > 0.0
    assert pla.latched_direction == DIR_LEFT
    # after a full engage ramp the levers reach their configured magnitudes
    assert abs(pla.t_follow - 1.1) < 1e-6
    assert abs(pla.accel_headroom - 0.4) < 1e-6
    assert abs(pla.overspeed - 2 * CV.MPH_TO_MS) < 1e-6

  def test_hard_cap_with_blinker_held(self):
    # Boost must end after `duration` even if the blinker is never canceled.
    pla = _make(LaneChangeAssistDuration=3)
    _prime(pla)
    _step(pla, 2.0, left_blinker=True)
    assert pla.active
    _step(pla, 3.5, left_blinker=True)  # past the 3s window + relax ramp
    assert not pla.active
    assert pla.t_follow is None

  def test_persists_after_blinker_off(self):
    # Hard cap is measured from the rising edge; canceling the blinker does not end it.
    pla = _make(LaneChangeAssistDuration=5)
    _prime(pla)
    _prime(pla, left_blinker=True)  # rising edge
    _step(pla, 2.0, left_blinker=False)
    assert pla.active

  def test_no_lead_no_engage(self):
    pla = _make()
    _prime(pla, lead_present=False)
    _step(pla, 1.0, left_blinker=True, lead_present=False)
    assert not pla.active
    assert pla.t_follow is None

  def test_lead_lost_after_latch_continues(self):
    pla = _make()
    _prime(pla)
    _prime(pla, left_blinker=True)  # latch with lead present
    _step(pla, 1.0, left_blinker=True, lead_present=False)
    assert pla.active

  def test_below_min_speed_no_engage(self):
    pla = _make()
    _prime(pla, v_ego=SLOW)
    _step(pla, 1.0, left_blinker=True, v_ego=SLOW)
    assert not pla.active

  def test_bsm_at_edge_no_engage(self):
    pla = _make()
    _prime(pla, left_blindspot=True)
    _step(pla, 1.0, left_blinker=True, left_blindspot=True)
    assert not pla.active

  def test_bsm_mid_boost_suppresses(self):
    pla = _make()
    _prime(pla)
    _step(pla, 1.0, left_blinker=True)
    assert pla.active
    _step(pla, 1.0, left_blinker=True, left_blindspot=True)
    assert not pla.active
    assert pla.engagement < 1.0

  def test_speed_drop_mid_boost_suppresses(self):
    pla = _make()
    _prime(pla)
    _step(pla, 1.0, left_blinker=True)
    _step(pla, 1.0, left_blinker=True, v_ego=SLOW)
    assert not pla.active

  def test_right_blinker_left_only(self):
    pla = _make(LaneChangeAssistDirection=LEFT_ONLY)
    _prime(pla)
    _step(pla, 1.0, right_blinker=True)
    assert not pla.active

  def test_right_blinker_both(self):
    pla = _make(LaneChangeAssistDirection=BOTH)
    _prime(pla)
    _step(pla, 1.0, right_blinker=True)
    assert pla.active
    assert pla.latched_direction == DIR_RIGHT

  def test_floor_never_exceeds_base(self):
    # min_t_follow param above the (aggressive) base must not grow the gap.
    pla = _make(LaneChangeAssistMinTFollow=1.6)
    _prime(pla, t_follow_base=1.25)
    _step(pla, 1.0, left_blinker=True, t_follow_base=1.25)
    assert pla.t_follow is not None and pla.t_follow <= 1.25 + 1e-9

  def test_hazards_no_engage(self):
    pla = _make()
    _prime(pla)
    _step(pla, 1.0, left_blinker=True, right_blinker=True)
    assert not pla.active
