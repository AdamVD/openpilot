"""Phantom-lead gate in `radard.match_vision_to_track` (2026-07-25, variant V1b-1S(5)).

Beyond ~50 m the Odyssey's NIDEC radar drops and re-creates points constantly (Track
lifetime 1-4 frames), so the vision-to-radar matcher routinely locks onto a DIFFERENT
vehicle. The worst recorded case accepted a track at 74 m / vLead 20.1 m/s while the
vision lead read 91 m / 29.7 m/s, and the planner's aTarget went -0.21 -> -1.24 m/s^2 in
0.4 s. The gate rejects the track only when falling back to the vision lead RELAXES the
braking demand, so it can never invent a brake event.

The three properties that had to hold for it to ship, one test each:
  * below `PHANTOM_LEAD_GATE_MIN_DIST` the path is bit-identical to stock,
  * it never rejects when the vision fallback is MORE braking-demanding,
  * `PHANTOM_LEAD_GATE = False` is an exact revert.
"""
import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls import radard
from openpilot.selfdrive.controls.radard import (RADAR_TO_CAMERA, KalmanParams, Track,
                                                 get_lead, match_vision_to_track)

from opendbc.car import structs


class FakeLead:
  """The `leadsV3[i]` fields `match_vision_to_track` / `get_lead` read."""

  def __init__(self, x, v, y=0.0, prob=0.9, x_std=5.0, y_std=1.0, v_std=1.2, a=0.0):
    self.x = [x]
    self.y = [y]
    self.v = [v]
    self.a = [a]
    self.xStd = [x_std]
    self.yStd = [y_std]
    self.vStd = [v_std]
    self.prob = prob


def make_track(d_rel, v_lead, v_ego, y_rel=0.0, identifier=1, n=3):
  t = Track(identifier, v_lead, KalmanParams(DT_MDL))
  for _ in range(n):
    t.update(d_rel, y_rel, v_lead - v_ego, v_lead, 1.0)
  return t


def x_obst(dist, v):
  """The planner's stopping-equivalence ranking of an obstacle (long_mpc)."""
  return dist + max(v, 0.0) ** 2 / radard.PHANTOM_LEAD_GATE_2CB


@pytest.fixture
def gate_off(monkeypatch):
  monkeypatch.setattr(radard, 'PHANTOM_LEAD_GATE', False)


def _cp():
  CP, CP_SP = structs.CarParams(), structs.CarParamsSP()
  CP.brand = 'honda'
  return CP, CP_SP


# ---------------------------------------------------------------------------------------
# it fires on the recorded phantom
# ---------------------------------------------------------------------------------------

class TestPhantomRejected:
  def test_worst_recorded_burst_is_rejected(self):
    """e9 t=14313.5: radar 74 m / vLead 20.1, vision 91 m / 29.7, v_ego 30.6."""
    v_ego = 30.6
    track = make_track(74.0, 20.1, v_ego)
    lead = FakeLead(x=91.0 + RADAR_TO_CAMERA, v=29.7)
    assert match_vision_to_track(v_ego, lead, {1: track}) is None
    # ... and the vision fallback really is the less braking-demanding obstacle
    assert x_obst(91.0, 29.7) > x_obst(74.0, 20.1)

  def test_rejection_falls_through_to_the_vision_lead_not_to_nothing(self):
    """Returning None must not blank the lead: get_lead() falls through to vision."""
    CP, CP_SP = _cp()
    v_ego = 30.6
    track = make_track(74.0, 20.1, v_ego)
    lead = FakeLead(x=91.0 + RADAR_TO_CAMERA, v=29.7)
    out = get_lead(v_ego, True, {1: track}, lead, v_ego, CP, CP_SP, low_speed_override=False)
    assert out['status'] is True
    assert out['radar'] is False
    assert out['dRel'] == pytest.approx(91.0)

  def test_gate_applies_to_both_radar_state_leads(self):
    """radard calls get_lead() twice and long_mpc takes the min of both obstacles, so a
    filter that only covered leadOne would be a bit-exact no-op. The gate lives inside
    match_vision_to_track, which both calls go through."""
    CP, CP_SP = _cp()
    v_ego = 30.6
    tracks = {1: make_track(74.0, 20.1, v_ego)}
    lead = FakeLead(x=91.0 + RADAR_TO_CAMERA, v=29.7)
    one = get_lead(v_ego, True, tracks, lead, v_ego, CP, CP_SP, low_speed_override=True)
    two = get_lead(v_ego, True, tracks, lead, v_ego, CP, CP_SP, low_speed_override=False)
    assert one['radar'] is False and two['radar'] is False
    assert one['dRel'] == pytest.approx(two['dRel'])


# ---------------------------------------------------------------------------------------
# it must never make braking more likely, and never act at close range
# ---------------------------------------------------------------------------------------

class TestGateIsSafe:
  def test_no_stock_accepted_match_can_be_rejected_below_37m(self):
    """The gate keys on the VISION distance, so state the radar-side bound it implies.

    Among matches stock would have ACCEPTED, `dist_sane` forces
    dRel > 0.75 * offset_vision_dist, and the gate needs offset_vision_dist > 50 m, so no
    radar track closer than 37.5 m can ever be rejected. (Cases stock rejects anyway are
    not interesting -- the gate only ever returns None where stock could also return None.)
    """
    v_ego = 30.0
    rng = np.random.default_rng(53)
    worst = 1e9
    fired = 0
    for _ in range(6000):
      d = float(rng.uniform(2.0, 220.0))
      d_vis = float(rng.uniform(2.0, 220.0))
      v_track, v_vis = float(rng.uniform(0.0, 45.0)), float(rng.uniform(0.0, 45.0))
      stock_ok = (abs(d - d_vis) < max(d_vis * .25, 5.0)) and \
                 ((abs(v_track - v_vis) < 10) or (v_track > 3))
      if not stock_ok:
        continue
      tracks = {1: make_track(d, v_track, v_ego)}
      lead = FakeLead(x=d_vis + RADAR_TO_CAMERA, v=v_vis, x_std=80.0, v_std=40.0)
      if match_vision_to_track(v_ego, lead, tracks) is None:
        worst = min(worst, d)
        fired += 1
    assert fired > 0
    assert worst > 0.75 * radard.PHANTOM_LEAD_GATE_MIN_DIST, worst

  def test_below_min_dist_is_a_no_op(self, monkeypatch):
    """Close range is believed immediately -- structurally, not by tuning."""
    v_ego = 30.0
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(3000):
      d = float(rng.uniform(2.0, radard.PHANTOM_LEAD_GATE_MIN_DIST))
      v_track = float(rng.uniform(0.0, 45.0))
      v_vis = float(rng.uniform(0.0, 45.0))
      # keep the vision distance below the gate threshold too (that is what the gate reads)
      d_vis = float(rng.uniform(2.0, radard.PHANTOM_LEAD_GATE_MIN_DIST))
      tracks = {1: make_track(d, v_track, v_ego)}
      lead = FakeLead(x=d_vis + RADAR_TO_CAMERA, v=v_vis, x_std=30.0, v_std=20.0)
      with monkeypatch.context() as m:
        m.setattr(radard, 'PHANTOM_LEAD_GATE', False)
        ref = match_vision_to_track(v_ego, lead, tracks)
      got = match_vision_to_track(v_ego, lead, tracks)
      assert got is ref, (d, d_vis, v_track, v_vis)
      checked += 1
    assert checked == 3000

  def test_never_rejects_when_vision_is_more_braking_demanding(self):
    """One-sided by construction: reject only if x_obst(vision) > x_obst(track)."""
    v_ego = 30.0
    rng = np.random.default_rng(23)
    rejections = 0
    for _ in range(5000):
      d = float(rng.uniform(50.0, 200.0))
      d_vis = float(rng.uniform(50.0, 200.0))
      v_track = float(rng.uniform(0.0, 45.0))
      v_vis = float(rng.uniform(0.0, 45.0))
      tracks = {1: make_track(d, v_track, v_ego)}
      lead = FakeLead(x=d_vis + RADAR_TO_CAMERA, v=v_vis, x_std=60.0, v_std=30.0)
      if match_vision_to_track(v_ego, lead, tracks) is None:
        # the only legal reason to return None with the gate on is that the swap relaxes
        # the demand, OR that stock's own dist/vel sanity check already rejected it
        stock_dist_sane = abs(d - d_vis) < max(d_vis * .25, 5.0)
        stock_vel_sane = (abs(v_track - v_vis) < 10) or (v_track > 3)
        if stock_dist_sane and stock_vel_sane:
          assert x_obst(d_vis, v_vis) > x_obst(d, v_track), (d, v_track, d_vis, v_vis)
          rejections += 1
    assert rejections > 0, "no gate rejections in the sweep -- test is blind"

  def test_matched_track_is_never_swapped_for_a_different_track(self):
    """The gate can only reject; it never picks a different radar point."""
    v_ego = 30.0
    rng = np.random.default_rng(31)
    for _ in range(500):
      tracks = {i: make_track(float(rng.uniform(50.0, 180.0)), float(rng.uniform(0.0, 45.0)),
                              v_ego, identifier=i) for i in range(1, 5)}
      lead = FakeLead(x=float(rng.uniform(50.0, 180.0)) + RADAR_TO_CAMERA,
                      v=float(rng.uniform(0.0, 45.0)), x_std=40.0, v_std=25.0)
      got = match_vision_to_track(v_ego, lead, tracks)
      assert got is None or got in tracks.values()


# ---------------------------------------------------------------------------------------
# exact revert + footprint
# ---------------------------------------------------------------------------------------

class TestExactRevert:
  def _sweep(self, v_ego=30.0, n=4000, seed=41):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
      cases.append((float(rng.uniform(2.0, 220.0)), float(rng.uniform(2.0, 220.0)),
                    float(rng.uniform(0.0, 45.0)), float(rng.uniform(0.0, 45.0))))
    out = []
    for d, d_vis, v_track, v_vis in cases:
      tracks = {1: make_track(d, v_track, v_ego)}
      lead = FakeLead(x=d_vis + RADAR_TO_CAMERA, v=v_vis, x_std=80.0, v_std=40.0)
      out.append(match_vision_to_track(v_ego, lead, tracks) is not None)
    return np.array(out)

  def test_gate_off_is_exactly_stock(self, monkeypatch):
    with monkeypatch.context() as m:
      m.setattr(radard, 'PHANTOM_LEAD_GATE', False)
      off = self._sweep()
    # stock, recomputed independently from the documented sanity rules
    rng = np.random.default_rng(41)
    ref = []
    for _ in range(4000):
      d, d_vis = float(rng.uniform(2.0, 220.0)), float(rng.uniform(2.0, 220.0))
      v_track, v_vis = float(rng.uniform(0.0, 45.0)), float(rng.uniform(0.0, 45.0))
      dist_sane = abs(d - d_vis) < max(d_vis * .25, 5.0)
      vel_sane = (abs(v_track - v_vis) < 10) or (v_track > 3)
      ref.append(bool(dist_sane and vel_sane))
    assert np.array_equal(off, np.array(ref))

  def test_gate_on_rejects_a_small_extra_share(self, monkeypatch):
    on = self._sweep()
    with monkeypatch.context() as m:
      m.setattr(radard, 'PHANTOM_LEAD_GATE', False)
      off = self._sweep()
    extra = int((off & ~on).sum())
    assert extra > 0, "the gate never fires on the sweep -- it would be dead code"
    assert int((on & ~off).sum()) == 0, "the gate accepted a track stock rejected"

  def test_constants_are_the_scored_ones(self):
    assert radard.PHANTOM_LEAD_GATE is True
    assert radard.PHANTOM_LEAD_GATE_MIN_DIST == 50.
    assert radard.PHANTOM_LEAD_GATE_DV == 5.
    # must stay equal to 2 * COMFORT_BRAKE in the planner, or the gate ranks obstacles
    # differently from the MPC it is protecting
    from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_BRAKE
    assert radard.PHANTOM_LEAD_GATE_2CB == 2 * COMFORT_BRAKE
