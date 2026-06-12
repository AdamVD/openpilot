"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from cereal import car

from openpilot.sunnypilot.selfdrive.controls.lib.curve_exit_pause_lateral import (
  CurveExitPauseLateral, ENTRY_TIME, RELEASE_GRACE)

DT = 0.01


class TestCurveExitPauseLateral:

  def setup_method(self):
    self.cepl = CurveExitPauseLateral(dt=DT)
    self.cepl.enabled = True
    self.frame = 0

    self.CS = car.CarState.new_message()
    self.CS.vEgo = 10.0  # ~22 mph

  def _step(self, angle, torque, rate, n=1, v=None):
    """Run n control cycles with the given carState; returns last pause decision."""
    if v is not None:
      self.CS.vEgo = v
    self.CS.steeringAngleDeg = angle
    self.CS.steeringTorque = torque
    self.CS.steeringRateDeg = rate
    out = False
    for _ in range(n):
      self.frame += 1
      out = self.cepl.update(self.CS, 0.0, self.frame)
    return out

  def _arm_frames(self):
    return int(ENTRY_TIME / DT) + 2

  def test_exit_fight_pauses(self):
    # left curve (+30 deg), driver pulling right toward center (-1500), wheel unwinding (-20 deg/s)
    assert self._step(30.0, -1500, -20.0, n=self._arm_frames())

  def test_right_curve_exit_pauses(self):
    # right curve (-30 deg), driver pulling left toward center (+1500), unwinding (+20 deg/s)
    assert self._step(-30.0, 1500, 20.0, n=self._arm_frames())

  def test_collaborative_turn_in_never_pauses(self):
    # driver torque INTO the deepening left turn — must never fire
    assert not self._step(30.0, 2500, 25.0, n=200)

  def test_curve_hold_correction_never_pauses(self):
    # mid-curve correction toward center but wheel NOT unwinding (rate ~0) — curve hold stays assisted
    assert not self._step(30.0, -1500, 0.0, n=200)

  def test_small_angle_never_pauses(self):
    assert not self._step(8.0, -2000, -20.0, n=200)

  def test_highway_speed_never_pauses(self):
    assert not self._step(30.0, -1500, -20.0, n=200, v=25.0)  # ~56 mph

  def test_brief_blip_does_not_pause(self):
    assert not self._step(30.0, -1500, -20.0, n=int(ENTRY_TIME / DT) - 5)
    assert not self._step(30.0, 0, -5.0, n=50)

  def test_flicker_tolerant_arming(self):
    # torque dips below threshold for single frames mid-arm; leaky debounce still fires
    # (net build is dt - 3*dt/5 per 5-frame cycle, so allow ~3s like a real unwind)
    fired = False
    for i in range(300):
      torque = -1500 if i % 5 else -400
      fired = self._step(30.0, torque, -20.0) or fired
    assert fired

  def test_resume_on_press_into_turn(self):
    assert self._step(30.0, -1500, -20.0, n=self._arm_frames())
    # driver decides to keep the curve: torque back INTO the turn -> immediate resume
    assert not self._step(25.0, 800, 0.0)

  def test_resume_after_release_grace(self):
    assert self._step(30.0, -1500, -20.0, n=self._arm_frames())
    # driver releases mid-unwind: paused through grace, then resumes (op takes it back)
    assert self._step(20.0, 0, -5.0, n=int(RELEASE_GRACE / DT) - 2)
    assert not self._step(20.0, 0, -5.0, n=5)

  def test_pause_maintained_while_helping(self):
    assert self._step(30.0, -1500, -20.0, n=self._arm_frames())
    # keeps pause with modest torque all the way down to small angle
    assert self._step(15.0, -400, -15.0, n=100)
    # straightened out below KEEP_ANGLE -> resumes after grace
    assert not self._step(2.0, -400, -2.0, n=int(RELEASE_GRACE / DT) + 5)

  def test_same_frame_double_call_is_stable(self):
    self.CS.steeringAngleDeg = 30.0
    self.CS.steeringTorque = -1500
    self.CS.steeringRateDeg = -20.0
    for _ in range(self._arm_frames()):
      self.frame += 1
      first = self.cepl.update(self.CS, 0.0, self.frame)
      second = self.cepl.update(self.CS, 0.0, self.frame)  # controlsd calls twice per cycle
      assert first == second
    assert self.cepl.paused

  def test_disabled(self):
    self.cepl.enabled = False
    assert not self._step(30.0, -1500, -20.0, n=200)

  def test_angle_offset_applied(self):
    # raw angle 10 deg with -5 offset = 15 effective -> can arm; with +5 offset = 5 -> cannot
    for _ in range(self._arm_frames()):
      self.frame += 1
      self.CS.steeringAngleDeg = 16.0
      self.CS.steeringTorque = -1500
      self.CS.steeringRateDeg = -20.0
      paused = self.cepl.update(self.CS, 5.0, self.frame)  # effective 11 deg < ENTRY_ANGLE
    assert not paused
