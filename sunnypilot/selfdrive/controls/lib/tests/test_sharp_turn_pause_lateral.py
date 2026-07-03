"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from cereal import car

from openpilot.sunnypilot.selfdrive.controls.lib.sharp_turn_pause_lateral import (
  SharpTurnPauseLateral, ENTRY_TIME, RELEASE_GRACE)

DT = 0.01


class TestSharpTurnPauseLateral:

  def setup_method(self):
    self.stpl = SharpTurnPauseLateral(dt=DT)
    self.stpl.enabled = True
    self.frame = 0

    self.CS = car.CarState.new_message()
    self.CS.vEgo = 3.0  # parking-lot creep

  def _step(self, angle, torque, rate=0.0, n=1, v=None):
    """Run n control cycles with the given carState; returns last pause decision."""
    if v is not None:
      self.CS.vEgo = v
    self.CS.steeringAngleDeg = angle
    self.CS.steeringTorque = torque
    self.CS.steeringRateDeg = rate
    out = False
    for _ in range(n):
      self.frame += 1
      out = self.stpl.update(self.CS, self.frame)
    return out

  def _arm_frames(self):
    return int(ENTRY_TIME / DT) + 2

  def test_parking_turn_pauses(self):
    # deep right lock, driver hand on wheel, creeping
    assert self._step(-300.0, -1200, -50.0, n=self._arm_frames())

  def test_left_parking_turn_pauses(self):
    assert self._step(300.0, 1200, 50.0, n=self._arm_frames())

  def test_torque_direction_agnostic(self):
    # holding full lock the driver may be pulling either way; both enter
    assert self._step(300.0, -1200, 0.0, n=self._arm_frames())

  def test_hands_off_sharp_angle_never_pauses(self):
    # wheel deep but driver not on it -> lateral stays active (Adam's low-speed-active clause)
    assert not self._step(300.0, 100, 0.0, n=200)

  def test_moderate_angle_never_pauses(self):
    # sharp-ish curve within lat control's working range
    assert not self._step(60.0, 1500, 20.0, n=200)

  def test_road_speed_never_pauses(self):
    assert not self._step(150.0, 1500, 20.0, n=200, v=15.0)  # ~34 mph, not a parking regime

  def test_brief_blip_does_not_pause(self):
    assert not self._step(300.0, 1200, 0.0, n=int(ENTRY_TIME / DT) - 5)
    assert not self._step(300.0, 0, 0.0, n=50)

  def test_pause_survives_torque_release(self):
    # wheel slipping through the hands while unwinding: no torque, still deep -> stay paused
    assert self._step(300.0, 1200, 0.0, n=self._arm_frames())
    assert self._step(200.0, 0, -80.0, n=300)

  def test_pause_survives_s_maneuver_through_center(self):
    # swinging lock-to-lock: angle crosses zero fast; rate keeps the pause alive
    assert self._step(300.0, 1200, 0.0, n=self._arm_frames())
    assert self._step(20.0, 800, 200.0, n=100)

  def test_resume_after_maneuver_ends(self):
    assert self._step(300.0, 1200, 0.0, n=self._arm_frames())
    # straightened out, wheel settled -> resumes after grace
    assert self._step(10.0, 0, 5.0, n=int(RELEASE_GRACE / DT) - 2)
    assert not self._step(10.0, 0, 5.0, n=5)

  def test_hard_release_on_speed(self):
    assert self._step(300.0, 1200, 0.0, n=self._arm_frames())
    # driving off while wheel still winding down: speed release wins immediately
    assert not self._step(100.0, 0, -60.0, v=13.0)

  def test_same_frame_double_call_is_stable(self):
    self.CS.steeringAngleDeg = 300.0
    self.CS.steeringTorque = 1200
    self.CS.steeringRateDeg = 0.0
    for _ in range(self._arm_frames()):
      self.frame += 1
      first = self.stpl.update(self.CS, self.frame)
      second = self.stpl.update(self.CS, self.frame)  # controlsd calls twice per cycle
      assert first == second
    assert self.stpl.paused

  def test_disabled(self):
    self.stpl.enabled = False
    assert not self._step(300.0, 1200, 0.0, n=200)
