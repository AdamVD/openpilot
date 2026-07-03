"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL

# Pause lateral control while the driver actively straightens out of a sharp curve.
#
# Characterized on the 2018 Odyssey (FINDINGS_lateral_exit_fight_2026-06-12.md): during
# driver-helped exits from saturated slow curves, the EPS executes ~none of the railed
# toward-center LKAS command (motor torque ~8 vs ~56 counts) and the LKAS-active state
# roughly doubles the wheel effort vs manual at matched angle/unwind rate. Dropping the
# LKAS REQUEST bit restores normal manual steering feel for the unwind.
#
# Trigger is exit-phase shaped — driver torque TOWARD CENTER while the wheel is actually
# unwinding at a meaningful angle. It cannot fire during collaborative turn-in (driver
# torque points into the turn there) and a rate gate keeps it from firing on in-curve
# corrections where the wheel is not unwinding (curve hold stays assisted).
#
# Replay-validated on routes 2e/2f/32/33/35: 11/12 known fighting-exits caught, zero
# pauses while holding a curve, pauses end <=0.6 s after the driver releases or
# immediately if the driver presses back into the turn.
#
# Thresholds softened 2026-07-02 from 600/250/0.6 after on-road feedback: light guiding
# needed "very confident moves" to fire and then grab/released chattily. Measured driver
# torque during lat-active unwinds over 3.9 h (26 routes): p25=328 / p50=691 / p10=175 —
# entry 600 missed half of real guiding and KEEP 250 sat above the light-touch floor, so
# the pause couldn't hold once the wheel went light (relay oscillation). Entry now
# catches ~75% of guiding torque, KEEP sits under the p10 floor, and a longer grace
# bridges torque dips. Cost in replay: no-unwind fires 3->8 per 3.9 h (rate gate still
# guards curve-hold).

ENTRY_TORQUE = 350     # driver torque counts toward center to arm
ENTRY_ANGLE = 12.0     # deg, offset-corrected steering angle magnitude
ENTRY_RATE = 2.0       # deg/s unwind rate toward center
ENTRY_TIME = 0.24      # s of (leaky) sustained trigger before pausing
ENTRY_LEAK = 3.0       # debounce decays this much faster than it builds
MIN_SPEED = 3.0        # m/s
MAX_SPEED = 18.0       # m/s (~40 mph) — phenomenon is a slow sharp-curve one
KEEP_TORQUE = 100      # counts toward center to maintain the pause (under light-guide p10)
KEEP_ANGLE = 4.0       # deg, below this the maneuver is over
RELEASE_GRACE = 1.2    # s after the driver releases before lateral resumes
RESUME_TORQUE = 400    # counts back INTO the turn -> resume immediately (driver wants the curve held)


class CurveExitPauseLateral:
  def __init__(self, dt: float = DT_CTRL):
    self.params = Params()
    self.dt = dt

    self.enabled = self.params.get_bool("CurveExitPauseLateralControl")
    self.paused = False
    self.entry_timer = 0.0
    self.grace_timer = 0.0
    self._last_frame_time = -1

  def get_params(self) -> None:
    self.enabled = self.params.get_bool("CurveExitPauseLateralControl")

  def update(self, CS, angle_offset_deg: float, frame_time: int) -> bool:
    if not self.enabled:
      self.paused = False
      self.entry_timer = 0.0
      return False

    # get_lat_active is called more than once per cycle; only step state on a new carState
    if frame_time == self._last_frame_time:
      return self.paused
    self._last_frame_time = frame_time

    angle = CS.steeringAngleDeg - angle_offset_deg
    sgn = np.sign(angle) if abs(angle) > 1e-3 else 0.0
    torque_to_center = -CS.steeringTorque * sgn       # positive = driver pushing toward center
    unwind_rate = -CS.steeringRateDeg * sgn           # positive = angle magnitude shrinking

    if not self.paused:
      arming = (torque_to_center > ENTRY_TORQUE and abs(angle) > ENTRY_ANGLE and
                unwind_rate > ENTRY_RATE and MIN_SPEED < CS.vEgo < MAX_SPEED)
      self.entry_timer = min(self.entry_timer + self.dt, ENTRY_TIME) if arming else \
                         max(self.entry_timer - ENTRY_LEAK * self.dt, 0.0)
      if self.entry_timer >= ENTRY_TIME:
        self.paused = True
        self.grace_timer = RELEASE_GRACE
    else:
      maintaining = torque_to_center > KEEP_TORQUE and abs(angle) > KEEP_ANGLE
      resume_into_turn = -torque_to_center > RESUME_TORQUE
      self.grace_timer = RELEASE_GRACE if maintaining else self.grace_timer - self.dt
      if resume_into_turn or self.grace_timer <= 0.0:
        self.paused = False
        self.entry_timer = 0.0

    return self.paused
