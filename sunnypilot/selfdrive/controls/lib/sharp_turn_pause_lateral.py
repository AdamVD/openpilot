"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL

# Pause lateral control during sharp low-speed maneuvers the driver is doing by hand.
#
# Characterized on the 2018 Odyssey: LKAS torque tops out well below what parking-lot
# curvatures need (EPS assist knee ~1500-2000 motor counts, FINDINGS_lateral_torque_
# ceiling_2026-06-12.md), and the model never plans such paths. At extreme wheel angles
# with the driver hand-over-hand, the railed LKAS request contributes nothing toward the
# maneuver and only makes the wheel heavier / less consistent. Drop the REQUEST bit and
# give the wheel back.
#
# Trigger is maneuver-shaped and direction-agnostic: an extreme steering angle no
# lane-centering plan produces, real driver torque on the wheel, at parking speeds.
# Driver torque is required to ENTER the pause (low-speed lateral stays assisted when
# the driver is hands-off and demand is within limits) but not to MAINTAIN it — during
# a parking maneuver the wheel routinely slips through the hands while unwinding, and
# S-maneuvers swing through center between locks, so the pause holds while the wheel is
# still deep or moving fast and ends a grace period after the maneuver is over.

ENTRY_ANGLE = 120.0    # deg — beyond any lane-centering demand; intersection/parking territory
ENTRY_TORQUE = 600     # driver torque counts, either direction — clearly hand-on-wheel
MAX_SPEED = 9.0        # m/s (~20 mph) — parking / tight-maneuver regime
ENTRY_TIME = 0.2       # s of (leaky) sustained trigger before pausing
ENTRY_LEAK = 3.0       # debounce decays this much faster than it builds
KEEP_ANGLE = 45.0      # deg — below this (and slow wheel) the maneuver is ending
KEEP_RATE = 45.0       # deg/s — wheel still swinging (S-maneuver through center) keeps the pause
RELEASE_SPEED = 12.0   # m/s — driving out of the regime releases regardless
RELEASE_GRACE = 1.0    # s after the maneuver ends before lateral resumes


class SharpTurnPauseLateral:
  def __init__(self, dt: float = DT_CTRL):
    self.params = Params()
    self.dt = dt

    self.enabled = self.params.get_bool("SharpTurnPauseLateralControl")
    self.paused = False
    self.entry_timer = 0.0
    self.grace_timer = 0.0
    self._last_frame_time = -1

  def get_params(self) -> None:
    self.enabled = self.params.get_bool("SharpTurnPauseLateralControl")

  def update(self, CS, frame_time: int) -> bool:
    if not self.enabled:
      self.paused = False
      self.entry_timer = 0.0
      return False

    # get_lat_active is called more than once per cycle; only step state on a new carState
    if frame_time == self._last_frame_time:
      return self.paused
    self._last_frame_time = frame_time

    # thresholds are far above any plausible angle offset; raw angle is fine here
    angle = abs(CS.steeringAngleDeg)
    torque = abs(CS.steeringTorque)
    rate = abs(CS.steeringRateDeg)

    if not self.paused:
      arming = angle > ENTRY_ANGLE and torque > ENTRY_TORQUE and CS.vEgo < MAX_SPEED
      self.entry_timer = min(self.entry_timer + self.dt, ENTRY_TIME) if arming else \
                         max(self.entry_timer - ENTRY_LEAK * self.dt, 0.0)
      if self.entry_timer >= ENTRY_TIME:
        self.paused = True
        self.grace_timer = RELEASE_GRACE
    else:
      maintaining = (angle > KEEP_ANGLE or rate > KEEP_RATE) and CS.vEgo < RELEASE_SPEED
      self.grace_timer = RELEASE_GRACE if maintaining else self.grace_timer - self.dt
      if CS.vEgo >= RELEASE_SPEED or self.grace_timer <= 0.0:
        self.paused = False
        self.entry_timer = 0.0

    return self.paused
