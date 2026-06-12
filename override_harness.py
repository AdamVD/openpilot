#!/usr/bin/env python3
"""Bench validation of the gas-override handoff (branch override-handoff).

Drives the REAL Honda CarController (HONDA_ODYSSEY, Nidec ALT_PCM_ACCEL) through an
engage -> gas-press -> release sequence and checks, frame by frame, what actually goes
on the wire (pcm_speed in ACC_HUD, COMPUTER_BRAKE in BRAKE_COMMAND):

  1. engaged+longActive: pcm_speed live (vEgo + pcm_off)
  2. gas override (enabled, longActive=False, gasPressed): pcm_speed STAYS live
     (the whole point), brake command exactly 0
  3. release: pcm_speed continuous -- no reset-to-zero re-ramp
  4. fully disengaged: pcm_speed = 0
  5. control case: same sequence with gas_override logic forced off (longActive
     False + gasPressed but enabled False) -> zeros, like today

Run from the openpilot repo root with its opendbc_repo on PYTHONPATH.
"""
import numpy as np

from opendbc.car import structs
from opendbc.car.honda.values import CAR
from opendbc.car.honda.interface import CarInterface
from opendbc.can import CANPacker  # noqa: F401  (ensures native parser available)


def mk_interface():
  CP = CarInterface.get_non_essential_params(str(CAR.HONDA_ODYSSEY))
  CP_SP = CarInterface.get_non_essential_params_sp(CP, str(CAR.HONDA_ODYSSEY))
  return CarInterface(CP, CP_SP)


def mk_cc(enabled, long_active, accel):
  CC = structs.CarControl()
  CC.enabled = enabled
  CC.latActive = enabled
  CC.longActive = long_active
  CC.actuators = structs.CarControl.Actuators()
  CC.actuators.accel = accel
  CC.actuators.torque = 0.0
  CC.hudControl = structs.CarControl.HUDControl()
  CC.hudControl.setSpeed = 25.0
  CC.hudControl.speedVisible = True
  CC.cruiseControl = structs.CarControl.CruiseControl()
  CC.orientationNED = [0.0, 0.0, 0.0]
  return CC


def mk_ccsp():
  return structs.CarControlSP()


def run_seq(ci, seq):
  """seq: list of (n_frames, enabled, long_active, accel_cmd, gas_pressed).
  Returns per-frame (phase_idx, pcm_speed, brake_cmd) using the controller's own state."""
  ctrl = ci.CC  # CarController built by the interface
  CS = ci.CS
  # attrs normally populated from CAN during CS.update
  CS.stock_brake = {"CHIME": 0}
  CS.acc_hud = {"FCM_OFF": 0, "FCM_OFF_2": 0, "FCM_PROBLEM": 0, "ICONS": 0}
  CS.lkas_hud = {"LKAS_PROBLEM": 0}
  out = []
  vego = 15.0
  for phase, (n, enabled, long_active, accel, gaspr) in enumerate(seq):
    for _ in range(n):
      cs_out = structs.CarState()
      cs_out.vEgo = vego
      cs_out.aEgo = 0.0
      cs_out.gasPressed = gaspr
      cs_out.brakePressed = False
      cs_out.cruiseState = structs.CarState.CruiseState()
      CS.out = cs_out
      CC = mk_cc(enabled, long_active, accel)
      CC = CC.as_reader() if hasattr(CC, 'as_reader') else CC
      CC_SP = mk_ccsp()
      CC_SP = CC_SP.as_reader() if hasattr(CC_SP, 'as_reader') else CC_SP
      _, can_sends = ctrl.update(CC, CC_SP, CS, 0)
      # pull what we care about straight from the controller (== what was packed)
      out.append((phase, ctrl.speed, ctrl.brake))
  return np.array(out)


def main():
  ci = mk_interface()
  print(f"car: {ci.CP.carFingerprint}  interceptor: {ci.CP_SP.enableGasInterceptor}")

  # phases: engage(2s) -> gas override(3s) -> release/re-engage(2s) -> disengage(1s)
  seq = [
    (200, True, True, 0.5, False),    # 0: normal long active, asking +0.5
    (300, True, False, 0.4, True),    # 1: driver gas override (shadow FF 0.4 from controlsd)
    (200, True, True, 0.5, False),    # 2: release -> long active again
    (100, False, False, 0.0, False),  # 3: disengaged
  ]
  r = run_seq(ci, seq)
  for ph in range(4):
    m = r[r[:, 0] == ph]
    # controller updates self.speed only every 10th frame; use nonzero-aware stats
    print(f"phase {ph}: pcm_speed med {np.median(m[:, 1]):6.2f}  min {m[:, 1].min():6.2f} "
          f" max {m[:, 1].max():6.2f} | brake max {m[:, 2].max():.3f}")

  p0, p1, p2, p3 = (r[r[:, 0] == ph] for ph in range(4))
  assert np.median(p0[:, 1]) > 14.0, "phase0: active command should be ~vEgo+pcm_off"
  assert np.median(p1[100:, 1]) > 14.0, "phase1 FAIL: servo command must stay LIVE during gas override"
  assert p1[:, 2].max() == 0.0, "phase1 FAIL: brake must be exactly 0 during gas override"
  assert np.median(p2[:, 1]) > 14.0, "phase2: post-release command live"
  # continuity at release: first 0.5s of phase2 should not dip toward zero
  assert p2[:50, 1].min() > 13.0, "phase2 FAIL: command dipped at release (state was reset)"
  assert p3[20:, 1].max() == 0.0, "phase3: disengaged must command zero"

  # control: gas pressed but NOT enabled -> zeros (no shadow when disengaged)
  ci2 = mk_interface()
  r2 = run_seq(ci2, [(100, False, False, 0.4, True)])
  assert r2[20:, 1].max() == 0.0, "control FAIL: shadow must require enabled"
  print("control (gas pressed, not enabled): pcm_speed stays 0  OK")

  # RELEASE RACE (route 35 ep2, 2026-06-11): carState gasPressed falls one cycle before
  # controlsd's longActive comes back. In the stale frames neither longActive nor the
  # plain gas_override condition holds; without the linger the controller resets its
  # pcm_off slew state every release and, if the 10Hz ACC_HUD tick lands inside the
  # window, puts PCM_SPEED=0 on the wire (servo dump -> ~3s rebuild -> -0.65 sag).
  ci3 = mk_interface()
  r3 = run_seq(ci3, [
    (200, True, True, 0.5, False),   # 0: normal long active
    (300, True, False, 0.4, True),   # 1: gas override (settles ~vEgo+3.6)
    (5,   True, False, 0.4, False),  # 2: STALE window -- gas up, longActive not yet back
    (200, True, True, 0.5, False),   # 3: handback
  ])
  stale = r3[r3[:, 0] == 2]
  post = r3[r3[:, 0] == 3]
  print(f"race: stale-window pcm_speed min {stale[:, 1].min():6.2f} | "
        f"post-handback first 0.5s min {post[:50, 1].min():6.2f}")
  assert stale[:, 1].min() > 17.0, "race FAIL: PCM_SPEED zeroed/dropped in the stale release window"
  assert post[:50, 1].min() > 17.0, "race FAIL: slew state was reset at release (re-ramp from vEgo)"
  # brake released mid-linger must clear it immediately
  ci4 = mk_interface()
  ctrl4 = ci4.CC
  r4 = run_seq(ci4, [
    (200, True, True, 0.5, False),
    (300, True, False, 0.4, True),
  ])
  cs_brake = structs.CarState()
  cs_brake.vEgo = 15.0
  cs_brake.gasPressed = False
  cs_brake.brakePressed = True
  cs_brake.cruiseState = structs.CarState.CruiseState()
  ci4.CS.out = cs_brake
  ctrl4.update(mk_cc(False, False, 0.0).as_reader() if hasattr(structs.CarControl(), 'as_reader') else mk_cc(False, False, 0.0), mk_ccsp(), ci4.CS, 0)
  assert ctrl4.gas_override_linger == 0, "race FAIL: brake/disengage must clear the linger"
  print("race: linger cleared on brake/disengage  OK")

  print("\nALL CHECKS PASSED")


if __name__ == "__main__":
  main()
