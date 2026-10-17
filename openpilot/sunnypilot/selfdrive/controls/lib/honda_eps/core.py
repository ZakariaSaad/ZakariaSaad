"""Honda Insight modified-EPS inversion, adapted from NRDR PR #22 (5777cf9).

The public 4096-count calibration preserves NRDR's transport constants. The active
core uses the explicit 3840-count canonical calibration below, with JamesL787
bfcbbf24bc857f9dbf93af52b7bb72852f4540bb's exact P lookup and post-override I decay.
The wrapper converts the COMPLETE output to MVL's 4096-count transport, preserving
physical feedforward and feedback gains. Clarity defaults remain only as source
numerical reference; vehicle admission supports the corrected Insight image only.

The firmware maps E4 to a target R5, subtracts rate feedback R6, and applies
P(R5) plus KFF. A measured column-load model predicts motor demand, the exact
piecewise quadratic inverse solves for R5, and the command map returns the E4
request. Residual-angle PID feedback closes the loop. Image identity, geometry
and transport normalization are separate contracts; an EPS part number alone
does not confirm the corrected firmware image.

"""
import math

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.geometry import NRDR_CLARITY_VGR_ANGLE_BP, NRDR_CLARITY_VGR_LINEAR_BP
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.integral_pid import IntegralScaledPIDController as PIDController

# 0xE4 value per unit of lateral output: torqueBP/V = [0, 3840] identity, apply_torque = -u * 3840
E4_PER_OUTPUT = 3840.0

# command map, row 0 (0x13810 / 0x1388E); key = trunc(trunc(E4 * 56756 / 32768) / 4) with the no-SHLL2 decode
R5_KEY_BP = [0, 111, 222, 333, 443, 665, 887, 1108, 1663]
R5_V = [0, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 30000]
KEY_CLAMP = 1663

# FUN_29C14 speed envelope: a CEILING on |key| (0x13660 / 0x136DE), speed axis in 0.5 km/h counts.
# 1774 up to 100 km/h, so the 1663 key clamp binds first until ~107 km/h; 1330 from 130 km/h.
ENVELOPE_BP = [0, 50, 100, 150, 200, 260, 300, 350, 400]
ENVELOPE_V = [1774, 1774, 1774, 1774, 1774, 1330, 1330, 1330, 1330]

# P row 0 (0x13B5E / 0x13BDC), indexed by the command key; project-added feedforward KFF45
KP_KEY_BP = [0, 222, 443, 665, 887, 1108, 1330, 1552, 1774]
KP_V = [117, 148, 184, 220, 245, 257, 263, 265, 265]
KFF = 45.0
R5_PER_KEY = 18.04
# the same P row as pieces over |R5|: (lo, hi, Kp at lo, dKp/d|R5|), flat past the last breakpoint
KP_PIECES = [(lo * R5_PER_KEY, hi * R5_PER_KEY, kp_lo, (kp_hi - kp_lo) / ((hi - lo) * R5_PER_KEY))
             for lo, hi, kp_lo, kp_hi in zip(KP_KEY_BP[:-1], KP_KEY_BP[1:], KP_V[:-1], KP_V[1:], strict=True)]
KP_PIECES.append((KP_KEY_BP[-1] * R5_PER_KEY, math.inf, KP_V[-1], 0.0))

# R6 comes from the column rate BEFORE the firmware's angle tables, the domain 0x18F STEER_ANGLE_RATE reports
# (R6 = -31.6 per 0x18F count, flat to 2% at every angle on route 369). The feedforward's rate is the derivative
# of the published (0x14A) angle, which the A (position) table compresses, so per deg/s of it the damping grows
# with angle by the A table's local slope (the chain rule): engaged, routes 363/365/366/369, -119 near center
# and -141..-146 past 60 deg, where a single -133 was 11% too strong near center and 10% too weak in big turns.
# Scaled by the slope it is flat at -120..-123 on each route (-122 pooled).
# steeringRateDeg (0x14A STEER_ANGLE_RATE) is NOT that derivative: the firmware publishes it through its second,
# B (rate) table, indexed by angle. It reads 1.1% fast at center (B[0] 16204 vs 16384) and 2-5% slow at 45-100
# deg, so don't feed it here without converting (see steer_ratio.py).
R6_PER_CENTRE_DEG_S = -122.0  # NORM 1650 / tracker-1 3200
# d(pre-table angle) / d(published angle) along the A020 angle table, 1.0 at center, ~1.19 from 150 deg
_VGR_SLOPE = np.gradient(NRDR_CLARITY_VGR_LINEAR_BP, NRDR_CLARITY_VGR_ANGLE_BP)
R6_ANGLE_BP = np.asarray(NRDR_CLARITY_VGR_ANGLE_BP)
R6_ANGLE_GAIN = _VGR_SLOPE / _VGR_SLOPE[0]
# 0x6A2 scale word while the request is held, re-measured on routes 35e/360/361: median 256 (252 on 352). Hands
# off it only moves on a fast column-torque RATE (A280) or above ~400 counts of torque (helper A).
SCALE_Q8 = 256.0       # helper A * B / 256 while the request is held

# Column load model, firmware output counts (A030 sign convention), fitted on route 00000352:
#   load = k0*th + k1*th*v^2 + c*thd + fr*tanh(thd/w) + bias + kroll*roll*v^2
# th/thd: steering-wheel deg, deg/s; v: m/s; roll: rad. R^2 0.82 in-route, 0.72 on the held-out route.
LOAD_K0 = -7.00387
LOAD_K1 = -0.21857
LOAD_C = -6.8317
LOAD_FRICTION = -314.07279
LOAD_BIAS = 20.46793
LOAD_KROLL = -7.20003
# The fit's friction width is 2 deg/s; 5 deg/s keeps the Coulomb term from flipping on desired-rate
# noise near straight driving (command roughness 0.0029 -> 0.0020 in replay, tracking nearly unchanged).
# Keyed on the DESIRED rate, even 5 deg/s turns the model's small path wiggles into a friction square wave
# in the city: closed-loop sim, a +/-2 deg 1 Hz wiggle at 10 m/s comes out at the wheel with gain 1.40; with
# 20 deg/s 0.96, and turn-ins (30-200 deg/s) keep the full friction. The price is lag on slow, small motions,
# which only pays off at city speed: replaying hands-off chunks of routes 35e/360/361, 20 deg/s cut the 0.6-2 Hz
# wheel/target transfer 0.93 -> 0.73 at 10-16 m/s for +25 ms, but above 16 m/s bought nothing (1.02 -> 0.97) for
# +45 ms of lag. So 20 deg/s up to 8 m/s, back to 5 by 15 m/s (10-16 m/s: 0.76 at +10 ms; above: unchanged).
FRICTION_WIDTH_DEG_S = 5.0
FRICTION_WIDTH_SPEED_BP = [8.0, 15.0]  # m/s
FRICTION_WIDTH_V = [20.0, FRICTION_WIDTH_DEG_S]  # deg/s


def friction_width(v_ego: float) -> float:
  _finite(v_ego)
  return float(np.interp(v_ego, FRICTION_WIDTH_SPEED_BP, FRICTION_WIDTH_V))


# Smoothing, picked in the closed-loop replay of route 352 over three plants (nominal, the 353 fit, +15 ms
# motor lag): unsmoothed, the command carried 3.8x vfn's 5-8 Hz content (the Clarity column's stutter band);
# with these it is 1.14x, for RMS error below 25 mph 2.3 deg vs vfn's 4.7, turn-in lag 10-20 ms vs 250 and
# exits 50 ms vs 130. A longer lead bought back turn-in but started 25-50 mph exits early, so it stays 0.07.
DESIRED_RATE_TAU = 0.10  # s, EMA on the frame-to-frame desired-angle rate
LEAD_S = 0.07            # s, covers the output LPF plus CAN/firmware transport (~20 ms)
FF_OUTPUT_TAU = 0.15     # s, first-order smoothing of the feedforward itself
R5_CAP = 27000.0         # stay clear of the 30000 rail, where the classic stutter lived (route 154)
R5_CAP_ENVELOPE_FRAC = 0.9

# HondaEpsLateralCore. The feedback path is vfn 35ddc44b's modified-EPS angle PID with the P/I trims the car
# ran on it (2026-09-26, LatPScale 125/100/125, LatIScale 70/95/35), fixed here because the replay validated
# the feedforward against exactly that PID. The output LPF is the NRDR setting (these are the car's values).
MPH_TO_MS = 0.44704
BAND_LOW_MAX = 25.0 * MPH_TO_MS
BAND_STD_MAX = 50.0 * MPH_TO_MS
P_SCALE = (1.25, 1.00, 1.25)
I_SCALE = (0.70, 0.95, 0.35)
INSIGHT_P_SCALE = (1.15, 1.25, 1.15)
INSIGHT_I_SCALE = (0.75, 0.95, 1.00)
OUTPUT_LPF_TAU = (0.07, 0.05, 0.01)
OVERRIDE_FADE_I_BLEED_TAU = 0.5  # James: decay held I during post-press actuator limiting
OVERRIDE_FADE_UP_S_DEFAULT = 1.5
INTEGRATOR_MIN_SPEED = 2.0  # m/s, below this the integrator is held at zero (as vfn)

# The feedforward asks for the torque that moves the wheel ALONG the desired path, so it is only right
# once the wheel is on it. Engaging mid-turn at low speed routinely starts 20-70 deg off (45 engagements on
# routes 352/353/34f), where it would have pushed against the PID at up to 0.9. So it joins only once the
# angle error is small, then fades in; a driver press or dropping below walking speed takes it out again.
FF_JOIN_ERROR_DEG = 10.0
FF_FADE_IN_S = 0.5
FF_SPEED_BP = [2.0, 4.0]    # m/s, faded in with speed; the desired angle is ill-conditioned near standstill

# At a crawl the feedforward also passes the model's small path wiggles to the wheel 1:1 (target -> wheel gain
# 1.02 measured below 4 m/s on route 361, 0.41 for the PID alone), and column stiction makes that gain grow with
# amplitude, so a near-straight crawl can build a ~0.9 Hz wobble through the camera and the model (route 361
# t 823-829, hands off). Below FF_CRAWL_SPEED_BP the feedforward therefore joins with |desired angle| (none under
# 5 deg, all from 20 deg: every real crawl turn), fading out of effect by 8 m/s so nothing changes at speed.
# Replay of that window through firmware + column plant: wheel/target 1.09 (logged 1.05) -> 0.54 (PID alone 0.51).
FF_CRAWL_ANGLE_BP = [5.0, 20.0]  # deg
FF_CRAWL_SPEED_BP = [5.0, 8.0]   # m/s


class EpsFirmwareCalibration:
  """Per-image command map, P key axis, envelope and rate feedback; no shared mutable state."""

  def __init__(self, e4_per_output, r5_key_bp, r5_v, envelope_bp, envelope_v, r6_per_deg_s,
               *, r5_per_key=None, r6_angle_bp=None, r6_angle_gain=None, kp_key_bp=None):
    self.e4_per_output = e4_per_output
    self.r5_key_bp, self.r5_v = tuple(r5_key_bp), tuple(r5_v)
    self.envelope_bp, self.envelope_v = tuple(envelope_bp), tuple(envelope_v)
    self.r6_per_deg_s = r6_per_deg_s
    self.r6_angle_bp, self.r6_angle_gain = r6_angle_bp, r6_angle_gain
    self.r5_per_key = r5_per_key
    # The P row's key axis is row-indexed in the image like the command map; KP_V is the same on every PTM row.
    self.kp_key_bp = tuple(KP_KEY_BP if kp_key_bp is None else kp_key_bp)
    if r5_per_key is not None:
      if self.kp_key_bp != tuple(KP_KEY_BP):
        raise ValueError("a linear command map uses the shared Clarity P pieces")
      self.kp_pieces = tuple(KP_PIECES)
    else:
      # Kp(key(R5)) has knots at BOTH maps' breakpoints. Keeping only the P
      # breakpoints would make the quadratic inverse wrong on C020's nonlinear map.
      knots = sorted(set(r5_v) | {float(np.interp(k, r5_key_bp, r5_v)) for k in self.kp_key_bp if k <= r5_key_bp[-1]})
      kps = [float(np.interp(np.interp(r, r5_v, r5_key_bp), self.kp_key_bp, KP_V)) for r in knots]
      pieces = [(lo, hi, kp_lo, (kp_hi - kp_lo) / (hi - lo))
                for lo, hi, kp_lo, kp_hi in zip(knots[:-1], knots[1:], kps[:-1], kps[1:], strict=True)]
      self.kp_pieces = (*pieces, (knots[-1], math.inf, kps[-1], 0.0))
    self.kp_r5_bp = tuple(p[0] for p in self.kp_pieces)
    self.kp_r5_v = tuple(p[2] for p in self.kp_pieces)


CLARITY_A020 = EpsFirmwareCalibration(
  E4_PER_OUTPUT, R5_KEY_BP, R5_V, ENVELOPE_BP, ENVELOPE_V, R6_PER_CENTRE_DEG_S,
  r5_per_key=R5_PER_KEY, r6_angle_bp=R6_ANGLE_BP, r6_angle_gain=R6_ANGLE_GAIN,
)
CLARITY_EPS_LOAD = (LOAD_K0, LOAD_K1, LOAD_C, LOAD_FRICTION, LOAD_BIAS, LOAD_KROLL)

# Insight 39990-TXM-A040, corrected PTM build only: 39990-TXM,A040-PTM-20261016b (RWD sha256 7ba37432...), a data-only
# fix of the released PTM (rwd 833a41bf, the 08-08 C020Surface Trk1-4000 build). The release wrote TargetMap-D onto
# row 0's key axis, which is the Clarity's (0, 111, ..., 1663), so it ran 1.14-1.53x the Clarity at outputs 0.10-0.30.
# The corrected build puts the Clarity's own command map and P table on rows 0-5, returns row 6 to stock and flattens
# A280 in rows 0-3: rows 0-1 are then the Clarity's tables exactly (SH-2A emulation matches the Clarity's output at
# every key). Row 0 is live: all three variant records (TXMA0/1/2) select row 0. Road-proven on JamesL787
# ns-bosch-radar-testing (f3a6a124bd) on this build. Both builds report 39990-TXM,A040: a car on the released image must
# be reflashed with this calibration.
# This tree sends the Insight 4096 E4 per unit of output (opendbc _EXTENDED_TORQUE_LIMITS), not 3840, so the 1663 key
# clamp binds at u ~0.94; the inversion uses that scale, so the firmware target is unchanged. The envelope was not
# located in the image: flat (the key clamp binds first). R6 is computed from its A table (center divisor 17613),
# per published deg/s. Load: the owner's konik route 0000001e, the Clarity recipe, in firmware output counts (so it
# holds across the reflash); route 00000020 held out scores R^2 0.70 / 0.75 against 0.70 / 0.75 for a fit on it.
INSIGHT_TXM_R6_ANGLE_BP = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180, 210, 240, 270, 300, 360, 420, 480)
INSIGHT_TXM_R6_ANGLE_GAIN = (1.0, 1.0247, 1.0326, 1.052, 1.0852, 1.1305, 1.1692, 1.207, 1.2242, 1.2239, 1.2156, 1.2234,
                             1.2254, 1.2249, 1.2273, 1.2253, 1.2279, 1.2436, 1.1917)
INSIGHT_TXM_A040 = EpsFirmwareCalibration(
  4096.0, R5_KEY_BP, R5_V, [0, 50, 100, 150, 200, 250, 300, 350, 400], [1774] * 9, -131.15,
  r5_per_key=R5_PER_KEY, r6_angle_bp=INSIGHT_TXM_R6_ANGLE_BP, r6_angle_gain=INSIGHT_TXM_R6_ANGLE_GAIN,
)
INSIGHT_TXM_EPS_LOAD = (-4.876, -0.1376, -4.452, -397.2, 65.41, -4.29)
# Compute in James's source normalized domain and convert once at the wrapper.
# Unlike NRDR's 18.04 linear shortcut, the union of command/P knots preserves
# every rounded table breakpoint from the corrected firmware image.
INSIGHT_TXM_A040_CANONICAL = EpsFirmwareCalibration(
  3840.0, R5_KEY_BP, R5_V, INSIGHT_TXM_A040.envelope_bp, INSIGHT_TXM_A040.envelope_v, -131.15,
  r6_angle_bp=INSIGHT_TXM_R6_ANGLE_BP, r6_angle_gain=INSIGHT_TXM_R6_ANGLE_GAIN,
)


def _finite(*values):
  if not all(math.isfinite(float(value)) for value in values):
    raise ValueError("nonfinite Honda EPS controller input or state")


def _calibration_signature(cal):
  return (cal.e4_per_output, tuple(cal.r5_key_bp), tuple(cal.r5_v), tuple(cal.envelope_bp), tuple(cal.envelope_v),
          cal.r6_per_deg_s, tuple(cal.r6_angle_bp), tuple(cal.r6_angle_gain), cal.r5_per_key,
          tuple(cal.kp_key_bp), tuple(cal.kp_pieces), tuple(cal.kp_r5_bp), tuple(cal.kp_r5_v))


# Immutable startup snapshots also catch mutations of shared calibration objects
# or constants before card/modeld/controlsd admits a controller. Table correctness
# against the image is verified independently by firmware_verify and unit tests.
_EXPECTED_INSIGHT_CALIBRATIONS = (
  _calibration_signature(INSIGHT_TXM_A040), _calibration_signature(INSIGHT_TXM_A040_CANONICAL),
)
_EXPECTED_INSIGHT_LOAD = (-4.876, -.1376, -4.452, -397.2, 65.41, -4.29)
_EXPECTED_INSIGHT_P = (1.15, 1.25, 1.15)
_EXPECTED_INSIGHT_I = (.75, .95, 1.)
_EXPECTED_P_VALUES = (117, 148, 184, 220, 245, 257, 263, 265, 265)
_EXPECTED_P_KEYS = (0, 222, 443, 665, 887, 1108, 1330, 1552, 1774)


def validate_calibration(cal=INSIGHT_TXM_A040):
  """Reject malformed or altered calibration data before steering initializes."""
  _finite(cal.e4_per_output, cal.r6_per_deg_s)
  if cal.e4_per_output <= 0.0 or cal.r6_per_deg_s >= 0.0:
    raise ValueError("invalid Honda EPS command or R6 scale")
  angle_gain = cal.r6_angle_gain
  if cal.r6_angle_bp is None or angle_gain is None:
    raise ValueError("Honda EPS calibration requires its matched R6 angle tables")
  tables = ((cal.r5_key_bp, cal.r5_v), (cal.envelope_bp, cal.envelope_v),
            (cal.kp_key_bp, KP_V), (cal.r6_angle_bp, angle_gain))
  for axis, values in tables:
    if axis is None or values is None or len(axis) != len(values) or len(axis) < 2:
      raise ValueError("Honda EPS calibration requires matching tables")
    _finite(*axis, *values)
    if axis[0] != 0 or any(b <= a for a, b in zip(axis, axis[1:], strict=False)):
      raise ValueError("Honda EPS calibration axes must start at zero and increase")
  if cal.r5_v[0] != 0 or any(b <= a for a, b in zip(cal.r5_v, cal.r5_v[1:], strict=False)):
    raise ValueError("Honda EPS command map must be invertible")
  if any(value <= 0 for value in (*cal.envelope_v, *angle_gain, *KP_V)):
    raise ValueError("Honda EPS envelope, P and rate gains must be positive")
  if not cal.kp_pieces or cal.kp_pieces[0][0] != 0 or not math.isinf(cal.kp_pieces[-1][1]):
    raise ValueError("Honda EPS P inverse must cover the full domain")
  # Detect stale derived tables if a mutable source calibration was altered.
  rebuilt = EpsFirmwareCalibration(
    cal.e4_per_output, cal.r5_key_bp, cal.r5_v, cal.envelope_bp, cal.envelope_v, cal.r6_per_deg_s,
    r5_per_key=cal.r5_per_key, r6_angle_bp=cal.r6_angle_bp, r6_angle_gain=cal.r6_angle_gain,
    kp_key_bp=cal.kp_key_bp,
  )
  if rebuilt.kp_pieces != cal.kp_pieces:
    raise ValueError("Honda EPS derived P table does not match its calibration")
  if cal is not CLARITY_A020 and _calibration_signature(cal) not in _EXPECTED_INSIGHT_CALIBRATIONS:
    raise ValueError("Honda EPS calibration differs from the verified Insight profiles")
  if (INSIGHT_TXM_EPS_LOAD != _EXPECTED_INSIGHT_LOAD or INSIGHT_P_SCALE != _EXPECTED_INSIGHT_P or
      INSIGHT_I_SCALE != _EXPECTED_INSIGHT_I or tuple(KP_V) != _EXPECTED_P_VALUES or tuple(KP_KEY_BP) != _EXPECTED_P_KEYS or
      KFF != 45. or SCALE_Q8 != 256. or KEY_CLAMP != 1663):
    raise ValueError("Honda EPS load, gains or firmware constants differ from the verified profile")



def command_key(e4: float) -> int:
  _finite(e4)
  return int(math.trunc(math.trunc(e4 * 56756 / 32768) / 4))


def key_ceiling(v_ego: float, cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  _finite(v_ego)
  return min(float(np.interp(v_ego * 3.6 * 2.0, cal.envelope_bp, cal.envelope_v)), KEY_CLAMP)


def r5_from_output(output: float, v_ego: float, cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  """What the firmware makes of a lateral output: forward model of 0xE4 -> key -> R5."""
  _finite(output, v_ego)
  key = command_key(-output * cal.e4_per_output)
  mag = float(np.interp(min(abs(key), key_ceiling(v_ego, cal)), cal.r5_key_bp, cal.r5_v))
  return math.copysign(mag, key) if key else 0.0


def output_from_r5(r5: float, cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  """Inverse of r5_from_output (up to integer truncation)."""
  _finite(r5)
  key = float(np.interp(min(abs(r5), cal.r5_v[-1]), cal.r5_v, cal.r5_key_bp))
  e4 = key * 4.0 * 32768.0 / 56756.0
  return -math.copysign(e4, r5) / cal.e4_per_output


def firmware_kp(r5: float, cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  _finite(r5)
  if cal.r5_per_key is not None:
    return float(np.interp(abs(r5) / cal.r5_per_key, KP_KEY_BP, KP_V))
  return float(np.interp(abs(r5), cal.kp_r5_bp, cal.kp_r5_v))


def firmware_r6(steering_rate_deg_s: float, angle_deg: float, cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  """The firmware's rate feedback for a published steering rate at a published angle."""
  _finite(steering_rate_deg_s, angle_deg)
  angle_bp, angle_gain = cal.r6_angle_bp, cal.r6_angle_gain
  if angle_bp is None:
    if angle_gain is not None:
      raise ValueError("Honda EPS R6 gains require an angle axis")
    gain = 1.0
  else:
    if angle_gain is None:
      raise ValueError("Honda EPS R6 angle axis requires gains")
    gain = float(np.interp(abs(angle_deg), angle_bp, angle_gain))
  return cal.r6_per_deg_s * gain * steering_rate_deg_s


def firmware_output(r5: float, steering_rate_deg_s: float, angle_deg: float = 0.0,
                    cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  """Steady-state firmware output for a target and a rate (D term omitted): scale*(Kp*(R5-R6) + KFF*R5)/1024/256."""
  r6 = firmware_r6(steering_rate_deg_s, angle_deg, cal)
  return SCALE_Q8 * (firmware_kp(r5, cal) * (r5 - r6) + KFF * r5) / 1024.0 / 256.0


def column_load(angle_deg: float, rate_deg_s: float, v_ego: float, roll: float,
                friction_width: float = FRICTION_WIDTH_DEG_S, coefficients=CLARITY_EPS_LOAD) -> float:
  _finite(angle_deg, rate_deg_s, v_ego, roll, friction_width, *coefficients)
  if friction_width <= 0:
    raise ValueError("Honda EPS friction width must be positive")
  k0, k1, damping, friction, bias, kroll = coefficients
  return (k0 * angle_deg + k1 * angle_deg * v_ego ** 2 + damping * rate_deg_s
          + friction * math.tanh(rate_deg_s / friction_width) + bias + kroll * roll * v_ego ** 2)


def r5_for_motion(load: float, rate_deg_s: float, r5_guess: float = 0.0, angle_deg: float = 0.0,
                  cal: EpsFirmwareCalibration = CLARITY_A020) -> float:
  """Solve the firmware law for the target that yields `load` while the wheel moves at `rate_deg_s` through `angle_deg`.

  load = scale * (Kp*(R5 - R6) + KFF*R5) / 1024 / 256, with Kp piecewise linear in |R5|. On each piece
  that is a quadratic in R5, so solve every piece exactly and keep the root nearest `r5_guess` (more than
  one root only exists when a fast unwind outruns a small target). A fixed-point iteration is not enough
  here: it is 1-4% off after three passes from rest and need not contract during a fast unwind.
  """
  _finite(load, rate_deg_s, r5_guess, angle_deg)
  x = 1024.0 * load * 256.0 / SCALE_Q8
  r6 = firmware_r6(rate_deg_s, angle_deg, cal)
  roots = []
  for lo, hi, kp_lo, slope in cal.kp_pieces:
    for side in (1.0, -1.0):
      # on this piece Kp = a + b*R5, and R5*(Kp + KFF) - Kp*R6 = x
      a, b = kp_lo - slope * lo, slope * side
      qa, qb, qc = b, a + KFF - b * r6, -(a * r6 + x)
      if abs(qa) < 1e-12:
        candidates = [-qc / qb]
      else:
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
          continue
        candidates = [(-qb + sq) / (2.0 * qa) for sq in (math.sqrt(disc), -math.sqrt(disc))]
      roots += [r for r in candidates if lo <= side * r <= hi]
  # g(R5) = R5*(Kp + KFF) - Kp*R6 - x runs from -inf to +inf, so a root always exists
  return min(roots, key=lambda r: abs(r - r5_guess))


class HondaEpsFirmwareFeedforward:
  def __init__(self, dt: float, rate_tau: float = DESIRED_RATE_TAU, lead_s: float = LEAD_S,
               output_tau: float = FF_OUTPUT_TAU, friction_width: float | None = None,
               *, cal: EpsFirmwareCalibration = CLARITY_A020, load=CLARITY_EPS_LOAD):
    _finite(dt, rate_tau, lead_s, output_tau)
    if dt <= 0 or min(rate_tau, lead_s, output_tau) < 0:
      raise ValueError("invalid Honda EPS feedforward timing")
    if friction_width is not None and (not math.isfinite(friction_width) or friction_width <= 0):
      raise ValueError("invalid Honda EPS friction width")
    validate_calibration(cal)
    if len(load) != 6:
      raise ValueError("Honda EPS load requires six coefficients")
    _finite(*load)
    self.dt = dt
    self.alpha = dt / (rate_tau + dt)
    self.output_alpha = dt / (output_tau + dt)
    self.lead_s = lead_s
    self.friction_width = friction_width  # None: the speed schedule above
    self.cal = cal
    self.load_coefficients = load
    self.reset()

  def reset(self):
    self.prev_angle = None
    self.rate = 0.0
    self.r5 = 0.0
    self.load = 0.0
    self.output = 0.0

  def update(self, desired_angle_no_offset: float, v_ego: float, roll: float) -> float:
    try:
      _finite(desired_angle_no_offset, v_ego, roll, self.rate, self.r5, self.load, self.output)
      output = self._update(desired_angle_no_offset, v_ego, roll)
      _finite(output, self.rate, self.r5, self.load)
      return output
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
      self.reset()
      raise ValueError("invalid Honda EPS feedforward input or result") from exc

  def _update(self, desired_angle_no_offset: float, v_ego: float, roll: float) -> float:
    first = self.prev_angle is None
    if not first:
      raw_rate = (desired_angle_no_offset - self.prev_angle) / self.dt
      self.rate += self.alpha * (max(min(raw_rate, 400.0), -400.0) - self.rate)
    self.prev_angle = desired_angle_no_offset

    angle = desired_angle_no_offset + self.lead_s * self.rate
    width = self.friction_width if self.friction_width is not None else friction_width(v_ego)
    self.load = column_load(angle, self.rate, v_ego, roll, width, self.load_coefficients)
    cap = min(R5_CAP, R5_CAP_ENVELOPE_FRAC * float(np.interp(key_ceiling(v_ego, self.cal), self.cal.r5_key_bp, self.cal.r5_v)))
    self.r5 = max(min(r5_for_motion(self.load, self.rate, self.r5, angle, self.cal), cap), -cap)
    target = output_from_r5(self.r5, self.cal)
    self.output = target if first else self.output + self.output_alpha * (target - self.output)
    return self.output


def speed_band(v_ego: float, values):
  return values[0] if v_ego < BAND_LOW_MAX else values[1] if v_ego < BAND_STD_MAX else values[2]


class HondaEpsLateralCore:
  """Angle PID on the residual + firmware-inversion feedforward, faded in once the wheel is on the path.

  Pure (no messaging, no params), so the closed-loop replay can drive exactly the code the car runs.
  """

  def __init__(self, kp_bp, kp_v, ki_bp, ki_v, dt: float, ff: HondaEpsFirmwareFeedforward | None = None,
               *, p_scale=P_SCALE, i_scale=I_SCALE):
    _finite(dt, *p_scale, *i_scale, *kp_bp, *kp_v, *ki_bp, *ki_v)
    if dt <= 0 or len(p_scale) != 3 or len(i_scale) != 3:
      raise ValueError("invalid Honda EPS feedback timing or trims")
    self.dt = dt
    self.p_scale, self.i_scale = p_scale, i_scale
    self.pid = PIDController((kp_bp, kp_v), (ki_bp, ki_v), pos_limit=1.0, neg_limit=-1.0, rate=1.0 / dt)
    self.ff = ff if ff is not None else HondaEpsFirmwareFeedforward(dt)
    # The NRDR torque-output LPF, run exactly as LatControlPID runs it (the car controller deliberately does
    # not filter, so this is the only one): same filter class, same per-band update_alpha, reset to 0.
    self.output_lpf = FirstOrderFilter(0.0, OUTPUT_LPF_TAU[0], dt)
    self.override_fade_up_s = OVERRIDE_FADE_UP_S_DEFAULT
    self.reset()

  def reset(self):
    self.pid.reset()
    self.ff.reset()
    self.since_press_s = math.inf
    self.ff_ramp = 0.0
    self.ff_weight = 0.0
    self.output_lpf.x = 0.0
    self.output_lpf.initialized = True
    self.output = 0.0

  output_lpf_enabled = True
  output_lpf_tau = OUTPUT_LPF_TAU

  def update(self, desired_angle_no_offset: float, angle_offset: float, angle: float, v_ego: float, roll: float,
             steering_pressed: bool, steer_limited: bool, *, optimized_lane_change: bool = False) -> float:
    try:
      _finite(desired_angle_no_offset, angle_offset, angle, v_ego, roll,
              self.pid.p, self.pid.i, self.output, self.ff_ramp, self.output_lpf.x)
      output = self._update(desired_angle_no_offset, angle_offset, angle, v_ego, roll,
                            steering_pressed, steer_limited, optimized_lane_change=optimized_lane_change)
      _finite(output, self.pid.p, self.pid.i, self.pid.f, self.ff_weight)
      return output
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
      self.reset()
      raise ValueError("invalid Honda EPS lateral input or result") from exc

  def _update(self, desired_angle_no_offset: float, angle_offset: float, angle: float, v_ego: float, roll: float,
              steering_pressed: bool, steer_limited: bool, *, optimized_lane_change: bool = False) -> float:
    error = desired_angle_no_offset + angle_offset - angle
    ff_full = self.ff.update(desired_angle_no_offset, v_ego, roll)

    if steering_pressed or v_ego < FF_SPEED_BP[0]:
      self.ff_ramp = 0.0
    elif self.ff_ramp > 0.0 or abs(error) < FF_JOIN_ERROR_DEG:
      self.ff_ramp = min(1.0, self.ff_ramp + self.dt / FF_FADE_IN_S)
    crawl = float(np.interp(v_ego, FF_CRAWL_SPEED_BP, [1.0, 0.0]))
    crawl_gate = 1.0 - crawl * (1.0 - float(np.interp(abs(desired_angle_no_offset), FF_CRAWL_ANGLE_BP, [0.0, 1.0])))
    self.ff_weight = self.ff_ramp * float(np.interp(v_ego, FF_SPEED_BP, [0.0, 1.0])) * crawl_gate
    # Owner-requested adaptation: no feedforward (including modeled EPS damping)
    # during optimized lane changes. Reset its rejoin ramp to prevent a jump on exit.
    if optimized_lane_change:
      self.ff_ramp = self.ff_weight = 0.0
    ff = self.ff_weight * ff_full

    # Latest James controller: a held integrator must not fight a driver turn
    # while the existing Honda actuator override fade limits the returned torque.
    self.since_press_s = 0.0 if steering_pressed else self.since_press_s + self.dt
    if steer_limited and not steering_pressed and self.since_press_s <= self.override_fade_up_s + self.dt:
      self.pid.i *= math.exp(-self.dt / OVERRIDE_FADE_I_BLEED_TAU)

    i_scale = speed_band(v_ego, self.i_scale)
    self.pid.update(error, speed=v_ego, feedforward=ff,
                    freeze_integrator=steer_limited or steering_pressed or v_ego < INTEGRATOR_MIN_SPEED,
                    integrator_gain_scale=i_scale,
                    reset_integrator=i_scale <= 0.0 or v_ego < INTEGRATOR_MIN_SPEED)
    output = max(min(self.pid.p * speed_band(v_ego, self.p_scale) + self.pid.i + self.pid.d + ff, 1.0), -1.0)

    if self.output_lpf_enabled:
      self.output_lpf.update_alpha(speed_band(v_ego, self.output_lpf_tau))
      output = float(self.output_lpf.update(output))
    else:
      self.output_lpf.x = output
      self.output_lpf.initialized = True
    self.output = max(min(output, 1.0), -1.0)
    return self.output
