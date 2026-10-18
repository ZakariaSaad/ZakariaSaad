# Insight firmware, geometry and source comparisons

References: JamesL787 `bfcbbf24bc857f9dbf93af52b7bb72852f4540bb`, including corrected
calibration `f3a6a124bdad3151f875f4f29c8d7396c7617702`; NRDR PR #22 merge
`5777cf9dc76b7b981e7f5f27cf4098df52b00aa2`. The NRDR core at inspected HEAD
`e1d7156b5d25ee38f24f63a4a04b125675448d39` is byte-identical to that merge.
`source-comparison.json` records the exact source-file hashes and measured results.

## Firmware image evidence

The read-only verifier accepted the supplied `39990-TXM,A040-PTM-20261016b.rwd`:

| Evidence | Value |
| --- | --- |
| RWD SHA-256 | `7ba3743248d30758f8423bcb2a1c65f0d2d2d3563a830b8268bf448659b28a38` |
| Decoded payload SHA-256 | `089a8a7251da04e4bbe693f764c4fe7a02ad6a630afde0fa700925801d9f9728` |
| Flash block | `0x4000`, length `0x6c000` |
| RWD checksum | `0x03853b3d`, valid |
| Payload checksums | `0x6bf80: 0x8805`, `0x6bffe: 0x981b`, valid |
| Command row 0 axis, flash `0x11b3c` | `0,111,222,333,443,665,887,1108,1663` |
| Command row 0 values, `0x11bba` | `0,2000,4000,6000,8000,12000,16000,20000,30000` |
| P axis, `0x11e8a` | `0,222,443,665,887,1108,1330,1552,1774` |
| P values, `0x11f08` | `117,148,184,220,245,257,263,265,265` |
| Insight A position table | X `0x11374`, Y `0x11338`: all 30 knots/divisors match the source |

These independently extracted tables agree with James's corrected calibration.
The full payload hash also matches the revision identified in his calibration
commit. No firmware was changed, flashed, or executed in an emulator here. The
source's SH-2A emulation claim was read as source evidence, not repeated locally.
File verification cannot establish the revision installed in the vehicle: the
released and corrected images report the same `39990-TXM,A040` identifier.

```sh
python -m openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.firmware_verify '/path/to/39990-TXM,A040-PTM-20261016b.rwd'
```

## Physical command domain

James's modified Insight interface uses a linear `[0,3840]` torque map. MVL and
NRDR use `[0,4096]`. The firmware-inversion equations agree when compared using
the same physical E4 command. NRDR scales only the feedforward inversion to
4096; its unchanged normalized residual PID consequently has 6.667% more
physical feedback gain. Its normalized saturation also permits another 256 E4
counts. Neither difference is needed to reproduce the proven James behavior.

The optional port runs the whole core in James's canonical 3840 domain and
converts its final filtered output by `3840/4096 = 0.9375` before MVL's existing
transport. Thus `-4096 * (u * 0.9375) = -3840 * u`, including PID feedback,
integrator state, feedforward, filtering and saturation. The public NRDR 4096
profile remains available for numerical reference; it is not the active core.
Existing carcontroller rate limits, CAN mapping and Panda safety remain intact.

Other intentional source adaptations: the active calibration uses James's exact
nonlinear `Kp(key(R5))` lookup rather than NRDR's constant `18.04 R5/key`
shortcut, and the integrator decays with a 0.5 s time constant during the 1.5 s
post-override fade. NRDR's optional optimized-lane-change feedforward suppression
is not imported. The rate and output filters and speed-dependent Insight trims
remain the source algorithms. Command timing includes NRDR's additional 25 ms
SunnyPilot interpolation compensation; James uses 0.15 to 0.00 s whereas this
port uses 0.175 to 0.025 s between 10 and 15 m/s. This remains an adaptation
requiring vehicle validation.

## Geometry choices

`mvl` is the default: it uses MVL's existing VehicleModel conversion, including
its learned steering ratio, stiffness and roll compensation. NRDR preserves its
own configurable steering-ratio selection, which is also different from MVL's
native conversion. Selecting the firmware controller does not transplant those
NRDR ratio features.

`james` explicitly selects the original Insight yaw-fitted rack conversion. It
combines the Insight A position table with an effective ratio of 17.86 near
center, decreasing to 17.17 past about 221 degrees, and slip factor -0.00066.
The ratio was fitted against the owner's route `0000001e` VSA yaw; the source
reports about seven minutes of turns, thin data past 50 degrees, and one fast
small-angle held-out cell with a ratio of 0.92. This port retains the exact
source grid and interpolation. It does not mutate VehicleModel or change
paramsd's learned values. The matching angle-to-curvature conversion is used
locally for measured-curvature initialization.

Illustrative desired steering angles at zero roll, using MVL's initial Insight
parameters (wheelbase about 2.7 m, ratio 15.0, model slip factor -0.00075758).
Live learned parameters will change the MVL column. Positive curvature requests
negative steering angle in both modes.

| Speed m/s | Curvature 1/m | MVL angle deg | James angle deg | James magnitude increase |
| --- | --- | --- | --- | --- |
| 3 | 0.001 | -2.336 | -2.779 | 18.96% |
| 3 | 0.005 | -11.682 | -13.656 | 16.90% |
| 3 | 0.010 | -23.363 | -27.244 | 16.61% |
| 10 | 0.001 | -2.496 | -2.945 | 17.99% |
| 10 | 0.005 | -12.481 | -14.458 | 15.84% |
| 10 | 0.010 | -24.963 | -28.855 | 15.59% |
| 20 | 0.001 | -3.024 | -3.492 | 15.50% |
| 20 | 0.005 | -15.118 | -17.111 | 13.18% |
| 20 | 0.010 | -30.237 | -34.099 | 12.77% |
| 30 | 0.001 | -3.903 | -4.392 | 12.54% |
| 30 | 0.005 | -19.513 | -21.566 | 10.52% |
| 30 | 0.010 | -39.026 | -42.459 | 8.80% |

Keeping MVL geometry is therefore not a claim of identical steering behavior.
Both modes remain behind the disabled-by-default controller and installed-image
confirmation gate. Geometry is latched with controller selection.

## Executed numerical comparison

`compare_sources.py` loads the original James and NRDR pure cores, James's actual
PID and rack-map implementations, and their source constants. Only project
imports are rebound for offline execution. A 100 Hz, 20-second deterministic
trace at each speed includes turn entry, reversal, driver press and release,
post-override fade, large residual errors and saturation recovery. These are
injected input traces, not recorded-route replay or a vehicle simulation.

| Speed m/s | Frames | Port vs James max E4 error | Integer E4 mismatches | NRDR vs James RMS E4 difference |
| --- | --- | --- | --- | --- |
| 3 | 2000 | 0 | 0 | 83.19 |
| 8 | 2000 | 0 | 0 | 73.05 |
| 12 | 2000 | 0 | 0 | 60.23 |
| 22 | 2000 | 0 | 0 | 97.80 |
| 30 | 2000 | 0 | 0 | 111.34 |

Maximum NRDR-vs-James difference was 256 E4 counts in the saturation portions.
At identical desired wheel angles, all 10,000 port commands match James exactly
before carcontroller delivery. This compares the identity torque-map transform;
carcontroller rate limits, CAN delivery and safety need separate tests. The
ported optional rack agrees with the source across 147 speed/roll/curvature
combinations with maximum angle error 0 degrees. The self-contained golden
trace test retains ten original source outputs, including override and recovery.

```sh
python -m openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.compare_sources \
  --james-root /path/to/pinned-james-openpilot --nrdr-root /path/to/pinned-nrdr-openpilot
```

The separately supplied route is evaluated in the
[recorded-route comparison](recorded-route-comparison.md); its results remain
distinct from the deterministic traces above. Its completed 58-rlog analysis
has one truncated integer E4 request mismatch across 287,710 active frames,
superseding the earlier route subset's zero-mismatch result. The 10,000
deterministic inputs above still match exactly. Neither comparison measures
closed-loop oscillation, overshoot, perception feedback or steering timing in the
vehicle with this port. Controlled vehicle validation is still required before
ordinary use.
