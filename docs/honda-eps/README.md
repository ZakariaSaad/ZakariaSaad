# Optional corrected-PTM Insight lateral controller

This development port adds a **disabled-by-default** controller for the 2022 Honda
Insight EX, Bosch A, with corrected `39990-TXM,A040-PTM-20261016b.rwd`. It retains
MVL's longitudinal planner/MPC, braking and acceleration adaptation, red-light
stopping, stop-and-go, lane-centering/lane-change flow, MADS, driver monitoring,
model selection/inference, and existing Honda/Panda code. The original port
validation records below describe its comparisons and limitations. Current UI
selection instructions are in [SunnyLink controls](sunnylink-model-and-eps-controls.md).

Base: [`mvl-boston/openpilot` `sp-honda-dev-202608`](https://github.com/mvl-boston/openpilot/tree/cb15c04e918b759688cb354b1bbc9b3c5a47a65c).
The existing opendbc gitlink stays `cf46117420f283b3e38241c00d728a191aa9fccc`;
**no opendbc or Panda modifications** are required.

## Source and behavior

The SunnyPilot integration follows [NRDR PR #21](https://github.com/nrdr/openpilot/pull/21)
and [PR #22](https://github.com/nrdr/openpilot/pull/22), merge
`5777cf9dc76b7b981e7f5f27cf4098df52b00aa2`. Insight calibration was introduced in
`5f54846231dde3b34ceb22bf7c9b637d1c692cf5`. Behavior is compared against
[JamesL787's calibration commit](https://github.com/JamesL787/openpilot/commit/f3a6a124bdad3151f875f4f29c8d7396c7617702)
and pinned branch HEAD `bfcbbf24bc857f9dbf93af52b7bb72852f4540bb`.

The port preserves the exact piecewise-quadratic inversion of the firmware P/KFF
law; corrected Clarity-equivalent command/P rows; Insight angle-dependent R6
(`-131.15` center scale); measured load `(-4.876, -.1376, -4.452, -397.2, 65.41,
-4.29)`; residual-angle PID with source speed gains/trims; FF joining/crawl gates;
rate limiting; filtering; override detection, release recovery and I decay; and
reset/saturation handling. Driver override uses the source default instant cut
and 1.5-second authority return, scoped to this controller. MVL's existing CAN
slew restrictions remain downstream.

Intentional differences from unmodified NRDR:

- Compute the whole core in James's 3840-count command domain, then convert the
  complete output to MVL's 4096-count transport. Scaling only FF would leave
  physical feedback 6.67% stronger than James.
- Use James's exact command-map/P-knot inversion, rather than NRDR's `18.04`
  linear shortcut, and preserve James's post-press integral decay.
- Add controller-local source-default override authority recovery: MVL's Honda
  transport lacks James's override actuator path. No other controller receives it.
- Preserve MVL geometry by default and offer the original Insight yaw-fitted rack
  explicitly. NRDR's configurable geometry manager is not transplanted.
- Keep MVL lane changes; NRDR's owner-specific FF suppression during optimized
  lane changes is absent.
- Require explicit installed-build confirmation, drive-latched admission and
  fail-closed invalid-data handling. NRDR's family check alone is insufficient.

See [integration plan](integration-plan.md), [source/output and geometry comparison](source-comparison.md),
[recorded-route comparison](recorded-route-comparison.md), and [validation results](validation.md).
The remaining broad-suite and process-replay failures are investigated in the
[exact MVL baseline comparison](baseline-comparison.md).

## Firmware evidence and admission

The uploaded image has SHA-256:

`7ba3743248d30758f8423bcb2a1c65f0d2d2d3563a830b8268bf448659b28a38`

Its read-only decoded payload SHA-256 is
`089a8a7251da04e4bbe693f764c4fe7a02ad6a630afde0fa700925801d9f9728`.
The verifier checks both digests, RWD/payload checksums, command/P rows and the
Insight A position table. It never writes firmware or changes controller settings:

```sh
python -m openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.firmware_verify /path/to/39990-TXM,A040-PTM-20261016b.rwd
```

**Verifying a file does not establish what is installed.** The old incorrect PTM
and corrected build both report `39990-TXM,A040`. Establish the installed revision
from your verified flash record/readback before making an installed-build
confirmation. The new UI preserves the existing confirmation; it cannot create one. Do not confirm based only on the diagnostic part number.
This project does not flash or modify the EPS.

Admission requires exact `HONDA_INSIGHT`, Honda Bosch A flags, torque steering,
native PID tuning, a comma-marked TXM-A040 EPS identifier, modified EPS in
`CarParamsSP.flags`, the existing identity `[0,4096]` map, a valid calibration,
explicit opt-in, and the exact confirmed build hash. Conflicting EPS records,
other Honda families, stock EPS, other transports, or absent confirmation keep
MVL's original controller. `CarParams.flags` is a different enum and is never
used as the modified-EPS flag.

Settings are sampled during interface setup, before torque/NNLC overrides, and
published synchronously as `HondaEpsFirmwareDriveConfig` before CarParams.
Controls and both model runners consume that same snapshot. The snapshot clears
on manager start/onroad transition; a same-drive process restart reuses it.
Changing the opt-in or geometry during a drive cannot switch the controller.
Malformed admitted configuration, calibration or initialization errors stop the
relevant process, preventing engagement with mismatched controller/model timing.
Nonfinite controller data clear history and raise before publishing an actuator
command. Small finite negative speed-estimator undershoot at stops is clamped
at the geometry/timing boundary.

## Command scaling, limits and delays

For canonical core output `u`, this wrapper returns `u * 3840/4096`. MVL retains
its normal rate limiter and computes
`int(interp(-transport_output * 4096, identity_BP, identity_V))` for 0xE4.
Thus, after the existing rate limiter settles, the physical request is
`int(-u * 3840)` for **FF and feedback together**. The core's +/-1 rail becomes
+/-0.9375 transport units, preserving the original +/-3840 physical command cap;
MVL's mapping remains 4096. The firmware-mode delivery-limit feedback threshold
is converted by the same ratio, preserving the source's physical E4 threshold
for integrator handling. The firmware's key clamp is 1663; feedforward stays
at or below R5 27000, 90% of its target rail. Physical saturation inside the EPS
is not measured by normalized carOutput feedback. No safety check is bypassed.

The separate command-history delay is .175 s through 10 m/s, linearly fading to
.025 s at 15 m/s. The extra .025 s is NRDR's SunnyPilot compensation for source
model-action interpolation absent in MVL. It is not added again to model lag.
The history warms while inactive and persists across normal disengagement.

Both model runtimes replace the lateral prediction lag with the same schedule:

| Speed (m/s) | 3.5 | 7 | 12 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| Prediction delay (s) | .15 | .08 | .10 | .20 | .30 |

Endpoints hold outside the table. Existing model-specific lateral smoothing and
frame/action offsets are added once by each model runtime. Longitudinal delay
is unchanged. Disabled mode retains MVL's existing live/manual delay behavior.
These timing fits originate in the source controller/models; source-equivalent
core commands do not establish identical closed-loop behavior on every selected
AI/Chestnut model. Validate each intended model and geometry combination.

## Geometry

`HondaEpsFirmwareGeometry = "mvl"` is the default. It uses MVL's existing learned
VehicleModel ratio/slip/roll conversion exactly. `"james"` uses the source Insight
A-table and yaw-fitted rack (center ratio 17.86, slip factor -.00066) for desired
angle and local measured-curvature/engagement feedback. Neither mode mutates the
shared VehicleModel, live learned parameters, or other vehicle consumers.

The modes are intentionally not declared equivalent. See the numerical comparison
for desired angles across speeds and curvatures. The source rack fit has limited
large-angle coverage and requires controlled validation on this vehicle.
The complete recorded route shows why the choice matters: at 1.696 m/s, native
MVL and James same-path angles differ by 74.419° during a large-angle turn.
At 25–50 mph, the difference is RMS 0.087° and maximum 0.506°. Select `james`
explicitly to use the original tight-turn conversion during controlled validation.

## Recorded response settings

The supplied route `f133facb1b9b7420|00000032--212d34d24d` records the corrected
calibration commit and the owner's actual controller settings. Its output LPF
was .1 s in every speed band; override fade-down was .1 s and recovery was 1.5 s.
Those settings differ from the source/NRDR defaults used in the initial numerical
comparison. Two explicit, drive-latched response presets preserve that distinction:

| `HondaEpsFirmwareResponse` | Output LPF low/medium/high (s) | Override fade-down (s) |
| --- | --- | --- |
| `source` (default) | .07 / .05 / .01 | 0 |
| `route32` | .1 / .1 / .1 | .1 |

Both use the original 1.5 s recovery and the same gains, FF, calibration and
command scale. During a filtered driver press both clear the CAN steering request
immediately; fade-down tracks the source's internal recovery authority. Choose
`route32` explicitly to match the recorded response settings. No settings from a
route are applied automatically. The admitted firmware mode uses JamesL787's
2400-count raw driver-press threshold in the published CarState and actuator
state, including for lane-change and override consumers. Equality at 2400 does
not trigger; values above 2400 do. The disabled MVL fallback retains its original
1200-count threshold. The robust direction/time filter follows that raw check.
The recorded comparison reports what could be reconstructed and the differences introduced by MVL's existing delivery limits.
The complete 58-rlog analysis compares 287,710 active frames: `route32` has one
truncated integer E4 request mismatch against reconstructed James, with maximum
continuous difference 0.003247 E4. This supersedes the earlier subset's zero-mismatch
claim; see the [recorded-route comparison](recorded-route-comparison.md).

## Select or restore the controller

Use **SunnyLink → Steering → Insight EPS Controller → Insight-specific EPS
lateral controller** while genuinely off road and disengaged. ON requests the
existing custom controller; OFF restores the existing ordinary selection path
on the next drive cycle. The adjacent status distinguishes requested and
currently effective state and explains eligibility failures.

Existing installed-firmware confirmation, geometry/response calibration and
ordinary torque/NN preferences are preserved. Toggling never writes firmware,
fabricates verification, retunes either controller or changes a drive's latch.
No SSH parameter editing is needed for this routine operation. See
[SunnyLink controls and recovery](sunnylink-model-and-eps-controls.md).

Installed-build confirmation remains an owner attestation: the controller cannot
read flash contents. Reassess that confirmation after an EPS reflash, ECU/vehicle
change or uncertainty about the installed build. Turning the controller OFF for
routine use does not erase valid confirmation or calibration.

## Added and modified files

New package `openpilot/sunnypilot/selfdrive/controls/lib/honda_eps/` contains the
firmware core, integral adapter, wrapper, driver-override delivery, geometry,
drive selection, timing, read-only firmware verifier, source comparison tooling,
and focused tests/fixtures. Documentation lives in `docs/honda-eps/`.

Existing files modified:

- `openpilot/common/params_keys.h`: default-off settings and drive snapshot.
- `openpilot/cereal/log.capnp`: append-only PID/FF/command-delay diagnostic fields.
- `openpilot/sunnypilot/selfdrive/car/interfaces.py`: latch before torque override.
- `openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py`: admitted constructor priority.
- `openpilot/selfdrive/controls/controlsd.py`: local geometry, timing and override-request hook.
- `openpilot/selfdrive/car/card.py`: publish the same latch for injected interfaces/replay.
- `openpilot/selfdrive/test/process_replay/process_replay.py`: prepare the latch before replay publishes CarParams.
- `openpilot/selfdrive/modeld/modeld.py`: optional shared prediction schedule.
- `openpilot/sunnypilot/modeld_v2/modeld.py`: same schedule for alternate/Chestnut models.
- `pyproject.toml` and `uv.lock`: declare pytest in the testing extra; retain existing dependency versions.
- `.github/workflows/tests.yaml`: run the firmware function tests explicitly alongside the existing unittest runner.

The [file manifest](files.txt) lists every added/modified path. No longitudinal,
Honda transport, Panda, submodule pointer, inference kernel, model weight or model asset file is
changed by this port.

## Navigation-model integration

The current workflow is [SunnyLink model and EPS controls](sunnylink-model-and-eps-controls.md).
Only corrected **Nav Model Integration Bigv2 — FP32 map correction** is supported.
Explicit selection prepares/reuses its installed local artifact and requests
navigation driving ON after admission; selecting an ordinary big model withdraws
that request. Explicit OFF and ordinary selections persist across reboot.
The original Bigv2 is retired. See [corrected artifact evidence](navigation-model-corrected-integration.md)
for immutable identities, approval and numerical limitations.
