# Validation record

Validation ran on Linux x86_64 with Python 3.12.14, pycapnp 2.1.0, NumPy 2.5.4,
pytest 9.1.1, Ruff 0.17.1 and SCons 4.10.1. Type checks used both the CI-pinned
ty 0.0.73 and host ty 0.0.89. Existing pinned opendbc,
msgq, rednose, Panda, tinygrad, teleoprtc and neural-network-data submodules were
initialized for testing without changing any gitlink. The testing extra now
declares pytest; its lockfile additions preserve every existing dependency
package and version.
Runtime import checks used the pinned tinygrad source, not a substitute model
engine. Local checks did not run inference, deploy to a device or flash EPS.

## Executed checks

| Check | Result |
| --- | --- |
| Firmware-controller and native baseline suite | 405 passed |
| Existing MVL controls suite | 44 passed, 1 skipped; 8 subtests passed |
| Existing Honda safety suite | 1,018 passed, 602 skipped; 1,156 subtests passed |
| Repository-wide Ruff | Passed |
| Type checks for the new package and edited integration code | Passed |
| Native Cap'n Proto C++ generation and Params build | Passed |
| PID diagnostics ordinals and native message round trip | Passed; old ordinals 0–9 retained, new fields 10–16 |
| Schema compatibility against the untouched MVL base | Passed using the repository's compatibility checker; 286 reachable structs |
| GitHub CI release and macOS builds | Passed at `d13f9df`; release checks and tests passed |
| GitHub CI static analysis, upstream schema and documentation | Passed at `d13f9df` |
| GitHub CI UI report | Passed at `d13f9df` |
| Manager parameter-default regression | Reproduced and fixed locally; existing native manager test passed |
| Existing unittest discovery of firmware tests | Zero collection errors with pytest installed; function tests have an explicit pytest CI step |
| Frozen testing dependency setup | Dry run passed; existing dependency packages preserved |
| Python compilation and whitespace checks | Passed |
| Read-only uploaded RWD verification | Passed: exact RWD/payload hashes, checksums and all checked tables |
| Pinned-source deterministic comparison | 10,000 core frames; zero physical-E4 error before delivery for the source preset |
| Original actuator-method comparison | 6,000 frames, both response presets, including opposing/same-direction/chattering/short/repeated presses |
| Actual MVL Honda CAN delivery | Passed on both Bosch A steering buses and both response presets; override torque/request cut and existing limits retained |
| Recorded route | See [recorded-route comparison](recorded-route-comparison.md) |

The controller/native suite covers exact profile admission and CP/SP flag
separation; absent/wrong build confirmation; drive-latched geometry/response;
malformed, corrupted or lost snapshots; both native CarParams representations;
firmware-law inversion; calibration tampering; physical command conversion and
rails; feedback/FF filters and gains; geometry/roll/offset; override/I decay;
disengagement/reset; fractional command delay; both model timing branches;
nonfinite data; and original MVL fallback including torque/NNLC preferences.
It constructs real Controls instances, exercises braking alongside lateral
control, and serializes PID diagnostics through native msgq.

A real Honda speed-filter stop regression produces a minimum filtered speed of
-0.03914 m/s from nonnegative wheel measurements. Both geometry modes remain
finite, clamp the undershoot at the geometry boundary, and emit zero torque
while inactive. Replay initialization publishes the same snapshot before CP/SP,
so strict live restart checks do not break the existing replay callback.

The existing MVL controls suite includes longitudinal following-distance
simulations, LongControl, lane-centering and existing lateral/radar checks.
It passed after the small existing neural-network-data submodule was initialized.
The one warning is Python's existing multithreaded-fork deprecation in the radar
replay test. Skipped Honda safety tests are reported as skipped, not passed.

Representative commands, after normal repository dependency/native setup:

```sh
DEV=CPU PYTHONPATH=tinygrad_repo pytest -q \
  openpilot/sunnypilot/selfdrive/controls/lib/honda_eps/tests \
  openpilot/common/tests/test_params.py \
  openpilot/selfdrive/controls/tests/test_longcontrol.py \
  opendbc_repo/opendbc/car/honda/tests/test_honda.py
DEV=CPU PYTHONPATH=tinygrad_repo pytest -q openpilot/selfdrive/controls/tests
pytest -q opendbc_repo/opendbc/safety/tests/test_honda.py
ruff check openpilot
ty check openpilot/sunnypilot/selfdrive/controls/lib/honda_eps
```

The completed upload provides all 58 rlogs and 58 qlogs, segments 0–57. They
contain 340,869 recorded controls frames. After the excluded five-second
initialization window per segment, 287,710 active frames were compared. The
explicit `route32` core differs from reconstructed James by RMS 0.00002543 E4
and maximum 0.00324710 E4, with **one truncated integer-request mismatch** in
segment 47. This supersedes the earlier four-log subset's zero-mismatch claim;
its original per-segment results reproduce exactly. Original source
reconstruction matches actual logged core requests with RMS 0.07018 and maximum
2.94302 E4 counts. These are core-request comparisons, not proof that the
complete MVL/model/transport pipeline will make identical on-road trajectories.

The compared frames include 124,500 below 25 mph, 163,210 at 25–50 mph, and
207 filtered driver override frames. Maximum recorded speed is 20.897 m/s,
below the highway gain boundary. There are zero logged saturation flags, but
192 recorded active requests have magnitude at least 0.999, with a peak of 1.0.
Both final logs have truncated Cap'n Proto tails. Segment 57 has 322 inactive
controls frames and no CarParams event; matching CarParams were independently
verified in segments 0–56, and source/settings/calibration match across all 58.
Complete preceding messages were retained; no missing messages were synthesized.

The full-route geometry comparison finds RMS 6.582° and maximum 74.419°
difference between native MVL and James conversions at the same path. The maximum
occurs at 1.696 m/s with source angle −339.616°; at 25–50 mph the difference is
RMS 0.087° and maximum 0.506°. These slow, large-angle turns were absent from the
earlier subset. Optional `james` geometry selects the original rack; default
`mvl` geometry does not reproduce its tight-turn conversion.

The admitted firmware mode uses the source route's 2400-count raw driver
threshold, as explicitly requested by the owner. Real packed-CAN and card-path
tests verify that equality at 2400 remains unpressed and 2401 triggers override
for both torque directions and response presets. The published and actuator
states agree. Disabled fallback retains MVL's original 1200 boundary; other
vehicle/controller configurations and Panda restrictions are unchanged.

See [source comparison](source-comparison.md) and [firmware verification JSON](firmware-verification.json)
for numerical evidence and provenance. Replaying supplied logs requires a local
copy of those logs and the pinned source references; signed download URLs and raw
vehicle logs are not committed.

## Checks with limitations

The complete GitHub unit run at `d13f9df` reported 1,510 passed, 47 skipped,
one expected failure, four failures and 12 collection errors. The feature's
collection errors came from pytest not being declared in the testing extra.
The manager-default failure came from registering an empty-string default for
the firmware confirmation hash: native Params reads empty content as absent.
The final follow-up declares pytest, explicitly runs the function tests in CI,
and leaves the confirmation hash unset until supplied. The existing manager
regression and all 405 controller/native tests pass after these changes.
These fixes have not been rerun remotely because creating the requested
independent publication repository was denied by GitHub permissions.

Three location tests also failed in that CI run: calibration auto-reset,
consistent gyro spikes and consistent timing spikes. Their implementations are
unchanged by this port. They are not reported as passed or as empirically
verified baseline failures. A local retry reproduced the calibration assertion;
the other two could not initialize because their separate reference route was
unavailable from the configured local log sources.

Full process replay reported five changed comparisons, 61 passed and zero
execution errors against upstream reference `27c132984df60e9ef42ebd1a805738b8c851356f`,
not MVL base `cb15c04`. Source inspection traces the changed startup alerts,
Honda longitudinal delay/telemetry, Nidec integral behavior and seven-frame
steering ramp to existing MVL code and its pinned opendbc revision. No replay
reference was rewritten. An unchanged-MVL baseline replay has not been run,
so empirical baseline parity is unverified.

The new package passes CI-pinned ty 0.0.73 with warnings treated as errors.
Repository-wide local checks with that version report eight diagnostics, all
reproduced with the same environment in a separate checkout of the untouched
target base. They concern the existing optional radar-interface assignment,
CasADi model typing, model-compiler monkeypatches, Toyota migration keys and
model metadata typing. Host ty 0.0.89 instead reports two existing errors and
five warnings, including the UI OSM path list and manager `should_run` override.
These local dependency/type-checker differences are not a claim about remote CI.
Unrelated baseline issues were not changed. The pinned-checker compatibility
fix preserves the deterministic source comparison output exactly.

A genuine `SKIP_TINYGRAD_COMPILE=1 DEV=CPU scons --minimal -j2` attempt compiled
common/msgq components, the existing longitudinal acados solver and locationd
filters, then stopped while building the model camera warp:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'clang'
scons: *** [openpilot/selfdrive/modeld/models/dm_warp_1344x760_tinygrad.pkl] Error 1
```

The host has no Clang executable. This local attempt is a partial build.
Neural-model compilation was excluded locally; no large model assets were
downloaded into this workspace. GitHub CI subsequently passed its complete
release build/checks/tests and macOS build at `d13f9df`. The later changes
declare and run the test dependency and remove the unusable empty hash default;
native Params was rebuilt and its manager-default regression passed locally.
Actual AI/Chestnut inference on a device, USB cameras, Panda hardware and
on-vehicle timing remain untested here.

## What the evidence establishes

The port is integrated and exercised offline; its numerical core can reproduce
the source law and recorded response settings under matched inputs. Physical
CAN scaling, configured safety gates, lifecycle transitions and MVL fallback
are tested. The source comparisons are not a closed-loop vehicle simulation.
Recorded-input replay evaluates commands against existing trajectories; it
cannot reveal how a different command would change the next camera image,
model plan or vehicle motion.

Controlled vehicle validation remains required for oscillation, overshoot,
tracking, internal EPS saturation, low-speed engagements/unwinds, driver
handover, highway regimes not covered by the supplied full-rate logs, and every
intended AI model/geometry/response combination. Keep the controller disabled
until the installed corrected build is independently confirmed. Do not infer
that build from the EPS part number or the route's software commit alone.
