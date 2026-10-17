# Validation record

Validation ran on Linux x86_64 with Python 3.12.14, pycapnp 2.1.0, NumPy 2.5.4,
pytest 9.1.1, Ruff 0.17.1, ty 0.0.89 and SCons 4.10.1. Existing pinned opendbc,
msgq, rednose, Panda, tinygrad, teleoprtc and neural-network-data submodules were
initialized for testing without changing any gitlink or dependency lockfile.
Runtime import checks used the pinned tinygrad source, not a substitute model
engine. No inference, device deployment or EPS flashing was performed.

## Executed checks

| Check | Result |
| --- | --- |
| Firmware-controller and native baseline suite | 394 passed |
| Existing MVL controls suite | 44 passed, 1 skipped; 8 subtests passed |
| Existing Honda safety suite | 1,018 passed, 602 skipped; 1,156 subtests passed |
| Repository-wide Ruff | Passed |
| Type checks for the new package and edited integration code | Passed |
| Native Cap'n Proto C++ generation and Params build | Passed |
| PID diagnostics ordinals and native message round trip | Passed; old ordinals 0–9 retained, new fields 10–16 |
| Schema compatibility against the untouched MVL base | Passed using the repository's compatibility checker; 286 reachable structs |
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

The supplied route has four full-rate rlogs (segments 5, 16, 19, 56) and 58
qlogs. After an excluded five-second initialization window per segment, 15,730
active full-rate frames were compared. The explicit `route32` core matches the
reconstructed original source to within 0.0000155 E4 counts, with zero truncated
integer-command mismatches. The original source reconstruction, initialized once
from observable state, matches actually logged core requests with RMS 0.0736 and
maximum 0.851 E4 counts. These are core-request comparisons, not proof that the
complete MVL/model/transport pipeline will make identical on-road trajectories.

The available active rlogs have low- and medium-speed data, no filtered driver
override frames and no logged saturation frames. The qlogs' sampled maximum speed
is 20.877 m/s, below the 50 mph highway gain boundary. One final qlog has a
truncated Cap'n Proto tail; complete preceding messages were retained and the
warning is recorded. Missing full-rate segments were not synthesized.

MVL's raw 1200-count press threshold remains below the source route's 2400-count
setting. Real packed-CAN tests verify that signed 1200 remains unpressed and 1201
triggers the original MVL threshold, with opposing torque clearing actual CAN
requests in both response presets. This deliberate preservation can cause earlier
handover than the source route; matched recorded booleans do not prove identical
raw-detector behavior.

See [source comparison](source-comparison.md) and [firmware verification JSON](firmware-verification.json)
for numerical evidence and provenance. Replaying supplied logs requires a local
copy of those logs and the pinned source references; signed download URLs and raw
vehicle logs are not committed.

## Checks with limitations

Repository-wide `ty check openpilot` reports two existing errors and five existing
warnings in untouched files. The errors are a `str` appended to a `list[Path]` in
`selfdrive/ui/sunnypilot/layouts/settings/osm.py:69` and a mutable `should_run`
override in `system/manager/process.py:193`. They are unchanged from the target
base and were not fixed as unrelated work. The feature and edited integration
files pass their scoped type checks.

A genuine `SKIP_TINYGRAD_COMPILE=1 DEV=CPU scons --minimal -j2` attempt compiled
common/msgq components, the existing longitudinal acados solver and locationd
filters, then stopped while building the model camera warp:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'clang'
scons: *** [openpilot/selfdrive/modeld/models/dm_warp_1344x760_tinygrad.pkl] Error 1
```

The host has no Clang executable. This is a partial host build, not a successful
complete build. Neural-model compilation was excluded; no large model assets
were downloaded. A complete device/release build, actual AI/Chestnut inference,
USB cameras, Panda hardware and on-vehicle timing remain untested here.

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
