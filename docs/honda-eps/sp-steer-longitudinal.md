# Authoritative sp-steer longitudinal integration

> **Model lifecycle follow-up:** [Bigv2 model lifecycle](bigv2-model-lifecycle.md)
> supersedes the separate navigation-driving selection/approval workflow described
> in this historical verification report. Current selection uses the ordinary Big
> Model slot; exact artifacts, EPS and longitudinal code remain preserved.

Repository: `ZakariaSaad/ZakariaSaad`, branch `mvl-sp-steer-eps-nav`.
Starting branch: `bc4af6400ff555a6d39eb7980714307347ba3d11`.
Authoritative MVL `sp-steer`: `0f5085fc3f713da58d236ed3822b9d27bc0e9917`.
The remote upstream head was rechecked before implementation and had not advanced.

The owner requested replacing the longitudinal behavior retained during the first
rebaseline with MVL's current implementation. This follow-up supersedes the old
ACC/Experimental/DEC preservation decision; it is not an EPS or model-runtime port.

## Exact baseline and adopted behavior

The following production files are byte-identical to the pinned upstream revision:

- `openpilot/selfdrive/controls/lib/longitudinal_planner.py`
- `openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`
- `openpilot/selfdrive/controls/lib/longcontrol.py`
- `openpilot/selfdrive/controls/lib/accel_boost.py`
- `openpilot/selfdrive/controls/plannerd.py`
- `openpilot/selfdrive/selfdrived/selfdrived.py`
- `openpilot/sunnypilot/selfdrive/selfdrived/events.py`
- `openpilot/selfdrive/car/cruise.py`

MPC, LongControl and AccelBoost already matched upstream; this change removes the
old planner gates around them. Planner process-replay subscriptions also match
upstream, including removal of the obsolete vehicleParameters subscription.
The replay initializer retains the necessary custom EPS drive-latch publication
before CarParams; that preexisting compatibility hook is unchanged.

| Area | Current behavior |
| --- | --- |
| Model participation | `modelV2.action.desiredAcceleration` and `shouldStop` participate on every planner update, without ACC/Experimental/DEC selection. |
| Acceleration arbitration | Upstream minimum of MPC, cruise and boosted model candidates; final `ACCEL_MIN`/`ACCEL_MAX` clipping. |
| Cruise target | Upstream -0.5 m/s² cruise-target lower bound and speed-dependent jerk limit. This is not a global braking limit: model/MPC braking and final limits remain separate. |
| Removed old constraints | No old throttle-probability/pitch-coasting gate, separate ACC acceleration table or steering-angle turn cap. The applicable upstream planning/speed-control logic is authoritative. |
| Boost | Upstream eligibility latches before accelerator override; up to 0.05 m/s² per override and 0.2 m/s² total, with low-speed/braking scaling and disengagement reset. No legacy-mode reset/gate remains. |
| Stopping and restart | Upstream OR of candidate stop requests, LongControl stopping/starting states and Honda-specific downstream actuation retained. |
| Stock longitudinal operation | MPC uses measured ego speed/acceleration when openpilot does not own longitudinal control. |
| Other settings | Longitudinal personality, Initial Max Speed, smart cruise/speed-limit controls and applicable driver overrides remain as upstream. |

This deliberately changes behavior from the starting branch. It does not claim
that the previous ACC coast/turn/DEC tuning survives under different names.

## Removed settings and migration

Removed parameters: `ExperimentalMode`, `ExperimentalModeConfirmed`, and
`DynamicExperimentalControl`. Removed the DEC implementation/tests, old
experimental gap-button helper, obsolete mode controls and gestures, widgets and
unused icons. Tests for the removed mode gate are replaced by tests of the current
model-participation and safety-bound contracts, rather than weakened assertions.

Existing manager startup calls native `Params.clear_all(CLEAR_ON_MANAGER_START)`
before parameter migration. That native operation also removes unregistered
keys under its existing lock/fsync protocol, so it removes files left by old
installations without a new migration service or marker. Backups use the current
BACKUP whitelist: old entries are ignored before decoding and cannot resurrect
the retired modes. Regression tests cover repeated startup cleanup, malformed
legacy backup values and preservation of model/EPS/navigation choices.

The corresponding cereal fields retain their wire ordinals in upstream's
`deprecated` groups; the event retains ordinal 14 under its upstream deprecated
name. Old recordings remain decodable. Custom EPS diagnostics and navigation
metadata fields are preserved. No hidden parameter edits are needed on the device.

## Preserved integration and intentional differences

There is **no longitudinal planner/MPC/LongControl/boost deviation** in the files
listed above. The existing downstream Bosch wind-learning observation-guard fix
remains in opendbc `84ce49fcd02904f324eff9252ae6c1c10b9ffb4b`, on top of upstream
`acb69016e46eb98c5d6442ad97953a882613d9e5`. It prevents an inherited uninitialized
variable exception on inactive/override/zero-error observations and preserves the
upstream set-speed guard. Reintroducing that exception would not improve parity.
No dependency or safety revision changes were needed for this follow-up.

Custom EPS code, controlsd/card lateral hooks, calibration, drive-cycle latching,
driver override and learned-state isolation are unchanged. Panda safety, Honda
actuation, Chestnut/tinygrad, Selector 21, model selection/download/registration,
corrected Bigv2 hashes and approval, native model execution, finite-output rejection
and ordinary single-active-model fallback are unchanged. Maps/destination delivery
remain independent of Navigation Model Driving.

Bigv2's native action head reaches the planner through the existing public
`modelV2.action` contract. The new regression uses the pinned model signature,
real Parser, NavigationModelState action extraction and real fill_model_msg,
then the actual Honda Insight planner. It tests braking and stopping, ordinary
fallback-shaped messages, legacy-mode files set to either value, acceleration
bounds, boost/reset and output publication. Those are synthetic predictions;
no model checkpoint is compiled, loaded on a GPU or represented as new inference.

Two help descriptions differed minimally from upstream: their obsolete advice
to enable the now-removed Experimental Mode was removed. Existing AEB/alpha and
drive-cycle warnings remain. Models, navigation and EPS settings retain their
custom functionality; the existing hosted SunnyLink frontend was not deployed.

## Verification

- Native builds passed: parameter library, cereal, msgq/VisionIPC, longitudinal
  MPC and host pandad. Building pandad does not run it or send CAN.
- EPS/controller/lane-centering regressions: **428 passed**.
- Navigation activation, selection, native publication, fallback and startup
  recovery: **55 passed, 22 subtests passed**.
- Native Params, migrations, settings schemas, SunnyLink settings, EPS owner
  controls and model-selection RPC: **143 passed, 1 skipped, 9 subtests passed**.
  The optional JSON Schema test initially skipped because its test dependency
  was absent; it subsequently **passed** with that dependency installed outside
  the checkout. No production dependency or lockfile changed.
- New Bigv2-to-planner cases and LongControl checks: **9 passed**.
- Focused following-distance, longitudinal maneuvers, lateral and EPS stopping:
  **19 passed, 2 skipped, 23 subtests passed**.
- Rendered settings/Models/navigation/EPS/camera tests against rebuilt schemas:
  **20 passed, 32 subtests passed**, in a network-disabled Xvfb container.
- Untouched upstream longitudinal/following-distance baseline: **11 passed,
  2 skipped, 23 subtests passed**. The same two upstream skips remain.

Groups overlap and must not be summed into a unique-test total. Generated settings
match their source, changed Python passes lint, and no active production/UI code
references the removed settings. The focused suites have no unresolved failures.
The complete unfiltered upstream CI suite and device builds were not claimed.

The [isolated replay report](sp-steer-longitudinal-replay.json) compares one
continuous recorded segment using official process replay, matching cereal and
private random messaging prefixes. Target planner runs with custom EPS ON and
OFF both produce 1,178 plans, 1,176 valid; every non-timing plan field matches
untouched sp-steer exactly. The two input/initialization-invalid publications are
identical to upstream. All numerical outputs are finite. Ordinary controlsd
commands match upstream; custom controlsd commands match the preceding project
branch across all 5,971 commands. No card, Panda, vehicle CAN transmission or
model inference runs in these replays. Raw route data and replay outputs are not
committed. Timing measurements are host test timings, not comma 4 performance.

## Updating and remaining limits

Use the normal **Settings → Software** check/download/install/restart while
remaining on `mvl-sp-steer-eps-nav`. No reinstall or manual parameter change is
needed. The old mode controls disappear and old mode values are cleaned on startup.
Navigation Model Driving retains its separate existing request and approval;
this update does not select a different model or remotely enable it.

The original working branch `mvl-honda-eps-selector21-chestnut-v28` remains
unchanged. If rolling back to a version with the old mode controls, review those
controls again: removed values are not secretly restored. A backup created before
this update can restore supported settings on that older version.

Software/replay equivalence does not establish every real-world driving outcome.
Physical AMD execution on the newer runtime remains the previously documented
uncertainty; this task does not change that runtime or ask for a new hardware
validation project. No vehicle access, firmware operation or remote activation
was performed.
