# Navigation (Alpha) on the existing comma 4 test branch

The known-working rollback point is **`525eafe95f5657ee9cafeb4876831873e6cf1c8f`** on
`ZakariaSaad/ZakariaSaad`, branch **`mvl-honda-eps-selector21-chestnut-v28`**.
This update adds informational navigation to that same branch. It does not change
model inputs, lateral control, longitudinal control, firmware, or model selection.
Navigation is **disabled by default**.

## Upstream scope and compatibility

The implementation comes from [commaai/openpilot PR #39066](https://github.com/commaai/openpilot/pull/39066),
merged as `4c362ca3272c7418a068d43c402c84bd0544f35a` (22 files), and inspected in
[SunnyPilot PR #2059](https://github.com/sunnypilot/sunnypilot/pull/2059), merged as
`0739fe823d7d9e7523999d26998b545dae9c5b6a` (132 commits, 301 files).
SunnyPilot master `1aa3a430fbe0bcf95f9b3a9dd6edfe34b92d0d63` was also inspected;
its navigation source and navigation panel remain identical to the original PR.
No wholesale merge or unrelated upstream control changes were applied.

Six existing-file patches needed context/signature adaptation: parameters, comma 4
Developer settings, road-view layout, torque-bar drawing, UI state, and process
registration. Other original patches apply directly. Additional compatibility fixes:

- Navigation uses this branch's existing `CommaConnectApi` directly; the shared API
  wrapper and all other consumers remain unchanged.
- The pre-existing SunnyPilot `setNavDestination` RPC retains optional `place_name`
  and `place_details`, adds coordinate validation, and writes the typed JSON parameter.
  The new `clearNavDestination` RPC is idempotent.
- The comma 4 toggle uses the existing `BigParamControl` signature. All SPLC controls
  and custom settings remain in place.
- The panel rejects zero-timestamp frames even during the first three seconds after
  UI startup. A real VisionIPC/UI regression test covers this case.

[Provenance and preserved Git tree IDs](navigation-alpha-provenance.json) record the
source revisions and exact preserved components.

## Functionality

- `navd`: GPS source selection, comma/Mapbox directions requests, destination and
  route processing, rerouting, maneuver transitions, remaining distance/time,
  instruction messages, and periodic route resends after a renderer restart.
- `mapsd`: route-relative map pose, PMTiles archive access, vector-road decoding,
  deterministic CPU map rendering, bounded disk cache and prefetching, archive
  version/range validation, retry after errors, and grayscale VisionIPC map frames.
- Comma 4 on-road panel: route map, maneuver arrow/distance, remaining time/distance,
  GPS/route/frame status, stale-frame handling. Driving alerts take priority over
  the navigation panel. Disabling navigation restores the original camera layout.
- Settings → developer → **navigation (alpha)** enables/disables the persistent
  `NavigationEnabled` setting. The manager runs both daemons only while on-road and
  enabled. `NavDestination` clears on manager restart and off-road transition,
  matching upstream behavior.
- `navInstruction` and `navRoute` messaging services are restored at the existing
  schema ordinals 82 and 83. Existing VisionIPC map stream 3 is reused. No new
  dependency, submodule, model, or firmware revision is introduced.

This is the upstream Alpha feature: it does not add address search on the device,
voice guidance, offline routing, or navigation-driven steering/turns. Destinations
come from comma connect's authenticated RPC. A working GPS fix, Internet connection,
comma registration/keys and backend access to directions and map tiles are needed.
A `MAPBOX_TOKEN` can override directions requests as upstream supports; it does not
replace comma authentication for the hosted tile archive. No tokens are bundled.

## Preserved behavior

The Git trees for both controls directories (including Honda EPS, MVL longitudinal,
SPLC, safety/admission checks, calibration, torque conversion/limits and fallback),
stock modeld, modeld_v2, model manager/catalog selector, both hardware directories,
updater, legacy SunnyPilot mapd/navd helpers, and custom cereal schema remain
unchanged from the rollback commit. All submodule revisions and both dependency
files remain unchanged. In particular, selector **21**, Chestnut catalog **v28**,
tinygrad **0.14.0 / `69fd4d34`**, firmware **`8516b768-CLEAN`**, both model slots,
big models, download/selection/fallback and EPS model-timing hooks are preserved.
No vehicle was deployed to or flashed during this work.

## Validation

Tests use isolated parameters/messaging and mocked GPS/directions/tile data; no
vehicle or production backend is controlled by the tests.

| Check | Result |
| --- | --- |
| Full Linux source build (`scons -j4`, Ubuntu 24.04, pinned dependencies/runtime) | Passed, including native services, model compilation, planners, UI support and replay tools |
| Navigation + real daemon lifecycle + real raylib/VisionIPC UI + destination RPC + updater branch migrations + Cereal messaging/services | 421 passed; UI tested under Xvfb, including alert priority, map loss/zero timestamps and original disabled layout |
| EPS controller/firmware/timing/recorded-route tests + existing controls + model manager/fallback/power/parser + Chestnut + native Params | 517 passed, 2 skipped |
| Cereal schema compatibility against the rollback baseline | Passed: 286 structs; original routes remain readable |
| Ruff on all changed Python source/tests; ty on navigation and panel; whitespace check | Passed |
| Preserved control/model/hardware/updater/submodule/dependency trees | Exact match to rollback commit |

The first desktop **minimal** build (`scons --minimal`) failed the existing 50 MiB
build-product guard: the CPU:LLVM driving pickle is 66.3 MiB (69,565,647 bytes).
An untouched source archive of the rollback revision compiled the same-size CPU
model successfully; executing its unchanged guard against both outputs rejects
both at the same limit. The build guard, models and runtime were not modified to
hide this inherited desktop limitation. The normal full source build passes.
This is not an AGNOS/QCOM build result; compiling/installing on the actual device
and measuring its resource use remain hardware validation.

The full historical MVL suite/replay comparison was not repeated for this additive
navigation update. The prior three MVL calibration/location unit failures and five
inherited replay differences are documented in [the existing verification report](selector21-chestnut-v28.md).
This update leaves their control/model sources untouched. The new navigation tests
include an end-to-end mocked GPS → route messages → HTTP tile ranges → map image
comparison, and real child processes publishing through native VisionIPC.

GitHub Actions currently run automatically for `master` pushes or pull requests;
this direct update to the existing test branch does not change those triggers or
open another PR. Results above are executed local/container checks, not a claim
that a new GitHub Actions run passed.

## Updating the installed branch

1. Park, disengage, turn the vehicle off, and connect the comma 4 to reliable Wi-Fi.
2. In Software settings, confirm the target branch is
   `mvl-honda-eps-selector21-chestnut-v28` and the installed repository is
   `ZakariaSaad/ZakariaSaad`. Keep **Disable Updates** off; changing that preference
   requires the normal reboot. Do not change the installation URL.
3. Select **check for update**, then **download** if offered. Once **install now**
   appears, select it and allow the normal restart/build to finish. Confirm the
   displayed new commit against the published navigation commit.
4. In Settings → developer, enable **navigation (alpha)** when ready to test it.
   After starting the vehicle, send a destination from comma connect. Destinations
   sent before an off-road transition or manager restart must be sent again.
5. First verify GPS, route, map frames and directions while parked. Check a clear
   destination, a changed destination, disabling/re-enabling the toggle, Internet/GPS
   loss and recovery, and that alerts retain their full original camera area.
6. During controlled hardware validation, check UI frame rate, model timing/drop
   rate, CPU/thermal load and Chestnut/big-model status with navigation both off and
   on. Routing and initial tile prefetch add CPU/network load; host tests cannot
   establish simultaneous QCOM/Chestnut performance or actual route correctness.

The updater and its safety/verification/finalization logic are byte-for-byte
unchanged. It uses the installed `origin`, reads remote branch heads, fetches the
selected branch, detects new commit IDs, prepares the existing overlay update and
restarts through the normal Software UI. The exact custom branch is not in any
migration/exclusion rule. Tests cover branch passthrough and commit detection.
This verifies source configuration; the actual device's current parameters,
connectivity and backend credentials were not accessed in this environment.
A fresh installation is not needed.

## Rollback without rewriting history

For an immediate return to the original on-road layout and process load, disable
**navigation (alpha)**. The manager stops both navigation daemons; controls and
models already use the unchanged baseline implementations.

For a complete software rollback, in an existing clean developer checkout of this
same branch, fetch/pull with `--ff-only`, then run:

```sh
git revert <published-navigation-commit-sha>
git push origin HEAD:refs/heads/mvl-honda-eps-selector21-chestnut-v28
```

Use the exact navigation commit SHA in the completion report. This creates a new
rollback commit on the same branch and restores the known-working tree when the
navigation commit is the only change since `525eafe95f5657ee9cafeb4876831873e6cf1c8f`.
Then use the same Software settings update flow. If subsequent feature commits
exist, review any revert conflicts before pushing. No reset, force-push, new branch,
reinstall, or EPS/Chestnut flashing is needed.

## Changed files

Status `A` means added, `M` means modified.

```text
A	docs/honda-eps/navigation-alpha-provenance.json
A	docs/honda-eps/navigation-alpha.md
M	openpilot/cereal/log.capnp
M	openpilot/cereal/services.py
M	openpilot/common/constants.py
M	openpilot/common/params_keys.h
A	openpilot/selfdrive/navigation/__init__.py
A	openpilot/selfdrive/navigation/helpers.py
A	openpilot/selfdrive/navigation/mapsd.py
A	openpilot/selfdrive/navigation/navd.py
A	openpilot/selfdrive/navigation/pmtiles.py
A	openpilot/selfdrive/navigation/renderer.py
A	openpilot/selfdrive/navigation/route.py
A	openpilot/selfdrive/navigation/tests/test_integration.py
A	openpilot/selfdrive/navigation/tests/test_navigation.py
A	openpilot/selfdrive/navigation/tests/test_ui.py
A	openpilot/selfdrive/navigation/tiles.py
A	openpilot/selfdrive/navigation/vector_tile.py
M	openpilot/selfdrive/ui/mici/layouts/settings/developer.py
M	openpilot/selfdrive/ui/mici/onroad/augmented_road_view.py
A	openpilot/selfdrive/ui/mici/onroad/navigation.py
M	openpilot/selfdrive/ui/mici/onroad/torque_bar.py
M	openpilot/selfdrive/ui/ui_state.py
M	openpilot/system/athena/athenad.py
M	openpilot/system/athena/tests/test_athenad.py
M	openpilot/system/manager/process_config.py
```
