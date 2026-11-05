# Predictive Acceleration Boost: integration review

Repository/branch: `ZakariaSaad/ZakariaSaad`, `mvl-sp-steer-eps-nav`.
Inspected baseline: `689d2b40a2d27237f50f844d7526828d91fc6933`, clean working
tree, matching the remote branch. Publication to the same branch was authorized
after the local implementation and verification; deployment remains manual.
Upstream: sunnypilot PR #2089, merged as
`8cf6ff9a33c94bdf2fcb02e228a9aef1a7e6dc7f`.

## Feature and controls

`AccelBoostSP` replaces the learned `AccelBoost` when enabled with
`CP.openpilotLongitudinalControl`. They are never applied together. OFF restores
the existing learned boost. Switching modes clears boost history as upstream
intends; the existing three-second parameter polling interval is retained.

The algorithm and its upstream tests are copied byte-for-byte. Personality caps
are 0.4/0.25/0.1 m/s² for Aggressive/Standard/Relaxed, with 0.2 m/s³ build and
1.0 m/s³ release. Speed, model acceleration, brake/gas predictions, stopping and
curve gating follow upstream without retuning. Brake/stop/curve gates release
existing boost at the upstream rate; disengagement or missing/short prediction
arrays reset it immediately. Missing prediction arrays produce unboosted model
acceleration, exactly as upstream, rather than inventing another model fallback.

Find **Acceleration Boost** under **SunnyLink → Cruise**. The existing tici UI
also has **Settings → Cruise**. This branch's comma 4/MICI layout has no local
Cruise panel; SunnyLink supplies its toggle without a new panel or frontend patch.
The parameter is `PERSISTENT | BACKUP`, Boolean, default **ON**. Existing manager
startup initializes an unset value to that default; an explicitly saved OFF
value survives reboots and lifecycle clears. Stock longitudinal disables the
UI control. Existing RPC/schema paths write the exact parameter read by runtime.

## Necessary compatibility adaptation

Upstream recreates the legacy boost on every toggle change even when stock
longitudinal makes predictive boost ineligible. This patch changes selection
only when openpilot longitudinal is enabled, preserving the original stock
boost object and its learned history. No other algorithm adaptation is made.

The unchanged base planner calls the selected boost exactly once, then selects
the minimum of boosted model, radar-MPC and cruise acceleration, ORs stopping
intentions, and clips to its existing acceleration bounds. `controlsd` and
`LongControl` retain engagement/override and Honda output limits. Honda's
low-speed brake logic, grade compensation, actuator delay and wind-factor fix
remain unchanged. Tests exercise native Bigv2 and ordinary model publication
through this planner/controller interface with synthetic predictions.

No changes were made to EPS tuning/transport, steering calculation modes,
SPLC/Blendful, model artifacts, selection/fallback, safety code or dependencies.
Pinned revisions remain:

- opendbc: `84ce49fcd02904f324eff9252ae6c1c10b9ffb4b` (wind-learning fix).
- tinygrad: `92b742f711c7d216fd3ed7a46a30a959e3c43bd4`.
- panda: `8a16e089b88593914823743df29e1c30b7dd3e00`.
- msgq: `e7396e76dadbb49e374d4b664ff6bbb43a39bcb0`.
- rednose: `28d4a7f69e80e1c3e0d24ca0733d7daeaeade3d0`.
- teleoprtc: `1aa8fc433bef1519a95c0700c96258c3be6dfb34`.
- neural-network data: `03cac2d30e111e0689c0429cb8c1fe6cb5a905af`.

The existing opendbc worktree has the correct pinned HEAD and is clean, though
the parent configuration lists it as uninitialized. That configuration was
left unchanged; no submodule update was required.

## Files and purpose

| File (under `openpilot/` unless noted) | Purpose |
| --- | --- |
| `sunnypilot/selfdrive/controls/lib/accel_boost.py` | Exact upstream predictive algorithm. |
| `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py` | Select one boost implementation; preserve stock history. |
| `common/params_keys.h` | Register the new persistent, backed-up default. |
| `selfdrive/ui/sunnypilot/layouts/settings/cruise.py` | Upstream toggle and eligibility. |
| `sunnypilot/sunnylink/settings_ui_src/pages/cruise.yaml` | Existing remote settings definition. |
| `sunnypilot/sunnylink/settings_ui.json` | Generated served settings. |
| `sunnypilot/selfdrive/controls/lib/tests/test_accel_boost.py` | Unmodified upstream algorithm tests. |
| `selfdrive/controls/tests/test_predictive_accel_boost.py` | Actual planner/controller, model metadata, replacement, reset and gate regressions. |
| `selfdrive/controls/tests/fixtures/predictive_boost_baseline_sp_update.txt` | Hash-pinned original selection-hook reference for OFF comparisons. |
| `selfdrive/controls/tests/test_upstream_longitudinal.py` | Explicitly select OFF for the existing learned-boost test. |
| `sunnypilot/sunnylink/tests/test_predictive_boost_settings.py` | Real defaults, persistence, schema and RPC/runtime linkage. |
| `selfdrive/ui/tests/test_predictive_boost.py` | Rendered toggle, persistence and eligibility checks. |
| `docs/honda-eps/predictive-acceleration-boost.md` (repository root) | Review, limitations and reversal instructions. |

## Verification and limits

- 30 upstream/existing algorithm, planner, longitudinal state and Honda wind
  tests passed (including all 14 upstream feature cases).
- 15 new integration cases passed. Two 360-frame OFF traces match the original
  planner and Honda `LongControl` exactly; a 250-frame stock toggle trace keeps
  identical learned history. ON tests verify one application, all three caps,
  model metadata, reset and real MPC/cruise/braking/stop/curve limits.
- 48 existing following-distance, EPS integration and stopping tests passed.
  The uninstantiated parameterized following-distance base class is intentionally
  skipped; all nine concrete speed/personality simulations passed.
- Six new parameter/RPC tests and two rendered UI tests passed. Seventy existing
  settings/compiler/capability tests passed. The JSON-schema check initially
  skipped in the venv was completed using installed system Python `jsonschema`.
- All ten settings semantic checks passed; generated JSON matches YAML.
- Native Params library rebuilt successfully. Changed Python syntax, Ruff and
  patch whitespace checks passed.

Test groups are reported separately. No road/hardware validation, model
recompilation, firmware work, device connection or hosted authenticated
SunnyLink browser test was performed. Source/RPC/rendering checks establish
software wiring; they do not reproduce upstream's reprojection simulation or
establish vehicle comfort or driving safety.

## Revert

Turning **Acceleration Boost OFF** selects the original learned boost after the
existing parameter refresh. To reverse the published feature completely, use
the feature commit SHA from the publication report:

```bash
cd /workspace/work/mvl-sp-steer-eps-nav
git revert <feature-commit-sha>
```

Review any conflicts with later edits before publishing the revert normally.
This preserves branch history. Rebuild the Params library if using the reverted
checkout locally. The feature publication does not install anything on a device.
