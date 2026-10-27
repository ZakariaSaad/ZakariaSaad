# sp-steer continuation: preservation inventory

Pinned custom source: `18106a1ddf26366236961b28509e6fa677bec6ef` on
`mvl-honda-eps-selector21-chestnut-v28`.
Pinned MVL upstream: `0f5085fc3f713da58d236ed3822b9d27bc0e9917` on `sp-steer`.
Common ancestor: `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56`.
The continuation is developed on `mvl-sp-steer-eps-nav`; the working branch and
published corrected-model assets remain separate rollback resources.

## Integration strategy

A regular two-parent merge was chosen on the new branch, starting from the
working custom head. The histories have 106 custom-side and 311 upstream-side
commits since the common ancestor, with 31 patch-equivalent pairs despite
different commit IDs. A disposable `git merge-tree --write-tree` trial found
33 conflicted paths (the real new worktree also needed the already-fast-forwardable
tinygrad gitlink initialized). Most overlaps are the same native-model and
Navigation Alpha additions, with the custom side's subsequent fixes already
present. Replaying the custom commits would replay retired experiments and
duplicate equivalent changes. The merge preserves both ancestries, adopts
nonoverlapping upstream additions, and resolves the shared interfaces by reviewed
hunks. Conflict-free files still require the feature and regression review below.

The upstream and source revisions above are fixed for this integration. New
commits published to either source during implementation are not automatically
included. No source branch is rebased, reset or overwritten.

This inventory was made from the final source, its history and regression tests
before implementation. Historical investigation documents do not override the
current runtime. Each disposition below was implemented and checked; the final
verification and remaining execution limits appear below and in the linked
controller/runtime reports.

| Feature | Integration disposition | Source and regression evidence |
| --- | --- | --- |
| Insight firmware law, corrected TXM-A040 calibration, geometry/response profiles | Retain algorithms unchanged; adapt infrastructure/transport deliberately | `sunnypilot/selfdrive/controls/lib/honda_eps/`; controller, core, source-golden, geometry, response, driver-delivery and timing tests |
| Insight 3840-count calculation / 4096 transport, driver override and drive-cycle latch | Adapt full steering path without stacking upstream learned correction; preserve threshold and latch | `card.py`, `controlsd.py`, `controlsd_ext.py`, car interfaces, Honda opendbc transport; `honda_eps/tests/test_integration.py`, `test_selection.py`, `test_raw_override.py` |
| Ordinary sp-steer controller and learning | Adopt upstream in ordinary mode; prevent custom mode from contaminating learned state | Upstream car interfaces/controller/steering learning; control-mode regression tests |
| Existing firmware attestation and calibration | Preserve valid local record; generic backups must not transfer firmware proof | `honda_eps/selection.py`, `sunnylink/insight_eps.py`, backup manager and EPS tests |
| Fresh-install owner firmware confirmation | Added explicit, VIN-bound, off-road UI recovery naming `39990-TXM,A040-PTM-20261016b`; never claims ECU readback | SunnyLink Steering schema, guarded EPS settings and tests |
| Corrected Bigv2 identity and experimental approval | Retain exact artifact and independent approval; retired original remains unavailable | `navigation/{candidate_identity,corrected_bigv2,activation}.py`, `driving_models.json`; activation/corrected-selection/retired-candidate tests |
| FP32 map precision, compiler history, camera warp and native contracts | Preserve provenance and bytes; review execution-runtime adaptation independently | `navigation/{experimental_catalog,software_compatibility,warp_compatibility}.py`, `modeld_v2/nav_map_precision.py`; native-contract/precision/warp tests |
| Public download, local reuse, atomic install and registration | Retain pinned public URLs/hashes, bounded sidecar extraction, portable manifest and notices | `navigation/{download,preparation,export_experimental_model}.py`; download/release-preparation/preparation tests |
| Single-active Bigv2 / ordinary-small fallback | Retain finite-output rejection, existing fallback and bounded numerical evidence; no shadow inference or new priming | `modeld_v2/modeld.py`, `numerics.py`; publication/fallback/startup-recovery/numerics tests |
| Native JIT recurrent alias handling, camera and action timing | Retain state-slot distinction and shared EPS timing schedule while adapting runtime | `modeld_v2/{helpers,model_adapters,frame_fusion}.py`, both model runners; diagnostic-state/timing/publication tests |
| Requested/effective model selection and cancellation | Retain transactional queue, superseded-request fencing, failure rollback, progress and persistent OFF/ordinary choices | `models/selection.py`, `navigation/selection.py`, `sunnylink/model_selection.py`; selection/manage/real-RPC tests |
| Model cache safety | Retain protection for selected model, fallback, proof/metadata and warp files | `models/manager.py`; navigation-cache-protection and manager-download tests |
| Model-slot migration | Retain one-time marker and modern-slot detection so Default/retired models are not silently reselected | `sunnypilot/system/params_migration.py`; params-migration tests |
| Models UI and SunnyLink | Retain native-navigation display overlay, existing hosted Activate/poll/Cancel protocol; adopt compatible upstream UI improvements | Both Models layouts, `model_info.py`, `model_selection.py`, SunnyLink schema/RPC; rendered-navigation-model-UI and RPC tests |
| Insight toggle and status | Retain fresh off-road admission, transactional calibration settings, requested/effective status and next-drive semantics | `sunnylink/insight_eps.py`, Steering schema; test_insight_eps |
| Settings backup/restore | Retain protected-proof exclusions, guarded model/EPS restore and native filtered-key lifetime fix | `sunnylink/backups/manager.py`, `common/params_c.cc`; native Params and SunnyLink backup tests |
| Navigation destination delivery | Retain coordinate checks and optional legacy labels; reconcile newer API entrypoints | `system/athena/athenad.py`, navd/tiles; Athena/navigation integration tests |
| Maps, maneuvers and route-generation consistency | Retain producer identity, monotonic route fencing, timestamp/frame association and independent neutral map/maneuver defaults | `navigation/{navd,mapsd,generation,model_inputs,temporal}.py`; route-generation/map-generation/maneuver/input/temporal tests |
| Navigation display and daemon lifecycle | Retain zero-timestamp rejection, alert priority and independent map enable; adopt newer UI descriptions/layouts | `ui/mici/onroad/navigation.py`, road view, UI state, process config; rendered UI and daemon integration tests |
| MVL stopping/stop-and-go/braking and E2E/ACC selection | Preserve explicit existing modes and DEC; reconcile upstream acceleration improvements only in their applicable mode | Both longitudinal planners, DEC, settings/schema; longitudinal maneuvers/DEC/following-distance tests |
| Lane centering, lane changes, MADS and monitoring | Adopt compatible upstream improvements while retaining existing settings/capabilities | lane-centering, controlsd, MADS, monitoring and settings; existing focused control tests |
| Updater, release and AGNOS transition | Adopt upstream compatibility improvements intentionally; keep same repository and new branch tracking | `system/updated/`, launcher, release/build files; branch-switch compatibility tests |
| Numerical investigation/replay tooling | Retain opt-in offline tools and evidence; do not install obsolete diagnostic services | `tools/navigation_*`, `tools/replay/nav_*`; offline/numerical replay tests |
| Separate hosted frontend patch | Retain unchanged; not deployed or reopened by this migration | `docs/honda-eps/patches/` |

All source paths above are relative to `openpilot/` unless otherwise shown.
See the completed dependency review, tests and installation instructions below.

## Owner firmware recovery and preserved UI behavior

The source branch had no supported action to recover a lost installed-firmware
confirmation. This continuation adds **SunnyLink → Steering → Insight EPS
Controller → I confirm corrected TXM-A040 firmware is already installed**.
On comma 4 the same recovery and toggle are under **Settings → Developer →
Insight EPS controller**. The larger legacy layout also exposes them under
**Steering**. All paths call the same guarded backend.

The action explicitly names `39990-TXM,A040-PTM-20261016b`, requires fresh genuine
off-road/disengaged admission and the identified supported Insight transport,
and records an owner installation attestation. It does not query the installed
firmware image, flash anything, enable the controller or replace a drive latch.
The exact family part number alone is insufficient. The new receipt is bound to
the controller-relevant vehicle identity and the individually identified VIN. A missing/placeholder VIN cannot create a
new statement; existing valid legacy confirmation is preserved.
It is protected from raw SunnyLink writes and excluded from generic backups, as
is the existing confirmed hash. Existing valid legacy confirmation is preserved
without creating fictitious new evidence. Corrupt newer receipts fail admission;
a confirmation write interrupted by loss of off-road state rolls back.

The confirmation action and enabling the controller are separate explicit UI
operations. A reinstall that erased the confirmation can recover it without SSH;
an update never creates the confirmation automatically. New device/vehicle
restores must use an explicit owner statement instead of importing another
vehicle's proof.

The UI integration preserves Experimental Mode and Dynamic Experimental Control
as explicit supported longitudinal modes while also retaining sp-steer's new
Initial Max Speed control, updated descriptions, alert/network rendering and
navigation panel layout. The map panel keeps the custom zero-timestamp rejection.
The branch-switch input says **enter branch name**, not a hardcoded upstream
repository name; updater repository tracking is not rewritten by that label.

Focused owner/RPC/backup regressions: **32 passed**. These are host software tests
with isolated parameters and mocked fresh off-road messages, not vehicle firmware
readback or physical-controller verification. Rendered Xvfb tests with networking disabled: **18 passed, 32 subtests passed**,
covering both Models layouts, navigation-panel stale frames, owner-confirmation
controls/refusals, and coexistence of Experimental Mode with Initial Max Speed.
All 24 new upstream UI/sound assets (444,151 bytes total) were fetched from the
configured public SunnyPilot LFS endpoint and independently matched their pinned
SHA256 values and sizes. Existing matching UI assets were reused locally.

## Replay reference selection after the rebaseline

The old CI preparation step called `tools/honda_eps_baseline_compare.py
prepare-replay`, downloaded exactly 66 historical references, and wrote an
untracked `ref_commit` selecting the original `cb15c04e`-era comparison set.
Those fixtures serve the original migration comparison; they are not the
reference set for the newer sp-steer process/schema/route configuration.

The new branch adopts pinned sp-steer's process-replay CI job without that
preparation step. The normal replay command, process selection, numerical
comparisons, tolerances, failure handling and diff artifacts remain intact. It
uses the replay runner's normal SunnyPilot CI-artifact reference selection. No
golden logs were regenerated, comparisons suppressed or expected differences
automatically accepted. The historical helper and verified fixture manifest
remain unchanged for explicit original-MVL investigations. The dedicated Honda
EPS CI unit-test step also remains. The separate isolated migration replay
results are recorded in `sp-steer-process-replay.json`.

## Final dependency decisions

| Dependency | Working source | New branch | Rationale |
| --- | --- | --- | --- |
| tinygrad | `69fd4d34d4430036378aba2b43f9d339cdd6ea73` | `92b742f711c7d216fd3ed7a46a30a959e3c43bd4` | Adopt sp-steer's runtime and ordinary native ABI adapter; separately preserve the historical compiler evidence and exact corrected-model compatibility scope. |
| opendbc | `cf46117420f283b3e38241c00d728a191aa9fccc` | `84ce49fcd02904f324eff9252ae6c1c10b9ffb4b` | Pinned sp-steer `acb69016e46eb98c5d6442ad97953a882613d9e5` plus the reproduced Bosch observation-guard fix and regression tests. No safety-file edits. |
| Panda | `74a0adced421e8b7acd728d0f9988ce225423f13` | `8a16e089b88593914823743df29e1c30b7dd3e00` | Adopt upstream's matching safety/process interface revision. |
| msgq | `e7396e76dadbb49e374d4b664ff6bbb43a39bcb0` | Same | Native messaging and VisionIPC rebuilt against the merged schemas. |
| rednose | `28d4a7f69e80e1c3e0d24ca0733d7daeaeade3d0` | Same | Vehicle and pose filters rebuilt against upstream model changes. |
| teleoprtc | `1aa8fc433bef1519a95c0700c96258c3be6dfb34` | Same | No adaptation required. |
| Neural-network data | `03cac2d30e111e0689c0429cb8c1fe6cb5a905af` | Same | Existing ordinary NNLC data retained. |
| AGNOS | 19.7 | 19.9 | Adopt upstream startup and OS-update requirements, including normal inactive-slot staging during upgrades and rollback. |

All seven submodule commits were independently fetched by exact SHA into empty
bare repositories from their final configured URLs. GitHub denied creation of a
separate dependency repository (`Resource not accessible by integration`), so the
small opendbc fix is published in the existing authorized repository on the
separate dependency branch `mvl-sp-steer-opendbc`. `.gitmodules` points there.
That branch contains opendbc, **not installable SunnyPilot**; select only the
project branch `mvl-sp-steer-eps-nav` on the device.

Selector 21, Chestnut catalog v28, the existing Chestnut firmware identity and
ordinary model selection remain. Python dependency/lockfile changes follow
upstream (including removal of Sentry and obsolete Qt-specific SCons pinning),
while preserving the project's pytest requirement. `uv lock --check --offline`
passed with Python 3.12.13. No dependency revision points to an unpublished local
commit. No firmware was flashed during development.

## Completed software verification

These groups overlap; their counts must not be added as a unique-test total.

| Focused check | Result |
| --- | --- |
| Full EPS controller suite plus DEC/boost gate tests | 405 passed |
| Longitudinal simulations, following distance, lane centering and lateral saturation | 58 passed, 2 skipped, 56 subtests passed |
| Honda safety | 1,046 passed, 612 parametrized/base-class skips, 1,190 subtests passed |
| Real Bosch CarInterface inactive/override/learning regression | 7 passed |
| Native model suite | 222 passed, 5 optional/obsolete-comparison skips, 54 subtests passed; added ABI regression also passed |
| Navigation runtime/export/warp admission | 54 passed, 115 subtests passed |
| Actual published corrected-model installer on the new runtime | Passed: existing model reused, exact metadata preserved, registered reuse made zero HTTP requests |
| Selection, preparation, registration, retirement, CLI, SunnyLink and publication | 138 passed, 36 subtests passed |
| Ordinary manager/default Chestnut downloader, cancellation and protected cache | 105 passed, 1 external-network skip, 9 subtests passed |
| Navigation services, route/maneuver generations, native daemon lifecycle and Athena | 126 passed, 13 subtests passed |
| Native Params, migrations and generated settings schemas | 89 passed, 9 subtests passed |
| Settings RPC and capabilities | 27 passed, 1 skipped |
| Firmware owner-confirmation backend | 32 passed |
| Rendered Models, navigation, EPS and longitudinal settings UI | 18 passed, 32 subtests passed |
| Updater/AGNOS branch-switch tests | 11 passed, 1 optional LFS-zipapp skip, 4 subtests passed |
| Original and migrated common parameter/branch migrations | 32 passed, 9 subtests in each checkout |
| Native builds | Params, cereal, msgq/VisionIPC, longitudinal MPC, vehicle/pose filters and host pandad passed |
| Generated schema, imports and source checks | Settings generator matches; changed Python parses; migration-specific Ruff/whitespace checks pass |

No vehicle process was launched by the native pandad build. The test runs used
isolated parameters/messaging namespaces. Safety tests construct and inspect CAN
messages in software; they do not transmit them. Rendered tests ran in a
network-disabled Xvfb container. The inherited upstream Cabana file
`ui/widgets/scrollabletabbar.cc` retains its extra blank line at EOF; no unrelated
source formatting was changed to suppress that warning.

Untouched pinned sp-steer baseline results were also recorded: native build
passed; model/native tests 110 passed and 3 skipped; ordinary downloader 51 passed
and 1 skipped; updater/migrations 35 passed and 1 skipped; controls 31 passed and
one stale Civic-PID fixture failed. That fixture was adapted to exercise the
actual upstream torque configuration while retaining explicit PID coverage.
The working branch's stale publication-guard AST test also failed before this
migration; the assertion now follows the existing successful try/finally startup
path without weakening the guard. The Bosch exception was reproduced directly
on upstream before the minimal dependency fix. No unresolved failure was found
in the executed migration-focused suites. This does not claim that the entire
unfiltered upstream CI suite or all-platform release builds were run here.

## Recorded replay and runtime limits

[The sanitized replay report](sp-steer-process-replay.json) records six isolated
full-segment runs using 1,178 recorded model frames. Custom-mode controlsd
produced 5,971 valid commands: source and migrated acceleration/activation flags
were identical, maximum torque difference was 3.7253e-8, and curvature difference
was 2.3283e-10. Ordinary-mode commands matched pinned sp-steer exactly. The 1,178
planner trajectories, target accelerations and stopping decisions matched the
working source exactly; the same eight initial/input-invalid publications were
invalid in both checkouts. Every numerical output checked was finite.

This is recorded prediction-consumer replay, not Bigv2 GPU inference or a closed
loop vehicle test. Raw routes, outputs and private capture files are not committed.
[The runtime review](sp-steer-runtime-compatibility.md) documents actual artifact
and buffer checks, historical-to-new CPU execution, and the remaining unmeasured
AMD HCQ/USB execution/performance on tinygrad 92b. The exact corrected artifact's
experimental authorization remains; that is not proof of real-world numerical
stability. The non-finite rejection, hardware-target checks, off-road admission,
single-active fallback and driver safeguards remain in force. No broad new
hardware validation workflow is required or installed.

[The control review](sp-steer-controls.md) documents the full ON/OFF steering
paths and intentional upstream behavior changes. In particular, E2E acceleration
boost is bounded and gated by the retained modes, and configurable Initial Max
Speed (default 40 km/h) replaces the old 120 km/h Experimental-mode initial
setpoint. These intentional changes are not described as bit-for-bit preservation
for every possible drive.

## Install, preserve settings, and roll back

No uninstall, installation URL change or routine SSH command is needed.

1. While genuinely off-road, use **Settings → SunnyLink → backup settings** and
   complete **slide to backup**. Keep the successful backup available. Model
   binaries and vehicle-specific firmware proof are deliberately not included
   in generic settings backups. Keep the already-exported model/sidecar files
   on the laptop; the published release is unchanged and remains downloadable.
2. On the current installed branch, open **Settings → Software**, select
   **mvl-sp-steer-eps-nav** using **Switch Branch**, and check/download the update.
   Complete the offered install/restart. After this migration, upstream's
   **target branch** control selects the branch for the next update check.
   If a newly published branch is absent, refresh update checks before reopening
   the branch list. The repository remains `ZakariaSaad/ZakariaSaad`.
3. Allow the normal updater's AGNOS 19.9 installation/restart when offered. The
   development task does not run it remotely, bypass its verification or flash
   EPS firmware. Use the normal device update power/setup procedure.
4. Check that Software shows **mvl-sp-steer-eps-nav** and the published commit.
   Existing parameter files, model selections, OFF requests, firmware
   confirmation, `/data/nav-bigv2` and `/data/navigation-models` stay in place.
   The corrected model is reused by SHA256; it is not automatically reselected.
5. SunnyLink **Models → Big Models → Navigation models** still selects corrected
   Bigv2 through Download/Verify/Install/Select as needed. An explicit admitted
   selection requests navigation driving ON; choosing Mountain Dew/another
   ordinary big model withdraws that request. Maps/destinations remain separate.
   Insight EPS remains under **SunnyLink → Steering → Insight EPS Controller**;
   missing confirmation can be recovered through the explicit owner-attestation
   control after the Insight's VIN has been identified. No firmware is flashed.

For rollback, while off-road choose
**mvl-honda-eps-selector21-chestnut-v28** in Software's target/switch-branch control,
check/download/install, and restart. Its preserved rollback commit is
`18106a1ddf26366236961b28509e6fa677bec6ef`. The newer updater stages the older
branch's required AGNOS 19.7 in the normal inactive slot when needed; it no longer
silently skips that OS transition. Reuse the same model files. If settings were
changed during experimentation, use **Settings → SunnyLink → restore settings**
with the saved backup, then review the requested controller/model modes while
off-road. Vehicle firmware confirmation is not imported from a different device's
backup; the existing same-device confirmation is preserved by a normal branch
switch. Do not select the opendbc dependency branch as device software.
