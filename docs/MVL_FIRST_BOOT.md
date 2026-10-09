# MVL fork startup audit and in-place switch

Target: `ZakariaSaad/ZakariaSaad`, `mvl-cinque-v3`. Exact source baseline:
`73dca34cece29339f43835f1c9296a5809586f4b` on `sp-honda-dev-202608`.
Hardware under consideration: comma 4, 2022 Honda Insight EX Bosch, Chestnut AMD.

This procedure is for a reviewed, parked, **non-actuating** startup test. It is
not an assisted-driving validation or authorization. No device was accessed or
modified during development. The original AMD V3 pickle remains untested here.

## Startup policy

| State | Selection and behavior |
| --- | --- |
| V3 disabled, no fault | Original MVL slot/default and Chestnut selection. No V3 imports or fallback verification. |
| V3 disabled, old fault | Same ordinary MVL selection. Fault remains diagnostic data; stale V3 runner cache is ignored. |
| V3 enabled, old fault | Reject the request and persistently disable V3. Never clear the fault automatically. |
| Explicit V3 request, Chestnut absent | Disable V3 and load ordinary MVL selection. |
| Explicit V3 request, fallback missing/corrupt | Disable V3 before candidate initialization; return to ordinary MVL selection, or stop this runner so manager can select stock modeld. |
| Explicit V3 request, verified fallback, bad/missing manifest or candidate failure | Latch V3 off; use the verified small fallback for the current session. |
| Active V3 inference/output/publication failure | Discard the frame, latch V3 off, reset fallback and publication/action state, then process a fresh frame. |
| Watchdog expiration | Persist the fault, exit modeld, request one manager restart, and reverify fallback. No repeated V3 initialization. |
| Manager restart/reboot/ignition transition after fault | Clear runtime recovery flags and restore ordinary MVL selection. Keep persistent fault and keep V3 disabled. |
| No usable ordinary model | Report initialization error and publish nothing. Existing model health/disengagement checks remain in effect. |

`CinqueV3UseFallback` and `CinqueV3RecoveryPending` are transient recovery state,
not settings inferred from a fault string. Manager clears them at startup and
on road-state transitions. The restart request is consumed before spawning;
ordinary model errors do not request repeated restarts. If reverification times
out too, recovery flags are cleared and one final ordinary-model start may be
requested; it cannot retry V3. A recovery exception stops publication.

`CinqueV3Fault` and `CinqueV3InitJournal` persist. Initialization is journaled
before fallback verification. Successful initialization removes only its own
journal, never a recorded fault. A later startup converts an interrupted journal
to a fault before model selection. No boot, update, installer, or successful
offroad check clears a fault or enables V3. Only a separately reviewed operator
action can clear a fault and request V3.

## Defects found and fixed

| Defect | Fix |
| --- | --- |
| A fault string forced modeld and manager into the QCOM recovery path while V3 was disabled. | Separate explicit requests, transient recovery, and persistent diagnostics; ordinary MVL selection is the default. |
| An unavailable V3 fallback prevented a working original model from starting. | Reject V3, then use ordinary MVL loading or hand off to stock modeld. |
| Runner cache could retain the V3 QCOM choice after failure or be malformed/unusable. | Invalidate cache on disabling V3; reject malformed/unsupported cached values and resolve ordinary selection when a fault is present. |
| Initialization reused the fault key as a journal and automatically cleared it; fallback verification was not journaled. | Dedicated persistent journal, established before verification; faults are never automatically cleared, and incomplete journal cleanup cannot replace the original fault. |
| The watchdog assumed manager would respawn dead modeld, but NativeProcess.start returns when a Process object already exists. | Modeld-specific process subclass consumes a bounded watchdog restart request; ordinary process behavior is unchanged. |
| A late GPU return could race watchdog persistence and overwrite recovery state. | Join the cancelled/expired timer before returning or raising, so durable fault/restart writes complete first. |
| A missing small model aborted even a successfully warmed Chestnut model. | Permit that selected big model to start; later GPU failure with no small model is explicitly fatal. |
| GPU-to-small fallback retained big-model metadata, delays, and publication/action history. | Reset those using the actual replacement model, for both V3 and ordinary Chestnut failures. |
| Queued asynchronous Chestnut status writes could arrive after completed startup/recovery and leave stale loading/active state. | Use blocking writes for modeld's startup and transition status, preserving their ordering. |
| Partial chunk checksum lists could pass fallback validation. | Require a valid checksum for every declared chunk before loading the fallback. |
| A catalog model with an AMD warp could appear to be an independent small fallback. | Reject AMD-dependent fallback execution/warps; verify real independence with Chestnut disconnected on the device. |
| Malformed selected-bundle Params JSON could crash manager selection or offroad catalog validation. | Record the parsing error and use MVL's hardware-default runner; retain diagnostic settings. |
| A V3 runner returning no predictions silently produced missing publications. | Treat it as a V3 failure and recover; legacy frame skipping remains supported. |
| A request withdrawn or fault written during initialization could still activate the warmed candidate. | Recheck request/fault before activation and before each live frame, then recover without publishing that frame. |
| A late serialization error could occur after the first V3 message had been sent. | Serialize every V3 message before sending any. Transport failures can still interrupt a multi-topic send; health handling must be tested on hardware. |

Honda EPS firmware, torque limits, safety, tuning, braking, CAN, longitudinal
logic, dependencies, and submodule pins are unchanged. Baseline source changes
are limited to Params keys, modeld, model-selection helpers, and the justified
modeld process descriptor. `manager.py`, the launcher, and updater are unchanged.

## What the updater actually does

The launcher uses `/data/safe_staging/finalized/.overlay_consistent` and the
checkout's `.overlay_init` timestamp to decide whether to replace the checkout.
A newer `.git` entry makes it skip a stale staged update. An existing
`/data/safe_staging/old_openpilot` prevents overwriting that backup.

Updater operates on an overlay of the checkout, fetches from **origin**, checks
out `UpdaterTargetBranch`, restores pinned submodules, and finalizes the new
tree. It may delete/recreate the staging root, so it is not a durable backup
location. It runs git reset/clean inside the staged tree: store private compiled
models and backups outside the checkout. Standard model storage is
`/data/media/0/models`; the local V3 source is `/data/media/0/cinque_v3_local_tinygrad.pkl`.

`UpdaterTargetBranch` is **CLEAR_ON_MANAGER_START**, not persistent. After a
reboot, absent a new user selection, updater targets the actual local checkout
branch. Editing this Param alone cannot reliably change the installed branch.
Changing `GitRemote` or `GitBranch` Params also does not change Git: manager
rewrites them from checkout metadata. The actual origin, local branch, tracking
ref, and staged tree must all agree. No normal reboot/update mechanism needs to
be removed or disabled to keep the fork selected.

## Before touching the device (requires device access)

1. Review the published changes and passing CI. Keep the vehicle harness
   disconnected and power comma externally. Engine off alone is not physical
   isolation. Do not run controls, CAN injection, maneuvers, or onroad simulation
   against a connected vehicle during this test.
2. Verify SSH works over the local network independently of cloud registration.
   Keep a second SSH session and a known way to power-cycle/recover the device.
3. Record `git -C /data/openpilot status --short --branch`, HEAD, origin URL,
   `git submodule status --recursive`, `/VERSION`, and the current launcher pane
   (`tmux list-panes -a`). Confirm the working baseline is the exact SHA above.
   If the checkout is a symlink/worktree, has a different layout, or HEAD differs,
   stop and adapt the backup/recovery plan before switching.
4. Record current model-slot Params, model filenames, catalog checksums, the
   local V3 file's independently computed checksum, and free space (`df -h /data`).
   Keep every compiled model and the original local V3 pickle. Do not install V3
   as part of this first boot. Confirm the baseline AGNOS version matches
   `launch_env.sh` (19.7); this branch does not change the OS manifest.
5. Stop the existing launcher/manager cleanly using Ctrl-C in its **identified**
   tmux pane. Confirm manager and updated have exited. Do not kill SSH/athena.
   Do not edit Git while updater is still running. This is temporary maintenance;
   reboot later restores the ordinary launcher and updater.

## Back up, neutralize old staging, and switch in place

Run these only after the preceding checks, in the existing installation and its
existing Python/SCons environment. These commands have **not** been run on your
device. Stop on any error. Do not use reset --hard, clean, uninstall, or force
checkout to overcome local changes.

```sh
export MVL_BACKUP_DIR="/data/mvl-switch-backup/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MVL_BACKUP_DIR"
# Ensure sufficient space; reflink falls back to a full copy when unsupported.
sudo cp -a --reflink=auto /data/openpilot "$MVL_BACKUP_DIR/openpilot"
sudo cp -a /data/params "$MVL_BACKUP_DIR/params"
git -C /data/openpilot remote -v > "$MVL_BACKUP_DIR/remotes.txt"
git -C /data/openpilot submodule status --recursive > "$MVL_BACKUP_DIR/submodules.txt"
git -C /data/openpilot status --short --branch > "$MVL_BACKUP_DIR/status.txt"
git -C /data/openpilot rev-parse HEAD > "$MVL_BACKUP_DIR/working-commit.txt"
```

Validate that the copy is complete and its HEAD is exactly `73dca34…`. The copy
preserves uncommitted/ignored files and native binaries. Separately back up any
models kept inside the checkout or staging tree to durable `/data/media/0`
storage. Models already outside these trees stay in place; do not duplicate or
delete them just to switch software. Preserve source/submodule modifications in
local rescue commits or stashes **in each affected repository**, after making
the full copy. Proceed only once tracked/untracked source changes are accounted
for and checkout/submodule status is clean. Never apply old experimental runtime
patches to the new branch automatically.

Acquire updater's real lock in an interactive shell; perform the rest of this
section while holding it:

```sh
flock -n /tmp/safe_staging_overlay.lock bash
set -e
if mountpoint -q /data/safe_staging/merged; then
  sudo umount /data/safe_staging/merged
fi
if [ -e /data/safe_staging ]; then
  sudo mv /data/safe_staging "$MVL_BACKUP_DIR/safe_staging"
fi
cd /data/openpilot
if [ -e .overlay_init ]; then mv .overlay_init "$MVL_BACKUP_DIR/overlay_init.old"; fi
if [ -e .overlay_consistent ]; then mv .overlay_consistent "$MVL_BACKUP_DIR/overlay_consistent.old"; fi
```

If the lock cannot be acquired or unmount is busy, stop and investigate active
processes; do not use a lazy unmount to hide the race. Preserve archived staging,
especially `old_openpilot`. Do not restore its finalized update on top of the
fork. Updater will create a fresh staging area normally after reboot.

For an installation whose only upstream remote is the original MVL origin:

```sh
git remote rename origin mvl-upstream
git remote add origin https://github.com/ZakariaSaad/ZakariaSaad.git
git fetch origin refs/heads/mvl-cinque-v3:refs/remotes/origin/mvl-cinque-v3
git switch --create mvl-cinque-v3 --track origin/mvl-cinque-v3
git submodule sync --recursive
git submodule update --init --recursive
git status --short --branch
git remote -v
git rev-parse HEAD
git submodule status --recursive
```

If a remote or local branch with either name already exists, inspect it and use
the matching existing ref; do not rename/replace/force it blindly. Do not change
the old `sp-honda-dev-202608` local branch. Verify the fetched commit is the one
reviewed and validated by CI, the fork is origin, and the old origin URL is
preserved as `mvl-upstream`.

Use the existing device environment (no dependency or model reinstall) to
rebuild Params **before** using new keys. Preserve the prebuilt marker so normal
first-boot build cannot accidentally be skipped:

```sh
if [ -e prebuilt ]; then mv prebuilt "$MVL_BACKUP_DIR/prebuilt.old"; fi
scons --minimal -j4 openpilot/common/libparams_c.so
python3 - <<'PY'
from openpilot.common.params import Params
p = Params()
assert p.get_bool('IsOffroad'), 'Maintenance must remain offroad'
p.put_bool('CinqueV3Enabled', False, block=True)
p.put_bool('CinqueV3Active', False, block=True)
p.put_bool('CinqueV3UseFallback', False, block=True)
p.put_bool('CinqueV3RecoveryPending', False, block=True)
p.put_bool('ExperimentalMode', False, block=True)
p.put_bool('OpenpilotEnabledToggle', False, block=True)
p.put('UpdaterTargetBranch', 'mvl-cinque-v3', block=True)
PY
exit  # release the flock maintenance shell
```

The Openpilot/Experimental toggles are explicitly disabled for the bench test;
their previous values are in the backup. No model slots, tuning, user profiles,
credentials, or persistent fault records are cleared. If Params rebuild fails,
stop and recover; do not bypass the build with `prebuilt`. QuickBoot may create
a new marker after the successful first build, as in ordinary MVL behavior.
Reboot normally only once the review/backup/preflight checks above are satisfied.

## First boot verification (requires device access)

- Keep vehicle communication physically disconnected and V3/Openpilot toggles
  disabled. Watch the normal build and launcher; save `/tmp/launch_log`, tmux
  output, boot/crash logs, and any build error. A build/import error is a failed
  test, not a reason to skip the build or safety checks.
- Check actual Git HEAD/branch/origin/submodules again after reboot. Confirm
  the updater targets `mvl-cinque-v3` and any new finalized tree belongs to the
  fork/branch. Repeat after a second reboot and later after a reviewed update.
- Confirm UI, manager, parked device state, preserved Honda settings, registration,
  model-slot selections, files/checksums, and local SSH. Check managerState and
  logs for repeated process failures. Modeld normally does not run offroad;
  its absence at parked boot is not proof that inference initialized.
- From the normal checkout environment, inspect ordinary selection:

  ```sh
  cd /data/openpilot
  python3 tools/mvl_startup_check.py
  python3 tools/mvl_startup_check.py --warmup --camera-size 1344x760
  ```

  The first command reports metadata/file presence without initializing a model.
  `--warmup` is offroad-only, rejects enabled V3/interrupted initialization, checks
  selected catalog artifacts where available, and loads/resets the **ordinary**
  model without publishing predictions or starting controls. Its separate process
  has a 90-second deadline and bounded native GPU waits. Match camera dimensions
  to the real device; use `1928x1208` only for that configuration. Capture the
  result with Chestnut connected and disconnected. Confirm the actual loaded
  device/model agrees with selection; a GPU-to-small fallback must be recorded.
- A pending initialization journal is a failed V3 preflight needing reviewed
  diagnosis; keep V3 disabled. Do not clear an old fault simply to pass a check.
- Before any assisted-driving consideration, validate real cameras and recorded
  frame replay, actual model messages/latency, hardware disconnects/watchdogs,
  manager recovery and vehicle health/disengagement, and Honda behavior. Perform
  these in a non-actuating harness first. Synthetic fixtures are not those tests.

Passing this checklist supports a parked startup result only. It cannot verify
the existing AMD V3 artifact, onroad temporal behavior, or assisted-driving safety.

## Recovery if custom boot fails (requires SSH/device access)

Keep the vehicle isolated. Stop launcher/manager/updater, reacquire the same
updater lock, unmount staging normally, and archive the new staging tree outside
`/data/safe_staging`. Save custom launch/crash logs and a fresh Params snapshot
for diagnosis. Do not restore a stale finalized overlay. Set the backup path to
the **verified** original copy created above, then preserve the failed checkout:

```sh
export MVL_BACKUP_DIR=/data/mvl-switch-backup/YOUR_VERIFIED_BACKUP
# While holding the updater lock, after stop/unmount/archive steps:
sudo mv /data/openpilot "$MVL_BACKUP_DIR/failed_custom_openpilot"
sudo cp -a --reflink=auto "$MVL_BACKUP_DIR/openpilot" /data/openpilot
git -C /data/openpilot rev-parse HEAD
git -C /data/openpilot branch --show-current
git -C /data/openpilot remote -v
git -C /data/openpilot submodule status --recursive
```

Require exact HEAD `73dca34cece29339f43835f1c9296a5809586f4b`, branch
`sp-honda-dev-202608`, the recorded original MVL origin, and recorded submodule
pins. The restored checkout includes the original native build and environment.
If these checks fail, do not boot a guessed ref. Do not reset a dirty rescue
checkout. A separate source-only recovery can fetch/check out the exact commit
and rebuild while offroad, but it is less reliable than the complete known-working
copy and requires its dependencies/build to pass.

Keep `/data/params` and `/data/media/0/models` in place. Do not wholesale restore
old Params over current settings; the saved snapshots are for diagnosis/selective
reviewed recovery. Before reboot, use the **restored baseline's** Params API to
set `OpenpilotEnabledToggle` and `ExperimentalMode` false and set
`UpdaterTargetBranch` to `sp-honda-dev-202608`. Baseline does not know the new V3
keys and will not run V3. Preserve their fault/journal data in the saved snapshot.
Release the lock and reboot normally. Recheck exact HEAD, original model, UI,
settings, and SSH. Future normal MVL updates may advance upstream; inspect staged
updates before installing them. Do not automatically restore assisted-driving
toggles or enable V3 after recovery.

If SSH is unavailable, physical access and the device's recovery facilities are
required. Software running on a failed checkout cannot guarantee remote recovery.
Do not begin the switch without a verified reachable SSH path and recovery plan.

## Validation boundary and report

Local validation passed **128 tests**, with **one existing legacy-suite skip**:
88 compatibility, startup, import, loop, manager, updater, and Honda tests, plus
40 legacy compiler/parser tests. Ruff, native Params rebuild, YAML parsing, and
diff checks passed. CI also enforces unchanged baseline safety/control, launcher,
updater, manager, hardware, dependencies, and submodule files. Consult the
published commit's GitHub Actions run for the independent remote result.

CI exercises production startup policy and runner predicates with real temporary
Params stores; hardware/model factories are replaced for failure injection.
Fresh interpreters import modeld/adapter/manager/updater in both orders. The real
modeld wrapper is executed with `--help`. Manager initialization runs with native
Params, while registration/bootlog external effects are isolated. Harmless real
child processes verify bounded restart consumption. The real frame loop and
Cap'n Proto serialization run with synthetic camera/vehicle/model fixtures.
Actual Tinygrad CPU recurrence, pickle roundtrips, warps, and adapter publication
are covered separately. Updater Git fetch/finalize and launcher overlay gates run
against temporary repositories; privileged overlay mounting and AGNOS are not run.

These tests cannot validate QCOM/AMD kernels or drivers, USB power/hotplug, real
camera buffers/calibration, AGNOS device dependency availability, full native
device build, real registration, actual overlay mounts/permissions/free space,
physical power-cycle persistence, or a live controls fallback/disengagement.
StarPilot success does not establish MVL artifact/runtime compatibility.

The branch is a candidate for a reviewed, parked, physically non-actuating startup
test after CI passes and the backup prerequisites are met. It is not validated
or approved for assisted driving; V3 remains optional and disabled by default.
