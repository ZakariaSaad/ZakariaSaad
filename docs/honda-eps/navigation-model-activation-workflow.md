# Nav Bigv2 selection and activation

> **Historical / superseded:** This document records the retired original Bigv2 activation and startup fixes.
> Its original-model selection/activation commands are no longer supported and
> must not be used as current instructions. Use [SunnyLink model and EPS controls](sunnylink-model-and-eps-controls.md)
> and [corrected artifact evidence](navigation-model-corrected-integration.md).
> The original `big-nav-v2` is retired; useful evidence and technical findings below are retained.


Use the already installed `/data/nav-bigv2/bigv2-amd.sptg` and its registered
manifest. No model download, recompilation, duplicate copy or fresh software
installation is needed. Update the installed
`mvl-honda-eps-selector21-chestnut-v28` branch through the normal Software
settings, then restart when prompted. The repository and update branch stay the
same.

## Startup-failure fix after `9a2c757`

If Navigation Model Driving was enabled during the failed startup, run the SSH
`disable` command below **before updating**. Software updates preserve that
setting; this fix does not remotely withdraw or enable device permission. Then
use the normal Software update and restart on the same branch. Leave navigation
driving off until you explicitly choose to enable it again.

The runtime now distinguishes tinygrad's empty automatic renderer/interface
configuration hints from actual AMD USB device identity. The physical
`gfx1200`/USB checks and the pinned artifact's LLVM target checks remain enforced.
The missing upstream `ChestnutRetry` parameter is registered with its original
reset flags, preserving the existing one-time retry and ordinary small-model
fallback. Loading and active flags are cleared synchronously on model-startup
failure, including failure to initialize the fallback; a late loader cannot
reactivate Chestnut after fallback selection. No model recompilation, new
hardware diagnostic or model-selection change is required.

The native Params-library build and focused startup/publication regressions
passed: **27 tests and 90 subtests**, plus lint and type checks. Tests use real
registered Params and the pinned tinygrad device/interface types without opening
a GPU. This is software regression evidence; the physical device was not accessed.

## SSH commands

From Windows PowerShell, inspect the installed model and its activation blockers:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage status"
```

After installing the owner-authorized approval update,
explicitly select Bigv2 and request Navigation Model Driving while the comma
genuinely reports off-road and disengaged:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage enable big-nav-v2"
```

To withdraw navigation-driving permission without changing either ordinary model
selection:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage disable"
```

Run these commands as `comma`, without `sudo` or a Python environment activation.
They use installed system Python and the existing native Params interface. They
do not stop `comma.service`, open the GPU, transmit CAN, flash firmware or change
control limits. `status` is read-only. A refused `enable` reports its exact reason
and leaves selection unchanged. `disable` can always withdraw permission.

Enabling uses the same file admission as runtime startup and Settings: branch
approval, exact registered model/warp/metadata hashes, compiler verification and
the configured ordinary QCOM fallback. It reads fresh `deviceState.started`,
`managerState` and any available fresh control-state messages, together with
`IsOffroad` and `IsEngaged`. The manager must confirm that vehicle-control and
driving-model processes are stopped. Physical ignition being off is not an
additional requirement. The state is checked again after file hashing, before
the driving flag is written. Enabling a setting does not engage vehicle controls.

## comma 4 Models interface

1. Open **Settings → Models → select model → big models**.
2. Select **Nav Model Integration Bigv2**, identified as **installed local ·
   navigation aware**. Its eligibility appears beside it. This selects only the
   local navigation candidate and leaves Navigation Model Driving off; ordinary
   small and big model selections are preserved.
3. Open **Models → navigation models → navigation model status** to see the
   registration, review and approval status.
4. After installing the approval update, explicitly turn on
   **navigation driving** from that same page while off-road. Settings runs the
   same admission checks as the SSH command. On another supported UI layout the
   toggle is labeled **Navigation Model Driving** directly in Models.

Turn that toggle off, or use the SSH `disable` command, to return to the existing
ordinary selections. Selecting an ordinary Big Model also withdraws navigation
permission before proceeding through its normal selection path. Bigv2 uses its
dedicated native navigation loader, not an ordinary Selector generation or a
downloadable catalog bundle. Selector 21, Chestnut catalog v28 and the other big
models remain available.

## Owner-authorized experimental approval

The collected records support explicit experimental activation of this
exact hash-pinned model: physical AMD source/reload compiler verification passed;
registration and saved-capture camera-warp verification passed; all seven actual
device file-compatibility checks passed; and the upstream-aligned runtime and
controller-preservation regressions passed. No demonstrated unresolved software
contract defect appears in these records. Repeating those checks or completing a
new hardware-validation program is not required for this approval decision.

**The owner explicitly authorized the approval change.** The applied change in
`openpilot/selfdrive/navigation/driving_models.json` is:

```diff
-      "activation_approved": false,
+      "activation_approved": true,
```

All pinned hashes, `runtime_compatibility: "unverified"` and the recorded runtime
scope remain unchanged. Branch approval does not enable the device setting.
After installing this update, the owner must explicitly enable Navigation Model
Driving through Settings or SSH while off-road. All registered-file, fallback
and fresh software-state admission checks remain in force. Neither the CLI nor
the local manifest can supply branch approval for an unapproved candidate. The
owner does not need to edit JSON, source files or hidden parameters.

The specific unobserved behavior is the composed live camera → registered warp →
native AMD Bigv2 → prediction-publishing path and its inference timing. The
physical compiler report starts with prewarped model inputs; saved-capture warp
execution was verified separately. These records therefore do not claim measured
live 20 Hz performance or physical inference-failure handover. Those are disclosed
experimental limitations, not evidence that compilation or registration needs to
be repeated. Existing startup camera/GPU checks, normal warmup, error handling
and ordinary small-model fallback remain in place.

The runtime executes one active model per frame. It does not add shadow
inference, continuous simultaneous small-model execution or a special activation
frame counter. The Honda EPS controller, MVL longitudinal tuning, Panda safety,
Chestnut firmware and normal updater are preserved.

## Focused verification for this workflow

The starting commit was `35f7946ccd1f23dd44ddfde82b203f364d265dc2`.
The focused CPU/Xvfb regression run passed **118 tests and 173 subtests**, covering
local model selection, responsive Settings activation, shared admission, native
message freshness, CLI behavior, system-Python Params access, native fallback
formats and existing runtime publication/fallback boundaries. Ruff and scoped
type checks passed. Controller, safety, model-runtime/compiler, production
catalog, updater and submodule revisions are unchanged. No GPU test, download,
compilation, service stop or deployment was performed for this change.

The owner-authorized approval update starts from
`a194dd92e4d3250825fcf94ccb1c894e34403c62`. Its focused admission, selection,
CLI, rendered UI and runtime-publication/fallback checks passed **59 tests and
41 subtests**. The sole production change is the approval boolean; all verified
hashes, device opt-in defaults, safety protections and model configurations are
unchanged. No device setting was activated remotely.
