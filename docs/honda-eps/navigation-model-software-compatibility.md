# Nav Bigv2 software compatibility

This is the current workflow for `ZakariaSaad/ZakariaSaad`, branch
`mvl-honda-eps-selector21-chestnut-v28`. The standalone navigation hardware
framework and its service orchestration have been removed. This workflow reuses
the successfully compiled and registered model. No model download, compilation,
registration repeat, GPU benchmark, service interruption or firmware operation
is part of validation.

Navigation Model Driving remains disabled and unapproved. Navigation Alpha maps
and directions, ordinary model selection and normal SunnyPilot updates continue
independently.

## The registered package

The supplied compiler report records physical AMD LLVM `gfx1200` execution over
USB using tinygrad `69fd4d34d4430036378aba2b43f9d339cdd6ea73`. All six outputs,
including the five recurrent queues, passed independent source/reload checks on
seeds 42 and 43. Its model contains 671 kernels. The later registration action was
intentionally file-only; `gpu_execution_performed: false` in that receipt does
not contradict the compiler's physical execution evidence.

| Identity | Reviewed value |
| --- | --- |
| Source ONNX | `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e` |
| Existing model | `/data/nav-bigv2/bigv2-amd.sptg` |
| Model SHA256 | `feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4` |
| Compiler report | `/data/nav-bigv2/bigv2-amd.verification.json` |
| Compiler source SHA256 | `d55aa36999ce05fa545d77dc1cc28e1e46fc28e60620c0b0db816e698bf79e7a` |
| OS04 warp SHA256 | `f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b` |
| Registered metadata SHA256 | `2f15fb98990afff3da873ada6180f8328e3af51baa771114ff4c9c6ff428fbf8` |
| Registration receipt | `/data/navigation-models/latest-registration.json` |

The receipt's exact manifest is:

```text
/data/navigation-models/big-nav-v2-7356adbbb1025695ae42a44462107d938059103c5fb446e8839cee34e9aac99c/manifest.json
```

Its status is `registered_disabled`, with `prepared: true`, `registered: true`
and `activation_allowed: false`. The current validator discovers this receipt;
you do not need to type the long manifest path or edit JSON.

The OS04 warp packs two 1344×760 NV12 camera frames into uint8
`[2,6,128,256]`. Its stride is 1408, aligned Y/UV heights are 768/384 and copied
extent is 1,622,016 bytes per camera. Saved-capture validation matched all 393,216
output bytes. Runtime checks actual camera geometry before candidate loading;
the hardware name alone does not identify the sensor.

## One command from Windows PowerShell

First obtain this source update through your existing SunnyPilot Software
settings and restart normally when ready. Keep the same repository and branch;
no reinstall or branch switch is needed.

From Windows PowerShell, run:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && PYTHONDONTWRITEBYTECODE=1 python3 -B -m openpilot.selfdrive.navigation.software_compatibility --output /data/navigation-models/software-compatibility.json"
scp comma@10.0.0.134:/data/navigation-models/software-compatibility.json .
```

Use your existing SSH key/password arrangement. Run as `comma`, without `sudo`.
This uses installed system Python, with no virtual environment. It suppresses
Python bytecode writes. Hashing reads the existing model; it does not duplicate
weights or create another compiled artifact. SunnyPilot remains running. The
validator never opens GPU, camera, Panda or CAN interfaces, writes Params, changes
model selection or enables driving. Because validation only reads files, it does
not require an off-road-status override or an ignition condition.

The command prints concise JSON and saves the same report at the path shown.
A failed check gives a nonzero exit status and a concrete reason. Send the JSON
report for review. Missing or changed files are compatibility failures; they are
not instructions to compile or register the model again.

## What the report establishes

File validation binds the receipt, existing manifest, compiled model hash,
SPTGPLN1 header, typed input/output declarations, recurrent pairs, kernel and
buffer layout, compiler evidence, runtime pin and registered camera profile.
It inspects data-only metadata and never executes artifact pickles or kernels.
It also reports the configured ordinary small-model fallback and driving status.

The fallback parser follows the production compiler and loader formats:

| Compiler layout | Native part keys |
| --- | --- |
| Native `compile_modeld_v2.py` | `run`; its external camera warp is fused by the loader |
| Legacy `compile_modeld.py` split, multiple-policy or supercombo | `run_policy` plus camera-resolution tuples such as `(1344, 760)` and `(1928, 1208)` |

`write_bundle()` stores tuple keys as JSON arrays; validation restores their
tuple identity before checking duplicates. Dimensions must be positive integers,
so booleans, floats, malformed arrays and duplicate resolution keys are rejected.
The legacy compiler combines vision and policy inside `run_policy`; there is no
required `run_vision` part. Tuple-only warp bundles are incomplete ordinary QCOM
fallbacks. Kernel signatures use compact slots into the kernel's `globals`
selection, which may omit unused outer arguments. All argument views still
receive bounds checks, and catalog hashes and QCOM target checks remain required.

The supplied on-device report for `e687b73c` passed all Bigv2, warp, compiler-proof
and software-review checks. Its sole fallback failure exposed the old parser's
string-only part-key assumption; it did not indicate an artifact or GPU failure.

The accompanying automated software suite exercises the actual navigation
producers/readers, tensor adapter, native output parsing and model lifecycle.
It covers route-generation identity, reroutes/destination changes, stale or
missing navigation, timestamp conversion, recurrent reset, 132 valid priming
frames, candidate/fallback initialization and failure, warm fallback, OFF-mode
behavior, settings, UI and controller/updater preservation. Those test results
are source-specific; the static command does not pretend to rerun every
integration test on the device.

## Activation decision

The [activation record](navigation-model-activation-workflow.md) separates artifact
evidence, software compatibility, runtime observations and explicit driving
authorization. Registration and a passing file report cannot enable the model.

The specific unmeasured runtime paths are native camera-to-AMD transfer and
inference at the 20 Hz scheduling contract while the ordinary QCOM small-model
fallback remains warm, together with actual invalid-navigation/failure handover.
The compiler report proves physical source/reload execution, but contains no
latency or concurrent-fallback observation. The static validator labels these
unverified; it does not replace that missing evidence with an endurance or
thermal-certification requirement.

After reviewing the report and integration results, a future reviewed branch
change must record the exact admitted hashes and relevant runtime evidence,
retain the correct 20 Hz contract and fallback policy, and carry explicit owner
authorization before branch activation approval can change. The device setting
is a separate opt-in. No manual JSON edits, hidden Params or source edits on the
comma are required. No authorization for live control is supplied by this update.

The custom Honda EPS controller, MVL longitudinal behavior, Selector 21, Chestnut
catalog v28, pinned tinygrad, USB/firmware handling, production protections and
normal updater are outside this compatibility tool's mutation scope. Navigation
instructions enter model tensors; they never directly issue actuator commands.

## Earlier reports

The investigation, Stage 2, temporal review and previous integration/audit files
are historical source and experiment records. Their old missing-artifact status
and retired diagnostic procedures are not current requirements. Use this page
for the installed-branch workflow.
