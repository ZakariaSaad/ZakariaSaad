# Nav Bigv2 software compatibility

> **Historical / superseded:** This document records the completed original-Bigv2 file validation and fallback-parser work.
> Its original-model selection/activation commands are no longer supported and
> must not be used as current instructions. Use [SunnyLink model and EPS controls](sunnylink-model-and-eps-controls.md)
> and [corrected artifact evidence](navigation-model-corrected-integration.md).
> The original `big-nav-v2` is retired; useful evidence and technical findings below are retained.


This records the completed validation workflow for `ZakariaSaad/ZakariaSaad`, branch
`mvl-honda-eps-selector21-chestnut-v28`. The standalone navigation hardware
framework and its service orchestration have been removed. This workflow reuses
the successfully compiled and registered model. No model download, compilation,
registration repeat, GPU benchmark, service interruption or firmware operation
is part of validation.

Experimental Bigv2 branch activation is owner-authorized. Navigation Model
Driving remains off by default and requires manual opt-in after updating; see the
[selection and activation workflow](navigation-model-activation-workflow.md).
Navigation Alpha maps and directions, ordinary model selection and normal
SunnyPilot updates continue independently.

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

The supplied on-device report for `f18366ef91b05339f3bc33526295d20d15da8e05`
passed all seven checks, including the existing Bigv2 artifact, OS04 warp, compiler
proof and configured ordinary QCOM fallback. That report records an 810,417,970-byte
model with 671 kernels and 17 input/output slots. The selected small fallback is
`driving_terrible_super_fantastic_do_over_model_august_15_2026_tinygrad.pkl`, SHA256
`dad05dca91ccb2302ff421b6d0dfd4c2fe93738f0de7e230c5d93dbd55a115ae`, with three native
parts. These are completed compatibility checks; they do not measure inference
performance. This source change preserves both installed model files and their
selection.

The accompanying automated software suite exercises the actual navigation
producers/readers, tensor adapter, native output parsing and model lifecycle.
It covers route-generation identity, reroutes/destination changes, stale or
missing navigation, timestamp conversion, normal recurrent-state operation,
selected-model initialization and failure, one-model-per-frame execution,
ordinary small-model fallback, OFF-mode behavior, settings, UI and
controller/updater preservation. Those test results are source-specific; the
static command does not rerun every integration test on the device.

## Execution and upstream parity

The reference is official comma.ai
[`nav-model-integration` at `afbf2e547e9319e2eb9f515e1e4330af63705cec`](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/modeld/modeld.py).

| Behavior | Current execution policy |
| --- | --- |
| Big-model initialization | Load the approved artifact and matching warp, perform normal initialization/warmup, then use it without an additional activation wait. |
| Ordinary small model | Initialize it as the existing fallback; do not schedule it while the healthy big model is selected. |
| Per-frame inference | Execute only the selected driving model and publish its output through the existing modeld path. |
| Big-model initialization failure | Retain ordinary model operation through SunnyPilot's existing startup path. |
| Big-model inference failure | Report the existing Chestnut failure, discard that failed frame, and run only the small model on subsequent frames. |
| Missing navigation data | Supply neutral tensor defaults rather than changing models or starting another execution path. |
| Recurrent state | Preserve normal initialization and per-inference state updates; no navigation activation counter, shadow samples, or route-driven re-priming policy. |

Fresh map and maneuver inputs follow upstream's independent receipt-age limits:
map delivery younger than 0.5 seconds and a valid instruction younger than
2 seconds. A missing or rejected map becomes a zero map; missing or rejected
guidance becomes a zero 47-element vector. Losing one does not discard the other
valid input or switch the driving model. Float32 map tensors receive raw 0..255
values; normalization remains inside the graph.

The custom 132-frame activation rule, Bigv2 shadow inference and continuous QCOM
inference are removed. The model's real `state_desire_q` still has shape
`[132,1,8]`; that ONNX input dimension is not an activation waiting period.

Deliberate SunnyPilot differences remain for technical compatibility: native
SPTGPLN1 registration and pinned compiler evidence instead of upstream's compiled
pickle format; Selector 21 and the existing ordinary recovery code; Honda EPS
prediction-delay hooks and MVL's existing prediction-consumption interfaces;
and explicit branch activation approval. SunnyPilot's existing `ChestnutRetry`
and on-road-cycle startup request remain for its established power-interruption
recovery; no additional retry mechanism is introduced. Route-generation,
camera-frame, source GPS timestamp and clock-domain guards reject mismatched input data instead of conditioning the
model with another route's pixels. These guards do not schedule another model
or reset learned state simply because navigation data is unavailable. The branch's
navd publishes fix-relative maneuver distances with source timestamps, so distance
extrapolation uses that physical fix time; upstream's untagged receipt-relative
message uses its receipt time. This preserves the meaning of each producer's
distance while preventing stale positions from appearing fresh.
Unsupported Mapbox maneuver types/modifiers remain neutral instead of guessing a
Valhalla turn encoding; the valid map can still condition the model independently.

## Activation decision

The [activation record](navigation-model-activation-workflow.md) separates artifact
evidence, software compatibility, runtime observations and explicit driving
authorization. Registration and a passing file report cannot enable the model.

The specific unmeasured runtime path is native camera-to-AMD transfer, fused
warp, Bigv2 inference and normal prediction publishing at the 20 Hz scheduling
contract, together with the existing inference-failure switch to the initialized
small model. The small model does not execute alongside healthy Bigv2 inference.
The compiler report proves physical source/reload execution, but contains no
measurement of that complete camera-to-prediction path. The static validator
labels it unverified; no combined dual-model workload, endurance or thermal
certification is a prerequisite.

After reviewing the report and integration results, the owner explicitly
authorized experimental branch approval for the exact pinned Bigv2 hashes. The
20 Hz contract, fallback policy and runtime-evidence limitations remain unchanged.
The device setting is a separate manual opt-in through Models or SSH. No manual
JSON edits, hidden Params or source edits on the comma are required. This
file-only validator cannot enable the model.

The custom Honda EPS controller, MVL longitudinal behavior, Selector 21, Chestnut
catalog v28, pinned tinygrad, USB/firmware handling, production protections and
normal updater are outside this compatibility tool's mutation scope. Navigation
instructions enter model tensors; they never directly issue actuator commands.

## Earlier reports

The investigation, Stage 2, temporal review and previous integration/audit files
are historical source and experiment records. Their old missing-artifact status
and retired diagnostic procedures are not current requirements. Use this page
for the installed-branch workflow.
