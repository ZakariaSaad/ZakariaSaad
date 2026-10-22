# Navigation-model research branch

For the current integrated source, selection/status and complete physical off-road workflow, use the [software integration report](navigation-model-integration.md) and [hardware guide](navigation-model-hardware.md). This earlier prototype guide is retained as historical investigation evidence.

The [Stage 2 report](navigation-model-stage2.md) is the current development guide,
including route/frame association, recurrent resets, file-only CPU/Chestnut
diagnostics and updated inference results. The sections below retain the Stage 1
record at `0556a2ea6`; its outstanding-work list is historical.

This is an experimental PC offline/replay integration, not a driving release.
Navigation Alpha's map and instructions remain available as before. This work
does not enable navigation-conditioned driving on a comma device.

## Branch and preserved baseline

- Repository: `ZakariaSaad/ZakariaSaad`.
- Experimental branch: `mvl-honda-eps-nav-model-testing`.
- Direct parent baseline: `mvl-honda-eps-selector21-chestnut-v28` at
  `4851eacac8fce2e875a0ef7703af5f5025fc0c15`.
- No merge into either existing working branch is authorized.

The baseline already contains Navigation Alpha. Honda EPS, MVL controls,
Panda/opendbc, the seven submodule pins, model assets, selector/catalog code,
Chestnut firmware handling, dependency locks, and the updater are unchanged.
Both runners retain their original firmware-aware action timing calculations.

## Implemented experiment

1. `navd` publishes future `allManeuvers` using geometric distance to each
   Mapbox step's starting maneuver. Incomplete future maneuver metadata produces
   an empty list while preserving the existing display instruction.
2. Navigation helpers encode the first supported future maneuver as 46 Valhalla
   command channels plus distance in hundreds of meters. Unknown/nonfinite
   instructions are rejected; this does not add direct actuator commands.
3. Native stock and SunnyPilot model adapters pack each input using its declared
   dtype, with the existing 128-byte alignment. A uint8 map remains uint8; float32
   map models receive a numeric cast of raw 0..255 pixels. The inspected graphs
   normalize internally. Recurrent inputs/outputs preserve their original types.
4. The shared optional input reader checks the display toggle, destination,
   GPS validity, instruction receipt age, map delivery age, source timestamp
   relative to camera capture, dimensions, and finite inputs. Missing/rejected
   data yields zero new navigation inputs in isolated experiments.
5. Navigation-capable artifacts require a PC, `NAV_MODEL_OFFLINE=1`, and the
   persistent `NavigationModelInputsEnabled` parameter. That new parameter
   defaults to **false** and has no on-device enable control.
6. Both normal `modeld.main()` entrypoints reject navigation-capable active or
   fallback models before creating prediction publishers, even on an opted-in
   PC. Direct `ModelState` use in the isolated harness is the supported experiment.

The public Big and Bigv2 input/output signatures are recorded as compact JSON
test fixtures. No model weights, compiled artifacts, catalog entries, firmware
updates, or dependency changes are included.

## Offline test commands

Use the existing supported development environment with dependencies and native
bindings built. The UI tests additionally require working Git LFS image/font
assets and an X server, such as Xvfb. From the repository root:

```sh
scons -j4
DEV=CPU python -m pytest -q openpilot/selfdrive/navigation/tests \
  openpilot/selfdrive/modeld/tests/test_navigation_inputs.py \
  openpilot/sunnypilot/modeld_v2/tests/test_navigation_inputs.py \
  tools/tests/test_navigation_model_offline.py
```

The contract tests inspect signatures and synthetic buffers; they do not execute
the public learned models. A separate research probe successfully executed the
public Bigv2 ONNX twice on CPU LLVM with identical synthetic cameras and zero
initial recurrent states. Changing only map/maneuver inputs changed finite plan,
lead, desire and action outputs. This is numerical conditioning evidence, not
recorded-route validation. A later native SPTG/CPU-warp ModelState comparison also passed for one synthetic NV12 frame, with independently warmed/reset arms and finite changed predictions. The broader
preservation and messaging/updater results are in the investigation report.

## Recorded inference harness

`tools/navigation_model_offline.py` runs two independently initialized model
instances on identical prepared recorded cameras/transforms/desires/action
timing: neutral navigation and conditioned navigation. It compares parsed
outputs. It starts no publisher, manager, controller, or CAN interface.

Prerequisites are trusted **CPU-compiled SPTG native driving and camera-warp
artifacts** compatible with this pinned tinygrad, plus a prepared NPZ recording.
The harness rejects GPU targets and checks the warp's copied NV12 extent,
transform shape and model output shape/dtype. An ONNX file or upstream OOB pickle
is not an interchangeable substitute. Public GPU-compiled artifacts must not
simply be renamed or blindly unpickled. No compiled artifacts are bundled.

```sh
NAV_MODEL_OFFLINE=1 DEV=CPU:LLVM python tools/navigation_model_offline.py \
  --artifact /absolute/path/to/trusted-cpu-native.sptg \
  --warp-artifact /absolute/path/to/trusted-cpu-warp.sptg \
  --recording /absolute/path/to/prepared-recording.npz \
  --output /absolute/path/to/comparison.json
```

The harness sets the experimental parameter only inside an isolated parameter
prefix. It loads NPZ with `allow_pickle=False`. Each array must have the same
nonzero leading frame count. Expected arrays are:

| Array | Per-frame content |
| --- | --- |
| `img`, `big_img` | Flat uint8 NV12 bytes, including the expected runtime layout/padding |
| `M_img`, `M_big_img` | Finite 3x3 transforms |
| `desire` | The model's desire input |
| `traffic_convention`, `action_t` | The model's existing auxiliary inputs |
| `nav_valid` | Optional explicit prepared-data validity, defaults to false |
| `nav_map` | When valid: raw uint8 256x256 pixels |
| `navigation` | When valid: float32 length 47; at most one command, nonnegative distance |

Prepared `nav_valid` is an offline annotation, not evidence of fresh GPS or
route identity. Preserve those facts when preparing a recording. The live-style
input reader is tested separately with mocked GPS/instructions/IPC timestamps.
The harness preserves a supplied recording's frame order; it does not decode
routes, retrieve maps, synchronize clocks, or claim a driving-quality score.
The explicit warp override applies only to the constructed offline model; normal
modeld callers retain their original asset paths. For 1344x760, the expected
copied NV12 extent is 1,622,016 bytes per camera (stride 1408, Y height 768, UV
height 384), distinct from the allocator's larger buffer size.

## Known limits before any hardware or driving use

- No learned comparison using recorded driving inputs or multi-frame route
  transitions has been completed. The two synthetic CPU ONNX runs establish only
  that combined map/maneuver changes can affect these learned outputs.
- CPU native compilation and one-frame ModelState inference passed. No QCOM or
  Chestnut AMD compilation, numerical parity, latency, thermal,
  USB transfer, or power behavior has been verified for these public nav models.
- The current Chestnut v28 catalog has no entry for either inspected nav model.
  Existing big-model support remains intact; adding these models needs separately
  validated artifact compilation and catalog metadata.
- Source freshness checks do not establish route-generation identity. During
  destination changes/reroutes, queued frames/instructions or `navd`'s existing
  recompute hold can refer to an earlier route.
- Zeroing new navigation inputs does not clear recurrent navigation history or
  prove a learned safe fallback. Big uses 32-frame raw navigation queues; Bigv2
  uses 128-frame feature queues. Reset policy remains unvalidated.
- The official experimental map renderer additionally extrapolates GPS pose.
  That change is not included; training-map pixel parity has not been proven.
- Unchanged controls can act differently on changed model predictions. Graph
  dependency analysis confirms that navigation can reach plan, lead, desire,
  and action outputs; it does not establish meaningful or safe driving behavior.

Production publication is deliberately unavailable for these artifacts. Do not
install this branch to evaluate navigation-conditioned controls on a vehicle.
No firmware flashing, deployment, PR, or merge was performed for this experiment.

## Provenance

- [SunnyPilot PR #2057](https://github.com/sunnypilot/sunnypilot/pull/2057),
  investigated at `f7f9a3efae3a6552068201bd9e4e4dbe66aae577`.
- [comma experimental navigation-model branch snapshot](https://github.com/commaai/openpilot/tree/afbf2e547e9319e2eb9f515e1e4330af63705cec).
- [comma Navigation Alpha PR #39066](https://github.com/commaai/openpilot/pull/39066).

The implementation selectively adapts maneuver publishing/encoding and typed
input handling. It does not wholesale merge the PR's 44-commit history or its
unrelated workflow change. See `navigation-model-investigation.md` for evidence,
exact model signatures, outstanding blockers, and validation results.
