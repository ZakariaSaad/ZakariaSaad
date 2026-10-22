# Navigation-model inference: Stage 2 technical report

This branch implements an isolated, reproducible navigation-model inference
experiment. **It does not enable navigation-conditioned driving.** No installation,
vehicle connection, firmware update, or live deployment was performed.

Starting branch: `mvl-honda-eps-nav-model-testing`, commit
`0556a2ea62dbf20de9c5b33cb4082a8c6da5260a`. Its known-working selector/Navigation
Alpha foundation remains `4851eacac8fce2e875a0ef7703af5f5025fc0c15`.
Work continues on the same experimental branch without rebasing or replacing
published history. Existing working branches are outside this change.

## Reference parity and actual incompatibilities

References were rechecked on October 21, 2026, America/New_York:

- Official [nav-model-integration](https://github.com/commaai/openpilot/tree/afbf2e547e9319e2eb9f515e1e4330af63705cec),
  head `afbf2e547e9319e2eb9f515e1e4330af63705cec`; initial input wiring
  `8b8ce13b407ddae14f2e415afad9c16f603bdb81`.
- [SunnyPilot PR #2057](https://github.com/sunnypilot/sunnypilot/pull/2057), head
  `f7f9a3efae3a6552068201bd9e4e4dbe66aae577`. It remains an **open draft**,
  unmerged, with 44 commits and six net changed files. Its unrelated LFS workflow
  change and duplicate import are not carried over.
- Public Bigv2 ONNX, 814,591,106 bytes, SHA256
  `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
  This is the official experimental ONNX checkpoint, also redistributed in the
  public `big-nav-model-integration-1d18c42e` release. Weights are not committed.

| Area | Official implementation / PR #2057 | Stage 2 implementation and remaining difference |
| --- | --- | --- |
| Maneuvers | Official publishes step-start maneuvers; the PR reader expects them but Alpha navd did not fill them | Existing geometric publisher is retained, now bound to fresh GPS and a route generation; malformed future metadata withholds the model list while retaining the banner |
| Map inputs | Raw 0..255 map pixels and 47 float32 command/distance values | Same convention; no extra `/255`. Bigv2 map is float32 `[1,1,256,256]`; its graph normalizes internally. Command vector is float32 `[1,47]` |
| Packing | Declared dtypes, 128-byte alignment, recurrent state pairs | Preserved native dtype-aware packed upload/warp fusion; state alias identities survive reset |
| Route identity | No strong route-generation association in the compared transport | Destination-coordinate key plus generation on route, instruction and exact map-frame metadata; destination changes immediately invalidate old routes and in-flight routing responses |
| Frame/source validity | Primarily receipt freshness; official renderer extrapolates GPS pose | Physical GPS timestamps, instruction source timestamps, exact IPC frame/SOF/EOF association, separate source/render/delivery ages, camera synchronization and pose extrapolation |
| Invalid navigation | Missing inputs become zero; old recurrent route state can persist | Diagnostic skips invalid navigation or explicitly runs a separate non-navigation fallback; it never calls zero input a trained safe fallback |
| Recurrent transitions | Generic state buffers; no route-transition policy | Full in-place reset on route/validity/session/clock discontinuity, and camera gaps over 75ms for the 20Hz checkpoint. Recovery starts cold; learned convergence remains unverified |
| Compiler | Official TinyJit pickle/OOB artifact; SunnyPilot native compiler pre-exists through #2050 | Native SPTGPLN1 capture/reload retained and strengthened. Official PKL is not interchangeable with an SPTG file |
| Runtime | Official tinygrad `d3f09c9bbd542fbfbe68c8569a1173550c408969` | Existing pin `69fd4d34d4430036378aba2b43f9d339cdd6ea73` retained. The three official compile/warp/helper recipes are byte-identical across pins; runtime internals and GPU results are not proven equivalent |
| Latest official compiled variant | `3ec09aed` overflow fix and `afbf2e54` n16 update change PKL pointers only | Public ONNX remains `1d18c42e…`; recompilation cannot establish reproduction of undocumented newer binary precision/flags |
| Publication and selection | Official/PR can feed ordinary prediction publishers | Both production modeld entrypoints reject navigation-capable or diagnostic active/fallback models. Experimental catalog is isolated and disabled; production selection/defaults remain unchanged |

This is selective compatibility work, not a wholesale upstream merge. The legacy
shape-only compiler is incompatible with Bigv2's rank-one desire input; the native
`compile_modeld_v2.py` path supports it and was used successfully.

## Route, frame and state implementation

`navd` emits positive route generations and a SHA256 identity of canonical routing
coordinates. Labels do not change identity. A destination change clears the old
route immediately, including during the recompute hold or GPS loss. A routing
response is discarded if its destination changed while HTTP was in flight.
Malformed responses are rejected atomically rather than partly replacing state.

`mapsd` uses the official pose extrapolation and checks destination identity around
map preparation. It publishes additive `navModelStateSP` metadata matching each
map's actual IPC frame ID, GPS source/SOF and render/EOF timestamps. This occupies
the fork's reserved custom slot 10, preserving that struct's wire ID. The historical
`mapRenderStateDEPRECATED` message is unchanged. NavRoute/NavInstruction additions
are additive schema fields. Existing map/instruction UI messages remain available.

The reader associates a received map with its own matching metadata, never with a
newer event merely because it arrived recently. A bounded pending-frame cache
handles asynchronous delivery; a previously verified frame can be held only within
its original freshness limits. Instruction, map and current destination must share
identity. Physical sources must be positive and within 2s of camera time, with
200ms future tolerance; render/delivery ages are separately limited to 500ms.
These are explicit diagnostic thresholds, not demonstrated hardware timing bounds.
Route distance is extrapolated from the instruction's physical GPS sample.

Clock-domain compatibility is an explicit remaining requirement: Python GPS message
and map timestamps use `CLOCK_MONOTONIC`, while native camera timing can use
`CLOCK_BOOTTIME`. Linux suspend can separate these clocks. The reader rejects such
unaligned sources and tests cover that rejection; this change does not rewrite
global clocks or prove post-suspend device recovery. Recorded inputs must share a
known clock domain or undergo a separately validated conversion.

Native producer-to-reader tests exercise real Msgq/VisionIPC transport. Prepared
file recordings independently supply instruction/map identities and frame IDs;
the diagnostic does not invent matching metadata for recorded navigation.
Synthetic navigation requires an explicit flag and is labeled in provenance.

All recurrent camera, desire, feature, map and navigation state buffers, packed
inputs and outputs are cleared **in place**, retaining captured-kernel aliases.
Reset also clears previous desire. Invalid input is skipped unless an explicitly
supplied non-navigation diagnostic fallback is available. Primary failures are
logged and quarantine that instance; failed fallback initialization/warmup is
discarded. Session revocation stops inference and clears history before another
frame. These policies prevent hidden route history crossing diagnostic sequences;
they do not prove cold-start or fallback driving behavior.

## Native compiler verification and artifacts

The pinned native compiler now uses independent seeds 42/43, full uint8 input
range, exact first-seed repeatability, finite checks for **every prediction and
state output**, and separately allocated source/reloaded buffers. Reloaded outputs
must match within `rtol=atol=1e-3`, with `equal_nan=False`. Seed-insensitive primary
outputs and known no-code backends are rejected. Artifacts are staged and promoted
only after verification; failures preserve an existing destination. Atomic JSON
records source/runtime/compiler identities, backend scope, checks and artifact SHA.

Actual Bigv2 CPU LLVM compilation completed in **375.21s**. All six named outputs
were finite and matched the independently reloaded artifact with **maximum absolute
difference 0 for both seeds**:

```
outputs                         float32 [1,18580]
next_state_img_q                uint8   [2,5,6,128,256]
next_state_desire_q              float32 [132,1,8]
next_state_feat_q                float32 [128,1,16384]
next_state_map_features_q        float32 [128,1,64]
next_state_navigation_features_q float32 [128,1,64]
```

The corresponding recurrent inputs omit `next_`. Other input contracts are
`new_img:uint8[2,6,128,256]`, `desire:float32[8]`,
`traffic_convention:float32[1,2]`, `action_t:float32[1,2]`, plus map/vector above.
The existing copied NV12 extent for each 1344×760 camera is **1,622,016 bytes**,
distinct from allocator size; dtype, transform and warp/model output compatibility
are checked before loading.

| Local artifact | SHA256 |
| --- | --- |
| Verified Bigv2 CPU SPTG | `df458e202b29deede75c3cc4be8aa9b4907a592b95b7cebd25d1714364a539aa` |
| Metadata-only safe derivative used for the final fallback experiment | `a280f8873a0391685bf4ea115504203b7448ce5a034b401788cc1e2921384790` |
| Existing non-navigation small model CPU safe artifact | `53ddb5500429688d9b1d0fe256bee0cfc879b74926dde10ff00e8e580282b11f` |
| CPU camera warp | `556cdca4c1740434000e52a0c57f9d0027359d62502deb0c539735af82dc0337` |

The fallback compiled in **49.74s**, with all four outputs finite and exactly
matching source/reloaded execution for both seeds. Model weights, SPTG artifacts,
raw videos and private logs remain outside Git. The safe derivative changes only
nested slice metadata; kernel/constants/check/spec fingerprints were preserved.
Diagnostic adapter loading now restricts slice deserialization and validates bounds.

The long Bigv2 run used a saved compiler source snapshot; only report I/O/provenance
was strengthened while it ran. Numerical verification functions were unchanged by
AST comparison. Final I/O paths were tested using real small CPU artifacts. The
original certificate and derived-artifact hash are recorded separately.

These checks prove local source-to-serialization consistency, **not an independent
ONNX numerical oracle**, CPU/GPU equivalence, learned behavior, or road readiness.

## AMD/Chestnut result

The known-working tracked Chestnut warp declares **AMD / LLVM / gfx1200 / USB**.
The environment exposes no AMD/Chestnut device nodes. A bounded mock attempt used
that target and real instruction emulation. Arithmetic preflight passed; five
learned-model kernel calls returned, without independent numeric verification.
Kernel six failed in the **CPU emulation lift**, producing a 2-byte-to-4-byte
`__builtin_bit_cast` rejected by Clang. The exception crossed a ctypes callback
and execution continued, so the attempt was deliberately stopped after 210s.

**No complete usable AMD artifact was produced.** The failure is not proof of a
physical AMD kernel defect. No partial or no-op artifact is offered for selection.
Full GPU outputs, CPU/GPU parity, USB execution, VRAM, latency, thermals, power and
firmware compatibility remain unverified. Tinygrad and firmware were not modified.

## Separate experimental catalog and non-actuating diagnostic

`experimental_models.json` is a disabled wrapper around a Selector 21-compatible
bundle. It has no top-level production `bundles`, so the ordinary catalog parser
selects zero entries from it. It does not change model defaults or persistent
production selection. Generation is explicitly unverified and cannot authorize
action conversion. Local artifact, warp and metadata hashes are required before
the separate loader admits an entry. JSON/header inspection never executes a public
pickle. A ready experimental entry still cannot enter normal modeld publication.

The AMD diagnostic CLI requires the exported compiler-verified manifest.
The exporter binds the real source ONNX, runtime, compiler, artifact and successful
non-emulated source/reload report before creating a ready local AMD manifest. It
rejects this environment's CPU reports; no ready AMD manifest is shipped. These
are diagnostic admission checks, not a signed trust chain or independent GPU
certification. Only trusted locally produced artifacts should be loaded.

`tools/navigation_model_diagnostic.py` accepts prepared **files only**. It starts
no manager/controller and denies Msgq publishers/subscribers and live VisionIPC
clients in its process. It runs in an isolated Params prefix with an explicit
`NAV_MODEL_DIAGNOSTIC=1` opt-in and scoped CPU/AMD capability; permission is checked
before loading and every inference. Normal navigation/display toggles never grant
this capability. The in-process guard is for trusted code, not an OS sandbox for
malicious model binaries. It does not access CAN or perform firmware operations.

Per-arm files contain parsed predictions, hashes, route/state decisions, failures,
inference time, total persistence time, process peak RSS and tinygrad logical
allocation counts. The latter is **not physical VRAM usage**. Primary and optional
non-navigation fallback each retain separate state. Invalid navigation does not
silently produce a navigation-conditioned prediction.

## Actual recorded-camera comparisons

The supplied route contains 58 inspected rlogs and **zero parsed navInstruction or
navRoute messages**. The selected recording also has an empty destination. At least
one log has a truncated tail warning. Therefore it cannot supply real navigation
conditioning or conclusively intact full-route replay.

Eight contiguous 20Hz recorded night/intersection camera pairs from segment 43
were decoded. Pairing used physical capture times, not equal file offsets; maximum
skew was **44,010ns**. Recorded calibration/intrinsics and telemetry were used;
neutral desire and target Honda timing were explicitly reconstructed. The action
schedule was `[0.225,0.575]` seconds for this sample. Its original source's smoothing
differs, so this is not an exact replay of the original model process. HEVC decoding
also differs from original raw sensor bytes.

Four independent arms used identical cameras/transforms/desire/action timing:
neutral, map-only, maneuver-only, and combined. Explicit synthetic overlays cover
left/right intersections, highway exit command and consecutive maneuvers. Five
valid samples per arm produced **20 finite predictions, zero failures**. Three
invalid samples per arm were rejected: old map generation after reroute, missing
GPS and a map source 3s old (**12 expected skips**).

| Arm versus neutral | Paired frames | Maximum plan difference | Maximum action difference |
| --- | --- | --- | --- |
| Map only | 5 | 5.7130127 | 0.0439777 |
| Maneuver only | 5 | 8.1710434 | 0.0887330 |
| Both | 5 | 10.9791565 | 0.0937215 |

These are raw parsed-output differences across cold-start synthetic navigation
contexts, not steering/braking units or a quality score. Nonzero differences also
occurred in leads/desire; large lead uncertainty changes reinforce that finiteness
is not a safety criterion. Camera-only heads remained identical. The actual
highway-exit/rerouting road scenes were **not** present in these eight frames.

Measured CPU frame-and-persistence medians were 26.17s neutral, 37.10s map-only,
26.01s maneuver-only and 25.95s both, under a 2CPU/6GiB container. Peak process RSS
was 2.65GB; peak tinygrad logical allocation was 1.62GB across the run. This run
used a saved pre-hardening tool snapshot; its `inference_seconds` included prediction
copy/persistence. The final tool separates those times. These are development CPU
results, far from 20Hz, and cannot predict Chestnut performance.

A subsequent actual run used the newly verified Bigv2 CPU derivative and final
diagnostic path: one valid frame, then the three invalid contexts with the existing
non-navigation CPU fallback. All **four predictions were finite, with zero failures**.
Bigv2 inference was 25.85s; fallback inference was 0.420–0.423s. That run had a
1.72GB peak RSS. Fallback predictions remained file-only and contained no navigation
inputs. Injected primary/fallback initialization and inference failures are covered
by orchestration tests, not claimed as naturally observed model failures.

Only 0.35s of camera data was used, with five valid model samples per arm. This is
shorter than Bigv2's 128-frame/6.4s feature history, and cannot establish steady
state, learned reroute recovery or prolonged fallback behavior.

## Preservation and automated validation

The verified implementation is commit `255ea8a94722f37afcb67fa9e7da192de444945a`.
The final frozen-source targeted run passed **640 tests and 270 subtests**, with
zero failures/errors/skips. The separate preservation suite passed **581 tests,
2 existing skips and 8 subtests**: **1,221 tests passed overall**, excluding subtest
counts. The skipped tests are a parameterized base class and the network-only
manifest URL availability check. Known multiprocessing fork warnings remain.
Ruff and Python syntax checks pass for all 28 changed/new Python files. With the
pinned tinygrad resolved explicitly, ty reports the same four pre-existing typing
diagnostics reproduced on untouched starting files, and no new diagnostics.
The standalone compiler checks also passed 40 tests with 3 existing skips; those
overlap the combined run and are not added to its total.

Machine-readable results, model hashes and comparison metrics are committed in
[evidence/navigation-model-stage2.json](evidence/navigation-model-stage2.json). The Linux native build completed successfully, including
generated Cereal/Msgq bindings and UI components. The broad preservation suite
passed **581 tests, 2 existing skips**. Controller source trees, steering gains,
torque curves, MVL acceleration/braking/stop behavior, safety submodules, firmware,
selector, production catalogs/assets, updater and dependency locks are unchanged
from the starting commit. Both model runners' action/timing function ASTs remain
identical; all seven submodule revisions match the baseline.

Passing software tests and unchanged source do not certify behavior with changed
model predictions. No full route process-replay or vehicle test was completed for
navigation-conditioned controls. Existing stable branch hardware results do not
transfer to this experimental model.

## Reproduction without deployment

The final review also fixed a maneuver-loop variable leak that would corrupt the
existing UI remaining-distance/time calculation. Its regression case now reports
175m / 17.5s, preserving the current step and future maneuver list.

Use a supported development environment with native dependencies built. Preserve
the exact tinygrad checkout. Download the official checkpoint into a local research
directory and verify its SHA above. The existing native CLI remains usable:

```sh
DEV=CPU:LLVM python openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py \
  /research/bigv2.onnx /research/bigv2-cpu.sptg --device-input '*' \
  --benchmark-runs 0 --verification-output /research/cpu-verification.json
```

Compile a matching CPU warp through the pinned tinygrad warp recipe, or reuse a
previously trusted verified warp. A GPU warp cannot be substituted in the CPU run.
Prepare local files with the recording adapter; it records which inputs were
reconstructed and rejects missing calibration or unsynchronized camera pairs:

```sh
python tools/navigation_recording.py --help
NAV_MODEL_DIAGNOSTIC=1 DEV=CPU:LLVM python tools/navigation_model_diagnostic.py \
  --backend CPU --artifact /research/bigv2-cpu.sptg \
  --warp-artifact /research/warp-cpu.sptg \
  --recording /research/prepared.npz --output /research/comparison \
  --synthetic-suite --max-frames 8
```

Omit `--synthetic-suite` only when the file contains independently recorded route,
instruction, map, GPS and frame provenance. `--fallback-artifact` optionally tests
a trusted compatible CPU non-navigation artifact; it does not change production
model selection. The older `navigation_model_offline.py` is retained as a bounded
Stage 1 sensitivity helper; it is not the route/transition validation entrypoint.

A future separately arranged physical AMD development host would compile using
`DEV=USB+AMD:LLVM` with the existing pin and full compiler checks, then export a
local manifest using `python -m openpilot.selfdrive.navigation.export_experimental_model --help`.
Only after a valid artifact/warp and metadata exist can the file-only diagnostic
use `--backend AMD --experimental-catalog /research/local-ready.json`. No mock
environment or production catalog activation is allowed by that diagnostic CLI.
These commands describe development interfaces; they are not instructions to
install this work on the user's comma 4 or enable live control.

## Requirements before separate live-integration authorization

1. Complete physical gfx1200/USB compilation and independent CPU↔GPU comparison,
   including every prediction/state tensor, raw map dtypes, recurrent aliases,
   fused warp/camera upload and official compiled-variant precision differences.
2. Measure genuine GPU execution, VRAM, 20Hz deadlines, thermal/power/USB behavior,
   restart/failure recovery and existing firmware compatibility. No firmware update
   is implied by this experiment.
3. Collect real route, instruction, map, physical GPS and camera metadata. Compare
   trained-palette/rendering and coordinate orientation against upstream using
   real intersections, exits, consecutive turns, reroutes and missing/stale sources.
   Establish common clock domains, suspend/resume handling and source-to-camera
   timing bounds on the actual device.
4. Run sequences exceeding the 128-frame history and controlled state transitions;
   establish a learned cold-start, disengagement and fallback policy with explicit
   accept/reject criteria. Current transport resets are only experimental policy.
5. Verify existing non-navigation models and EPS/MVL timing on appropriate hardware
   without navigation-conditioned actuation, and independently review the full
   prediction-to-control safety effects. Unchanged controls can react differently
   to changed lateral **and longitudinal** model outputs.
6. Obtain a **new explicit authorization** for a separately reviewed live integration.
   This branch retains production publication blocks, default-off permission and
   no automatic model selection. It is not ready for navigation-conditioned road
   validation.

No other branches were intentionally changed. Publication uses a normal
fast-forward push of this existing experimental branch only.

## Changed files

The implementation commit changes these 32 source, schema, fixture and test files:

```text
M	openpilot/cereal/custom.capnp
M	openpilot/cereal/log.capnp
M	openpilot/cereal/services.py
M	openpilot/selfdrive/modeld/modeld.py
A	openpilot/selfdrive/navigation/experimental_catalog.py
A	openpilot/selfdrive/navigation/experimental_models.json
A	openpilot/selfdrive/navigation/export_experimental_model.py
A	openpilot/selfdrive/navigation/generation.py
M	openpilot/selfdrive/navigation/mapsd.py
M	openpilot/selfdrive/navigation/model_inputs.py
M	openpilot/selfdrive/navigation/navd.py
A	openpilot/selfdrive/navigation/temporal.py
A	openpilot/selfdrive/navigation/tests/fixtures/chestnut_v28_baseline.json
M	openpilot/selfdrive/navigation/tests/test_experimental_boundaries.py
A	openpilot/selfdrive/navigation/tests/test_experimental_catalog.py
A	openpilot/selfdrive/navigation/tests/test_experimental_export.py
M	openpilot/selfdrive/navigation/tests/test_maneuver_contract.py
A	openpilot/selfdrive/navigation/tests/test_map_generation.py
M	openpilot/selfdrive/navigation/tests/test_model_input_admission.py
A	openpilot/selfdrive/navigation/tests/test_native_route_admission.py
A	openpilot/selfdrive/navigation/tests/test_route_generation.py
A	openpilot/selfdrive/navigation/tests/test_temporal.py
M	openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py
M	openpilot/sunnypilot/modeld_v2/model_adapters.py
M	openpilot/sunnypilot/modeld_v2/modeld.py
A	openpilot/sunnypilot/modeld_v2/tests/test_diagnostic_output_slices.py
A	openpilot/sunnypilot/modeld_v2/tests/test_diagnostic_state.py
A	openpilot/sunnypilot/modeld_v2/tests/test_native_compiler_verification.py
A	tools/navigation_model_diagnostic.py
A	tools/navigation_recording.py
A	tools/tests/test_navigation_model_diagnostic.py
A	tools/tests/test_navigation_recording.py
```

The accompanying report commit adds this report and its evidence JSON, and adds
Stage 2 links to `navigation-model-testing.md` and
`navigation-model-investigation.md`. No model weights or recordings are committed.
