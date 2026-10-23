# Bigv2 map precision correction

This continues [the numerical investigation](navigation-model-numerical-replay.md)
from `74414354db1b12f767ba957ea24fffb25d04f366`. The correction is a guarded,
opt-in compiler transformation. Updating Git alone **does not repair the kernels
inside the registered `.sptg` file**. The existing source, artifact, registration,
approval, model selections, runtime finite checks and fallback remain unchanged.

## What the actual AMD export establishes

The supplied kernel archive was read as JSON and ELF data. No uploaded executable
or pickle was executed. The header SHA256 is
`684a58521055e8c4269e9fc40a9164e0fb7579d82388b3293303661f3b63becb`.
All 671 library references and 123 distinct ELF libraries verify against their
exported lengths and SHA256 hashes; every target declares AMD/LLVM/gfx1200/USB.
The export identifies registered artifact
`feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4`
and source ONNX
`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
The archive excludes weights, so the complete 810,417,970-byte artifact hash
cannot be independently recomputed from it; that identity is the exporter's
attestation, supported by the matching header and individually verified kernels.

Tracking pooled scratch buffers by byte range and last writer identifies the
map-only chain from calls 2–202. Call 180 is structurally the stage-3 block-0
2048→512 contraction and bias/scale/residual epilogue: its dimensions, constant
sizes and dependencies match the source overflow span. There are no original
node labels or weight bytes in this export; this is a strong structural mapping,
not a recovered ONNX debug symbol.

LLVM 19 disassembly of that actual gfx1200 binary, SHA256
`e4056c0a0103d082cb35c447ce3d73e5c598b0f218dd3673168c409cce234b3b`, shows:

| `.text` offsets | Operation |
| --- | --- |
| `0x300–0x3c8` | `v_wmma_f32_16x16x16_f16`, with float32 partial accumulation |
| `0x460–0x580` | `v_cvt_f16_f32`: narrow the completed dots to float16 |
| `0x614–0x68c` | `v_pk_add_f16`: add bias after that narrowing |
| `0x698–0x714` | `v_pk_fma_f16`: apply existing scale and residual |
| `0x720–0x744` | Store packed float16 outputs |

The early conversion occurs in registers even though these operations are fused.
It therefore survives fusion. Call 192 has analogous early conversions despite
its float32 output buffer; a buffer descriptor alone cannot prove internal
precision.

The synthetic source fixture has finite operands whose dot is approximately
**−65535.56**, beyond float16's finite range. The existing scale would reduce it
to approximately −14288. This makes premature narrowing a demonstrated defect
on valid-domain inputs, with independent evidence that the registered AMD code
contains the implicated arithmetic. It is the strongest explanation for the
recorded all-NaN map features and downstream plan/lead/action outputs. The exact
failing drive map and pre-inference state remain absent, so the original physical
frame has not been reconstructed and other independent failures are not ruled out.

The prior `locationd` warnings occurred over a second **after** each numerical
failure. They do not support changing the location filter. The existing fallback
successfully switched models in the same process; its first small-model invocation
took about 1.08–1.10 seconds. That separate initialization/link/transfer/execution
breakdown is not measurable from these records and is not changed here.

## Correction and replacement scope

The compiler option `--nav-map-precision-fix` applies only to the exact reviewed
source hash and guarded map stage-3 block-0 graph span. It retains float16 matrix
operands and weights, requests float32 dot accumulation/output, and preserves
float32 through the existing bias and channel scale. Explicit float32
materialization boundaries make the precision inspectable. It also materializes
the conversion back to float16 before the existing residual addition; otherwise
the optimizer can fold that conversion into a wider residual operation. No weights, scale values, model
inputs, external outputs, or recurrent-state layouts are changed.

The transformation is performed in the pinned tinygrad graph in memory, without
rewriting the ONNX. The generated candidate carries transformation/compiler
provenance and new hashes. New output/report paths are required; registration and
driving approval are not performed by compilation. Ordinary compilation and
ordinary models do not opt into this transformation.

## What was tested

The existing deterministic working/overflow fixtures contain synthetic map pixels
only. No recordings, consumed navigation maps, kernel archives, model weights or
device captures are committed.

- The original source map probe with exposed intermediates identifies the first
  infinity at `node_MatMul_599` / `val_711`; its map features become all NaN.
- The in-memory source correction keeps both map fixtures finite. The working
  map-feature maximum absolute change is `0.0009765625`.
- Real **tinygrad CPU:LLVM**, native SPTGPLN1 serialization and reload of the
  actual contraction/bias/scale span reproduce one negative infinity out of
  32,768 values originally, and 32,768 finite values after correction. The working
  span and all 32,767 originally finite values in the overflow span are bitwise
  unchanged in that comparison.
- The **complete stateless map encoder**, using the actual production lowering
  and weights on CPU:LLVM, produces 64/64 NaN features originally on the overflow
  fixture and 64/64 finite features after correction. Native reload matches
  TinyJit exactly. Both working maps remain finite; 17/64 features change, with
  maximum absolute difference `0.00048828125` and mean `0.000034213`. This is the
  final production-lowering result, distinct from the ONNX experiment above.
  The residual addition is checked against an explicit float16 addition of the
  materialized scaled value and the unchanged residual. Rounding can differ
  from the original fused epilogue; the entire encoder is not bitwise unchanged.
  In the working residual tensor, 346/32,768 elements change (maximum absolute
  difference 8, mean `0.00233454`). On the overflow fixture, 322/32,767 originally
  finite residual elements change (maximum 8, mean `0.00306097`); the previously
  infinite element becomes finite. These are intermediate float16 values, not
  steering commands. The final map-feature differences are the smaller values
  reported above.
- Full **ONNX Runtime 1.31 CPU** comparison runs working/overflow/working/overflow
  with identical camera/navigation inputs and the original sequence's exact
  pre-state supplied to each paired corrected call. All six outputs, every
  prediction section and every next-state array are bitwise equal. The input/
  output declarations, metadata and initializers are unchanged. This CPU backend
  already keeps sufficient internal precision without exposed intermediates, so
  full-model equality does **not** establish AMD equivalence.
- **Offline native gfx1200 compilation** of the exact production helpers, with
  the original operand shapes and float16 dtypes, generates four real ELF kernels
  using the pinned tinygrad AMD LLVM renderer and host LLVM 19. Disassembly shows
  float32 WMMA accumulation/stores, then float32 bias/scale without narrowing,
  then a separate float32→float16 conversion, then float16 residual addition.
  Both generated LLVM and actual AMD instructions were checked. Shape-only lazy
  placeholders supplied no weights or values; no device was opened and no tensor
  was executed. This verifies isolated AMD code generation, not the complete
  replacement's execution or performance.
- Neither the original nor corrected AMD binary was **executed** here. No AMD
  device exists in this development environment. There is no newly approved AMD
  replacement and no AMD performance or exact-road-frame success claim.

The compiler correction addresses the demonstrated numerical mechanism. The
complete replacement remains a candidate until its native AMD execution and
compiled precision checks pass. This is separate from the already completed
registration/camera-warp/software compatibility work; none of those workflows
needs to be repeated merely to implement this correction.

The complete CPU map test also exercises pooled-buffer aliases and both legitimate
downstream consumers of the restored float16 residual input. The full candidate
compiler checks post-optimization tinygrad IR, generated LLVM, exact byte ranges
through copies/overwrites, initializer bytes and kernel hashes. It refuses a build
whose dot or scale narrows early, or whose residual boundary disappears.

Focused regression results: 76 tests and 42 subtests passed, including compiler
reload, correction guards, off-road lifecycle, finite-output rejection and normal
publication checks. Four optional exact-source CPU tests and one native AMD
cross-compilation test passed separately. Ruff
and whitespace checks passed. To reproduce the optional source tests using an
already available source file and development dependencies:

```bash
NAV_BIGV2_SOURCE_ONNX=/path/to/existing/big_driving_supercombo.onnx \
NAV_BIGV2_FULL_REFERENCE=1 DEV=CPU:LLVM python3 -m pytest -q \
  openpilot/tools/tests/test_navigation_precision_source.py
```

This developer command needs ONNX/ONNX Runtime and approximately 6 GB RAM for the
optional full reference. It does not download anything and is not an additional
device-validation requirement.

The static AMD check uses shape-only placeholders and prohibits opening any
device. On a development host with LLVM supporting gfx1200:

```bash
NAV_BIGV2_AMD_CROSSCOMPILE=1 PYTHONPATH=.:tinygrad_repo python3 -m pytest -q \
  openpilot/sunnypilot/modeld_v2/tests/test_nav_map_amd_codegen.py
```

## Build the separate candidate on the existing device

Use the normal Software settings update on
`mvl-honda-eps-selector21-chestnut-v28`, then restart normally. No reinstall,
branch switch or service modification is needed. Keep Navigation Model Driving
off while evaluating the candidate. From Windows PowerShell:

```powershell
ssh comma@10.0.0.134
```

On the comma, parked and genuinely reporting off-road, with Chestnut powered:

```bash
cd /data/openpilot
DEV=USB+AMD:LLVM PYTHONDONTWRITEBYTECODE=1 python3 -B \
  -m openpilot.sunnypilot.modeld_v2.compile_modeld_v2 \
  /data/nav-bigv2/big_driving_supercombo.onnx \
  /data/nav-bigv2/bigv2-amd-map-fp32.sptg \
  --nav-map-precision-fix --offroad --device-input '*' --benchmark-runs 0 \
  --verification-output /data/nav-bigv2/bigv2-amd-map-fp32.verification.json
```

This is a full candidate build, not a short smoke test; changing embedded AMD
kernels requires recompilation. Its duration and peak memory on this device have
not been measured. It reuses the existing local ONNX, downloads no model and does
not create a CPU reference artifact. The serializer can temporarily hold both its
blob and the staged candidate: allow roughly **1.7 GB additional free space**,
plus compiler caches and other working space. The retained candidate is about
811 MB; no original file is deleted to make room. Use the vehicle's existing power arrangement without
deliberately exhausting the accessory battery. Physical ignition OFF is not an
admission condition; genuine software off-road status is.

The compiler uses fresh off-road/control-process information continuously and
tinygrad's exclusive USB GPU lock. It aborts if admission disappears, state goes
stale or another process owns Chestnut. `comma.service` remains running. It does
not start controllers, publish model messages, change selections, register the
candidate, flash the USB bridge, or approve driving. Ordinary initialization loads
the already cached GPU microcode into volatile memory; missing cache files are
reported instead of downloaded.

Back in Windows PowerShell, export only the small verification report:

```powershell
scp comma@10.0.0.134:/data/nav-bigv2/bigv2-amd-map-fp32.verification.json .
```

A successful candidate report is evidence for review, not activation approval.
The replacement artifact's exact new hash and matching registration/approval
would require a subsequent reviewed change. Do not rename it over
`bigv2-amd.sptg` or edit expected hashes to accept it.

## Rollback

No runtime replacement occurs in this task: the original ONNX,
`/data/nav-bigv2/bigv2-amd.sptg`, its approved SHA256, camera warp, registration and
model settings remain intact. To abandon the candidate, leave it unregistered;
the running software continues using the original registered model or existing
ordinary fallback. There is no service state to restore. Do not re-enable the
original failing model merely as a rollback test. Reverting this opt-in compiler
change is not required to preserve the current driving implementation.
