# Corrected Bigv2 on the pinned sp-steer runtime

> **Model lifecycle follow-up:** [Bigv2 model lifecycle](bigv2-model-lifecycle.md)
> supersedes the separate navigation-driving selection/approval workflow described
> in this historical verification report. Current selection uses the ordinary Big
> Model slot; exact artifacts, EPS and longitudinal code remain preserved.

The migration retains the existing corrected model and registration bytes. It adopts
sp-steer's native SPTG loader adapter and tinygrad revision for ordinary models;
it does not replace Bigv2, recompile it, or rewrite its historical compiler report.

| Identity | Revision or SHA256 |
|---|---|
| Working custom source | `18106a1ddf26366236961b28509e6fa677bec6ef` |
| Pinned MVL sp-steer source | `0f5085fc3f713da58d236ed3822b9d27bc0e9917` |
| Historical compiler tinygrad | `69fd4d34d4430036378aba2b43f9d339cdd6ea73` |
| Migrated execution tinygrad | `92b742f711c7d216fd3ed7a46a30a959e3c43bd4` |
| Corrected model | `4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c` |
| OS04 camera warp | `f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b` |
| Registered metadata | `cff885b4ef1af332582f6df847351b8d69c2da6e70979718a03e1931f7dbad95` |
| Historical compiler report | `3f776eafc180c87fc85333bd029c4ee2bddf206b00ce7df695eba54e306af591` |

The model and sidecars remain at the existing
[release](https://github.com/ZakariaSaad/ZakariaSaad/releases/tag/nav-bigv2-map-fp32-sidecars).
No release asset was changed by the rebaseline.

## What changed and what did not

`openpilot/sunnypilot/modeld_v2/helpers.py` adopts upstream's field-aware
`ProgramInfo`, `ParamArg` and `Target` reconstruction. The stored native program
binaries, argument descriptions, buffer extents and saved initializers are retained.
The ordinary compiler/converter also adopts upstream's updated kernel-signature and
scratch-`ALLOC` handling. The native container remains `SPTGPLN1`.

The custom dtype-aware frame packing, registered warp loading, all five recurrent
state pairs, navigation input admission, finite-output rejection and single-active
model fallback remain unchanged from the working source. The FP32 correction
implementation is unchanged. The camera recipe, layout source and their recorded
hashes are unchanged. The global runtime advances to the sp-steer revision, so
newer ordinary models are not forced onto the historical compiler runtime.

`export_experimental_model._runtime_checkout` distinguishes historical provenance
from execution compatibility. Historical reports must still name the exact clean
69fd compiler checkout. The exact corrected artifact and reviewed camera assets
also admit the exact clean 92b execution revision. Other model hashes, other new
revisions and dirty runtime checkouts do not inherit that compatibility. Warp
validation still requires the exact camera-specific artifact, source bindings,
layout and saved NumPy reference. This is a bounded compatibility adapter, not a
wildcard revision exemption. Existing experimental approval for the corrected
artifact remains; the retired original Bigv2 remains unavailable.

The status registry labels the newer runtime `reviewed_native_binary_abi_adapter`
and distinguishes its software evidence from historical physical compiler evidence.
Compiler, lowering, model, metadata and warp hashes remain unchanged.

## Checks performed

The actual published 810,439,301-byte corrected artifact and released sidecars were
inspected with the migrated validators. All compiler/report identities, protected
FP32 kernel byte ranges, saved initializer digests and tensor contracts passed.
The actual model contains 673 program calls, 759 internal buffers and 17 input/output
slots. All 673 model program descriptors and all seven OS04 warp program descriptors
were reconstructed by the newer loader without GPU execution. Kernel bytes,
launch dimensions and global/input/output argument indices matched their saved
native descriptions exactly. The warp's saved camera capture matched its independent
NumPy reference. An independent review also checked all 2,717 argument descriptions,
including 1,794 nonzero-offset views and 839 dtype casts, retaining alias bases and
byte alignment. Bigv2's recorded internal/input buffers are AMD buffers, so the
new host enqueue-device selection does not reroute its saved program calls.

For cross-runtime execution, a small deterministic native CPU model with a matrix
operation, retained weights, a primary output and next-state output was compiled
under the pinned historical runtime. That unchanged SPTG was loaded under the new
runtime with its saved reference check enabled. Fresh independent inputs/state for
seeds 42, 43 and 100 produced bit-exact matching outputs. A separate nine-frame
continuous sequence used the production adapter's distinct buffer-view aliases
for recurrent input/next-state output; every consumed state and output matched
between the historical and new runtimes. This exercised CPU LLVM, not AMD.

Focused regression results:

- 222 modeld-v2 tests and 54 subtests passed, with five existing optional/obsolete
  stock-comparison skips; the subsequently added historical AMD-descriptor ABI
  regression also passed.
- 54 runtime admission, export, software-compatibility, activation and camera-warp
  tests and 115 subtests passed.
- The ordinary native suite exercised roundtrip execution, corruption rejection,
  cache invalidation and fused warp equivalence. Its camera-packing fixture was
  adapted to the retained dtype-aware interface without changing its comparison.
- A stale publication-guard AST assertion failed identically on the original
  working source. It now checks the unconditional successful startup body inside
  the existing try/finally, while retaining active/fallback rejection and
  guard-before-publisher assertions.
- The two upstream pickle-converter tests explicitly require physical AMD USB;
  running them with a CPU backend produces their hardware-target assertion. They
  were not represented as successful physical tests.

## Existing-file installation on the new runtime

The actual installer was also exercised in an isolated device-like directory
layout, using the published model and ZIP bytes and a clean checkout of tinygrad
92b. Only the HTTP response transport and fresh off-road messaging were mocked;
all file hashing, ZIP inspection, native layout, compiler provenance, camera warp,
metadata construction and atomic registration used the real implementation.

The existing 810 MB model was reused without copying or redownloading it. The
sidecar ZIP was streamed through the real downloader once, registration retained
the exact approved metadata hash, and notices were installed. A subsequent
registered selection made zero HTTP requests. The model inode and ordinary model
selection stayed unchanged, Navigation Model Driving remained OFF, and temporary
staging directories were removed. No model kernel ran during preparation. The
permanent release-download lock file is intentionally retained.

## Remaining limits

The uploaded physical source/reload and working/overflow map comparisons were made
with tinygrad 69fd, not 92b. Tinygrad's newer host/HCQ/USB code has changed; descriptor
identity and CPU execution do not demonstrate physical AMD transport equivalence,
new-runtime latency, power-loss recovery or vehicle stability. No physical Chestnut
execution, firmware action, vehicle access or driving test was performed for this
migration. Runtime target checks, integrity checks, non-finite-output rejection and
ordinary fallback remain active. Experimental artifact authorization is not a claim
of on-road safety.
