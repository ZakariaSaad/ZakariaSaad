# Bigv2 map overflow: isolated numerical investigation

This continues the investigation from diagnostic baseline `086af922`, using the
three original rlogs from build `1db06a2` and the repeat failure from build
`086af922`. It does not change the registered model, driving admission, model
selection, controllers, firmware, or production inference/fallback code.

## What the recordings establish

All **427,555 events** were decoded using this fork's
`openpilot.tools.lib.logreader.LogReader` and matching cereal schemas. The three
reuploaded original segments have the same SHA256 hashes as the previous files.
Wall times below use the error record's `created`; publication gaps use
`logMonoTime`, so logger delivery latency is not mistaken for inference latency.

| Measurement | Original failure | Repeat failure |
| --- | --- | --- |
| Error UTC | 2026-10-22 16:06:57.651051 | 2026-10-23 01:39:40.668232 |
| Error logMonoTime, ns | 426587945506 | 804693128304 |
| Model PID | 48650 | 49989 |
| Rejected camera frame | 7523, inferred | 15119, explicitly logged |
| Last Bigv2 frame | 7522 | 15118 |
| First small frame | 7524 | 15120 |
| First small invocation | 1075.122 ms | 1098.524 ms |
| Model publication gap | 1141.559 ms | 1166.550 ms |
| Next small frame / invocation | 7545 / 41.730 ms | 15142 / 39.334 ms |
| Later small median | 34.676 ms | 34.312 ms |
| Location warning after error | 1.106541 s | 1.115412 s |

Each route contains one failure, duplicated into the log/error streams. The same
model process remained running through each fallback. There was no recorded
camera clock jump, unhealthy GPS/location state, USB reset, supply fault, or
non-finite calibration/control-source value in the preceding ten seconds.
The repeat route generation changed approximately 30 seconds earlier, not at the
failed frame. Both failures occurred approximately **5.95 metres apart**; no
coordinates or route geometry are published here.

The repeat diagnostic records **1,213 NaNs and no infinities** in 18,580 float32
outputs. Every value in these sections is NaN:

| Section | Slice | Count |
| --- | --- | ---: |
| plan | 917:1907 | 990 |
| lead | 1907:2051 | 144 |
| lead_prob | 2051:2054 | 3 |
| desire_state | 2054:2062 | 8 |
| action | 2062:2066 | 4 |
| map_features | 18450:18514 | 64 |

The remaining **17,367 values are finite**, including vision heads, hidden_state
and navigation_features. Host transform, desire, traffic convention, map,
maneuver and action-time arrays are finite. Action times are
`[0.2595362067222595, 0.625]`; map/maneuver reasons are both `valid`.
The failed inference path took 27.877 ms and logged run_count 14,184.
Finiteness summaries do not contain exact pixels or input ranges.

Reconstructed maneuver distances follow the official `distance / 100`
convention: approximately 42.615 and 13.551, respectively. These are not
normalized probabilities; clipping them to `[0,1]` would change valid inputs.

The `locationd` warning occurs after the failed Bigv2 inference and delayed small
observation in both drives. There is no evidence that it initiated either NaN
failure. No location-filter change is warranted.

## Upstream lead and the numerical reproducer

The relevant source is ONNX SHA256
`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
The installed AMD artifact remains SHA256
`feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4`.

Upstream commit
[`3ec09aed`](https://github.com/commaai/openpilot/commit/3ec09aed2ea589cbd3234137ae6fb74aecde3c4c)
changes a compiled PKL pointer. Static interpretation of already available
pickle opcode data, without importing or executing pickle constructors, finds
a changed kernel constant: `1e-5` becomes `6.25e-7` (`1e-5 / 16`). The Bigv2 ONNX
already contains matching scaling/epsilon changes. Its model name and Git
ancestry alone would not establish that relationship; the graph and weight
comparisons supply the narrower evidence. This is not a reason to substitute
the old official PKL for the registered native artifact.

The exact ONNX map encoder can be isolated without cameras, navigation commands
or recurrent history: its `map_features` dependency is the current `nav_map`
and constant weights. An independent ONNX Runtime CPU execution with every
intermediate exposed finds a reproducible float16 overflow from a deterministic
synthetic map whose pixels are all in **0–255**. A nearby comparison map remains
finite. The failing matrix multiplication is **`node_MatMul_599`**, output
**`val_711`**, in `map_encoder.stages.3.blocks.0.mlp.fc2`. Its float16 output
shape is `[1,8,8,512]`; the first invalid coordinate is `[0,1,5,374]`. An
independent float64 dot product of its finite operands is
**−65535.557566018295**, outside float16's maximum magnitude **65504**. The
existing channel scale is about 0.218, but the graph stores the unscaled
intermediate in float16 first. One negative infinity propagates into all 64
NaN map features. This is a
source-model numerical defect on valid-domain inputs, not an invented
non-finite input or a downstream parser failure.

The exact pinned tinygrad `69fd4d34`, running **CPU:LLVM**, independently produces
the same first invalid intermediate on the same input. Its working map features
differ from ONNX Runtime's by at most **0.00048828125**. Both probes expose every
map intermediate, which changes fusion/materialization opportunities. These
results establish source-graph behavior, not the registered AMD binary's exact
execution. The CPU trace times include compilation/scheduling and many
readbacks; they are not Chestnut performance measurements.

The **complete** ONNX was also executed in ONNX Runtime CPU, without adding
intermediate outputs, for the working map followed by the overflow-pattern map.
All five queues began at zero and the first frame's returned state was carried
into the second. **Both full-model outputs were finite**, including all 18,580
primary values and all recurrent outputs. The only reference adaptation replaced
`org.tinygrad::Contiguous` with the value-equivalent ONNX `Identity` in memory;
unknown custom operators are rejected and the source bytes are unchanged.
Graph optimization was disabled, but CPU operator implementations can still
retain float32 temporaries across declared float16 boundaries. This important
difference prevents calling the exposed-intermediate result a full AMD replay.

| Actual execution | Result on the synthetic failing map |
| --- | --- |
| ONNX Runtime 1.31 CPU, map intermediates exposed | First −Inf at `val_711`; 64 NaN map features |
| Pinned tinygrad CPU:LLVM, map intermediates exposed | Same first invalid tensor and 64 NaN map features |
| Full ONNX Runtime 1.31 CPU, ordinary outputs only | All primary and recurrent outputs finite |
| Previously compiled full CPU SPTG | SIGILL during loading/verification; no inference result |
| Registered AMD SPTG | Not executed in this environment |

Full CPU reference initialization took 5.754 seconds; the two calls took
1.254 and 1.369 seconds. These include host execution/readback and are reported
only to identify the experiment, not as an estimate of Chestnut latency.

The previously prepared complete CPU SPTG was also attempted without
recompilation. It exited with **SIGILL / status 132** during loading/verification,
before either input was run. Its target was `CPU:LLVM x86_64,native`; compatibility
with this execution host is not established. No numerical result is claimed
from that attempt, and the failure is not attributed to the installed AMD model.

The committed fixture is **3,597 bytes**, contains only two generated rasters,
and has a deterministic generation recipe and pinned pixel hashes. It contains
no camera images, route recordings, coordinates or model weights. Developers
with the existing exact source and optional `onnx`/`onnxruntime` packages can run:

```bash
python3 -B -m openpilot.tools.navigation_map_numerics \
  --source-onnx /path/to/existing/big_driving_supercombo.onnx \
  --output /path/to/new/map-numerics.json
```

Expected result: `working: all map intermediates finite`, followed by
`overflow: node_MatMul_599`. This is an isolated numerical reproducer, not another
on-device compatibility or hardware test.

An optional **`--check-precision-repair`** experiment keeps only this MatMul,
bias addition, transpose and existing channel-scale multiplication in float32,
then converts back to float16 before the unchanged residual addition. It
preserves weights, bias, scale and output contracts, and changes no source or
artifact file. On the same failing input, every exposed map intermediate and
all 64 features become finite. The working input also stays finite; its largest
map-feature change is **0.0009765625**. This is a concrete CPU-tested precision
correction candidate. It changes rounding and requires validation of the
compiled AMD implementation before replacing a driving artifact; it is not
applied to production or used to fabricate replacement predictions.

This reproducer strongly supports a content-sensitive map-encoder failure as
the most likely cause of the drives: the same map area failed twice and the
recorded output sections match that path. It does **not** establish the exact
first operation executed on Chestnut during either drive. The consumed map
was not saved, and the registered AMD binaries have not been executed here.

## Exact replay boundaries

`openpilot.tools.replay.nav_numerical` starts from the existing model/process
replay design but invokes the native SPTG runner directly, without launching
modeld or using vehicle messaging. It uses the production loader, typed input
buffers, and the same input/output aliasing for the five recurrent queues.
It does not use the upstream 60-frame CI limit.

- Each JSONL record names a SHA256-bound NPZ of the exact consumed, prewarped
  model inputs: `new_img`, navigation map/vector, desire pulse, traffic
  convention and action timing, as required by the artifact's actual signature.
- Select explicit zero initialization at the true model-start boundary, or a
  matching **pre-inference** checkpoint. Segment labels never reset history.
- Every reference comparison receives identical input bytes and a copy of the
  primary runner's incoming state. The reference does not accumulate a separate
  divergent history and then compare unlike states.
- Input/state finiteness and ranges, output sections, discrepancies and elapsed
  times are recorded. No invalid output is repaired or published.
- An opt-in window exports at most eight frames, including state copied before
  inference overwrites the aliased `next_*` buffers. Captures are private files;
  no hook is installed in the live inference or fallback path.
- The optional camera preparation helper uses the fork's `LogReader`,
  `FrameReader`, encode indexes and process replay's NV12 layout. HEVC decoding
  and rewarping are explicitly reconstructed inputs, not byte-identical proof of
  the tensor consumed during a previous drive.

Numerical execution is restricted to an isolated development host. This does
not require the owner to acquire another workstation: only the read-only export
command below runs on the comma. No GPU exercise or service change is requested.

## What remains missing

The supplied evidence archive contains compiler verification, registration,
static compatibility, approved model identities and selected log events. It
contains no weights, kernel binaries, consumed map pixels, camera tensors,
recurrent checkpoints or intermediate values.

The rlogs contain camera metadata/indexes, but no camera image or encoded-video
payloads and no raw model prediction bytes. Map pixels travel through VisionIPC
and were not recorded in these logs. A current route render cannot be presented
as the exact map previously consumed.

The subsequently supplied segment-12 narrow/wide HEVC files each exceed the
attachment tool's **32 MiB** transfer limit. The tool refused both, so their
actual camera data has not been decoded in this investigation. This is a
transfer limitation, not an absence of owner-supplied camera recordings.

For the stateless map-path hypothesis, the smallest exact drive reproducer is
the **single consumed 256×256 map** from the failed call, plus the exact AMD
program. It does not need 164 camera frames or a long priming sequence. For a
full recurrent-model reproduction, the required alternative is the consumed
input sequence from genuine initialization, or the exact incoming five-queue
checkpoint and subsequent input sequence. Neither can be recovered from the
provided rlogs or from a post-failure aliased state dump.

The offline tools now support bounded pre-inference capture when running such
a sequence. No new on-road failure or live synchronous state dump is requested.

## Small, read-only compiler evidence export

After updating this branch normally, run in **Windows PowerShell**:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m tools.navigation_numerical_export"
scp comma@10.0.0.134:/data/nav-bigv2/numerical-kernels.tar.gz "$env:USERPROFILE\Downloads\bigv2-numerical-kernels.tar.gz"
```

This system-Python command runs as `comma`, without sudo. It checks the existing
registered artifact's SHA256 and exports only bounded native headers and
deduplicated program binaries. It reads the existing 810 MB file once for its
digest; it does not duplicate or upload the weights. The export is capped at
32 MiB, private, and refuses to overwrite an existing archive. If needed, use
`--output /data/nav-bigv2/numerical-kernels-2.tar.gz` and SCP that exact filename.

No GPU, firmware, CAN, Params, controller, messaging, network download, model
compilation, or comma service operation is performed. The archive contains no
credentials, route recordings or input pixels. It will permit static inspection
of the exact AMD lowering; it cannot retrospectively recover the failed map.

To transfer the already supplied camera files without re-encoding, run this
PowerShell command in the Windows folder containing the two original HEVC files:

```powershell
$videos = Get-ChildItem -File '*00000039--7e32df3151--12--?camera.hevc'
foreach ($video in $videos) {
  $stream = [System.IO.File]::OpenRead($video.FullName)
  try {
    $buffer = New-Object byte[] (24MB)
    $part = 0
    while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
      $path = '{0}.part{1:D3}' -f $video.FullName, $part
      $out = [System.IO.File]::Open($path, [System.IO.FileMode]::CreateNew)
      try { $out.Write($buffer, 0, $count) } finally { $out.Dispose() }
      $part++
    }
    Get-FileHash -Algorithm SHA256 $video.FullName
  } finally { $stream.Dispose() }
}
```

The `.part000`, `.part001`, ... files preserve every original byte and can be
uploaded privately with the printed original hashes. They are not standalone
videos; concatenate them in numeric order before `FrameReader` use. No route
recording or capture is included in the Git commit.

## Fallback latency is a separate issue

The small model is constructed at startup; fallback switches to the existing
instance and discards the rejected frame. There is no model reload in the
failure handler. Ordinary adapters pre-run the camera warp but not the policy.
In the pinned tinygrad, `CapturedJit.linear` links lazily on first invocation,
and the resolved property is omitted from serialized caches. That is a concrete
possible first-use cost, but the recordings do not measure its share of the
approximately 1.1 seconds.

`OfflineInferenceProfile` can surround one isolated model call and records
camera copy/binding, warp/policy submission, first-use linking, runtime creation,
output readback and an optional completion barrier. Its host durations are
inclusive; they must not be added together or described as separately measured
GPU kernel/transfer costs. The exact selected QCOM fallback binary is not
available here, so its initialization/transfer/kernel split remains unmeasured.
The different local CPU fallback is not used to invent a QCOM timing result.

Finite rejection, the single-active-model handover, Honda EPS, MVL longitudinal
behavior, Panda safety, model selection, firmware and registered artifact hashes
remain unchanged. These changes are an isolated numerical reproducer and
instrumentation, **not a deployed stability fix**.

## Verification and publication scope

Focused regressions passed **54 tests and 15 subtests**, including actual
SPTG serialization/loading and recurrent aliases over 67 frames; continuous
reference-state comparisons across segment labels; pre-inference checkpoint
resume; first non-finite rejection; live messaging denial; generated HEVC
decoded by real FrameReader/LogReader; the CPU precision experiment and graph
guards; kernel-export bounds, integrity and privacy; and the existing modeld
finite-output/fallback publication tests. Ruff and scoped type checks passed.

These test counts are separate from the real source-model experiments reported
above. Synthetic fixture tests do not turn the supplied drive into an exact
replay. No AMD or QCOM timing measurement was performed here. The next useful
compiler evidence is the small registered-kernel export, not another drive or
an arbitrary endurance test. Any eventual replacement artifact must demonstrate
the same correction on its actual AMD execution path before its hash or driving
registration is changed.
