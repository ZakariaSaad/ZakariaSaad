# Corrected Bigv2: official model-behavior parity

Branch: `mvl-sp-steer-eps-nav`. Starting commit:
`4100c6563e46fdf84be04c56a54d3de334ee259d`.

The later [Blendful integration](blendful-integration.md) adds an optional
upstream curvature correction after inference. It defaults OFF; OFF preserves
the behavior reviewed here. Enabling it does not change model inputs or buffers.

## Two distinct authorities

Bigv2 model behavior follows official comma.ai/openpilot
`a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9`. The official GitHub commit and
Git blob independently confirm the source ONNX pointer:

`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`

The pointer names 814,591,106 bytes. The a03 compiled PKL and its tinygrad
revision are separate identities; later official PKL updates are not assumed
equivalent merely because the ONNX pointer or model name remained unchanged.
See [source provenance](evidence/bigv2-official-reference.json).

The already integrated MVL `sp-steer` revision
`0f5085fc3f713da58d236ed3822b9d27bc0e9917` remains authoritative for model
selection/startup/fallback and longitudinal control. This change does not merge
another branch, replace that lifecycle, or change the longitudinal controller.

## Removed model-behavior differences

- Restored official route calculation, held-position behavior, maneuver
  publication and geometry-based progress. Removed destination identity and
  route-generation fencing from the model-facing production path.
- Restored receipt-relative producer extrapolation and maneuver consumption.
  The model consumer uses instruction receipt age under two seconds and latest
  map receipt age under half a second, independently, exactly as a03 does.
- Removed extra GPS/source timestamp checks, camera epoch barriers and exact
  map-frame/metadata associations from production inference admission. Fresh
  receipt of a black map has official semantics even if diagnostic metadata is
  invalid. Route changes do not invent recurrent resets.
- Restored the official maneuver encoder, including its unknown type/modifier
  behavior, distance normalization, ordering and number of encoded maneuvers.
- Bigv2 now uses the official DesireHelper. SunnyPilot auto-lane-change,
  road-edge lane-change gating and lane-turn conditioning remain available for
  ordinary models but do not change Bigv2's desire pulses.
- Bigv2 uses the official camera selection/calibration transforms without
  CameraOffset tuning. Single-camera extra-stream intrinsics follow a03.
- With custom EPS disabled, Bigv2 reads live `lateralDelay` each frame as a03
  does. With custom EPS enabled, the existing `prediction_delay(vEgo)` schedule
  remains in both model runners. Longitudinal action timing, frame/action delay
  constants and model-side smoothing follow a03.
- Restored the official output parser, raw action interpretation, hypothesis
  handling and publication functions. Bigv2 no longer inherits the ordinary
  parser's inferred MDN layout or extra lane/edge time interpolation. Inactive
  lateral control does not overwrite Bigv2's previous action curvature.
- Removed the production recurrent reset on admission failure; failure still
  rejects inference and takes the ordinary fallback path. Warmup reset and
  per-frame recurrent-buffer updates retain the official sequence.

The strict source/frame inspector remains an opt-in offline diagnostic tool.
Its checks do not participate in `read_for_model` or live Bigv2 inference.

## Timing producer and clock evidence

Official navigation ages compare Python monotonic receipt times. Road-camera
capture timestamps are not operands in that calculation. Restoring the entire
producer/consumer interface therefore removes the need for the prior production
camera-clock translation and epoch rejection rules; it does not assert that
MONOTONIC and BOOTTIME are universally identical. Capture timestamps remain
untouched for frame publication and diagnostics.

The existing EPS integration explicitly requires its prediction schedule in both
model runners ([integration plan](integration-plan.md#timing-and-geometry)). The
dependency is `prediction_delay(vEgo)` → lateral prediction time → Bigv2 `action_t`
→ predicted action curvature → common curvature-consumption interface → firmware
controller → separate `command_delay(vEgo)` → steering output. A downstream
controller need not directly read `lat_delay` for its input curvature to depend
on that model conditioning. That observation cannot justify removing the schedule.

The prediction schedule, command-delay schedule, firmware-mode estimator guards
and delay-cache isolation remain unchanged. No command delay is added to
`action_t`, and no prediction delay is added to the command buffer. We have no
end-to-end model/vehicle evidence establishing that removing the prediction
schedule preserves the required steering response, so this change retains it.

## Necessary remaining adaptations

### 1. Corrected FP32 map computation

1. **Upstream:** the source checkpoint contains the map computation. Its original
   local AMD lowering was implicated in the demonstrated overflow investigation;
   the exact official a03 PKL's numerical behavior was not established here.
2. **Local:** the existing corrected artifact retains the FP32 map correction.
3. **Required functionality:** run the owner's already verified corrected native
   artifact without restoring the defective original model.
4. **Evidence:** [map precision investigation](navigation-model-map-precision.md),
   pinned compiler/provenance records and corrected artifact SHA256
   `4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c`.
5. **Minimality:** unchanged weights and tensor contracts; only the demonstrated
   precision correction is retained. No new clipping, zeroing or smoothing.
6. **Affected values:** map features can affect predictions and recurrent outputs;
   bit-identical numerical equivalence to the original official artifact is not
   asserted.

### 2. Native artifact/runtime and shared model lifecycle

1. **Upstream:** a03 loads its compiled TinyJit/OOB artifact using its pinned
   tinygrad and stock model startup.
2. **Local:** the same model contracts use the existing SPTGPLN1 artifact and
   pinned native-binary adapter inside SunnyPilot's shared Chestnut loader.
3. **Required functionality:** preserve the installed corrected artifact,
   Chestnut runtime, ordinary models, downloads, selection and fallback.
4. **Evidence:** distinct artifact formats/runtime revisions in the provenance
   report; [native runtime compatibility](sp-steer-runtime-compatibility.md);
   differential host packing, state-alias and publication tests.
5. **Minimality:** no fabricated model generation, alternate model controller or
   replacement fallback policy. The adapter retains the artifact's exact input,
   output, kernel and camera-warp contracts rather than renaming an incompatible
   official PKL or recompiling the model.
6. **Affected values:** intended model-facing tensors are unchanged; compiler,
   backend and numerical equivalence remain separately bounded by actual
   execution evidence. Ordinary fallback restores ordinary-model conditioning.

### 3. Required custom EPS prediction timing

1. **Upstream:** a03 forms lateral `action_t` from live `lateralDelay`, model
   smoothing and frame/action offsets; its controller is not this custom EPS
   integration.
2. **Local:** only when the latched custom EPS configuration is enabled, both
   runners substitute the unchanged speed-based `prediction_delay(vEgo)` for
   the lateral prediction delay. Existing smoothing and frame/action offsets
   remain. The controller independently applies its unchanged command delay.
3. **Required functionality:** preserve the owner's established custom EPS
   model/controller timing interface and steering response, explicitly required
   by the integration plan and the owner's clarification during this review.
4. **Evidence:** `honda_eps/timing.py`, both runners' actual input-construction
   paths, `latcontrol_fw.py`, and timing regressions trace the distinct model and
   command paths. Direct non-use of a controller argument does not establish
   independence from model-predicted curvature. No full prediction comparison
   demonstrates that removing this schedule is compatible.
5. **Minimality:** retain the existing substitution only in custom EPS mode;
   do not alter its values, add delay twice, or change `lagd` or ordinary learned
   caches. EPS-off Bigv2 uses the official live-delay path.
6. **Affected values:** lateral `action_t` (or legacy lateral-control parameters
   for ordinary models), predicted curvature and recurrent state may differ
   from stock-controller upstream. Longitudinal timing is unchanged. Numerical
   prediction equivalence is not claimed for this required adaptation.

API import relocation (`CommaConnectApi`) and extra SunnyPilot diagnostic schema
fields adapt existing interfaces without changing model pixels, maneuvers or
predictions. Optional map metadata never gates production inference. Controller
law/calibration, longitudinal planner/AccelBoost, Panda, firmware, selected model
records, download assets/notices and the updater are preserved.

## Verification and execution limits

Differential references are exact, MIT-licensed official source files pinned by
SHA256 and Git blob IDs. Tests execute their original functions/classes or inline
modeld blocks with explicit inert transports and clocks. They do not substitute
a reimplementation of local policy as an upstream oracle.

Comparisons cover route production, reroutes, republishing, absent/asynchronous
data, maneuver vectors, actual synthetic-vector-tile map pixels, desire pulses,
camera transforms and packed bytes, timing, parser/action/publication, and
recurrent-buffer semantics. Discrete inputs and map pixels use exact equality,
not loose tolerances.

- Navigation producer/consumer and isolated Msgq/VisionIPC comparisons: **103
  passed, 250 subtests passed**. Map renderer comparisons: **7 passed**.
- Native runtime, actual main-loop publication, startup/fallback, finite-output
  rejection and exact-source model comparisons: **59 passed, 38 subtests passed**.
- EPS control/timing/transport and Bigv2-to-sp-steer longitudinal regressions:
  **411 passed**. Synthetic curvature tests exercise actual controlsd and both
  firmware geometry choices; they are not learned-model comparisons.
- Existing warp/camera-offset checks: **13 passed, 34 subtests passed**. The
  `compile_warp.py` recipe is identical between official a03's tinygrad revision
  and the retained runtime; compiled GPU execution is a separate question.
- SunnyLink/model identity: **38 passed, 2 subtests passed**. All **12 rendered
  navigation UI tests and 32 subtests passed** in Xvfb with networking disabled.
- The extended navigation run exposed obsolete helper and AST-location
  expectations. They now compare official helper behavior and the actual shared
  loader boundary; targeted reruns passed (**8 helper tests**, **9 boundary tests
  and 62 subtests**). No production admission check was weakened to fix a test.
- Model selection/download/cache regressions: **118 passed, 9 subtests passed**,
  with one external-network integration test skipped. One ordinary-default
  hash test fails because this checkout contains the 133-byte ONNX Git LFS
  pointer rather than the 60,918,562-byte weights. The identical failure was
  reproduced on untouched `sp-steer`; the stored expected hash agrees with the
  LFS content OID. No ordinary weights, model hashes or selection logic changed.

Groups overlap and are not summed into a unique-test total. Changed Python passes
Ruff and syntax checks, and Git whitespace checks pass. Protected production
files and dependency pointers match the starting commit; the existing corrected
artifact was independently rehashed to the SHA256 above without executing it.

The CPU:LLVM runtime comparison executes the exact official packing/run code
against the native adapter using a small trusted synthetic tensor graph. After
warmup and reset, nine continuous frames have identical packed host bytes,
18,580 float32 raw outputs per frame, all five checkpoint-shaped recurrent
buffers and parsed outputs: **zero differences**, exact array equality. This
tests the runtime interface and state sequence, not Bigv2 weights or its warp
kernel. A separate actual-main test advances state exactly once per delivered
frame across twelve frames with missing IDs; upstream frame-drop accounting
matches, with no fabricated frames or recurrent resets.

An attempted full-model CPU comparison used a previously compiled, uncorrected
CPU Bigv2 artifact solely as a host adapter experiment. It terminated with SIGILL
during native loading, before model inference or output comparison. The cause is
not established; this is not an AMD result or a demonstrated model regression.
The corrected AMD artifact was not recompiled, replaced or executed here.

Recorded-controller replay is not used as evidence of prediction equivalence.
Host/source comparisons do not establish bit-identical corrected-AMD predictions,
physical camera/GPU execution or every real-world numerical outcome. No new
vehicle session, on-road failure or hardware-validation framework is required.

## Updating

Update `mvl-sp-steer-eps-nav` normally through **Settings → Software**, then restart
when prompted. Existing model selection persists. **Models → Big Models** remains
the sole selection workflow; no separate navigation-driving toggle, SSH activation,
recompilation or hidden parameter change is needed. Maps/guidance remain separate.
