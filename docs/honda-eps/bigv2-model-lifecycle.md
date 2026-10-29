# Corrected Bigv2 in the ordinary Big Model lifecycle

Branch: `mvl-sp-steer-eps-nav`. Starting commit:
`8a2f81b3c411437d4e356291c08fd8a2f81252ad`.
Authoritative upstream lifecycle: MVL `sp-steer`
`0f5085fc3f713da58d236ed3822b9d27bc0e9917`.

The subsequent [official Bigv2 parity review](bigv2-official-parity.md) restores
the checkpoint's navigation, camera, desire and output semantics while retaining
this lifecycle and the required custom EPS prediction schedule. Its test results
and execution limits are documented separately from the lifecycle results below.

## What changes

Corrected Bigv2 now occupies `ModelManager_ActiveBundleChestnut`, the same canonical
slot used by Flip Flop, Mountain Dew and other big models. The shared selection
queue downloads missing files, verifies exact bytes, prepares registration and
commits that slot only while genuinely off road and disengaged. Background
preparation does not select a model. The previous selection remains authoritative
until the replacement is ready.

The separate Navigation Model Driving toggle, independent writable activation
preference and branch activation-approval flag are removed from normal operation.
A stale activation setting cannot compete with the selected model. Maps, route
guidance and destination sending remain separate.

The selected Bigv2 uses the normal big-model loader thread, timeout and upstream
retry/fallback policy. Its model-specific admission runs inside that loader;
an admission failure cannot silently start a preserved alternate ordinary big
model. The normal small model is prepared as upstream requires and used after
big-model failure. Only one model executes at a time. There is no shadow inference,
continuous dual-model operation, new priming policy or new fallback framework.

Model status distinguishes requested/selected from actually publishing. A ready
big model is not reported as running before publication, and a prepared small
fallback is not reported as working before its first successful publication.
Failures in small initialization or inference are reported as failures. The device
UI also requires live `modelV2` messages with the matching big/small identity.
SunnyLink describes the last confirmed runtime report rather than claiming a
continuous model heartbeat.

## Preserved adapters and identities

The corrected artifact remains:

`4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c`

The existing FP32 map correction, compiler provenance, camera warp, native tensor
dtypes/shapes, navigation inputs, recurrent-state handling, action interpretation
and non-finite-output rejection remain. The canonical descriptor identifies the
native model; it does not assign a fabricated ordinary model generation. The
registered manifest supplies the actual native loader paths and integrity pins.

Honda Insight EPS, installed-firmware recovery, sp-steer longitudinal planning and
AccelBoost, Panda safety, driver overrides, Chestnut firmware/runtime, ordinary
models and the updater are outside this lifecycle change. The retired original
Bigv2 remains unavailable. Published model/sidecar bytes and redistribution notices
are unchanged; no recompilation or physical-device action is required here.

## Migration and restart behavior

The one-time migration recognizes a prior completed, explicit corrected-Bigv2
selection and its matching enabled state. A stale activation flag, candidate name
or local model file alone cannot override an ordinary selection. Old OFF state
preserves the ordinary slot. Interrupted transactions are recovered before further
selection, and conflicting pending intent is not silently promoted. Once migrated,
only the canonical slot controls startup. Saved legacy flags/remote writes cannot
reintroduce the separate activation preference.

Ordinary Big Model selections persist across restarts. Selecting Bigv2 again uses
the same mechanism. Registration alone only records verified files; it cannot
select a model or change the ordinary small-model choice.

## Supplied-log findings

The five supplied segments cover two successful startup windows at the starting
commit, not a reproduced terminal startup/fallback failure. Matching-schema
`LogReader` analysis orders each route by `Event.logMonoTime`:

- Ordinary Flip Flop initialized in approximately 22 seconds; its recovered
  sequence contains 1,485 big-model predictions.
- Corrected Bigv2 initialized in approximately 24 seconds; its recovered sequence
  contains 2,461 big-model predictions, 2,460 marked valid.
- No complete recovered record establishes a big-model-failed alert or a runtime
  fallback/non-finite-output exception in those windows.
- The ordinary route has a wall-clock correction during startup, so wall-clock
  log creation timestamps do not establish reliable elapsed durations.
- The last Bigv2 upload ends with incomplete compressed and Cap'n Proto data.
  Complete recovered records remain useful; upload truncation is not proof of a
  device/process crash.

The [sanitized LogReader evidence](evidence/bigv2-lifecycle-log-review.json)
records counts, monotonic timing and the incomplete tail without route positions,
VINs or raw parameter snapshots.

A small-model initialization line during startup is expected upstream preparation,
not evidence that fallback actually executed. The owner's later restart symptom
remains unconfirmed by these recordings. The source-level split selection and
misleading early runtime status were concrete cleanup targets; these should not
be represented as a proven cause of the unrecorded symptom.

## Normal use

1. While parked, genuinely off road and disengaged, update this same branch through
   **Settings → Software**, then restart when prompted. Refresh SunnyLink afterward.
2. Open **Models → Big Models**, select **Nav Model Integration Bigv2 — FP32 map
   correction**, and use the existing activation/download action. Wait for selection
   to complete. Verified local bytes are reused; missing files are downloaded and
   prepared automatically.
3. To switch away, select Flip Flop, **Mountain Dew Model v3 (Default)** or another
   ordinary big model through that same menu. No navigation OFF action is needed.

No SSH command, manual JSON/hidden-parameter edit, service stop, firmware change or
additional approval is part of this workflow. Selecting a model never engages
controls. See [the controls guide](sunnylink-model-and-eps-controls.md) for download
URLs, cancellation, status and EPS controls.

## Verification scope

- Rebuilt the native Params library with the migration-only marker.
- Native admission, downloader/registration, fallback formats, native execution
  compatibility, route/temporal input handling and ordinary fallback regressions:
  **102 passed, 67 subtests passed**.
- Preparation/registration, optional CLI and canonical selection compatibility:
  **56 passed, 14 subtests passed**.
- Real modeld startup/publication, native input handling and numerical rejection:
  **42 passed, 37 subtests passed**. These exercise the actual shared loader body
  and main publication loop with inert devices or synthetic CPU tensors.
- EPS and Bigv2-to-longitudinal compatibility: **407 passed**.
- SunnyLink RPC and model identity: **37 passed**, followed by **3 identity tests
  and 2 subtests passed** after adding live-message health gating.
- Both rendered Models layouts, maps and EPS controls: **20 passed,
  32 subtests passed** in a network-disabled Xvfb container.
- Shared selection, downloader, cache/default models and migration:
  **128 passed, 1 skipped, 18 subtests passed**. The skip is the existing optional
  external-network manifest-availability test; no release bytes changed.
- Changed Python passes Ruff; generated settings match YAML; Git whitespace
  checks pass. Controller/firmware/runtime pins and corrected model hashes are
  unchanged from the starting commit.

Groups overlap; these are not summed into a unique-test total.
Synthetic host tests do not establish physical AMD execution or every real-world
restart outcome. No new on-road failure or broad hardware-validation project is
required by this cleanup.
