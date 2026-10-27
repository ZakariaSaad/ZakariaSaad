# Corrected Bigv2: integration and retained evidence

Current model: **Nav Model Integration Bigv2 — FP32 map correction**,
`big-nav-v2-map-fp32`. It is approved for the owner's experimental activation of
the exact artifact below. The original `big-nav-v2` is retired: it is neither a
selection option nor a rollback target. Historical files and evidence are kept.

Use [SunnyLink model and EPS controls](sunnylink-model-and-eps-controls.md) for
current update, switching and recovery instructions. This supersedes the earlier
registration-first, separate-SSH-enable workflow.

## Normal selection

Update the same installed `mvl-honda-eps-selector21-chestnut-v28` branch through
**Settings → Software** and restart when prompted. In SunnyLink, open **Models →
Big Models → Navigation models**, choose the corrected FP32 model and **Activate**.
The entry identifies itself as **installed local** or **downloadable**. The equivalent comma menu is **Settings → Models → Big Models**.

Explicit selection reuses an installed `.sptg` whose SHA256 matches the approved
artifact. If needed, it downloads and verifies the published model and sidecars,
installs complete files atomically, and performs missing local registration.
The existing SunnyLink progress/cancel controls track this request. It verifies
approved provenance/warp/fallback and requests Navigation Model Driving ON only
after admission succeeds off road and disengaged. No separate receipt preparation,
compilation, GPU diagnostic, hidden parameter edit or new approval is needed.
Existing valid registration is reused. A failed, cancelled or superseded request
preserves the prior valid selection and reports its reason.

Selecting **Mountain Dew Model v3 (Default)** or another ordinary big model
withdraws navigation driving after that replacement is ready. **Models →
Navigation Model → Navigation Model Driving → OFF** withdraws the request without
altering saved ordinary model selections. OFF and ordinary choices persist across
reboot. Maps remain available. Selecting any model never engages controls.

Existing files used by automatic preparation:

- `/data/nav-bigv2/bigv2-amd-map-fp32.sptg`
- `/data/nav-bigv2/bigv2-amd-map-fp32.verification.json` (the existing
  `.sptg.verification.json` report name is also accepted)
- The compatible registered camera warp, or its existing tracked native copy.

The corrected package remains separate under
`/data/navigation-models/big-nav-v2-map-fp32-…/`; its receipt is
`/data/navigation-models/big-nav-v2-map-fp32-registration.json`. No old model,
report, warp or source ONNX is automatically deleted. Published distribution is
available through the existing Models interface; full URLs, verified sizes/hashes
and cancellation/recovery instructions are in the
[controls guide](sunnylink-model-and-eps-controls.md#published-assets-and-portable-installation).

The device-created sidecar archive has SHA256
`ebe109ce6352e8d144b60b623c047320b923d6493921f860e48f0efd32a291de`
and size 3,279,412 bytes. Its copied absolute paths and receipt claims do not
authorize installation or activation. The installer verifies the pinned contents
and regenerates local manifest/registration paths. The published model is
810,439,301 bytes. Both assets retain their exact published bytes.

The sidecar ZIP contains no license texts. The accompanying
[release notices](navigation-model-release-notices.md) provide full comma MIT,
tinygrad MIT and SunnyPilot custom notices and acknowledgments for redistribution
and the installed package.

## Evidence and identities

The supplied verification report records physical **AMD / LLVM / gfx1200 / USB**
execution with clean tinygrad `69fd4d34d4430036378aba2b43f9d339cdd6ea73` and compiler
revision `b9aba01d3a25f0a6c86707fe3aa52b29dcb918de`. Its status is
`passed_source_reload_consistency`, with `artifact_promoted: true` and
`candidate_activation_approved: false`.

All 30 reported source/reload output comparisons passed with equal output hashes
and zero numerical difference. These include both working/overflow synthetic map
cases, predictions and all five recurrent outputs. Compiled IR/native buffer
checks confirm FP16 operands, FP32 dot/bias/channel scaling, and the original FP16
residual boundary. The external input/output and recurrent contracts are unchanged.

| Binding | SHA256 |
| --- | --- |
| Corrected artifact | `4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c` |
| Uploaded compiler report | `3f776eafc180c87fc85333bd029c4ee2bddf206b00ce7df695eba54e306af591` |
| Exact retained compiler source | `0a1c5174d7a27faf03b0304defd2dac6fa1b1cec036165b7dfd42ef0e6a06ce0` |
| Exact precision-lowering source | `696d9c3b9949618d051ed6fafcfe69c7cd8bc0e2ae17ae0d94d7eeea8e174c4a` |
| Expected portable metadata | `cff885b4ef1af332582f6df847351b8d69c2da6e70979718a03e1931f7dbad95` |
| Unchanged camera warp | `f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b` |
| Unchanged source ONNX | `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e` |

The published artifact was independently downloaded and its complete bytes
matched the approved SHA256 above. Installation and registration check those bytes
again. The metadata pin is derived from the
reviewed report, existing camera profile and native contract using the actual
exporter's deterministic format. Registration checks the bytes it writes against
that pin. It also binds the report's protected native kernel and initializer
hashes to the actual artifact, without executing pickle data or kernels.

Once explicitly selected and admitted, the existing native loader receives this
candidate's manifest and artifact path. It cannot substitute the original Bigv2.
The original input preparation, parser, recurrent state, finite-output rejection,
single-active-model execution and ordinary small-model fallback are unchanged.
Honda EPS and MVL algorithms, safety, firmware, Selector 21 runtime and routing
are unchanged. The current UI/catalog overlay and shared selection queue are
described in the controls guide.

## Approval, limitations and recovery

The final review of integration commit
`c2c0d72a9451823e980249203672597e74f7aa30`, followed by owner authorization,
approved this corrected candidate in
`086fd710f3ea3b5c1088003a1d7e1132f96c2cfe`. The historical compiler report's
`candidate_activation_approved: false` remains unchanged: compilation evidence
and later branch authorization are different records. Current selection preserves
the exact approved identities; it cannot approve unknown artifacts.

The physical compiler/source-reload comparisons establish the reported synthetic
results, not sustained real-world numerical stability or reproduction of the
original failing road frame. Existing finite-output rejection and single-active-
model small fallback remain. The previously observed roughly one-second first
small-model invocation during fallback is not eliminated by this cleanup.

For recovery, turn Navigation Model Driving OFF and select Mountain Dew or a
known-working ordinary big model through the same UI while off road. Do not
restore original Bigv2. No package deletion, firmware restoration, service change
or fresh installation is required.

## Regression history

The original corrected-artifact integration passed 126 focused tests and 175
subtests, plus 17 rendered UI tests and 8 subtests. Its later approval checks
passed 79 tests and 76 subtests, including synthetic CPU:LLVM publication/fallback
behavior, finite-output rejection, off-road admission and CLI persistence.
These are historical results, not new AMD execution in this cleanup.

The earlier UI cleanup (`99c175065`) also covered real RPC and persistent Params,
registration/admission before commit, failed-selection recovery, stale queued jobs,
ordinary switching, explicit OFF persistence, retired candidate handling and
active-file protection. Current download tests cover pinned assets, cancellation,
atomic installation, local reuse and portable registration; see the current
change's test report for run counts.
The [prepared hosted frontend patch](patches/README.txt) remains a separate
maintainer deployment. This download integration uses existing hosted interfaces
and does not modify or deploy that patch.
