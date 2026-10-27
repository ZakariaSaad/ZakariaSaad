# SunnyLink model and Insight EPS controls

Current workflow for `ZakariaSaad/ZakariaSaad`, branch
`mvl-honda-eps-selector21-chestnut-v28`. Routine selection uses SunnyLink or the
comma's Models settings. No SSH, recompilation, firmware operation or new approval
is required for the already-approved corrected model.

## Update the installed branch

With the vehicle parked and the comma genuinely off road and disengaged, use
**Settings → Software → Check for updates**, download/install the update, and
restart when prompted. Keep the same repository and branch. No reinstall or
service changes are needed. The normal updater is unchanged.

Refresh SunnyLink after the device reconnects so it reads the new settings schema
and model list. A browser's cached page is not confirmation of device state.

## Insight EPS controller

In SunnyLink, select your device, then **Steering → Insight EPS Controller →
Insight-specific EPS lateral controller**.

- **ON:** request the existing Insight controller for the next drive cycle.
- **OFF:** restore the existing ordinary lateral-controller selection path on
  the next drive cycle.
- **Insight EPS requested and effective status** reports eligibility, preserved
  calibration, requested state and the current drive's latched controller.

The device requires genuine off-road/disengaged status, the supported identified
Insight configuration and the existing installed-firmware confirmation. The
toggle cannot create that confirmation or flash firmware. Missing prerequisites
and conflicting vehicle configuration are explained in the status. Existing
geometry/response values and ordinary torque/NN preferences are preserved. An
admitted Insight controller retains its existing priority; turning it off does
not erase those ordinary preferences or the firmware confirmation.

## Corrected Bigv2 and ordinary models

In SunnyLink, open **Models → Big Models → Navigation models**, choose
**Nav Model Integration Bigv2 — FP32 map correction**, then **Activate**. Its
availability reads **installed local** when present, or **downloadable**. On the
comma, use **Settings → Models → Big Models** and choose the corrected FP32 model.
Both paths submit the same guarded request.

Selection first checks `/data/nav-bigv2/bigv2-amd-map-fp32.sptg`. If its SHA256
matches the approved artifact, it is reused without redownloading or duplicating
the model. Otherwise, the manager downloads the published model and required
sidecars into temporary files, checks their pinned sizes and SHA256 values, and
installs complete verified files atomically. Required local preparation and
registration happen automatically while off road. **Activate** therefore covers
**Download → Verify → Install → Select**; no SSH, manual receipt editing or
compilation is needed.

The existing SunnyLink download progress and cancellation controls track the
request. **Model selection status** explains preparation, failure or cancellation.
A partial, cancelled or superseded download cannot replace the verified installed
model or activate a different selection. Wait until the requested switch completes
before starting a drive.

Before committing selection, the device checks approved identity, compiler
provenance, compatible warp and the ordinary small-model fallback. It then selects
the native navigation loader and requests **Navigation Model Driving ON**. Selecting
a model never engages controls.

To return to Mountain Dew, choose the **Mountain Dew Model v3** default entry in **Models → Big Models**,
or another ordinary big model, and activate it. The comma labels the default
entry **Mountain Dew Model v3 (Default)**. The comma's equivalent
is **Settings → Models → Big Models**. After the requested ordinary model is ready,
the switch withdraws navigation driving automatically. The ordinary small-model
selection and Navigation Alpha maps, routes and destination sending are preserved.

To keep navigation driving off without changing the saved ordinary selections,
use **Models → Navigation Model → Navigation Model Driving → OFF**. This explicit
OFF survives reboot. Choosing corrected Bigv2 again is a new explicit ON request.

Use **Model selection status** to distinguish pending/preparing, selected,
loading, active and failed states. Selected/requested while off road does not
mean inference or controls are active. A failed preparation keeps the previous
valid configuration; its exact reason is shown. Do not start a drive assuming a
pending request has completed. Unsafe on-road changes are refused by the device,
including when the browser offers Force Offroad.

## Persistence, migration and recovery

Corrected registration, ordinary model choices and the EPS toggle are preserved.
Reboot, reconnect, update and catalog refresh do not create a new Bigv2 selection.
An explicitly chosen Default big model also survives the legacy-slot migration
check on subsequent boots.
Interrupted switches use a persisted transaction to finish or restore the previous
selection; a superseded preparation cannot later overwrite a newer choice.

The original `big-nav-v2` is retired and cannot be selected or executed. An old
saved request to drive that model is withdrawn, then migrated to corrected Bigv2
only after corrected-file admission succeeds. Missing/corrupt corrected files
leave navigation driving off with a failure explanation. A saved old-model OFF
does not become ON. Migration is repeatable after interruption.

No model, ONNX, report, recording, firmware evidence or shared warp is deleted by
this migration. Required active corrected/fallback files are protected from model
cache cleanup. There is no automatic legacy-file removal.

For recovery, remain off road, turn **Navigation Model Driving OFF**, and select
**Mountain Dew Model v3 (Default)** or a known-working ordinary big model. Disable
the Insight EPS toggle only if you intend to change the controller for the next
drive. Use the normal update mechanism for subsequent fixes. Do not restore the
retired original Bigv2. The pre-cleanup source rollback point is
`086fd710f3ea3b5c1088003a1d7e1132f96c2cfe`; reverting source would also restore its
older selection semantics, so prefer the ordinary-model recovery above.

## Published assets and portable installation

Both assets were downloaded independently from the
[published release](https://github.com/ZakariaSaad/ZakariaSaad/releases/tag/nav-bigv2-map-fp32-sidecars)
and verified by hashing their bytes:

| Asset | Size | SHA256 |
| --- | --- | --- |
| [bigv2-amd-map-fp32.sptg](https://github.com/ZakariaSaad/ZakariaSaad/releases/download/nav-bigv2-map-fp32-sidecars/bigv2-amd-map-fp32.sptg) | 810,439,301 bytes | `4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c` |
| [bigv2-sidecars-from-comma.zip](https://github.com/ZakariaSaad/ZakariaSaad/releases/download/nav-bigv2-map-fp32-sidecars/bigv2-sidecars-from-comma.zip) | 3,279,412 bytes | `ebe109ce6352e8d144b60b623c047320b923d6493921f860e48f0efd32a291de` |

This ZIP is the owner's device-created registration package. It differs from the
earlier workspace ZIP and has its own checksum. It contains the camera warp,
compiler report, compiler/lowering source, metadata, manifest and registration
receipt. The installer verifies the approved component hashes and provenance.
It does not trust copied absolute paths, the old receipt's registration claims or
any activation flag from the archive. It generates the local manifest and receipt
for this installation using the existing reviewed exporter and admission checks.

The original ZIP lacks license notices. The accompanying
[complete release notices](navigation-model-release-notices.md) supply comma's
MIT license, tinygrad's MIT license, and SunnyPilot's separate custom license and
acknowledgments. The published release description now includes the complete
notices, and new registration packages retain a copy. The published model and ZIP bytes are unchanged;
adding notices does not change their approved hashes. The entire package must
not be represented as covered by one unqualified MIT license.

## Hosted SunnyLink limitations

The existing hosted frontend already understands the device-provided Steering
schema, Models schema and cached-catalog interface used here. The native
navigation entry is a response-only display overlay; it does not falsify an
ordinary model generation or overwrite ordinary catalog/slot parameters.

Current frontend requests contain only a ref or default `{}` without a client
revision. The device fences superseded jobs, but cannot distinguish a fresh action
from identical stale browser intent. The hosted frontend also briefly displays
optimistic checkbox/default state. The adjacent device status is authoritative.

The [prepared frontend patch](patches/README.txt) adds observed-revision checking,
dynamic status polling and non-optimistic queued model acknowledgments. It needs
deployment by the maintainers of `sunnypilot/sunnylink-frontend`; updating this car
branch does not deploy it. No hosted deployment or live website/device session
was performed here. The download/selection flow and EPS schema use the existing
hosted interface and do not depend on deploying that patch; strict stale-client
detection does. This download integration does not modify or deploy that patch.

The corrected model remains experimental. Compiler/source-reload evidence does
not establish real-world numerical stability. Non-finite-output rejection and
the existing single-active-model small fallback remain in place, including its
previously observed first-invocation delay. Controller algorithms, MVL tuning,
Panda safety, firmware, Chestnut runtime and updater are unchanged by this cleanup.

## Download integration verification

- Independently downloaded both public assets and verified their exact byte
  sizes and SHA256 values above. Existing data-only validators accepted the
  native tensor contracts, protected FP32 kernel/initializer bytes, compiler
  provenance, pinned tinygrad revision and saved warp NumPy comparison.
- Isolated installation with the real model and ZIP rebuilt registration and
  preserved the approved metadata hash after staging files were removed. The
  existing model's inode and selection parameters were preserved; registered
  reuse made zero HTTP requests. HTTP transport replayed the downloaded ZIP and
  off-road messaging was mocked; no GPU ran.
- **275 focused tests and 36 subtests passed**. One existing optional external
  network test was skipped; both actual release downloads were checked separately.
  Coverage includes download size/hash failures, HTTP errors/timeouts,
  cancellation, atomic publication, registration, fallback admission, model
  switching, persistent choices and cache protection.
- Both rendered Models layouts: **9 tests and 32 subtests passed** under Xvfb
  with networking disabled. Generated schema matches YAML; all **10** schema
  checks, Ruff and Git whitespace checks pass.
- Verified the unpatched hosted frontend's Activate, polling and Cancel protocol.
  No hosted frontend changes, model recompilation or physical-device actions.

## Earlier UI-cleanup verification history

These results describe the earlier UI cleanup in commit `99c175065`; they are not
the current download integration test results.

- Combined host regressions: **423 passed, 128 subtests passed, 2 skipped**.
  Skips are the existing optional external-network download test and a generic
  settings test requiring the unavailable `jsonschema` package.
- Rendered comma Models interfaces under Xvfb, with networking disabled:
  **9 passed, 32 subtests passed**.
- Native C++ Params library rebuilt successfully. The filtered-key lifetime fix
  is covered with actual native backup key enumeration.
- Generated settings schema matches its YAML sources; all **10** dedicated
  schema validation checks pass. Ruff and Git whitespace checks pass.
- Prepared external frontend: Svelte checks report **0 errors, 0 warnings**.

Coverage includes real SunnyLink RPC with persistent Params, guarded EPS settings
and drive latching, retired-model admission, preparation rollback, shared CLI/UI
selection, superseded jobs, reboot/Default persistence, atomic download failures,
cache protection and immediate startup runner resolution after a model switch.
Synthetic publication/fallback tests use **CPU:LLVM**, not physical AMD execution.
No model was recompiled, no private route/capture was published, and no physical
device, firmware or live vehicle controls were accessed.
