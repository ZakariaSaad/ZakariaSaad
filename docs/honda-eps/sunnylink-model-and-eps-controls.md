# SunnyLink model and Insight EPS controls

Current workflow for `ZakariaSaad/ZakariaSaad`, branch
`mvl-sp-steer-eps-nav`. Routine selection uses SunnyLink or the comma's Models
settings. Corrected Bigv2 uses the same selected Big Model slot and startup/fallback
lifecycle as the ordinary big models. There is no separate navigation-driving
toggle, SSH enable command or manual approval step.

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
corrected Bigv2 in the normal Big Model slot. The shared loader recognizes its
native navigation tensor/warp contract when that model starts. Selecting a model
never engages controls.

To return to Mountain Dew, choose the **Mountain Dew Model v3** default entry in **Models → Big Models**,
or another ordinary big model, and activate it. The comma labels the default
entry **Mountain Dew Model v3 (Default)**. The comma's equivalent
is **Settings → Models → Big Models**. After the requested ordinary model is ready,
the ordinary model replaces Bigv2 in that same slot. The ordinary small-model
selection and Navigation Alpha maps, routes and destination sending are preserved.

Use **Model selection status** to distinguish pending/preparing, selected,
loading, active and failed states. Selected/requested while off road does not
mean inference or controls are active. A failed preparation keeps the previous
valid configuration; its exact reason is shown. Do not start a drive assuming a
pending request has completed. Unsafe on-road changes are refused by the device,
including when the browser offers Force Offroad.

## Persistence, migration and recovery

Explicitly selecting Bigv2, Flip Flop, Mountain Dew or another big model writes
one canonical `ModelManager_ActiveBundleChestnut` choice. Reboots, ordinary ignition
cycles and model-process restarts retain that choice. The model manager verifies
files before committing a replacement. Interrupted or superseded transactions
cannot silently select a different model. A background download or an installed
Bigv2 file does not create a selection.

For this update, a completed explicit corrected-Bigv2 selection from the old
workflow migrates once to the canonical slot. The migration requires the matching
completed selection record and prior ON state; a bare stale ON flag or candidate
name cannot override an ordinary selection. Old OFF state keeps the ordinary
selection. Conflicting/incomplete old transactions are recovered conservatively.
Legacy activation writes from a cached SunnyLink page are refused; refresh the
page and use Big Models. The original `big-nav-v2` remains retired and is never
silently promoted to corrected Bigv2.

The loader follows sp-steer's normal big-model loading and one-cycle retry policy.
It prepares the selected small model for fallback but executes only one model at
a time. A big-model failure switches to that prepared small model. Loading a small
model during startup does not by itself mean fallback occurred. Runtime status
separates loading/ready, actual published predictions, fallback pending, working
small fallback and terminal failure; an initialized model is not called running.

No model, ONNX, report, recording, firmware evidence or shared warp is deleted by
this migration. Required corrected/fallback files remain protected from model
cache cleanup. Registration alone records verified files and never selects them.

For recovery, remain off road and select **Mountain Dew Model v3 (Default)**,
Flip Flop or another known-working ordinary big model. No separate navigation OFF
action is needed. Disable Insight EPS only if you intend to change the lateral
controller for the next drive. Use the normal Software updater for subsequent
fixes. The source rollback point for this lifecycle cleanup is
`8a2f81b3c411437d4e356291c08fd8a2f81252ad`; reverting source restores the former
split-selection behavior, so review model settings if doing so.

See [Bigv2 model lifecycle](bigv2-model-lifecycle.md) for supplied-log evidence,
upstream comparison and test scope.

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
navigation catalog entry is a native-format descriptor. It occupies the existing
Big Model selection slot; the loader uses its verified manifest instead of
pretending it has an ordinary model generation. Runtime status does not control
selection.

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

## Earlier download integration verification

These results describe the preceding downloadable-model integration, not the
current lifecycle cleanup. Current checks are in the lifecycle report linked above.

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
