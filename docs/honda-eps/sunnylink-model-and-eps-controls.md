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

In SunnyLink, open **Models → Big Models → Installed local navigation models**,
choose **Nav Model Integration Bigv2 — FP32 map correction (installed local)**,
then **Activate**. On the comma, use **Settings → Models → Big Models** and
choose the corrected FP32 model. Both paths submit the same guarded request.

Explicit selection reuses `/data/nav-bigv2/bigv2-amd-map-fp32.sptg`, checks the
approved identity, provenance, compatible warp and ordinary small-model fallback,
and performs file-only preparation if registration is missing. Once preparation
succeeds off road, it selects the corrected native navigation loader and requests
**Navigation Model Driving ON**. Selecting a model never engages controls.

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

## Installed-local availability and optional publication

Corrected Bigv2 currently supports **installed-local reuse only**. If the file is
absent, the entry says **local file required; download unavailable**, and admission
refuses it. No download URL is fabricated. The required artifact SHA256 is:

`4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c`

The full corrected binary was not available in this development workspace. The
prepared `nav-bigv2-map-fp32-release-sidecars.zip` contains the reviewed report,
native warp, source/provenance, metadata and licenses, **not** the model. Archive
SHA256: `176f714e646e45318a44d493a9fbaf1a0af3dc53f7dabc9dbc853b97cf3771fa`.

Optional one-time publication, separate from using the installed model:

1. In WinSCP, connect using SFTP to **10.0.0.134**, user **comma**, with your existing
   SSH authentication. Copy only
   `/data/nav-bigv2/bigv2-amd-map-fp32.sptg` to your Windows laptop.
2. Open [your GitHub Releases page](https://github.com/ZakariaSaad/ZakariaSaad/releases)
   in a browser. Create/edit the intended release and upload that model file and
   the provided sidecar ZIP as assets. Do not commit the large binary to source.
3. Provide the actual asset URLs. Download support remains incomplete until the
   downloaded model checksum is verified and the downloader/catalog is bound to
   the verified model and sidecars. Uploading alone does not add download support.

Keep the supplied comma/tinygrad MIT notices and SunnyPilot's distinct custom
license/acknowledgments with the release. The bundle is not wholly under an
unqualified MIT license. No credentials, private routes or firmware files belong
in the release. This publication step requires no JSON editing or compilation.

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
was performed here. Installed-local selection and the EPS schema do not depend
on deploying that patch; strict stale-client detection does.

The corrected model remains experimental. Compiler/source-reload evidence does
not establish real-world numerical stability. Non-finite-output rejection and
the existing single-active-model small fallback remain in place, including its
previously observed first-invocation delay. Controller algorithms, MVL tuning,
Panda safety, firmware, Chestnut runtime and updater are unchanged by this cleanup.

## Focused verification of this cleanup

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
