# Register the corrected Bigv2 already on the device

This completes the integration of the separately compiled FP32 map correction
on `mvl-honda-eps-selector21-chestnut-v28`, starting from
`b9aba01d3a25f0a6c86707fe3aa52b29dcb918de`. No further compilation, download,
GPU diagnostic or on-road reproduction is needed for registration.

The corrected candidate is **Nav Model Integration Bigv2 — FP32 map correction**
(`big-nav-v2-map-fp32`). It appears separately from the original Bigv2 under
**Models → Big Models**, as an installed local navigation-aware model after
registration. Selecting it keeps Navigation Model Driving **OFF**. Its branch
approval is **false**; the original artifact's approval does not apply to it.

## Update and register

While parked and genuinely off road, leave Navigation Model Driving off. Use
**Settings → Software** to check for updates, download/install the update on
the existing `mvl-honda-eps-selector21-chestnut-v28` branch, and restart when the
normal updater requests it. No reinstall, branch switch or service changes are
needed. This integration does not modify the updater.

From Windows PowerShell, run this one command after the update:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage register big-nav-v2-map-fp32"
```

It uses system Python as `comma`, with no sudo. It checks fresh software
off-road/disengaged state, hashes the existing files, validates the native
artifact and proof, retains a small registration package, and selects the
corrected candidate with driving OFF. It does not open the GPU, execute a model,
download or copy model weights, compile anything, stop `comma.service`, flash
firmware or alter ordinary model selections. Physical ignition OFF is not the
definition of off-road status.

Required existing files are:

- `/data/nav-bigv2/bigv2-amd-map-fp32.sptg`
- `/data/nav-bigv2/bigv2-amd-map-fp32.verification.json` (the `.sptg.verification.json`
  report name is also accepted)
- The existing validated camera warp already used by the original registration.

Registration retains exact compiler and precision-lowering source snapshots,
the verification report, camera warp, metadata and a manifest under a distinct
`/data/navigation-models/big-nav-v2-map-fp32-…/` package. The model path continues
to refer to the existing artifact. The new receipt is
`/data/navigation-models/big-nav-v2-map-fp32-registration.json`;
the original `latest-registration.json`, package and artifact are preserved.
Failed selection writes restore the previous candidate/manifest and leave
driving OFF.

Inspect the result with:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage status big-nav-v2-map-fp32"
```

Expected: matching registration, installed corrected artifact, selected candidate,
reviewed compiler/software evidence, **Branch activation approval: not approved**,
and **Navigation Model Driving: OFF**. The precise approval blocker is
`Explicit user authorization for Navigation Model Driving is still required`.
If another blocker appears, use its exact message; do not rerun compilation.

In **Settings → Models → Big Models**, choose **Nav Model Integration Bigv2 —
FP32 map correction**. Both Models layouts use the same candidate-specific
registration and admission logic as SSH. The corrected model is selectable while
approval-blocked; its Navigation Model Driving toggle cannot bypass approval.

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

The artifact's actual local bytes are checked during registration; they were not
available in this development environment. The metadata pin is derived from the
reviewed report, existing camera profile and native contract using the actual
exporter's deterministic format. Registration checks the bytes it writes against
that pin. It also binds the report's protected native kernel and initializer
hashes to the actual artifact, without executing pickle data or kernels.

Once explicitly approved and admitted, the existing native loader receives this
candidate's manifest and artifact path. It cannot substitute the original Bigv2.
The original input preparation, parser, recurrent state, finite-output rejection,
single-active-model execution and ordinary small-model fallback are unchanged.
Honda EPS, MVL controls, safety, firmware, catalogs, Selector 21 and routing are
unchanged.

## Remaining approval

Registration completes local file binding. The remaining approval step is an
**explicit authorization for experimental driving with corrected artifact
`4f4c215d…4c1a919c`**, followed by a reviewed branch change setting only this
candidate's `activation_approved` to true. This task does not grant that approval.
After that separate authorized update, the existing Settings toggle or
`manage enable big-nav-v2-map-fp32` will still require off-road/disengaged state
and ordinary runtime admission. No hidden parameter or JSON editing is needed.

The report establishes successful physical compiler execution and the reported
synthetic comparisons. It does not establish sustained corrected-model driving
performance or reproduce the exact original failing road frame. This is a limit
of the evidence, not a request for another compilation or validation framework.

## Roll back without deleting either package

To withdraw navigation driving permission and resume the ordinary selections:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage disable"
```

For ordinary **Mountain Dew Model v3**, keep navigation driving off and select
**Settings → Models → Big Models → Mountain Dew Model v3 (Default)**. The normal
Selector 21 interface remains in charge of ordinary big/small models.

To restore the original Bigv2 candidate and its retained manifest while off road:

```powershell
ssh comma@10.0.0.134 "cd /data/openpilot && python3 -B -m openpilot.selfdrive.navigation.manage select big-nav-v2"
```

This also leaves driving OFF; it neither deletes the corrected package nor
changes ordinary model slots. Inspect it with `manage status big-nav-v2`.
The original Bigv2 retains its known map-overflow defect, so restoring its
selection is not a claim that the numerical failure is fixed in that artifact.

## Focused regression scope

The integration checks cover real data-only native container/proof parsing,
corrected precision provenance, unchanged external model contracts, candidate
identity binding, failed-registration rollback, original-package preservation,
non-inheritance of approval, shared CLI/UI selection and both rendered Models
interfaces. Fixtures are inert and do not execute the uploaded artifact.
No physical GPU validation was repeated in development.

Results: **126 focused tests and 175 subtests passed**, plus **17 rendered UI
tests and 8 subtests** under Xvfb. Ruff and Git whitespace checks passed.
The compiler and precision-lowering source files retain exactly the hashes above.
