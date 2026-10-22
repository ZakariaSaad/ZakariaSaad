# From SSH validation to reviewed navigation-model activation

Navigation Model Driving remainsOFF, and the Bigv2 candidate is not approved. This
change prepares diagnostics and the review process; it does not grant live activation.
The user never needs to edit JSON/source, compile by hand or set hidden Params.

## Operator workflow

1. Update the same installed v28 branch normally, run the [simple SSH smoke command](navigation-model-ssh-smoke.md)
   and return `latest-results.tar.gz`. Run the offered `prepare` command only if a
   verified Bigv2 artifact is missing and the measured resources allow it.
2. Codex checks source/runtime/firmware/device identities, actual compiler proof,
   output/state contracts, timing, power/resource failures and service restoration.
   Probe-only or smoke-only evidence is labelled accordingly, never treated as full
   activation approval. Missing evidence is returned as a precise next-test list.
3. Codex supplies any additional automated validation commands needed for the observed
   setup, reusing its artifact/cache. It must not require a GPU workstation, external
   bench, manual model compilation, user-assembled manifests or hand-edited settings.
   Genuine power/off-road incompatibility, memory shortage or faulty hardware remains
   a blocker; source changes cannot safely bypass it.
4. Once sufficient results exist, Codex prepares a concrete review: exact checkpoint,
   native model/warp/compiler-metadata hashes, firmware/runtime profile, compatible
   configured small fallback, review evidence and intended activation changes. The
   review must identify remaining road-quality/behavior limits and proposed operating
   constraints. No larger generation number or firmware upgrade is inferred from names.
5. The user explicitly authorizes that reviewed activation change. Only then may Codex
   implement/test the approval and automatic preparation changes and normally push
   `mvl-honda-eps-selector21-chestnut-v28`. Other branches remain untouched.
6. The user performs a normal Software settings update/restart, then prepares/selects
   the approved candidate and enables **Navigation Model Driving** in Models settings
   while off-road. Existing ordinary model slots remain available as before. Turning
   the switchOFF and the normal off-road process cycle restore ordinary selection.

A short smoke pass alone cannot advance step4 to approval. The returned report includes
`activation_readiness.normal_activation_allowed=false` and a machine-readable checklist.
There is no diagnostic CLI switch that flips this approval.

## Evidence required before approval

| Gate | Initial smoke establishes | Additional evidence still required |
| --- | --- | --- |
| Real off-road state / exclusive access | Fresh normal device/panda messages; continuously read physical ignition, silent/noOutput safety; ownership and restoration | Repeated maintenance/startup/recovery checks on the actual setup |
| Firmware / actual GPU / runtime | Exact USB product, discovered USB AMD LLVM gfx1200 and pinned runtime import | Compatibility during repeated operation; no speculative firmware change |
| Exact native artifact | On prepare: pinned public ONNX, native compiler hash, two independent source/reloaded verification seeds, actual AMD target | Retained model/warp/metadata identities and reproducible final artifact chain |
| Tensor/state execution | Eight artificial prewarped frames; finite typed outputs, five state aliases and image-queue appends | Independent equivalent-input CPU/reference versus GPU comparison of all prediction/state tensors |
| Camera/navigation integration | Not exercised by this short prewarped smoke | Correct warp/upload extent, real VisionIPC/frame identity, map palette/orientation, maneuver geometry and route-generation association |
| Full recurrent history / transitions | Two samples per arm, intentionally cold-start diagnostic policy | More than132 learned samples, reroutes, consecutive maneuvers, GPS/map/instruction/permission loss, restarts and reacquisition |
| Clock handling | Off-road admission source clocks only | Actual camera/GPS clock domains, suspension/resumption, valid age/skew and rejection/recovery |
| Performance / power / cooling | Short inference times, RAM and sampled power/thermal safeguards | Sustained20Hz, stable USB, memory, thermal/power budget on this180W outlet; transient/dropout behavior |
| Driving-runtime coexistence | No production inference runs during smoke | Non-actuating measurement of simultaneous AMD navigation plus continuously warm QCOM fallback, resource headroom and genuine failure handover |
| Controls and fallback | No CAN/vehicle controllers run; ordinary selections preserved | Verified installed fallback identity, EPS/MVL timing, driver override/disengagement/safety preservation and mode-dependent prediction consumption review |
| User approval / delivery | Remains unapproved | Explicit authorization of the exact reviewed activation commit and artifact set |

The existing [advanced package](navigation-model-hardware.md) supplies compiler/warp,
reference/GPU, recurrent-state and long-sequence tooling. Synthetic repeated inputs
are useful execution tests but do not substitute for actual route/camera provenance or
learned driving quality. Where additional capture/testing is needed, Codex must supply
an automated, non-actuating procedure and describe its scope before any execution.
No navigation-conditioned driving or remote deployment is authorized by these steps.

## What the future approval update must deliver automatically

The current driving registry deliberately has `hardware_reviewed=false`,
`activation_approved=false` and null model/warp/metadata hashes. The runtime also
requires a verified local manifest and a compatible ordinary native small-model fallback.
The present selection helper does not install/register those artifacts by itself.
Those are explicit pending delivery work, not something the user should repair through
hidden parameters.

The authorized approval change must therefore include all of the following, tested
with successful/failed/partial downloads, unavailable resources, restarts and OFF mode:

- Pin the exact reviewed native artifact/warp/metadata hashes and source evidence in
  the branch-controlled candidate registry, without altering the public production
  catalogs or inventing a SunnyPilot generation.
- Provide an off-road UI preparation action that discovers/reuses the verified local
  diagnostic artifacts or downloads the exact approved bytes from a reviewed source,
  checks hashes, prepares/relocates the manifest and registers its path automatically.
  Hashes alone cannot recreate a lost compiled model: an approved artifact must be
  retained, or its exact reproducible preparation must be automated and reverified.
- Preserve compiler-proof paths/runtime identity when relocating; the diagnostic
  manifest is path-bound. Validate the relocated bundle before making it available.
- Keep the permission switchOFF through update and preparation. Require the user's
  explicit off-road Settings action to enable it. Fail closed with a readable reason
  and retain ordinary model operation if any artifact/profile/fallback check fails.
- Keep the configured small fallback and model-specific timing correct; verify the
  normal updater and rollback on the exact final source. Do not change steering,
  braking, safety, firmware or normal model settings to make admission succeed.

Codex performs this work only after the evidence and explicit authorization in steps4–5.
No unsupported artifact, local diagnostic manifest or manually forced checkbox can
approve operation. Initial road validation, if later authorized, remains a separate
controlled activity; no short off-road test establishes safe navigation driving.
