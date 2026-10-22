# From software compatibility to reviewed navigation-model activation

Navigation Model Driving remains OFF. The Bigv2 candidate is not approved, and an
SSH smoke pass cannot approve it. The goal is a correct SunnyPilot implementation
of comma's navigation-model software, not certification that Chestnut can execute
comma models. Existing ordinary big-model operation is preserved.

## Operator workflow

1. Update the same installed v28 branch normally, run the
   [short SSH compatibility check](navigation-model-ssh-smoke.md), and send
   `latest-results.tar.gz`. This step has no automatic model download or Bigv2
   compilation. A missing compatible AMD artifact is a specific software-preparation
   blocker, not a request to spend 30 minutes compiling on the car battery.
2. Codex reviews identities, off-road/control evidence, GPU communication, actual
   artifact proof if present, predictions/timings and service restoration. A probe
   result is labelled probe-only; short synthetic inference is labelled as such.
3. Codex completes the remaining software work and targeted integration checks.
   Any additional on-device check must be automated and scoped to a real unresolved
   compatibility issue. No user-assembled manifests, hand compilation, source edits,
   hidden Params, external bench or separate AMD workstation are required of the
   operator. Long endurance/reference tools remain available for development.
4. Once sufficient evidence exists, Codex presents the exact checkpoint and native
   model/warp/metadata hashes, runtime/firmware profile, fallback identity, intended
   changes and remaining limitations for explicit activation authorization.
5. Only after that authorization may Codex implement/test the reviewed approval
   change and normally push `mvl-honda-eps-selector21-chestnut-v28`. No other branch
   is changed and no device deployment is performed remotely.
6. The user performs a normal Software settings update/restart, prepares/selects the
   approved candidate in Models settings and explicitly enables **Navigation Model
   Driving** while off-road. Turning it OFF and the normal off-road process cycle
   restore ordinary selection. Existing ordinary model slots remain available.

The user must not manually edit JSON/source or set hidden Params at any step. There
is no diagnostic CLI switch that grants approval. The report keeps
`activation_readiness.normal_activation_allowed=false`.

## Remaining software requirements before activation

| Requirement | Existing evidence / implementation | What still needs to be delivered or verified |
| --- | --- | --- |
| Exact native AMD Bigv2 artifact | Public Bigv2 ONNX identity and native compiler contract are known; CPU execution has been exercised | A real AMD `SPTGPLN1` artifact, pinned runtime/compiler identity, complete source/reloaded output/state proof, retained artifact hash and reviewed distribution/reuse path. No compatible precompiled native artifact is currently verified |
| Navigation inputs and model I/O | Raw 0..255 map tensor, 47-value maneuver vector, dtype/shape/state contracts and synthetic independent input arms are implemented | Keep parity with the selected official checkpoint; exercise any corrected contract with targeted regression tests. Do not infer compatibility from a model name or rename a comma PKL |
| Stateful model/runtime integration | Five recurrent pairs, reset/priming, generation checks, prediction parsing and fallback paths have automated coverage | Verify the exact delivered native artifact's aliases/outputs and short non-actuating execution; resolve any actual software discrepancy. Test route/GPS/map/permission/restart transitions in reproducible fixtures rather than requiring the operator to assemble 164 frames |
| Camera and timing interfaces | Existing warp/frame association and EPS/MVL timing interfaces are preserved | Pin and admit the correct warp/metadata; confirm camera upload dimensions, navigation/frame timestamp domains and rejection/recovery for the final bundle |
| Model selection and fallback | Ordinary model slots remain intact; candidate permission is default OFF and a compatible native small fallback is required | Automatic off-road UI preparation/manifest registration; verify failed/partial preparation keeps ordinary selection. Validate exact-bundle candidate/fallback loading, handover and timing/resource coexistence without vehicle actuation |
| Consumer/safety compatibility | Existing Honda EPS, MVL longitudinal and safeguard paths remain preserved | Final intended-diff and regression review for the approved bundle, including model-specific delays, mode-dependent prediction consumption and driver override/disengagement. No steering/braking tuning change is part of this feature |
| Explicit authorization | No activation approval exists | Present exact hashes and reviewed changes, obtain separate authorization, then deliver the tested update and an explicit off-road Settings switch |

The short SSH check supplies actual USB/GPU/runtime information and, only when a
verified artifact is already available, a few native predictions. It does not
exercise the live camera warp, production daemon coexistence or navigation driving.
If one of those has an actual unresolved software issue, Codex must identify it
specifically and supply the smallest useful automated check. A short non-actuating
production-runner/coexistence check may be needed for the final artifact; that is
distinct from demanding an endurance or thermal-certification session.

## Optional development checks

The [existing comprehensive package](navigation-model-hardware.md) retains
equivalent-input CPU/GPU comparisons, long recurrent sequences and detailed
power/thermal measurements. They can help investigate a compiler discrepancy,
memory leak, timing problem or reset bug. They are **not** the default SSH procedure,
and a 164-frame run, thermal certification, dedicated test bench or prolonged
ACCESSORY-mode session is not a blanket prerequisite for software compatibility.

Measured short inference timing still matters to the model's 20 Hz contract; existing
hardware capability does not excuse a wrong artifact, excessive memory ownership
or broken fallback. If a concrete runtime budget problem appears, report it and
fix the responsible software before activation. Synthetic repeated inputs do not
prove learned driving quality, and no off-road smoke test proves safe navigation
driving. Any later controlled road validation remains a separate authorized activity.

## What the future approval update must deliver automatically

The driving registry deliberately retains `hardware_reviewed=false`,
`activation_approved=false` and null model/warp/metadata hashes. The runtime requires
a verified local manifest and compatible ordinary native small-model fallback.
The current selection helper does not install/register those artifacts by itself.
This is pending software delivery work, not a problem for the user to repair.

Before the final enabling update, the implementation must:

- Pin the exact reviewed model/warp/metadata hashes and source/compiler evidence in
  the branch-controlled candidate registry. Preserve production catalogs and do not
  invent a SunnyPilot generation.
- Provide an off-road UI preparation action that discovers/reuses verified local
  artifacts or downloads the exact approved bytes from a reviewed source, validates
  hashes and registers the manifest automatically. Hashes cannot recreate a lost
  artifact: retain the reviewed bytes or provide an automated, reverified preparation.
- Preserve proof/runtime identities when relocating the path-bound manifest and
  validate the relocated bundle before admitting it. Test successful, failed and
  interrupted preparation, unavailable resources, restarts and OFF mode.
- Keep permission OFF through update/preparation; require the user's explicit off-road
  Settings action to enable it. On any missing artifact/profile/fallback check, show
  the precise reason and retain ordinary model operation.
- Preserve model-specific timing, Honda EPS/MVL behavior, Chestnut/Selector 21 and
  normal same-repository/same-branch updates and rollback. Do not alter firmware or
  unrelated safety/control settings to make admission succeed.

Software implementation and verification can proceed while approval remains OFF.
The approval flags and exact approved artifact set change only after the user has
reviewed the evidence and explicitly authorized live integration. Nothing in this
workflow authorizes remote deployment, firmware flashing or navigation actuation.
