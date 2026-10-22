# Navigation-model integration: historical software review

**Historical snapshot:** this report records the earlier implementation and test
revision named below. Its missing-artifact and approval-state statements describe
that revision. The [current workflow](navigation-model-software-compatibility.md)
supersedes its retired hardware procedures and activation requirements. The
custom activation counter, shadow execution and continuous small-model inference
described here have been removed; they are historical implementation details.

This review continues the existing implementation; no upstream driving-system merge,
model-runtime upgrade, new Git branch, device installation or firmware operation was
performed. The source implementation reviewed here is commit
`6908d2132c3f30504e20ae3240cc925ee978358f`. The final publication receipt and tests of
the actual integrated target revision accompany this report separately.

The fetched starting heads were:

- Experimental `mvl-honda-eps-nav-model-testing`:
  `036fd8ca9afb5240fa163c377bcfa2b961a38fe0`.
- Working integration target `mvl-honda-eps-selector21-chestnut-v28`:
  `4851eacac8fce2e875a0ef7703af5f5025fc0c15`.

The target is an ancestor of the experimental implementation, permitting a normal
history-preserving fast-forward after refetch and verification. Publication must
retain any concurrent target work; this document does not authorize resetting it.
No other branch, including `mvl-honda-eps`, `mvl-cinque-v3` or `master`, is a push target.

## What was implemented at the reviewed revision

Navigation Alpha display, file-only diagnostics and navigation-conditioned driving
selection have separate permissions and lifecycles. Existing map/destination display
remains available independently of model selection.

The branch-controlled `driving_models.json` registry exposes **Nav Model Integration
Bigv2** in the existing Models settings. Candidate preparation changes only new
navigation-selection Params. It preserves the ordinary small/big slots, downloads,
runner cache and `NavigationEnabled` setting. `NavigationModelDrivingEnabled` defaults
to false. Both comma 4 and comma 3 layouts display status, eligibility reasons and an
off-road-only opt-in control. Runtime identity reflects successful candidate or small
fallback inference rather than relabelling an unchanged ordinary selection.

At the historical revision reviewed here, Bigv2 was visible for preparation and
supported by the file-input inference code, but was ineligible for activation.
The physical compiler evidence and registered hashes were not yet available.
Those outstanding preparation items are now completed; see the current workflow
for the received evidence. A UI checkbox or local manifest still cannot approve
vehicle control.

The completed runtime path checks branch approval, exact source/artifact/metadata
identity, native tensor layout, actual USB firmware/product and discovered AMD LLVM
`gfx1200` target before candidate execution. It requires the configured ordinary
native tinygrad small model as a compatible fallback. Diagnostic models cannot
publish through either production runner; only the reviewed permit can admit the
candidate through the native SunnyPilot runner. The ordinary publication path and
ordinary action-conversion function are preserved.

A candidate-specific native 20 Hz action contract uses lateral acceleration in
m/s², curvature conversion by `max(vEgo,1)²`, longitudinal acceleration in m/s² and
second-based horizons. It retains the audited reference's lateral smoothing of0
and longitudinal smoothing of0.3s. It does not invent a SunnyPilot numeric generation.
Existing EPS timing hooks continue to supply the appropriate prediction delay.

**Historical policy, since removed:** the candidate performed132 valid shadow
samples before its outputs were eligible.
The ordinary configured small model runs during priming and continuously alongside
active candidate inference so its recurrent history remains current. Invalid route,
permission or synchronization data resets candidate history and uses that small
model. Reacquisition requires priming again. Genuine GPU/inference failures retain
failure reporting and existing disengagement safeguards; deliberate small-model
handover is distinguished from a failed Chestnut. An intentional permission
withdrawal keeps the warm small fallback until the normal process cycle restores
ordinary selection; UI changes are off-road-only. The combined AMD/QCOM workload
and learned transition quality remain physical activation gates.

Producer and reader fixes reject delayed/rolled-back generations, same-destination
reroute races, changed destinations during HTTP/rendering, map/frame association
mismatches, missing/unsupported maneuver metadata and invalid physical GPS time.
GPS loss cannot advance the route or delete an arrived destination. Display-only
legacy position retention cannot authorize model inputs. Clock discontinuities
and unaligned camera/GPS clocks are rejected rather than hidden by larger tolerances.

The exporter accepts numerically identical float signed-zero results while retaining
exact repeatability/integer checks. Catalog admission now rejects mixed-device
native buffers, including hidden CPU/QCOM/mock buffers in an AMD candidate. Recurrent
state snapshots copy all five native queues and reject non-finite states.

## Exact compatibility

The full [primary-source matrix](navigation-model-compatibility.md) and its
[evidence JSON](evidence/navigation-model-compatibility.json) distinguish enforced,
executed, source-compatible and unknown requirements.

| Component | Established requirement / result |
| --- | --- |
| Initial checkpoint | Bigv2 ONNX SHA256 `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`; official comma commit `a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9` |
| Official latest reference | `afbf2e547e9319e2eb9f515e1e4330af63705cec`; its n16 PKL declares a different checkpoint despite unchanged ONNX pointer/shapes |
| Firmware / USB | Existing source expects `custom 8516b768-CLEAN`; driver/firmware interfaces agree with the official reference. No additional Bigv2-specific firmware dependency found; current physical product is unobserved |
| GPU / backend | Physical USB AMD LLVM `gfx1200` required by this reviewed profile; production initialization discovers and verifies it rather than forcing its architecture |
| Runtime | Existing tinygrad `69fd4d34d4430036378aba2b43f9d339cdd6ea73`, unchanged; official reference pins `d3f09c9bbd542fbfbe68c8569a1173550c408969`, with inspected compile recipes equal but runtime internals not generally equivalent |
| Native artifact | Branch `SPTGPLN1`, full typed input/output/state metadata and compiler source/reload verification; official TinyJit PKL is not interchangeable |
| Selector / catalog | Existing Selector21 and v28 production catalog retained; separate branch extension and local verified diagnostic manifest. The catalog's advertised tinygrad ref is not the actual runtime pin |
| Generation / cadence | Numeric SunnyPilot generation is null; explicitly audited native action contract at20Hz, 11 inputs, five state pairs, six outputs |
| Navigation pixels | Raw float32 `nav_map` at0..255; graph performs `/255` internally; float32 `[1,47]` maneuver input |
| Warp / fallback | Model-matched native AMD warp, actual NV12 copied extent and pinned hash required; configured ordinary native small model must exist. Neither is silently substituted |

There is no usable verified AMD model artifact from this development environment.
The earlier AMD emulation failure is not a physical GPU/firmware verdict. No
firmware, submodule, dependency lock, production model or public catalog was changed.

## Verification at the committed implementation

| Check | Result at `6908d2132c3f30504e20ae3240cc925ee978358f` |
| --- | --- |
| Native x86_64 SCons build, generated schema/bindings | Passed, including new Params bindings |
| Navigation/UI/selector/lifecycle/messaging/diagnostic/compiler-boundary suite | 693 passed, 310 subtests passed, zero failures/skips;7 existing fork deprecation warnings |
| EPS/controls/DEC/model-loading/parsing/fallback/recovery/Chestnut/Params preservation suite | 589 passed, 8 subtests passed,2 existing skips, zero failures;1 existing fork warning |
| Ruff across all project Python changes since target baseline | Passed |
| Pinned-runtime type analysis | Four inherited diagnostics; no new diagnostics. Two tensor-shape typing sites and two variants of one stock Device-index typing site |
| Protected source and submodule audit | All12 protected path groups unchanged; seven indexed and checked-out submodule SHAs match the target baseline |

The2 preservation skips are reported as skips, not passes: an abstract parameterized
base test and the opt-in external-manifest check. That external check was then
explicitly enabled and passed separately (1 test,80.81s, HEAD requests without model
bodies). The untouched4851 source archive passed588 preservation tests plus8 subtests;
its sole initial failure hashed an unhydrated LFS pointer. Hydrating only
the exact baseline OID and rerunning that case passed (1 test,0.70s). This preparation
failure and correction are retained, not suppressed. All8 DEC checks passed on both
sources. No inherited regression failure was found in this scope. The four type
diagnostics were also reproduced from exact4851 source. Focused precommit runs
in the temporal/runtime reports remain historical; the table above is the coordinated
committed-source run. Test commands, JUnit, hashes and logs are retained in the
accompanying evidence package. Final target tests use their own recorded SHA.

The [temporal report](navigation-model-temporal-review.md) explains which tests use
actual RouteEngine/MapsD/NavigationInputs, Msgq, VisionIPC, native Params and raylib.
A300-decision sequence has140 valid samples before and140 after rerouting. Native
queue tests exercise full recurrent history; mocked trusted model execution is
identified as such. These are not learned driving-quality tests.

A final file-only CPU/LLVM run at the implementation SHA used the verified native
Bigv2 artifact `a280f8873a0391685bf4ea115504203b7448ce5a034b401788cc1e2921384790`
and CPU warp `556cdca4c1740434000e52a0c57f9d0027359d62502deb0c539735af82dc0337`.
All8 predictions (two recorded camera frames × four synthetic-navigation arms)
completed without failure, with all five recurrent queues finite and saved. All27
parsed output arrays per prediction matched the corresponding Stage2 output exactly
(maximum absolute difference0). This is reuse/regression evidence, not an independent
ONNX oracle. Inference took 26.37–27.11s/frame
(median 26.68s); maximum process RSS was
2.52GiB. It is not real-time CPU or GPU evidence.
The fixture uses neutral desires and reconstructed timing, including its original
zero smoothing; it does not reproduce the newly reviewed live candidate's0.3s
longitudinal smoothing or original on-device model inputs. Navigation is synthetic;
full132-frame learned inference was not executed in that historical CPU run.

Existing Stage2 compiler and four-arm camera experiments are reused with their
original artifact/source identities; they are not relabelled as final-source or
physical GPU results. The final audit matched all67 frozen Stage2 source hashes
and separated the compiler source snapshot from later report-only changes.

## Preserved behavior and control consumption

The [independent final audit](navigation-model-final-audit.md) verifies unchanged
Honda EPS calculations, calibration, torque conversions, limits, override logic,
model timing hooks, MVL planner/braking tuning, safety/vehicle submodules, hardware
firmware logic, updater, production assets, launches and dependency locks against
the exact working baseline. Shared schema/compiler/adapters from the existing
experimental history are retained and regression-tested; no wholesale upstream
control merge is performed.

Navigation-conditioned predictions enter the existing prediction interfaces, not
direct steering/braking commands. Direct desiredAcceleration/shouldStop planner
consumption requires `ExperimentalMode` and either DEC inactive or DEC `blended`.
ACC and DEC `acc` exclude that direct candidate, but model leads, predicted trajectory,
SCC-V and DEC mode selection can still affect longitudinal behavior indirectly.
Unchanged controller code is not proof of unchanged behavior with different predictions.

## Retired diagnostic procedure

The former standalone hardware-test framework and its operational instructions
were removed after successful physical compilation and existing-model
registration. They are not prerequisites for this branch. Use the
[software compatibility workflow](navigation-model-software-compatibility.md).

## Update and rollback

On the already-installed target branch, remain parked/off-road, connect networking,
use the normal Software settings update check/download and restart when ready.
Confirm repository `ZakariaSaad/ZakariaSaad`, branch
`mvl-honda-eps-selector21-chestnut-v28` and the published revision. No reinstall,
installation-URL change, branch switch or updater bypass is needed. This is source
configuration and regression verification; an actual device update has not been run.
The updater and branch migration implementation are byte-unchanged. Actual
Updater/native Params checks verified this exact custom branch on mici both from
current-branch fallback and an explicit UpdaterTargetBranch; both preserve the Param
and return the same branch. No download/staging/install action was triggered. Existing model slots/settings persist;
new navigation driving permission defaults off and Bigv2 remains ineligible.

The known-working pre-integration rollback revision is
`4851eacac8fce2e875a0ef7703af5f5025fc0c15`. A rollback must use **new revert commits**
on the same branch, not reset/rebase/force-push. In a separate development checkout,
first inspect `git log --oneline 4851eacac..HEAD` and the intended diff. Revert the
navigation-project commits in reverse order, latest first; omit any unrelated
concurrent commits. For the linear integration reported here, the fixed reviewed
range is `4851eacac..<full-publication-SHA-from-completion-receipt>`; use
`git revert --no-commit 4851eacac..FULL_PUBLICATION_SHA` (substitute that full SHA),
inspect/test, then commit and normally push the revert to the same target. Stop on conflicts and
review them. The recorded known-working baseline already contains Navigation Alpha, which
the rollback therefore preserves. An off-road normal update then applies the rollback without a fresh
installation. No rollback was executed as part of this task.
