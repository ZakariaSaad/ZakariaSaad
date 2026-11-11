# Selective NRDR Bosch-A integration

Target branch: `mvl-sp-steer-eps-nav` in `ZakariaSaad/ZakariaSaad`.
Dependency branch: `mvl-sp-steer-opendbc-nrdr-bosch-a` in the same repository.
Main base: `d0d58d2844af0cf6f003c965a8b397723e2fc610`.
Dependency base: `84ce49fcd02904f324eff9252ae6c1c10b9ffb4b`.
Tested dependency commit: `3489464a5623fcea5a30646987d110c899459b65`.

This selectively ports NRDR PR27 and its prerequisites. It does not include the
earlier project's optional stop-command guard, observation-only mode, custom
master gate, brake-preservation clamp, extra fallback/filter policies, or an
Insight calibration prerequisite. Earlier analysis and separate branches remain
preserved; they are not merged into this branch.

## Source authority

| Source | Revision |
| --- | --- |
| Current `nrdr/openpilot` reference | `622a1479bd27598a949d5ca601e486380b0369a0` |
| PR27 head | `fd40efb737251d2734e2d62837e37f47235b9df8` |
| PR27 merge | `4aa4d5349878742d1df395b1dbd55d3d21284949` |
| Current `nrdr/opendbc` reference | `f3e0e2904f388bdf17d7779e7f9ad6191062e2ad` |
| Parser PR3 | `461f81cddfa96be4fbf2b9c01e821790f8110f27` |
| Actuator PR4 | `427f1986ed241f111d106cc72cdefbda5efe5f7f` |

Earlier required dependency behavior is present from
`0dae94409be7570cfb0a760ca8bb218aca78f797` (DBC/harness),
`184629dfd58769ad77b527dfa7ab648071f4f5a6` (range scale),
`52fc0a285c6e507baabef47b3ff4fb8a1707e716` (factory-ACC safety/cruise reconciliation),
and `1cdf9e41a25c66770dbc7b7013a1c8c12a94606d` (legacy PID arrays).
Radar host prerequisites include `0bd55916397ab9c01c7daad7a10006222d8a365f`
and `c2828dd8599bcf9022ea2c6f0a3b913965680604`.

The shared parser, DBC, RadarInterface, Bosch longitudinal helper, low-speed
brake helper, brake onset and gas learner match the current dependency source.
The later Civic PCM experiments are unrelated to the Insight shared path and
are not imported. Civic-only gas limits/learning remain Civic-only.

## Enabling and mode selection

| Setting | Full imported Insight path | Default/missing behavior |
| --- | --- | --- |
| Alpha Longitudinal | ON | Existing target preference and confirmation are retained |
| Enforce Factory Longitudinal Control | OFF | Existing target default OFF; saved choice retained |
| Live Learning Gas — Full MVL Longitudinal | OFF | Missing Honda choice initializes ON; saved OFF retained |
| Honda Bosch-A Radar | ON | Donor default ON; saved OFF retained |
| Experimental Mode | Optional | OFF; no new confirmation or prerequisite |
| Following profile | Driver choice | Existing saved value; Econ available only in imported mode |

`HondaLiveLearningGas` selects full MVL versus the imported NRDR planner. The
immutable startup `HondaFlags.MVL_LONGITUDINAL` flag (131072) carries this choice
to the planner and controller. Changing a saved preference does not mutate
current CarParams or grant live control authority.

The NRDR planner is selected for classic Bosch-A + openpilot longitudinal + MVL
OFF. Turning radar OFF does not select the old planner: donor radar-independent
HumanFollowing, smooth and onset behavior remain, while radar-dependent pieces
disable themselves. To select the established MVL path, turn Live Learning Gas
ON. To retain factory ACC, use the existing factory-longitudinal selection.

The radar preference is independent of longitudinal authority. It can decode
with factory ACC or MVL. Consequently, MVL's original formulas are preserved,
but enabling decoded radar can change its lead inputs. Equality tests of the
old and retained planners use identical inputs and do not establish identical
driving behavior when radar inputs change.

### Comma 4 procedure, keeping the Insight in READY

1. Keep the car parked in READY, disengaged, and select the existing forced
   offroad control while installing/configuring. Accessory mode is unnecessary.
2. Use branch `mvl-sp-steer-eps-nav` with its pinned submodules. The repository
   gitlink, rather than an independently selected opendbc head, identifies the
   tested dependency. Update submodules using the normal branch installation
   procedure and allow the device build to finish.
3. Under Settings → Honda longitudinal, turn Enforce Factory Longitudinal
   Control OFF, Live Learning Gas OFF, and Honda Bosch-A Radar ON. This page
   requires a valid current/cached Honda identity. The larger UI exposes the
   same choices on its existing Honda vehicle settings page.
4. Under Settings → Developer, enable the existing Alpha Longitudinal control
   and complete its existing confirmation. This is the usual authority choice;
   this port adds no separate activation confirmation.
5. Exit forced offroad while staying in READY and parked. Allow the normal car
   process restart/onroad cycle to load the saved startup choices. A setting
   that says “next drive” is not proof that current CarParams changed.
6. Check current CarParams: Insight fingerprint, `openpilotLongitudinalControl`
   true, `pcmCruise` false, `radarUnavailable` false, and MVL flag131072 absent.
   Confirm the expected parameters above. A stale factory-ACC CarParams cache
   means the new process selection has not yet taken effect.
7. In the active imported mode, Settings → Toggles exposes optional Experimental
   Mode and Econ. Leave Experimental OFF for donor ACC planning or turn it ON
   for donor model-assisted planning. A distance-button hold of 50 control
   frames (about0.5s) also switches it when cruise is available; a short release
   cycles the four profiles. The donor speed-limit-assist reservation consumes
   the button gesture instead when applicable.
8. While parked and disengaged, verify fresh `radarTracks`, paired
   `carStateSP.radarTracksMonoTime`/`radarPointStatus`, and `radarState`; inspect
   parser errors/validity rather than expecting a stationary scene to provide a
   lead. Check `selfdriveState.experimentalMode` against the selected setting
   and fresh finite `longitudinalPlan` output. Do not engage merely to test
   recording. These are device checks to perform, not checks already performed
   remotely by this integration.

No car was contacted, configured or deployed to during this work. The optional
donor Reset Live Gas action retains its ignition-OFF/card-stopped guards. It is
not part of enabling this feature and is unnecessary for the READY procedure.

## Why the scoped Experimental Mode signal is required

The donor's `selfdrive/controls/lib/longitudinal_planner.py` consumes
`selfdriveState.experimentalMode` in four relevant places:

* `bosch_a_human_accel_base`: `ACCEL_MAX` versus the normal profile envelope.
* Model launch: launch enhancement requires Experimental mode and the donor's
  additional armed/motion/non-stopping conditions.
* `get_cruise_accel`: Experimental versus ACC acceleration/coast/turn rules.
* Candidate selection: MPC/cruise are always candidates; E2E/model is a candidate
  only when Experimental is ON. Candidate stop flags follow the same selection.

The original target always includes E2E in the planner candidates. It had
retired the persistent Experimental setting and did not publish an effective
Experimental boolean. Alpha Long is an authority choice, MVL is an algorithm
choice, and `longitudinalPlanSource` is a result; none is an equivalent signal.
Inferring Experimental from any of them would silently change donor behavior.

Required restoration is therefore limited to a manual OFF/ON signal in the
imported mode, not an upstream Experimental Mode subsystem:

| Item | Change and scope |
| --- | --- |
| `ExperimentalMode` Params key | Persistent/backup Boolean, missing default false; saved choice retained |
| `SelfdriveState.experimentalMode @10` | Move existing deprecated field to active group, retaining wire ordinal |
| `selfdrived` | Read/publish effective setting only for Bosch-A + OP long + MVL OFF; false elsewhere |
| Normal/comma4 Toggles | Manual control visible only in that imported mode |
| Distance button | Donor hold toggles mode, short release cycles profiles, SLA reservation/consumption preserved |
| `OnroadEventSP.experimentalModeSwitched @14` | Reuse existing deprecated ordinal for donor latched ON/OFF warning |
| `LongitudinalPersonality.econ @3` | Fourth profile only in imported mode; stale Econ maps to Relaxed outside it |

Donor stock-longitudinal deactivation clears a Bosch-A saved Experimental true
choice. MVL selection publishes false without destroying the saved choice.
No DEC/automatic mode switching, ExperimentalModeConfirmed prompt, experimental
camera selection, global UI architecture or default-ON mode is restored.
Experimental OFF still runs the imported planner: this setting changes its
decisions, not which planner instance runs. Outside the imported mode, its
parameter does not change planning, button behavior or camera selection.

The detailed [Experimental Mode source audit](NRDR_EXPERIMENTAL_MODE_SCOPE.md)
records the exact donor and target files, conditions and line references.

## Processing and compatibility boundaries

The imported path preserves donor candidate selection and this post-MPC order:
Bosch-A long → coast-first → release-coast → smooth → onset-ramp → stop-gap,
then the donor final clipping/filter updates. It replaces overlapping legacy
gas/wind/brake shaping on its selected actuator path rather than applying both.
The Insight uses the shared learner and its separate JSON persistence; the
Civic-only learner/750 gas ceiling are not broadened to Insight.

Intentional adaptations are:

1. Select the new planner/control law only for shared Bosch-A + OP long + MVL
   OFF. Retain this target's existing MVL and other vehicle implementations.
   A copy of the original planner is retained as `baseline_planner.py`.
2. Reuse the existing host parameter snapshot/startup interface rather than
   importing NRDR's complete typed configuration architecture. Seed deprecated
   PID arrays only where the imported LongControl requires them, including
   radar-OFF mode. Other startup defaults remain target-owned.
3. Keep target navigation, map/SCC/speed-limit services. Apply the directly
   required donor boost semantics, Econ bypass and SCC-V active/enable check
   only in the imported path. There is no double boost or legacy brake shaping.
4. Preserve existing schema ordinals: radar status/time use CarStateSP@1/@2
   instead of donor@5/@6; lead trajectories use LongitudinalPlan@41..44 instead
   of donor@40..43 because this branch already uses@40 for accelBoost.
5. Remove one unused `v_ego_stopping` keyword at the otherwise identical host
   model-interpolation call. Keep Econ constants locally instead of importing
   an unrelated MVL cruise module. Numeric behavior is unchanged.
6. Import only the necessary parameter groups while retaining donor snapshot
   coercion, atomicity and0.5s refresh. Use the target registry for the donor's
   missing Experimental=false initialization.
7. Fit required controls into normal and comma4 settings. Preserve existing
   Alpha Long confirmation, camera behavior, unrelated controls and the target
   parameter API. Optional reset uses a raw-byte compatibility read and the
   existing manager lifecycle; unrelated remote reset RPC is not imported.

EPS/lateral, lane-change logic, model support and navigation implementations
are retained. Panda Honda safety source is unchanged. The accepted donor range
formula remains `raw/16 - 346/128` metres for Insight: offset−2.703125m. This
acceptance is not a claim of a physical Insight calibration.

## Validation and limits

The final main suite passed 440 tests plus 21 subtests; the complete Honda
dependency suite passed 581 tests. Native cereal/Params, messaging, MPC and loggerd build completed in this
checkout. Tests exercise defaults/startup, decoder→RadarInterface→RadarD,
native MPC parity, stopped/restart/driver-pedal/cancellation behavior, actual
controller CAN packing, mode controls, parameter migration and optional reset.
Frozen donor sources and source-hash fixtures are included with the tests.

Private replay completed 203,859 native solves across seven complete prefixes,
with zero raw solver errors or nonfinite targets. Active target versus exact
donor main/MPC matched 36,983 updates, including native arrays. Retained baseline
stock and MVL comparisons matched 9,472 and 36,983 updates respectively. Radar
receipt qualification unavailable in rlogs is explicitly treated as assumed or
conservative; software counterfactuals are not measured vehicle responses.

Important retained donor behavior: ACC mode excludes the model stop candidate;
an existing event therefore changes from baseline +0.090114/stop=true to donor
+1.427885/stop=false. Another existing braking target changes from −2.425615
to −0.780. These match the donor and are reported rather than silently clamped.
A positive planner target during stopping can still produce positive encoded
ACCEL_COMMAND with braking/standstill bits asserted. Encoded requests do not
prove positive physical force, stopping distance or ECU acceptance.

The real-controller 270-tick comparison matches 270 packed donor ACC frames;
it validates port wiring, not the physical correctness of the donor law. The
separate custom stop-command guard remains unmerged. B raw segments35–36 remain
missing, beyond the tested prefixes ending in31. No new drive was required.

Actual GUI rendering and comma4 hardware/vehicle validation were not performed.
UI imports, policy/state tests and source checks pass; this cloud has no working
display/Xvfb. Native cloud tests do not establish device build or road behavior.

### Reproduce repository checks

From the main checkout, after its normal development setup and native build:

```sh
export PYTHONPATH=.:tinygrad_repo:opendbc_repo:msgq_repo:rednose_repo
export DEV=CPU:LLVM
python -m pytest -q openpilot/nrdr/tests \
  openpilot/selfdrive/controls/tests/test_upstream_longitudinal.py \
  openpilot/selfdrive/controls/tests/test_predictive_accel_boost.py \
  openpilot/selfdrive/controls/tests/test_longcontrol.py \
  openpilot/sunnypilot/system/tests/test_params_migration.py
python -m pytest -q opendbc_repo/opendbc/car/honda/tests
```

Native donor comparisons live in
`openpilot/nrdr/tests/test_bosch_a_native_parity.py`; control mode tests are in
`test_driver_controls_port.py` and `test_honda_longitudinal_settings.py`;
LongControl parity is in `test_longcontrol_port.py`. The private route replay
harness, receipts, per-event tables and original raw inputs are preserved
outside the public source tree. Their method report identifies the exact input
paths, source hashes, receiver assumptions and reproduction commands.
