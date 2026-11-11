# Why this faithful Bosch-A port restores a scoped Experimental Mode selector

## Conclusion

The selector is needed to reproduce the donor's two existing longitudinal
planning modes. It is **not an extra switch required to activate Bosch-A**.
The donor Bosch-A chain operates with Experimental Mode either OFF or ON. OFF
is the donor's default. The setting changes several inputs and candidate
choices inside that chain; omitting it or fixing it to one value would remove
one of the donor's behaviors.

The target previously removed this selector and always admitted the model's
E2E action into its candidate set. That behavior is not equivalent to the
donor's OFF mode. The current port restores only the state, manual controls,
publication and feedback required by the imported path. It retains the
target's previous planner outside Bosch-A + openpilot longitudinal + MVL OFF.

This audit inspected the completed source port, not a deployed car. Production
files were not edited by this independent review.

## Source authorities and reproducible evidence

- Donor: `nrdr/openpilot` development
  `622a1479bd27598a949d5ca601e486380b0369a0`.
- Target baseline: `ZakariaSaad/ZakariaSaad`, `mvl-sp-steer-eps-nav`,
  `d0d58d2844af0cf6f003c965a8b397723e2fc610`.
- Inspected working tree: `/workspace/bosch-a-faithful`, based on that target.
- Exact donor/baseline source snapshots, with original line numbers preserved,
  are preserved under the private validation directory `experimental-mode-source/`;
  `SOURCE_MANIFEST.json` records their hashes. They were extracted using `git show COMMIT:path`, not reconstructed
  from prose or PR descriptions.

Paths and line numbers below are relative to the corresponding source root.
Working-tree line numbers can move as the final implementation is assembled.

## The donor actually consumes the Boolean

The donor publishes `selfdriveState.experimentalMode` from the saved
`ExperimentalMode` preference, qualified by actual
`CP.openpilotLongitudinalControl`: `selfdrive/selfdrived/selfdrived.py:623` and
`:690`. The class initializes the runtime value false at `:150` and writes a
saved false marker when OP longitudinal is unavailable at `:129`.

`nrdr/params/defaults.py:31` sets the fresh-install value to **False**. The nearby
comment in selfdrived about an earlier default enabling it again is not the
current default's authority; the actual defaults initializer is.

In donor `selfdrive/controls/lib/longitudinal_planner.py`:

| Function / location | OFF | ON |
|---|---|---|
| `bosch_a_human_accel_base`, lines99–114; called at229 | Uses NRDR's acceleration envelope and personality ceiling | Uses `ACCEL_MAX` as the base before HumanAcceleration's cap |
| `LongitudinalPlanner.update`, lines262–267 | Model action is read, but E2E launch enhancement is inactive | Passes true to donor launch enhancement, subject to its own conditions |
| `get_cruise_accel`, lines68–90; called at271 | Uses the donor cruise envelope, turn/throttle-coast restrictions and positive personality ceiling | Uses the E2E branch's `ACCEL_MAX` cruise cap; those OFF-only restrictions do not execute in this function |
| Candidate construction, lines285–291 | MPC and cruise candidates participate | Adds model E2E acceleration and its `shouldStop` candidate |
| Bosch-A engage initialization, lines297–304 | Initializes with the OFF branch of the same cruise law | Initializes with its ON branch |

The chosen acceleration is the minimum of participating candidates. Stop intent
is the OR of participating candidates' stop flags. Therefore toggling the
Boolean can change both acceleration and stopping without changing the radar
decoder or constructing a different Bosch-A post-MPC chain.

The launch helper's actual guard is also important:
`nrdr/features/longitudinal/longitudinal_planner.py`,
`NrdrLongitudinalPlanner.launch_accel`, lines128–155. It returns the original
action unless launch is armed, E2E is active, should-stop is false, and the model
predicts sufficient forward motion; it also requires complete model velocity
and acceleration arrays. ON alone does not command a launch.

This is not an exclusive “radar versus camera” switch. Radar/MPC candidates
remain in ON mode, the same Bosch-A post-MPC processing still runs, and OFF
mode can still use qualified camera/model information for lead processing.
The final donor acceleration clipping also remains after candidate selection.

## Why existing target controls are not equivalent

At target `d0d58d284`, `selfdrive/controls/lib/longitudinal_planner.py:114–119`
always constructs all three candidates: MPC, cruise, and model E2E. There is
no runtime mode test around the E2E candidate. Its cruise law at lines26–31
also has no E2E argument.

A Git search over the baseline Params registry, controls, selfdrived and UI
finds no active `ExperimentalMode` selection or publication. It finds only the
deprecated schema fields and tests asserting the previous mode removal.

Existing controls have different responsibilities:

- Alpha Long selects longitudinal actuation authority; it cannot also express
  the donor's independent ACC-versus-E2E planning choice.
- Full MVL selects the existing MVL planning/control path; turning it off selects
  the imported NRDR Bosch-A path, not one of that path's two planning modes.
- Following personality selects gap/acceleration/jerk profiles. It is independent
  of Experimental Mode in the donor.
- Existing model/navigation selection chooses model execution or navigation
  support. It does not implement the donor's candidate/acceleration policy.

Keeping the target's unconditional E2E candidate would silently make the
donor OFF mode unavailable. Hard-coding false would remove the donor ON mode.
Reusing an unrelated switch would conflate controls the donor keeps separate.

## Exact restored surface in the current port

| Surface | Current implementation | Purpose / scope |
|---|---|---|
| Saved preference | `common/params_keys.h:412`, `ExperimentalMode`, PERSISTENT + BACKUP, BOOL, default `0` | Equivalent missing-value default through the target's normal registry/manager mechanism; not an activation prerequisite |
| Published state | `cereal/log.capnp:826`, `SelfdriveState.experimentalMode @10` | Restores the existing wire ordinal from the deprecated group to the active API; enables the donor's direct field reads |
| Manual settings | `selfdrive/ui/mici/layouts/settings/toggles.py:26–29,102–104`; large layout `:52–57,149–150,176–181,199–203` | Reuses the target's existing Toggles pages; control is visible/enabled only for the imported mode |
| Saved-choice callback | `nrdr/ui/honda_longitudinal.py:108–120` | Rejects non-Boolean and out-of-scope changes; writes the normal saved preference |
| Runtime qualification | `selfdrive/selfdrived/selfdrived.py:127–129,675–678` | Creates the driver-control adapter only for the imported mode and publishes false otherwise |
| State publication | `selfdrive/selfdrived/selfdrived.py:608` | Publishes the qualified runtime value to the planner |
| Wheel-button control | `sunnypilot/selfdrive/car/cruise_helpers.py:17–64` | Exact donor helper: 50-frame distance-button hold toggles once per press |
| Short-press/profile coordination | `nrdr/features/longitudinal/driver_controls.py:37–57` | Retains donor long-hold versus short-release distinction and SLA reservation |
| Event feedback | `custom.capnp`, `experimentalModeSwitched @14`; `sunnypilot/selfdrive/selfdrived/events.py:30–31,192–194` | Reuses the deprecated event ordinal; gives the donor's 1.5-second On/Off alert |
| Latched event arguments | `selfdrive/selfdrived/selfdrived.py:591–594`; `events_base.py:123–136` | Alert uses the button's latched result, not an asynchronously polled stale Params value |

The Econ personality enum at `log.capnp:149` is a separate donor profile
dependency, not a new activation switch. Four choices are exposed only inside
the imported mode; existing modes retain three. The port maps a stale saved
Econ to Relaxed outside the imported scope rather than wrapping to Aggressive.
The additional lead-trajectory schema fields belong to donor stopping/planning
context and are not required merely to display Experimental Mode.

## Whole-path gate, not only UI hiding

`nrdr/features/longitudinal/bosch_a/gate.py:15–27` defines the donor platform
and OP-long condition. `driver_controls.imported_mode` at lines20–21 and
`ui/honda_longitudinal.donor_bosch_a_mode` at lines108–109 both use:

```text
Honda in HONDA_BOSCH_A
AND CP.openpilotLongitudinalControl
AND NOT startup MVL longitudinal
```

The planner uses the same gate at
`selfdrive/controls/lib/longitudinal_planner.py:120–127`. Outside it, the class
delegates to `BaselineLongitudinalPlanner`. Inside it, the Boolean controls the
donor's internal modes; it does not decide whether that class is selected.

This intentionally does not require radar availability: the donor's applicable
radar-independent Bosch-A processing must also work with the radar toggle off.
The gate is based on actual startup CarParams, not just a saved UI preference.

The target's baseline/MVL planner formulas remain selected for stock longitudinal,
MVL on, other platforms and non-Bosch-A Honda. Restoring a registry key and wire
field globally does not by itself activate this mode-selector code for those
modes. This is not a claim that the entire radar integration leaves MVL outputs
unchanged: shared radar decoding/lead processing can supply different lead
inputs to MVL while its planner/control formulas are preserved. On a Bosch-A
transition to stock longitudinal, the saved preference becomes false, matching
the donor's disabled-marker behavior within the imported scope.

## Distance button and speed-limit assistance

Donor `sunnypilot/selfdrive/car/cruise_helpers.py:31–64` requires actual OP-long
and cruise availability before a hold can toggle. It keeps observing releases
while an action is blocked and latches a consumed press until a fresh press.
The current port's helper is byte-identical to the donor622 helper.

Donor `nrdr/hooks/selfdrived.py:139–148` reserves the button while speed-limit
assistance is preActive and for 0.75 seconds afterward. Its profile handling at
lines151–196 prevents one hold from also changing following personality.
The port's small adapter preserves those rules without importing the donor's
unrelated HUD submode/personality synchronization architecture:

- A normal short release cycles the four donor profiles.
- A 50-frame hold toggles Experimental once; its release does not cycle profile.
- An SLA-reserved or cruise-unavailable press remains consumed even if the
  reservation/availability changes before release.
- Alert text uses the toggled value, avoiding an async write/read race.

Outside the imported gate, the existing target short-release three-profile
handling remains in selfdrived. Its behavior is not replaced by the donor helper.

## Explicit exclusions verified in the working tree

- `DynamicExperimentalControl` is not registered or restored. No automatic mode
  switching controller was imported.
- `ExperimentalModeConfirmed` remains retired. No confirmation/acceptance flow
  was restored.
- No donor wide/narrow camera switching rule was restored. The two target
  augmented-road-view source files are unchanged; the target's camera rule
  remains based on its previous longitudinal-capability logic.
- No donor navigation/model selection, full UI architecture, HUD mode selector,
  unrelated Civic PCM experiments or custom Bosch-A master opt-in was added as
  part of this selector.
- Source does not automatically set Experimental Mode true to enable Bosch-A.
  It is optional and its fresh default is false.

## Review findings and validation boundary

No scope escape was found in the inspected Experimental Mode routing: setting
visibility, saved-choice callback, wheel helper construction, runtime state and
planner dispatch agree on the imported-mode gate. No production changes were
made by this review.

Meaningful existing tests cover this surface:

- `nrdr/tests/test_driver_controls_port.py`: short press, one-toggle hold,
  consumed SLA/unavailable press, latched alert value, stock transition, MVL
  exclusion and unrelated-vehicle preference preservation. The recorded focused
  run has 8 passing tests.
- `test_honda_longitudinal_settings.py` and migration tests: false default,
  scoped saved-choice admission, native persistence/backup, retired DEC and
  confirmation keys remaining absent.
- `test_bosch_a_native_parity.py:97–121`: exact donor planner fixture comparison
  across both Experimental values, both radar states and all four profiles;
  also checks excluded paths against the retained target baseline.
- `test_upstream_longitudinal.py:86–99`: the baseline/MVL path still accepts
  model E2E braking irrespective of obsolete-mode-file values.
- `test_insight_eps_ui.py:118–134`: unchanged target camera-choice rule.

These are software and source assertions. They do not establish actual vehicle
response, driver comprehension of the rendered UI, or physical road validation.
The final combined validation receipt should supply the final test totals; this
report does not reuse an older failing intermediate parity XML as a final result.
The parent reports the latest main suite at 440 passed plus 21 subtests and
dependency suite at 581 passed; those totals are integration-wide, not tests
executed by this read-only review.
