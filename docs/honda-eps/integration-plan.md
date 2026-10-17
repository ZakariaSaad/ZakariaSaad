# Honda Insight firmware controller: dependency and integration plan

This plan was completed before implementation against MVL openpilot
`cb15c04e918b759688cb354b1bbc9b3c5a47a65c` and its existing opendbc pointer
`cf46117420f283b3e38241c00d728a191aa9fccc`.

## Dependencies and boundaries

Use NRDR PRs #21 and #22 as the SunnyPilot integration reference and JamesL787's
Insight calibration commit `f3a6a124bdad3151f875f4f29c8d7396c7617702` as the behavior
reference. Port the firmware-inversion core, residual-angle PID, exact Insight
calibration and measured column load, driver-override logic, command-delay and
prediction schedules, and geometry helpers. Preserve the algorithms and compare
their outputs; an import passing does not establish equivalence.

Keep the port in a separate SunnyPilot lateral-control package. Add small hooks
to interface setup, lateral-controller construction, and model timing. Preserve
MVL's longitudinal planner/MPC, red-light stopping, LongControl, Honda braking and
acceleration adaptation, stop-and-go, MADS, lane centering, lane changes, driver
monitoring, model loading/inference, Chestnut support, and panda safety. Do not
copy NRDR's longitudinal implementation or replace MVL control files.

## Admission and selection

The Insight uses native PID tuning and a 4096-count identity steering map. No
opendbc modification is needed for this platform's transport or firmware
controller. Honda's generic comma-based modified-EPS detection belongs to
`CarParamsSP.flags` (`HondaFlagsSP.EPS_MODIFIED = 2`), not `CarParams.flags`, where
the same bit means `BOSCH_ALT_BRAKE`.

Require the exact Honda Insight platform, compatible modified TXM-A040 EPS family,
correct PID tuning and steering map, valid calibration, explicit optional-controller
selection, and explicit confirmation of the corrected supplied firmware revision.
Both TXM firmware revisions report the same EPS identifier, so a part-number match
cannot establish the installed revision. Default the new controller to disabled.

Select before `setup_interfaces` converts PID tuning to torque for
`EnforceTorqueControl` or NNLC. Give an admitted firmware controller priority over
`ControlsExt.initialize_lateral_control`, which otherwise substitutes SunnyPilot's
torque V0 controller. Keep the original MVL path for disabled or unsupported
selections. Never switch the controller while steering is active.

Publish an explicit enabled or disabled immutable selection snapshot for the drive,
including the admitted profile, geometry, and calibration identity. Register it as
a JSON parameter cleared on manager start and onroad transition, without backup.
Publish it synchronously before `card` publishes `CarParams` and `CarParamsSP` so
their blocking reads provide an initialization barrier. Validate snapshot identity
against the actual car parameters; final byte digests, if used, must be recorded
after `card` finishes its MADS, alternative-experience, and passive-state changes.
Reuse a valid same-drive selection during process restarts. Missing, malformed, or
mismatched enabled configuration must prevent firmware-controller engagement.

## Physical command and saturation

MVL's chain for the Insight is:

1. Firmware lateral output `u` uses normalized steering units.
2. Honda CarController rate-limits `u` by 0.03 per 10 ms.
3. It computes `int(interp(-u * 4096, identity_BP, identity_V))`.
4. `create_steering_control` sends that signed value in the 0xE4 torque field,
   or zero when lateral control is inactive.
5. Bosch A sends steering on radar bus 0 with stock longitudinal, or powertrain
   bus 1 when openpilot longitudinal is enabled.

Keep 4096 in MVL's interface and the corrected SunnyPilot Insight calibration.
Compare JamesL787's 3840-unit implementation using equivalent physical E4 commands,
not equal normalized output values. Confirm the supplied image's command/P tables,
R6 damping and physical key clamp against PR #22.

CarController returns the rate-limited normalized transport command as actuator
feedback. It cannot report internal EPS saturation. The calibrated physical
envelope and controller anti-windup must therefore recognize that limit directly.
Honda panda safety already rejects nonzero E4 while lateral control is disallowed;
preserve its existing allowlists, engagement checks, and forwarding unchanged.

## Timing and geometry

Read the same drive snapshot once in controlsd and both model entry points:
`selfdrive/modeld/modeld.py` and `sunnypilot/modeld_v2/modeld.py`. The alternate
runner serves selected AI models and supports both `action_t` and older
`lateral_control_params` inputs. Apply the firmware prediction schedule consistently
to all lateral prediction inputs while preserving each model's existing smoothing
and frame/action offsets. Keep command-delay compensation separate from prediction
timing and do not add either twice. Leave longitudinal delay unchanged.

Preserve the existing model timing exactly when firmware mode is disabled. In
particular, stock modeld currently writes its SP delay to `model.lat_delay` while
using raw `lateralDelay` for prediction; this port must not silently change the
disabled path. Alternate modeld must retain its periodic PlanPlus/camera updates
and pass the selected delay to both old-model and action-time inputs.

Evaluate the James yaw-fitted Insight rack as an optional firmware-controller-local
geometry mode. NRDR instead uses configurable steering-ratio geometry. Do not
change MVL's shared VehicleModel or live parameter estimation to make the rack
mode work. Document desired-angle differences over speeds and curvatures and
verify sign conventions, offsets, roll compensation, and conversions.

## Verification

Test selection/fallback, calibration constants and supplied-image correspondence,
feedforward inversion, physical E4 scaling and saturation, geometry, override and
integral decay, reset/disengagement, delay interpolation, and consistency across
both model runners. Include nonfinite inputs and low/medium/high-speed regimes.
Compare actual command sequences with NRDR and James implementations. Run available
unit, real-import, schema, lint, and build checks; record unavailable checks
explicitly. Use recorded-route replay where available and distinguish offline
verification from controlled vehicle validation. Do not flash firmware, deploy to
the comma device, merge the development branch, or claim road validation.
