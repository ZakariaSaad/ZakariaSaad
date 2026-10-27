# sp-steer controller integration

The migration pins the working source at `18106a1ddf26366236961b28509e6fa677bec6ef`
and MVL sp-steer at `0f5085fc3f713da58d236ed3822b9d27bc0e9917`.
This is an interface adaptation, not a new firmware image or a new Insight control law.

## Steering paths and mode boundaries

`HondaEpsFirmwareEnabled` remains an owner request. `prepare_drive_config()`
admits the supported Insight and installed-image attestation, then writes the
shared drive snapshot before publishing CarParams. Card, controls, both model
runners, torque learning and delay learning consume that same snapshot. Changing
the request does not change controller mode during a drive or process restart.

| Stage | Custom Insight controller ON, admitted | Custom controller OFF |
|---|---|---|
| Vehicle configuration | sp-steer's native Insight torque CarParams and linear 4096 transport, verified from the current CarControllerParams and EPS lookup | Upstream native Insight torque configuration |
| Model prediction | Existing firmware prediction-delay schedule, for ordinary and corrected navigation models | Ordinary upstream delay path |
| Curvature and control | Existing firmware geometry (`mvl` or `james`), response (`source` or `route32`), feedforward/PID, integral history and driver-delivery law | sp-steer/SunnyPilot torque controller, including applicable owner torque/NNLC settings |
| Command units | Existing 3840-domain complete output converted exactly once by 3840/4096; unchanged 3840-count ceiling | Ordinary normalized upstream request |
| Honda downstream controller | `FirmwareTransport`: no additional feedforward or shape; original +/-0.03 normalized final rate limit each 10 ms; actual delivered torque reported back | Original `HondaLateralModel.update`, learned response/inverse shape, acceleration-space limiter and reporting |
| Driver input | Existing strict `abs(torque) > 2400` raw threshold plus existing opposing-press/override logic; request disabled on override | Upstream `HondaSteerThresholdLearner` and effective threshold |
| CAN and safety | Native Honda packing and Panda safety remain in force | Same native path |

The Insight is **not** listed in upstream `EPS_MOD_TORQUE_LOOKUP`; its transport
is linear 4096. No Civic/Clarity lookup is borrowed. The admission no longer
reads removed `CarParams.lateralParams` or requires the old native PID tuning.
The firmware controller itself remains the same PID/feedforward algorithm and
still publishes the PID diagnostic union. Controlsd dispatches that union using
the actual controller, and never invokes ordinary live-torque updates on it.

Custom mode is selected before constructing inactive ordinary torque/NNLC
controllers. Unused ordinary model files and settings cannot prevent custom
controller initialization. Ordinary lateral choices are preserved, including
conflicting NNLC/LJTC selections that their ordinary setup will evaluate when
that mode is actually used.

The custom mode does not update the ordinary Honda plant model or adaptive
steering threshold. It also does not load, delete or overwrite ordinary
`LiveTorqueParameters`, `LiveDelay` or `LagdValueCache`. Existing estimator
heartbeat messages continue; no new services or per-tick parameter reads were
added. A subsequent ordinary drive reconstructs its upstream models from its
own persisted state. The shared mass-normalized upstream VehicleModel retains
the same bicycle-model equations; the optional James rack geometry remains
controller-local.

The firmware calibration, `core.py`, `latcontrol_fw.py`, `geometry.py`,
`response.py` and `timing.py` are unchanged from the pinned working branch.

## Longitudinal reconciliation

sp-steer removes ACC/Experimental/DEC selection and always includes the model's
acceleration candidate. This migration deliberately retains the working fork's
ACC/Experimental/DEC selection, throttle-probability/coasting limits, turn
acceleration limits, stop combination, gap-button mode switching, and related
UI/events. The old planner's braking/acceleration constants and longitudinal
MPC/LongControl algorithms remain.

Two upstream planner improvements are adopted deliberately:

- Stock longitudinal operation initializes MPC from measured ego speed and
  acceleration instead of a simulated openpilot acceleration target.
- Upstream `AccelBoost` applies only when the existing E2E gate admits the model
  candidate. A qualifying driver accelerator override can add up to 0.05 m/s²
  per override and 0.2 m/s² total; it scales out during braking/low speed and
  resets on disengagement or leaving E2E. It cannot bypass MPC/cruise minima or
  stop requests. This is a documented behavior change, not bit-for-bit planner
  preservation.

Upstream's configurable **Initial Cruise Speed** replaces the old hardcoded
120 km/h Experimental-mode initial setpoint. The new default is 40 km/h; initial
set speed is the larger of current speed and that setting, bounded by cruise
limits. Existing resume-setpoint behavior and stock PCM cruise remain intact.

The newer Honda dependency retains the Bosch low-speed brake PID, stopping
release/ramp and gas learning for the Insight, while adding upstream Nidec
learning/governor/launch improvements for the platforms they apply to. A concrete
inherited Bosch indentation regression was fixed: when below the cruise setpoint,
inactive/override/zero-error frames could reference undefined `gas_error` or
`wind_adjust`. The correction restores the original observation guards around
learning, retains the upstream set-speed guard, and changes no safety limits.
Its dependency regression test invokes the real CarInterface/CarController for
seven active/inactive/override/zero-speed/error cases; generated CAN is not sent.

## Verification and limits

The focused EPS regression suite exercises the native schemas, both geometry and
response options, source-domain golden outputs, finite-value rejection, drive
latching, model timing branches, real controlsd update/publication, actual Honda
CAN packing, opposing driver override and new transport/learner isolation. The
transport sequence is compared with the original final rate limiter and proves
that the source→transport conversion and 3840-count bound are applied once.
The same native constructors also verify the ordinary upstream controller mode.

The inherited generic lateral saturation test assumed the Civic still used PID.
It now tests the actual upstream Civic torque path and retains separate PID
coverage with an explicit PID fixture, rather than reading the inactive tuning
union. This changes a stale test assumption, not vehicle tuning.

Longitudinal simulation retains both ACC and E2E coverage, pitch/throttle
scenarios, stopping and starting maneuvers, following-distance tests, and the
DEC gate assertions. Native Honda safety tests exercise the new dependency's
unchanged safety implementation on CPU. See the migration report for final
counts and any inherited failures. These are software tests with synthetic
inputs; no CAN transmission, vehicle access, GPU inference or on-road safety
claim is involved.
