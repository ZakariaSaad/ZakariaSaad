# Optional Honda Bosch-A integration

This testing branch integrates the Bosch-A object-bank decoder and selected NRDR
lead/planning components with the existing MVL planner. `BoschALongitudinalEnabled`
defaults to **OFF** and is read at drive startup. This work does not enable a
vehicle, change firmware, or establish closed-loop vehicle performance.

## Source provenance

| Component | Pinned source |
|---|---|
| Protected main baseline | `d0d58d2844af0cf6f003c965a8b397723e2fc610` |
| Protected dependency baseline | `84ce49fcd02904f324eff9252ae6c1c10b9ffb4b` |
| Imported main checkpoint | `a0b0a4c32d46d168dc7123d9cf9e801764315bb8` |
| Imported dependency checkpoint | `40927b6f99e787fe9ce1fb8e76bd0e2e06aeceb9` |
| NRDR openpilot PR27 head | `fd40efb737251d2734e2d62837e37f47235b9df8` |
| NRDR opendbc PR4 head | `427f1986ed241f111d106cc72cdefbda5efe5f7f` |
| Earlier radar PR3 head | `461f81cddfa96be4fbf2b9c01e821790f8110f27` |

The main and dependency testing branches are respectively
`insight-bosch-a-long-integration` and `insight-bosch-a-long-opendbc`, in the same
ZakariaSaad/ZakariaSaad repository. The parent gitlink is authoritative; initialize
submodules without `--remote`. The donor and protected baseline MPC generators
both use **six parameters**. Their `gen_long_model` and `gen_long_ocp` ASTs are
identical. An earlier transfer document's eight-parameter claim was incorrect.

## Startup and data flow

The dependency sets the non-safety `HondaFlagsSP.BOSCH_A_LONG` flag only for a
supported platform with explicit opt-in and openpilot longitudinal control. Stock
longitudinal, passive, dashcam, notCar and unsupported radar configurations are
excluded. The main eligibility gate also requires radar availability for the
radar-dependent planner/policy. Final passive eligibility must be resolved before
constructing the optional normal-path radar interface and publishing CarParams.
Explicitly injected replay interfaces remain caller-owned.

The decoder consumes the existing camera-bus object bank: 16 slots and 80 frame
IDs, with nominal 14.35 Hz sweeps. RadarD uses that nominal interval for its
Bosch-A track filter. The decoder, DBC, lifecycle constants and geometry have not
been retuned by these repairs.

`card` publishes point status in `carStateSP`, referencing the exact `radarTracks`
publication timestamp. RadarD accepts status only from a valid companion whose
reference equals the held tracks timestamp. A nearby sweep with a reused track
ID is insufficient. Missing optional status never substitutes for valid tracks.
Existing radar errors and core input validity continue to propagate.

The two model leads have independent vision-acceleration histories. The FCW
carrier requires a present radar lead, the strict vision/track match, filtered
model probability strictly above 0.9, valid core inputs, and no decoder errors.
That carrier is authorization for the existing crash-count path, not a warning
or evidence of physical braking by itself.

## Planner composition and deliberate donor differences

The enabled path qualifies recorded/model lead horizons using clock, receipt,
health, frame and radar/model pairing checks. HumanFollowing supplies only
qualified paths; otherwise the existing radar extrapolation is used. A newer
carState publication than a delayed model is not alone a reason for rejection.

The planner combines:

1. Qualified lead paths and HumanAcceleration cap into the native MPC.
2. Existing MPC, cruise and model candidates, including ordinary and Bigv2 model
   output and the existing predictive-acceleration-boost lifecycle.
3. The minimum acceleration candidate and the OR of candidate stop requests.
4. `bosch_a_long`, `coast_first`, `release_coast`, `smooth`, `onset_ramp`, and
   `stop_gap` in that order.
5. Preservation of the selected negative acceleration and incoming stop intent,
   followed by the existing acceleration limits and optional cap.

Comfort stages may add braking but cannot raise an already-selected negative
request. This also covers braking with no stop flag; a stop-only clamp did not.
Positive boost is not applied to an existing negative model request or model
stop request, while boost update/reset behavior continues.

The donor MVL early return is not copied because it would bypass this chain.
Obsolete ACC/Experimental/DEC selection and unrelated launch/personality tuning
are not imported. Malformed optional MPC caps/paths fall back to the baseline
envelope/extrapolation rather than crashing.

The inherited OCP constrains acceleration at running stages, not terminal stage
N. A terminal predicted acceleration can exceed the optional cap; the final
planner target cap remains necessary. No generator dimensions or constraints
were changed to conceal this limitation.

## Command-level boundary

The original Insight Honda controller, gas/wind learning, grade compensation,
low-speed brake integrator, actuator delay, CAN packing and Panda safety are
retained. Civic-fitted gas floors, brake relief and a second gas learner are not
imported. Existing EPS/lateral, model/navigation/fallback, lane-change, Chestnut,
SunnyLink and updater source paths are preserved.

Planner braking preservation does **not** guarantee an immediately nonpositive
Honda CAN acceleration request. The existing stopping ramp applies:

```python
targetaccel = max(targetaccel, self.last_applied_brake - 0.002)
```

With a positive prior internal value, negative incoming LongControl requests can
therefore coexist briefly with positive internal and encoded requests. Offline
controller tests reproduce this with the feature OFF and ON, and cover restart
state-field lag, cancellation, disabled authority and stock selection. They
characterize existing behavior; they do not claim positive physical force or a
stopping-distance effect. A stricter command-level invariant would require a
separately defined stopping/restart policy and corresponding controller change.

`ACCEL_COMMAND` is a request in m/s². `GAS_COMMAND` is in OEM raw units, including
the inactive sentinel -30000; compensation can make the two fields differ in
sign. Protocol STANDSTILL is not measured wheel standstill. Panda returned USB
metadata is not independent wire acknowledgment or actuator application.

## Reproducible build and focused tests

Build in the testing checkout using its pinned lockfile and submodules. Do not
copy another checkout's virtualenv or native libraries.

```bash
git submodule update --init --recursive
uv sync --frozen --extra testing --extra tools
export PYTHONPATH=.:tinygrad_repo:opendbc_repo:msgq_repo:rednose_repo
export PATH="$PWD/.venv/bin:$PATH"
export DEV=CPU:LLVM
scons -j4 \
  openpilot/common/libparams_c.so \
  openpilot/cereal/libcereal.a openpilot/cereal/services.h \
  msgq_repo/msgq/ipc_pyx.so msgq_repo/msgq/visionipc/visionipc_pyx.so \
  openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code/acados_ocp_solver_pyx.so

python -m pytest -q \
  openpilot/sunnypilot/selfdrive/controls/lib/bosch_a/tests \
  openpilot/selfdrive/controls/tests/test_bosch_a_*.py \
  openpilot/selfdrive/controls/tests/test_upstream_longitudinal.py \
  openpilot/selfdrive/controls/tests/test_predictive_accel_boost.py \
  openpilot/selfdrive/controls/tests/test_longcontrol.py \
  opendbc_repo/opendbc/car/honda/tests/test_bosch_wind_learning.py \
  opendbc_repo/opendbc/car/honda/tests/test_bosch_a_integration.py \
  opendbc_repo/opendbc/car/honda/tests/test_bosch_a_decoder.py \
  opendbc_repo/opendbc/car/honda/tests/test_bosch_a_radar_interface.py
```

Verify `openpilot`, `opendbc`, cereal, Params, msgq and native MPC import origins
before interpreting results. Parser/packer tests use the actual Python backend.
Planner/MPC tests use the actual native solver; observers record raw solve
returns before inherited reset can clear the public status field. Synthetic
clocks, SubMaster fixtures and publication capture replace external I/O, not
numerical planning. Controller tests construct CAN bytes offline and replace
only asynchronous Params persistence.

Private recorded-input validation is kept outside Git. It advances state from
route start to the selected windows, uses the preserved final decoder snapshots,
and compares feature-OFF/stock results against a separate protected-baseline
process on identical inputs. Unrecoverable receiver clocks/health must be marked
as assumptions; a qualified-clock replay is not a reconstruction of live process
scheduling. No inference or closed-loop vehicle experiment is implied.

## Remaining evidence limits

Insight's donor-inherited range origin is not independently calibrated. Model
distance and video are useful diagnostics but are not independent range truth.
The known saturation-entry sentinel suppression remains characterized without
globally accepting invalid sentinel values. Missing full-rate tail segments do
not acquire full-rate coverage from sampled logs. Command delivery/application,
physical effects, and independent range/velocity/grade calibration require their
own targeted evidence. The optional feature remains OFF by default.
