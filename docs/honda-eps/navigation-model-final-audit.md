# Navigation-model final audit and MVL consumption boundaries

**Historical source/experiment record.** Claims, branch restrictions, test counts
and outstanding requirements below apply to the stated historical revision.
They are not current operator instructions. See the
[current software compatibility workflow](navigation-model-software-compatibility.md)
for the compiled and registered model, current status and activation boundaries.

The [subsequent software correction](navigation-model-software-status.md) fixes the clock-domain and extra big-model allocation limitations documented in this historical audit. The current compatibility workflow supersedes the retired diagnostic procedures.

Coordinated committed-source results and the authorized target integration are recorded in the [software integration report](navigation-model-integration.md). Focused working-tree results below retain their original scope.

This audit continues the existing implementation. Its initial inspected
experimental tip is `036fd8ca9afb5240fa163c377bcfa2b961a38fe0`; the fetched working
selector/navigation target is
`4851eacac8fce2e875a0ef7703af5f5025fc0c15`. These are audit starting points, not a
claim that final integration or physical validation has already occurred.
The user's newer authorization permits a reviewed, history-preserving integration
into the existing selector branch. Historical Stage 1/2 restrictions describe
their original scope and do not override that authorization.

## Stage 2 claims checked against evidence

The committed [Stage 2 report](navigation-model-stage2.md) accurately describes
an isolated CPU/diagnostic implementation. It does not claim a complete usable
AMD artifact or working navigation-conditioned driving. Those distinctions must
remain clear as the remaining source integration is completed.

| Evidence | What actually ran | What it does not establish |
| --- | --- | --- |
| Typed packing and frame-fusion tests | Production packer/adapter code with real CPU/NPY storage, declared public shapes and inert/mock kernels | Learned inference, USB transfer correctness or physical AMD execution |
| Recurrent reset tests | Real CPU/NPY state buffers and in-place aliases, including explicit native/legacy camera, feature, desire and navigation queues | Learned cold-start quality, convergence or safe vehicle fallback |
| Native route transport tests | Production navd/mapsd/reader functions over real Msgq/VisionIPC transport; synthetic GPS, mocked HTTP response and renderer | Actual GPS/camera clock alignment, live map pixel parity or road routing quality |
| Compiler verification | Actual Bigv2 CPU compilation; two seeds; finite prediction/state outputs; repeatability and independently loaded SPTG agreement | An independent ONNX-runtime oracle, latest official binary equivalence or CPU/GPU parity |
| Recorded-camera sensitivity | Eight real HEVC camera pairs, logged calibration and telemetry, reconstructed neutral desire/Honda timing, explicitly synthetic navigation | Real recorded navigation replay, real exit/reroute scenes or 128-frame steady state |
| Real CPU fallback experiment | One primary Bigv2 frame plus three invalid navigation contexts handled by a separate actual non-navigation CPU model | Physical GPU failure/recovery or safe vehicle transitions |
| Orchestration failures | Mocked primary/fallback initialization, warmup and inference failures exercise the diagnostic's actual orchestration | Naturally observed failures in the learned model or GPU driver |
| 581-test preservation suite | Existing EPS/control, selector, fallback/loader/parser/recovery, Chestnut helper and Params checks | A complete unit suite, physical firmware compatibility, or unchanged behavior after model predictions change |

The two preservation skips remain skips: a parameterized test base and the live
manifest URL-availability check. The original 581-test command did not include
the separate DEC planner gate tests; those belong in final regression selection.

### Source and experiment provenance

All **67** frozen-source hashes from the final Stage 2 targeted run match the
committed implementation `255ea8a94722f37afcb67fa9e7da192de444945a`. The three
later focused-run hashes also match that commit. This is stronger evidence than
the recorded `HEAD=0556a2ea6` field alone: the run occurred on an uncommitted
working tree, and its exact contents were subsequently committed.

Some long experiments used saved source snapshots. The evidence explicitly
records this: numerical compiler verification functions were AST-identical while
report I/O changed; the eight-frame tool snapshot included persistence in its
timing; the later CPU fallback snapshot differed from the final tool by an
AMD-only manifest-proof rejection. Their results retain those limits and must
not be relabeled as final-source physical hardware runs.

The exact Bigv2 source remains ONNX SHA256
`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
The source ONNX and older/later official compiled binaries are separate objects.
Later bounded, non-executing extraction of their leading metadata establishes a
stronger distinction: the a03 Bigv2 binary checkpoint matches the public ONNX,
whereas the latest afbf n16 binary declares a different checkpoint despite the
same tensor shapes/slices. It therefore cannot be described as this ONNX with
only different compiler flags. The v1/overflow-fix pair retains matching metadata;
that still does not disclose every compilation setting. Prefix inspection does
not independently recompute the complete large PKL's hash.

## MVL longitudinal consumption by mode

The existing control path is:

```text
modelV2 → MVL longitudinal planner → longitudinalPlan.aTarget/shouldStop
        → existing LongControl state machine/PID → existing vehicle interface
```

Raw route instructions are not steering or braking commands. The planner polls
`modelV2`; the proposed navigation integration changes model inputs, not the
planner's subscriptions or actuator authority.

The effective direct-model gate is exactly:

```python
experimental_mode and (not dec.active() or dec.mode() == "blended")
```

The planner reads the model action fields every update, but adds them to the
acceleration/stop candidate list only when this gate is true.

| Effective state | Direct `modelV2.action.desiredAcceleration` / `shouldStop` candidate | Other possible prediction effects |
| --- | --- | --- |
| ACC: Experimental mode off | Excluded, even if the model action changes | Radar/vision lead matching and MPC; SCC-V when enabled; model-based alerts. DEC's internal mode is not active for control |
| Experimental mode on, DEC disabled/inactive | Included, regardless of a stale/internal DEC mode label | Same indirect paths; existing radar/cruise candidates remain competing constraints |
| Experimental mode on, DEC active in `acc` | Excluded | Model trajectory can influence DEC's decision to switch to blended; leads/MPC and enabled SCC-V can still influence speed |
| Experimental mode on, DEC active in `blended` | Included | Same indirect paths and competing constraints |
| DEC parameter enabled, Experimental mode off | Excluded; DEC is not active | Same ACC limitations; setting DEC alone does not authorize direct model acceleration |
| `CP.openpilotLongitudinalControl` false | Selfdrived does not expose Experimental mode as active through its current parameter gate, so the direct E2E candidate is not enabled by that toggle | Existing OEM/cruise/alternative longitudinal paths require their own configuration; no claim that a model directly commands the OEM brake actuator |
| Longitudinal control inactive/disengaged | Planner predictions may still exist; existing LongControl off-state/actuator gating applies | No new navigation authority bypasses engagement, driver override or vehicle/safety checks |

An included model acceleration remains a **candidate**, not the unconditional
command. The planner chooses the minimum acceleration among radar/MPC, cruise
and eligible E2E candidates, then clips to its existing acceleration bounds.
The stop request is the `any()` of eligible candidate stop flags. A more positive
model request cannot defeat a lower radar/cruise request; a more negative model
request can change the chosen deceleration within the existing bounds. Stop-state
behavior must be evaluated separately from acceleration magnitude.

Source references on the exact target baseline:

- [`LongitudinalPlannerSP.is_e2e`, lines 39–44](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py#L39).
- [`LongitudinalPlanner.update`, lines 134–155](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/selfdrive/controls/lib/longitudinal_planner.py#L134).
- [`DynamicExperimentalController.update`, lines 374–388](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/selfdrive/controls/lib/dec/dec.py#L374).
- [`selfdrived.params_thread`, line 668](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/selfdrive/selfdrived/selfdrived.py#L668).
- [`controlsd`, lines 126–147](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/selfdrive/controls/controlsd.py#L126), and [`LongControl.update`, lines 54–80](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/selfdrive/controls/lib/longcontrol.py#L54).

The existing four DEC planner-gate tests were independently run during this
audit: **4 passed**. They execute the production planner/native MPC with synthetic
Cereal inputs and a deliberate DEC-mode test double. They validate gate behavior,
not the actual DEC heuristic's learned-model response or vehicle behavior. Final
acceptance tests must be recorded against the final committed/integrated SHA.

### Indirect paths that remain relevant in ACC

1. **Lead/MPC:** `radard` consumes model `velocity.x[0]` and `leadsV3` to build
   radar/vision leads; the longitudinal MPC consumes those leads. The raw plan
   velocity profile is not an unconditional ACC acceleration command.
2. **SCC-V:** when enabled and longitudinal control is enabled, it derives
   predicted lateral acceleration from `abs(orientationRate.z) * velocity.x`,
   computes a curve target and can lower the requested cruise target. Its own
   state machine and driver override remain relevant.
3. **DEC:** its slowdown heuristic uses the 33-point model trajectory endpoint
   against a speed-dependent expected distance and can request blended mode.
   A shorter nav-conditioned path can therefore change eligibility for direct
   model acceleration. This is not proof of correct red-light recognition.
4. **Throttle/coast and physical steering:** the planner uses model gas-press
   probabilities for its existing throttle gate and actual steering angle for
   its non-E2E lateral-acceleration budget. The inspected nav graph does not
   condition every output head, so graph/head reachability must be checked
   rather than assuming all these values necessarily change.

Relevant unchanged files are `radard.py:246–263`,
`smart_cruise_control/vision_controller.py:82–100,163–201`,
`dec/dec.py:242–303,305–388`, and `longitudinal_planner.py:39–51,98–100`.

## Artifact semantics and activation evidence

The official experimental native consumer maps the two action means as
`action[0] / max(1, v_ego)**2` for desired curvature and `action[1]` for desired
acceleration. Bigv2 has two means plus two uncertainty values in flat slice
`2062:2066`; the loop is 20 Hz. The public ONNX includes checkpoint and slice
metadata but no SunnyPilot numeric generation. Neither catalog v28, Selector 21,
nor the later compiled binary's "n16" title justifies inventing that generation.
An explicitly pinned native contract must preserve these semantics and timing.

CPU source/reload agreement and finite output differences do not authorize normal
activation by themselves. A final candidate needs model/warp/runtime identities,
the actual discovered GPU target, required firmware/protocol checks, finite and
tolerance-bounded equivalent-input CPU/GPU evidence, temporal behavior checks and
an explicit default-off enable path. Missing physical evidence must gate that
candidate rather than disable unrelated existing models.

### Completion review findings

The source now separates ordinary Selector 21 choices, a branch-reviewed driving
candidate, and file-only diagnostics. The checked-in Bigv2 driving record has
`hardware_reviewed: false`, `activation_approved: false`, and no approved compiled
hashes. Neither the display switch nor a locally exported diagnostic manifest can
make it eligible. When the driving switch is off, admission returns before opening
the candidate registry or artifacts, and the ordinary runner cache and selected
bundle slots remain in use.

Independent review identified the following concrete issues; the completion work
addresses them in their relevant components:

- A source/reload comparison could correctly accept floating-point `-0.0` versus
  `+0.0`, while manifest export rejected the different byte digests despite zero
  numerical error. Export now permits this finite float distinction under its
  numerical tolerance policy; exact source repeatability remains byte-exact.
- AMD kernel targets alone did not exclude foreign-device internal buffers.
  Experimental admission now restricts captured buffers and inputs to the approved
  AMD/NPY devices before execution.
- A normal driving candidate needed the same real-device checks as its diagnostic:
  exact enumerated Chestnut firmware presence, discovered `gfx1200` USB/LLVM
  target, and a reviewed warp copied extent matching the actual NV12 camera
  geometry. These checks occur before candidate artifact execution.
- On a navigation inference failure, switching to the existing small fallback also
  needs to recompute its longitudinal model delay and refresh its lateral delay
  where the Honda firmware schedule does not supply it. The feature-enabled
  fallback path now performs that refresh; the ordinary feature-off path retains
  its existing behavior.
- A fallback that stops receiving frames during navigation inference would resume
  with stale recurrent/camera history. The navigation session now uses the
  configured small model as its ordinary fallback and executes it on every active
  candidate frame, without publishing those shadow outputs. Model selection slots
  are preserved; enabling this candidate explicitly chooses this small-fallback
  policy instead of continuing an ordinary selected big model during invalid
  navigation.
- Candidate handovers refresh the existing Chestnut status used by selfdrived.
  Intentional small-model fallback clears the active status rather than reporting
  a GPU failure; actual inference failure retains the existing failed status.
  Active candidate inference sets the status needed for the existing big-model
  disappearance monitoring.

These fixes remove implementation errors. They do not replace physical compiler,
numerical parity, sustained timing, or learned handover validation. In particular,
the initial 132-frame candidate history policy is a source policy, not evidence
that its output is ready to drive.

Continuous small-model inference adds a production workload that the isolated
single-model AMD diagnostic cannot measure by itself. An ordinary big model may
also remain allocated after candidate admission. Combined model memory, frame
deadline, power and transition behavior therefore require explicit physical
review before normal activation approval, even if the isolated candidate passes
its CPU/GPU comparison.

### Source-clock and physical replay limits

The producer/reader contract pairs exact map frame ID, source/render timestamps,
route generation and destination identity. Route invalidation, a restarted map
producer, camera gaps, source regressions and permission loss cause explicit state
boundaries. Invalid model data is not relabeled as a trained zero-navigation
fallback. Legacy map display may retain a permissive GPS choice without making
that source eligible for model input.

The existing Python GPS and map publishers use a monotonic clock, while native
camera captures can use a boot-time clock. No unverified offset conversion or
global clock rewrite was added. If those clocks disagree beyond the strict source
or render windows, model input is rejected. Hardware verification must therefore
establish common capture/source clock behavior, including any suspend/resume
condition; passing synthetic-clock tests does not establish that fact.

## Preservation invariants and final acceptance

At the initial current-source audit, the Honda/MVL control trees, hardware and
firmware code, updater trees, production model assets, dependency locks and launch
configuration match the exact target baseline. All seven submodule pins match,
including tinygrad `69fd4d34`, Panda `74a0adce`, opendbc `cf461174` and msgq
`e7396e76`. Existing production selector/catalog source also matches at that
snapshot; later branch-local selector extensions must be reviewed separately
for default-off behavior, ordinary model availability and settings persistence.

The Stage 2 UI remaining-distance/time regression was already fixed: a maneuver
loop no longer overwrites the active step. The focused fixture retains 175 m /
17.5 s remaining and its future maneuver list. Stage 2 clock rejection is an
explicit limitation, not proof of post-suspend hardware synchronization.

The current completion work must be judged by its final code and evidence:

- Test actual production selection/admission/fallback functions, with mocks
  clearly limited to external devices and controlled failures.
- Keep normal updates, selected models/settings and automatic firmware behavior
  unchanged by installing the new source with all feature gates off.
- Keep real transport tests distinct from renderer/network/physical-device tests.
- Bind source/compiled/warp/checkpoint identities and exact test commands to the
  final committed source; retest the history-preserving integrated result.
- Report pending physical results and candidate-specific gates explicitly.

This audit does not certify learned turn preparation, stopping, reroute behavior
or vehicle safety. Existing controllers can react differently to changed model
predictions even when every controller source file remains identical.
