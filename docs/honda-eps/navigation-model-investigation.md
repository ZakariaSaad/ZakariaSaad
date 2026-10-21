# Navigation-aware model integration: technical investigation and offline testing boundary

Inspected October 21, 2026 (America/New_York). Source and PR states below are pinned snapshots, not promises about future upstream revisions.

**Navigation Alpha is a working display/routing stack, but it is not navigation-conditioned driving on the current working branch.** PR #2057 adds much of the model-input plumbing; it still omits the maneuver publisher and has confirmed freshness, admission and robustness gaps. A more complete official comma.ai experimental branch exists. Both requested public big models are real navigation-input models, byte-identical to artifacts on that official branch. None of these facts establishes learned driving quality, AMD/QCOM execution parity, or vehicle readiness.

The original request authorized research only. You later authorized a **separate experimental branch**, then clarified that its base must be the existing selector/Navigation Alpha branch. That corrected base is `4851eacac8fce2e875a0ef7703af5f5025fc0c15`. The experimental work is confined to `mvl-honda-eps-nav-model-testing`; the existing `mvl-honda-eps-selector21-chestnut-v28`, `mvl-honda-eps`, `master`, `mvl-cinque-v3` and other existing branches must remain unchanged. No device deployment, EPS flashing, Chestnut flashing, live control or model artifact replacement is part of this work.

This report distinguishes **confirmed by source**, **reproduced with isolated CPU/mocked tests**, and **not yet established by meaningful recorded-route/differential inference or hardware**. Experimental commits and automated results are recorded below. Final branch-tip publication and remote-ref verification are included in the accompanying completion record.

## Source snapshots

| Component | Immutable revision |
|---|---|
| Existing working selector/Navigation Alpha branch | [`4851eacac8fce2e875a0ef7703af5f5025fc0c15`](https://github.com/ZakariaSaad/ZakariaSaad/commit/4851eacac8fce2e875a0ef7703af5f5025fc0c15) |
| SunnyPilot PR #2057 head | [`f7f9a3efae3a6552068201bd9e4e4dbe66aae577`](https://github.com/sunnypilot/sunnypilot/commit/f7f9a3efae3a6552068201bd9e4e4dbe66aae577) |
| PR API comparison base | `922513f3128b00611d695ce465989a9bc8af9d59` |
| SunnyPilot current master inspected | [`1aa3a430fbe0bcf95f9b3a9dd6edfe34b92d0d63`](https://github.com/sunnypilot/sunnypilot/commit/1aa3a430fbe0bcf95f9b3a9dd6edfe34b92d0d63) |
| comma.ai current master inspected | [`815a90f60603989150ccd0909b5bd2bdeb57b7ca`](https://github.com/commaai/openpilot/commit/815a90f60603989150ccd0909b5bd2bdeb57b7ca) |
| Official experimental `nav-model-integration` | [`afbf2e547e9319e2eb9f515e1e4330af63705cec`](https://github.com/commaai/openpilot/commit/afbf2e547e9319e2eb9f515e1e4330af63705cec) |
| Navigation Alpha introduction, PR #39066 | [`4c362ca3272c7418a068d43c402c84bd0544f35a`](https://github.com/commaai/openpilot/commit/4c362ca3272c7418a068d43c402c84bd0544f35a) |

## A. PR #2057 status and why it is not merge-ready

[SunnyPilot PR #2057, “modeld_v2: prep for nav models”](https://github.com/sunnypilot/sunnypilot/pull/2057) is **open, draft, unmerged**, with `mergeable=false` and `mergeable_state=dirty` at the inspected head. The API reports **44 commits and six net changed files**. The title “prep” and an empty PR description do not establish a complete feature or an author's private reason for keeping it in draft.

### Actual CI results

The head's [test run 39199917587](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587) reports:

| Job | Result | Evidence |
|---|---|---|
| Release build | Passed | [117890552655](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890552655) |
| macOS build | Passed | [117890552788](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890552788) |
| Unit tests | Passed: 1492 passed, 50 skipped, 1 xfailed | [117890552831](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890552831) |
| Process replay | Passed | [117890552821](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890552821) |
| UI report | Passed | [117890552799](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890552799) |
| Static analysis | Failed | [117890552760](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890552760) |
| Simulator driving | Skipped | [117890553745](https://github.com/sunnypilot/sunnypilot/actions/runs/39199917587/job/117890553745) |

The [replay bot comment](https://github.com/sunnypilot/sunnypilot/pull/2057#issuecomment-6199456642) says **0 changed, 66 passed, 0 errors**. Auxiliary review/comment jobs pass; label/sync jobs are skipped. These successes are real but do not demonstrate that a nav-capable model received a nonzero maneuver vector or followed a route. The net change does not add dedicated inference/navigation integration tests.

The complete static-analysis job log identifies **one Ruff F811 error**, duplicate `lower_and_compile` imports in `openpilot/selfdrive/modeld/modeld.py` at lines 12/14. The same run shows the remaining static checks, including ty and codespell, passing. Local Ruff independently reproduces that exact error; removing the duplicate in an isolated copy clears it. This is a small merge-hygiene defect, separate from the functional gaps below.

### Exact merge conflicts reproduced without touching the working branch

Git's merge-tree operation in an isolated repository reproduced:

- Against PR API base `922513f3`: **one conflict**, `.github/workflows/lfs-maintenance.yaml`.
- Against inspected SunnyPilot master `1aa3a430`: **the same one conflict**.
- A wholesale merge into the custom working commit `4851eacac`: **19 conflicted paths**, including an opendbc submodule conflict, both model pipeline files, navigation/API/UI files and updater/configuration files.

The 19 custom-base paths are `.github/workflows/test_models.yml`, `.gitmodules`, `launch_chffrplus.sh`, `opendbc_repo`, stock `modeld.py`, navigation `helpers.py`, `navd.py`, `tiles.py`, mici Developer settings, on-road `navigation.py`, `torque_bar.py`, mici `widgets/button.py`, `ui_state.py`, modeld_v2 `frame_fusion.py`, `model_adapters.py`, `modeld.py`, Athena `athenad.py` and its tests, and `system/updated/updated.py`.

The upstream PR's current merge conflict is an unrelated LFS workflow issue. The 19 custom-base conflicts demonstrate why a full merge is inappropriate for this branch; they are not 19 independent navigation requirements.

### Reviews and discussions

Five review submissions by `Discountchubbs` are COMMENTED, not approvals or requests for changes. All **five review threads remain unresolved and not outdated**; there are no resolved threads in the captured data. The comments concern:

1. `(shape, dtype, device (QCOM/AMD))` tuple semantics — [thread](https://github.com/sunnypilot/sunnypilot/pull/2057#discussion_r4286771253).
2. Using graph dtype so uint8 nav inputs do not get float32 offsets — [thread](https://github.com/sunnypilot/sunnypilot/pull/2057#discussion_r4286778105).
3. The helper being unchanged official experimental code, expected eventually to become a no-op diff — [thread](https://github.com/sunnypilot/sunnypilot/pull/2057#discussion_r4286788989).
4. Allowing uint8 adapter views — [thread](https://github.com/sunnypilot/sunnypilot/pull/2057#discussion_r4286791544).
5. Continuing to watch the upstream development branch — [thread](https://github.com/sunnypilot/sunnypilot/pull/2057#discussion_r4286795121).

These are useful source-intent comments. They do not record that trained behavior, firmware, compiler parity or on-road safety has been approved. There is no public developer discussion conclusively explaining every reason the PR remains unmerged. The observable draft state, conflict and failed check are confirmed merge blockers; the functional omissions below independently make a blind merge unsuitable.

### Six net changed files and meaningful commits

| File | Actual remaining scope |
|---|---|
| `.github/workflows/lfs-maintenance.yaml` | +77 lines of unrelated LFS maintenance; causes the upstream merge conflict |
| `openpilot/selfdrive/modeld/modeld.py` | Duplicate-import-only change; no stock-runner nav implementation |
| `openpilot/selfdrive/navigation/helpers.py` | Mapbox→Valhalla mapping and 47-value maneuver vector |
| `openpilot/sunnypilot/modeld_v2/frame_fusion.py` | Typed, 128-byte-aligned packed layout and fused upload mapping |
| `openpilot/sunnypilot/modeld_v2/model_adapters.py` | Actual graph dtype in native input views |
| `openpilot/sunnypilot/modeld_v2/modeld.py` | Capability detection, map IPC input and instruction preparation |

The feature is primarily [`2a84e148`](https://github.com/sunnypilot/sunnypilot/commit/2a84e1481c8ce347bd69e6a3eca1714fb2a584af). [`84fda467`](https://github.com/sunnypilot/sunnypilot/commit/84fda467b5552ab08de9fffb987628c178d8c6bb) corrects native camera-key handling already present in the customized base. Subsequent merges include broad runtime/master work. [`f7f9a3e`](https://github.com/sunnypilot/sunnypilot/commit/f7f9a3efae3a6552068201bd9e4e4dbe66aae577) is the LFS workflow change. Appendix 1 lists all 44 reported commits so that history scope is reviewable without equating the entire history with a necessary nav backport.

## B. What comma.ai actually implemented

### Current master: Navigation Alpha, without direct navigation inference

The inspected current [master modeld](https://github.com/commaai/openpilot/blob/815a90f60603989150ccd0909b5bd2bdeb57b7ca/openpilot/selfdrive/modeld/modeld.py#L169-L206) allocates generic metadata-declared inputs but populates camera transforms, desire, traffic convention and action timing only. Its [subscriptions](https://github.com/commaai/openpilot/blob/815a90f60603989150ccd0909b5bd2bdeb57b7ca/openpilot/selfdrive/modeld/modeld.py#L301-L305) exclude `navInstruction`; it has no mapsd VisionIPC client. Generic zero-initialized buffers do not constitute a navigation feed, particularly when that packing still assumes float32 auxiliary inputs.

The current default small ONNX was inspected read-only. Its SHA256 `65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce` and 60,918,562-byte length exactly match the official master LFS pointer. Its seven inputs are image/desire/traffic/action timing and three state queues; **no `nav_map` or `navigation` input exists**. The main output is float32 `[1,2576]`. This says nothing about every possible model artifact, but rules out an implicit nav feed into that default graph.

[Navigation Alpha's navd](https://github.com/commaai/openpilot/blob/815a90f60603989150ccd0909b5bd2bdeb57b7ca/openpilot/selfdrive/navigation/navd.py#L143-L169) publishes the UI banner, maneuver distance and ETA but not `allManeuvers`. Alpha's presence on master therefore does not establish re-enabled navigation-conditioned driving.

### More complete official experimental branch

The public `commaai/openpilot:nav-model-integration` branch at `afbf2e54` is **five commits ahead and 44 behind** inspected master. Its common ancestor is Alpha introduction `4c362ca3`. It contains a model feed, a maneuver publisher, typed packed buffers, render-pose extrapolation and big model artifacts.

| Commit | Meaningful scope |
|---|---|
| [`a52113eea9a81c79c9309dbc379f0ce25ffa943d`](https://github.com/commaai/openpilot/commit/a52113eea9a81c79c9309dbc379f0ce25ffa943d), “nav driving model” | Introduces v1 big ONNX/PKL pointers |
| [`8b8ce13b407ddae14f2e415afad9c16f603bdb81`](https://github.com/commaai/openpilot/commit/8b8ce13b407ddae14f2e415afad9c16f603bdb81), “modeld wired” | Adds direct model inputs, maneuver helper/publisher, typed packing and map pose extrapolation |
| [`3ec09aed2ea589cbd3234137ae6fb74aecde3c4c`](https://github.com/commaai/openpilot/commit/3ec09aed2ea589cbd3234137ae6fb74aecde3c4c), “fix overflow inputs in map model” | Changes only the compiled PKL pointer; no inspectable Python/compiler fix in that commit |
| [`a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9`](https://github.com/commaai/openpilot/commit/a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9), “rl nav driving model (80eedcdd/12864)” | Introduces v2 ONNX and updated PKL |
| [`afbf2e547e9319e2eb9f515e1e4330af63705cec`](https://github.com/commaai/openpilot/commit/afbf2e547e9319e2eb9f515e1e4330af63705cec), “n16 nav-wm model” | Updates only the PKL pointer again |

The latest official compiled PKL has SHA256 `52205044e136efb64a6a841c9b016cf35b1db81186dd0f941b62eee5ebdcde24`, length 822,465,506. It was **not downloaded or executed** in this investigation. The artifact-only overflow/n16 changes mean the ONNX graph and a particular compiled binary cannot be assumed to behave identically without compiler/runtime parity checks.

That official branch pins tinygrad `d3f09c9bbd542fbfbe68c8569a1173550c408969`; your base pins `69fd4d34d4430036378aba2b43f9d339cdd6ea73`, and inspected current master pins `02e698160eed1751738212144a7ea2cfb160b3f1`. Its other submodule revisions also differ. A wholesale branch merge or binary pickle transplant would therefore be inappropriate.

### Current direct-input architecture

The official [mapsd](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/navigation/mapsd.py#L47-L80) publishes 256×256 uint8 grayscale over VisionIPC stream 3 at 20 Hz. The [renderer](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/navigation/renderer.py#L1-L39) is styled after its training counterpart: roughly 1 metre/pixel, ego at `(128,192)`, white roads 255, grey route 128, black background, and route-relative heading. Public source expresses training intent; it does not publish complete training data or prove pixel-for-pixel parity.

The model reader consumes raw grayscale pixels, not the UI's green/grey recoloring. It performs no host normalization, resize or transpose; per-graph views supply the declared rank/dtype. Typed 128-byte-aligned input packing combines those bytes with camera data into one upload. Official mapsd additionally projects ahead by `max(gps.speed,0) × GPS_receipt_age`; that change is absent from basic Alpha and PR #2057 and should be independently evaluated rather than copied as an unexplained timing change.

### Maneuver tensor: 47 values, not historical 150-value instructions

The official [helper](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/navigation/helpers.py#L15-L43) and PR helper are identical. They emit float32 `(47,)`: 46 Valhalla command slots and a final distance divided by 100. Distance is measured in 100-metre units, **not clipped to 0–1**. Model input `(1,47)` is accommodated by broadcasting when declared.

Only the first eligible future maneuver is encoded. Depart/arrive IDs 1–6 are ignored; merge is omitted; common commands are straight 8, slight/right/sharp-right 9/10/11, U-turn 13, sharp/left/slight-left 14/15/16, on-ramps 17–19, exits 20–21, forks 22–24, roundabout 26, exit-roundabout 27 and notification 28. Roundabout exit number/lane choice is not encoded. The list message may contain many turns, and the map may depict multiple route segments, but the vector is not a multi-maneuver sequence.

The helper subtracts approximate distance travelled since instruction receipt (`v_ego × receipt_age`), skips passed commands and advances to the next. Unknown modifiers/types can default to straight in the original helper. Matching these names/shapes does not prove arbitrary weights were trained on the same semantics.

### Historical support is a separate model interface

Official v0.9.4 release notes described Navigate on openpilot keeping left/right at forks/exits in Experimental mode; v0.9.5 added navigation instructions as model inputs. In [v0.9.5 `87fbeb87`](https://github.com/commaai/openpilot/blob/87fbeb8787e7a29fa190232f9117192824a193c1/selfdrive/modeld/navmodeld.py#L19-L60), a separate nav network consumed a 256×256 raster and produced **256 learned nav features**. The driving model accepted `nav_features[256]` and `nav_instructions[150]`, representing 50 distance bins × three turn channels, about a ±500 m window. Historical navd did populate `allManeuvers`.

[PR #32106](https://github.com/commaai/openpilot/pull/32106) disabled Navigate on openpilot in April 2024; [PR #32773](https://github.com/commaai/openpilot/pull/32773) removed navigation in June 2024. The historical SNPE/navmodeld/150-vector pipeline should not be imported into this current 47-vector/raw-map integration. It is evidence of prior functionality, not evidence that today's master or current models already enable it.

Search coverage: 155 public official branch refs, public nav/navigation PR and commit searches, relevant historical releases and the five official experimental commits. No public PR was associated with the experimental branch head in GitHub's commit→PR endpoint. That does not rule out private development, discussions elsewhere or unindexed work.

## C. Architecture comparison

| Component | comma master | Official nav experimental branch | SunnyPilot PR #2057 | Your existing branch at `4851eacac` |
|---|---|---|---|---|
| Alpha routing/map/UI | Present | Present with map-pose extrapolation | Inherited | Working backport |
| `allManeuvers` producer | Missing | Present | Missing | Missing before this experiment |
| 47-value helper | Missing in default inference path | Present | Present | Missing before this experiment |
| Map IPC consumed by model | No | Stock modeld | modeld_v2 only | No |
| Typed auxiliary packing | Existing generic stock packing assumes float32 | Per graph dtype | Native modeld_v2 per graph dtype | Native adapter host packing assumed float32 |
| Stock runner nav support | No | Yes | Duplicate import only | No before this experiment |
| Legacy compiler fully handles new UINT8 state interface | Not a general solution | Uses native artifact contract | Not solved by helper changes | Must use native compilation/conversion path |
| Trained nav model artifacts | Default small graph has no nav inputs | Experimental big artifacts | No model artifacts added | Current 17-model v28 catalog does not list the two requested artifacts |
| Proven input validity or road behavior | No nav inference claim | Experimental gaps remain | Experimental gaps remain | Requires later evidence; offline prototype is deliberately barred from production publication |

Proposed source reuse is feasible: use the official publisher/helper concepts and PR native packing/runtime concepts, adapted around your existing timing and fallback hooks. Generic upstream modeld replacement would discard important branch-specific integration. No change to Panda, opendbc, EPS firmware, tinygrad pin, Chestnut firmware, catalog versions or dependency files is shown necessary simply to experiment with input plumbing.

## D. Ranked defects and missing features

Severity here means relevance to a future production integration, not a claim that an incident was observed.

| Priority | Finding and evidence | Classification | Consequence |
|---|---|---|---|
| P0 before vehicle use | No recorded-route inference, native-target parity or integrated GPU/CPU performance evidence | Unverified deployment prerequisite | Passing builds/graphs cannot establish route-following or braking behavior |
| P1 | `allManeuvers` is never assigned by Alpha navd; actual empty-list message produces zero vector | Confirmed source and CPU/native-message tests | Vector-conditioned part receives no upcoming maneuver signal |
| P1 | Reader treats fresh receipt as validity and ignores map `timestamp_sof=0`/old source | Reproduced against PR reader | Invalid/stale producer content can be admitted as recent |
| P1 | GPS loss can retain valid instructions published from old position | Reproduced with 30-second-old/invalid GPS inputs | Fresh message age does not imply fresh physical route progress |
| P1 | No strong admission/feature-off boundary in PR | Confirmed source | Zeroing inputs is not an instant non-nav fallback; recurrent state persists |
| P1 | Nav preparation exceptions occur outside model.run fallback boundary | Confirmed source; malformed-data risk conditional | Bad frame/shape/helper input can terminate modeld before normal Chestnut fallback |
| P1 | Artifact/compiler/adapter contract can be mismatched | Verified two distinct public contracts; legacy shape-only assumptions | Wrong dtypes/state queues or format can invalidate loading/inference |
| P1 | No route-generation association in map/instruction transport | Confirmed schema/stream limitation | A queued old-route frame/instruction can be paired with a new destination |
| P2 | Unknown maneuvers default to straight | Reproduced | Finite but invented guidance can reach model |
| P2 | Missing future step metadata breaks faithful publisher fixture | Reproduced 18 pass/1 fail | Existing mocks need migration; malformed metadata should not invent commands |
| P2 | No explicit shape/dtype admission before copies in PR | Reproduced unsupported shape rejection | Only inspected signatures are known to fit |
| Merge blocker | Duplicate import, draft state and LFS workflow conflict | Local Ruff/API/merge-tree evidence | Upstream PR is not ready for a regular merge |

Map-only conditioning is computationally possible in both inspected graphs. It is not evidence that a zero maneuver vector is acceptable, safe or equivalent to a trained “no route” condition. Incorrect but finite route information need not throw an exception, so model-load/Chestnut fallback cannot detect all semantic failures.

Recurrent-state nuance: v1 raw map/vector queues hold 32 samples, about 1.6 s at 20 Hz; v2 feature queues hold 128 samples, about 6.4 s. Other recurrent features can carry effects longer. Input zeroing does not establish an exact behavioural cutoff. A route toggle or destination change needs a separately defined and validated policy for model state, fallback and temporal consistency.

Timestamp nuance: the map source timestamp is the originating GPS message's logMonoTime, not necessarily the render time. A blanket “source younger than 0.5 s” rule would reject legitimate 1 Hz GPS. Delivery age, source age and camera/source skew need distinct thresholds, with consistent or rebased replay clocks.

## E. Minimal fixes, proposed patches and authorized offline proof

### Changes that can be made independently of controller logic

| Problem / root cause | Files and minimal proposal | Upstream reuse / dependencies | How to validate and what remains |
|---|---|---|---|
| Empty maneuver list | Populate `allManeuvers` in `navigation/navd.py` from geometric step-start distances minus projected route progress; preserve raw Mapbox type/modifier | Official experimental publisher and existing RouteTracker; existing Cereal field is sufficient | Native synthetic left/right/exits/roundabouts/consecutive turns, route clear and malformed metadata tests. No trained-response claim |
| Unknown command invented as straight | Fail closed on unrecognized future maneuver/nonfinite latency in `navigation/helpers.py` | Optional hardening beyond original helper | Correct handling of known command IDs and no invented guidance; zero sentinel still needs training validation |
| Wrong buffer size/dtype | Per-graph itemsize in `modeld_v2/frame_fusion.py` and native adapter views; retain 128-byte alignment and fused upload | Scoped PR hunks; pinned runtime already understands the tensor types | CPU byte-layout/offset/roundtrip and ordinary-model invariance tests; GPU parity still required |
| Missing/bad input can bypass fallback | Shared contained navigation input reader with explicit validity and signature checks; integrate at stock/native model-input boundaries without changing actions/timing | New small helper plus existing Params/GPS/VisionIPC | Zero source timestamp, stale/future source, missing GPS/destination, wrong shape, malformed frame, producer restart, enable/revoke tests |
| Experimental code can publish to live control | Separate permission, default false; require PC plus exact `NAV_MODEL_OFFLINE=1`; reject any nav-capable active/fallback model in both production main entrypoints before PubMaster construction | Additional offline boundary; no new service or firmware required | Native parameter test, all AGNOS flag combinations reject, main-guard AST probes, permission revocation. Dedicated direct ModelState harness only |
| Duplicate import / unrelated workflow | One-line import removal in isolated proposal; exclude LFS workflow from a selective integration | No runtime dependency | Ruff passes; no need to change custom LFS/updater workflows |
| Native versus legacy artifact mismatch | Recognize exact signature/state/output contract; compile/convert through the existing native path, rather than shape-only legacy compiler | Existing `compile_modeld_v2.py`/converter, explicit target runtime | Static signatures, CPU native compilation and a one-frame ModelState A/B passed; converter/reference parity, temporal replay and hardware inference remain |

### Correct maneuver publishing proposal

A Mapbox step's `maneuver` occurs at the **start** of that step. Its current banner often describes the following step; copying banner text/distance into every future entry is wrong. The proposed algorithm concatenates step geometry, records each step's starting vertex into cumulative geometric distance, projects GPS with the existing RouteTracker, and emits remaining starts. Duplicate junction vertices do not add distance. Incomplete future metadata produces an empty model maneuver list while leaving display instructions available.

Standalone research patches are `proposed-navd-maneuver-publisher.patch`, `proposed-helper-fail-closed.patch` and `duplicate-import.diff` in the supporting evidence bundle. They were checked against isolated source copies. The publisher and optional helper tests use left/right turns, both highway exits, ramps, forks, U-turns, roundabouts and their exits, consecutive turns, GPS loss, unknown commands, and missing fields. No patch was applied to the original working branch.

### Authorized separate branch: deliberately offline

The separate experimental branch uses the corrected selector/Navigation Alpha base `4851eacac`, not a new merge into the stable branch. The small source implementation adds typed native packing, validated map/vector preparation and test fixtures while retaining the model timing/action functions. Its enforced boundary is stronger than a PC flag alone:

- `NavigationModelInputsEnabled` defaults false and is independent of Alpha's display toggle.
- Direct model-state experiments require actual PC platform, exact offline environment opt-in and that parameter.
- Normal stock modeld and modeld_v2 `main()` reject navigation-capable active **or fallback** models before constructing publishers, even on an opted-in PC. A PC can otherwise be bridged to a vehicle, so “PC only” by itself would not be a complete live-control boundary.
- No catalog entry, default model file, dependency pin, firmware or production on-device permission is changed.
- Unsupported signatures and malformed/stale map data fail closed at the input boundary. This is a transport/test safeguard, not proof that zero input is a trained safe fallback.

The implementation does not solve learned behaviour, route identity, temporal-state reset policy or actual GPU parity. It does not authorize testing this model integration by driving the vehicle. The runtime was tested and committed separately from documentation; see the validation and publication record below.

## F. Public model compatibility and exact contracts

Both requested releases were fully downloaded. Byte counts and SHA256 agree with GitHub release digests and official experimental LFS pointers:

| Model | Bytes | SHA256 | Official experimental source |
|---|---:|---|---|
| [Nav Model Integration Big](https://github.com/happymaj11r/openpilot-models/releases/tag/big-nav-model-integration-6ea13501) | 764,686,430 | `6ea13501cd6682321ee303c2fe425667988fa43a8aa326d35845602e22e4460d` | [`a52113ee`](https://github.com/commaai/openpilot/blob/a52113eea9a81c79c9309dbc379f0ce25ffa943d/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx) |
| [Nav Model Integration Bigv2](https://github.com/happymaj11r/openpilot-models/releases/tag/big-nav-model-integration-1d18c42e) | 814,591,106 | `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e` | [`a03bb1d4`](https://github.com/commaai/openpilot/blob/a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx) |

They are **official experimental ONNX weights redistributed by a third party**, not independently trained third-party weights merely because of the release host. Nor does redistribution constitute an official supported production release. The model-checkpoint strings corroborate lineage: v1 `cf8f6120-49ee-48f8-b2d7-d663a4708271/56320`; v2 appends `/80eedcdd-f97b-46d0-9b94-780685292acf/12864`.

### All common inputs

ONNX FLOAT is float32. Exact signatures are also supplied as machine-readable `compact-signatures.json` and the new static fixture; no model weights or executable pickle data are embedded there.

| Input | Dtype | Shape in both models |
|---|---|---|
| new_img | uint8 | `[2,6,128,256]` |
| desire | float32 | `[8]` |
| traffic_convention | float32 | `[1,2]` |
| action_t | float32 | `[1,2]` |
| state_img_q | uint8 | `[2,5,6,128,256]` |
| state_desire_q | float32 | `[132,1,8]` |
| state_feat_q | float32 | `[128,1,16384]` |

| Navigation contract | Big v1 | Bigv2 |
|---|---|---|
| nav_map | uint8 `[1,1,256,256]` | float32 `[1,1,256,256]` |
| navigation | float32 `[47]` | float32 `[1,47]` |
| Map history | state_map_q uint8 `[32,1,256,256]` | state_map_features_q float32 `[128,1,64]` |
| Maneuver history | state_nav_q float32 `[32,47]` | state_navigation_features_q float32 `[128,1,64]` |
| Main output | float32 `[1,18452]` | float32 `[1,18580]` |

Each state input has a `next_` output of the same dtype/shape: image, desire and feature states in both, plus the two generation-specific nav states. Those recurrent queues must be retained and updated; the two immediate inputs alone are not a complete runtime interface.

### Map values and layout

The map is one-channel NCHW. Both actual ONNX graphs **divide numeric map values by 255 internally** before casting to float16 for the map encoder. Big v1 first casts uint8 to float32; Bigv2 begins with float32. PR native assignment from uint8 H×W data into a float32 `[1,1,256,256]` view numerically converts 0..255 and broadcasts the singleton dimensions correctly. A float map input is therefore not inherently incompatible with mapsd's uint8 frame. Reinterpreting bytes would be wrong, and pre-dividing externally would double-normalize.

### Output schemas and proved graph connectivity

The metadata's base64 `output_slices` was interpreted using restricted pickle **opcodes as data**, never `pickle.loads`, constructor execution or a compiled model PKL. Both flat vectors share:

`lane_lines[0:528]`, `lane_lines_prob[528:536]`, `road_edges[536:800]`, `meta[800:855]`, `desire_pred[855:887]`, `pose[887:899]`, `wide_from_device_euler[899:905]`, `road_transform[905:917]`, `plan[917:1907]`, `lead[1907:2051]`, `lead_prob[2051:2054]`, `desire_state[2054:2062]`, `action[2062:2066]`, `hidden_state[2066:18450]`.

V1 ends in `pad[18450:18452]`. V2 adds `map_features[18450:18514]`, `navigation_features[18514:18578]`, then `pad[18578:18580]`.

Graph dependency analysis was tied to the exact inputs of the final output Concat and matching metadata ranges, rather than only the opaque combined vector:

- Both `nav_map` and `navigation` have computational paths to **plan, lead, lead probability, desire state and action** in both artifacts.
- Lane-lines/probabilities, road edges, meta, desire prediction, pose/transforms and hidden-state heads are camera-only in this static dependency analysis.
- V2 action combines two baseline action means plus learned tanh residual corrections, followed by action spread values. Both baseline and residual branches depend on navigation.
- V2's map and maneuver feature heads depend on their respective nav inputs.

That static analysis establishes navigation-aware architecture. The separate Bigv2 CPU experiment below additionally establishes a nonzero numerical response to combined navigation inputs for synthetic images. Neither proves useful interpretation of commands, correct intersection stopping, or safe acceleration.

### Runtime, compiler and Chestnut requirements

Both files are ONNX IR 10, default opset 20 plus `org.tinygrad` opset 1, exported by PyTorch `2.13.0+cu130`. The only custom-domain op is `org.tinygrad::Contiguous`, supported by pinned [tinygrad ONNX support](https://github.com/tinygrad/tinygrad/blob/69fd4d34d4430036378aba2b43f9d339cdd6ea73/tinygrad/nn/onnx.py#L544). Other required op names have implementations/mappings in that pinned reader. Static operator availability alone is not a compile result. The separate later CPU native compile and ModelState comparison passed as recorded below; AMD/QCOM kernel correctness remains unverified.

These `.onnx` files are not drop-in selector artifacts. The branch uses both stock native out-of-band PKLs and catalog `SPTG` plain artifacts; the formats are not interchangeable. The existing native model compiler/converter must preserve true dtype, state pairs, output slices and target runtime/device. The legacy temporal compiler `openpilot/sunnypilot/modeld_v2/compile_modeld.py:get_policy_npy_shapes(..., is_supercombo=True)` fails before compilation with `IndexError`: it indexes `desire.shape[2]`, while both public graphs declare rank-one `(8,)`. This was reproduced from the actual unchanged base function and signature. The separate native `compile_modeld_v2.py` has the modern ONNX/native-SPTG path; it must not be described as having that legacy failure.

The current standard catalog is v25 and Chestnut catalog is v28, with selector 21. The exact inspected [v28 catalog](https://github.com/sunnypilot/sunnypilot-models/blob/a89041fcbc13582893f81971214ff4120795b1d4/docs/driving_models_chestnut_v28.json) has 17 entries and contains neither requested nav release nor its hash/ref. Existing big-model selection does not already make these two models selectable or correctly compiled. No catalog change was made. This is not a graph audit of all 17 existing entries.

Raw logical input storage totals about 12.92 MB for v1 and 11.08 MB for v2. Shared feature history alone is 8,388,608 bytes. V1 raw map history is 2,097,152 bytes; v2's two learned feature queues total 65,536 bytes. GPU-resident state can avoid transferring all of this every frame. Those figures are allocation arithmetic, not full VRAM, bandwidth or latency measurements.

## G. Honda EPS, MVL longitudinal and existing-feature impact

### Source preservation and lateral controls

The existing custom timing and drive snapshot must remain in both runners. The branch's [model action/timing code](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/modeld_v2/modeld.py#L232-L255) and [firmware timing hooks](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/modeld_v2/modeld.py#L411-L462) latch the drive calibration and apply prediction delay/action timing. An unrelated upstream overwrite must not remove them.

Navigation inputs can be added alongside existing `action_t`/lateral-control parameters without changing EPS gains, torque curves, geometry, override threshold, actuator safeguards or the 3840→4096 conversion. SPLC and existing curvature/jerk limits still constrain desired curvature. A different model trajectory can still request different steering within those limits; controller preservation does not preserve numerical predictions.

The separate offline branch is independently reviewed for unchanged controls trees, hardware/updater/catalog trees, all seven submodule pins, and the executable action/timing assignments in both model pipelines. Final publication verification must repeat this comparison against `4851eacac`.

### Longitudinal model outputs and current MVL consumption

The native parser/modeld treats parsed action component 1 as desired acceleration and component 0 as lateral action converted to curvature using speed. Without an explicit action head, existing code derives acceleration from predicted plan velocity/acceleration at the relevant action time. It publishes `modelV2.action`, plan position/velocity/acceleration/orientation, and leads through existing message interfaces.

The [MVL planner](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/selfdrive/controls/lib/longitudinal_planner.py#L137-L155) directly considers model desiredAcceleration/shouldStop **only in E2E mode**, alongside existing cruise/radar/MPC candidates. [E2E gating](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py#L39-L44) requires Experimental mode, and blended mode when DEC is active. It then uses existing candidate selection, limits, stop state machine and actuator controller. Stock ACC mode does not thereby acquire direct model acceleration control.

There are other existing indirect routes:

- [SCC-V](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py#L82-L100) computes a curvature speed target from predicted orientationRate and velocity, when enabled.
- DEC/red-light heuristics inspect model trajectory length/orientation, so an altered trajectory could change mode/urgency decisions. A turn-induced trajectory change must not be mistaken for improved red-light detection.
- Radar/vision matching consumes model velocity and leads; the static graphs confirm leads can depend on nav.
- Existing physical steering-angle/lateral-acceleration limits independently constrain acceleration.

A compatible model can therefore influence desired speed, acceleration, intersection/turn/exit preparation and potentially braking **through predictions already consumed by MVL**, without adding raw nav→brake commands or changing the planner. None of those useful behaviours is demonstrated merely by graph reachability. Wrong finite guidance may also alter predictions without activating fallback.

Isolated CPU tests of extracted existing functions confirm the data path: an injected slowing plan gives -1 m/s² instead of zero; a synthetic -1.5 m/s² model action affects the mock planner in E2E but not with E2E off; -10 m/s² is clamped to the tested existing -3.5 m/s² bound; synthetic stop flags enter the model candidate list only in E2E; and increased predicted yaw rate at 20 m/s lowers the existing SCC-V speed target. These are five mocked interface/limit checks, not learned-model or vehicle tests.

### Chestnut, timing and fallback

Existing big→small fallback handles loading/execution exceptions, not semantically incorrect but finite nav-conditioned predictions. Missing input defaulting and hardware fallback are distinct mechanisms. Test active/fallback combinations, destination changes, producer restart and malformed input without erasing the custom delay hooks.

At 20 Hz the raw map stream is 65,536 bytes/frame, about 1.25 MiB/s. Typed fused upload adds 65,792 bytes/frame for uint8 map plus padded vector, or 262,400 bytes/frame for a float32 map plus vector. This is roughly 1.316 MB/s or 5.248 MB/s respectively, excluding other work. Mapsd rasterization, initial route prefetch, extra neural encoders and USB/GPU synchronization can cost much more than those payload sizes imply.

The 64 MiB disk tile-cache limit does not bound route geometry, in-memory road cache or prefetch CPU/RAM bursts. If Alpha is already enabled, its rendering load already exists; model integration adds ingestion/casting/upload and neural computation. Existing model-execution-time counters do not necessarily include preparation before the measured inference region. Full frame-to-output latency, dropped frames, thermal/power and USB state require direct measurement.

## H. Tests, failures, unresolved evidence and hardware limits

### Stable research evidence

| Investigation/check | Result | What it establishes |
|---|---|---|
| Complete public ONNX download and SHA/size checks | Both passed | Exact artifact identity and official LFS lineage |
| Streaming protobuf inspector vs ONNX 1.23.2 | Exact input/output/metadata agreement | Independently verified signatures, no model execution |
| ONNX structural checker | Both passed: v1 861 nodes/658 initializers; v2 1194/790 | Structural ONNX validity, not runtime/kernel correctness |
| Safe output-slice and per-head graph analysis | Passed | Static nav paths to plan/action/leads, internal /255 normalization |
| Original official publisher against inherited 19 tests | 18 passed, 1 failed | Existing synthetic fixture lacks `maneuver`, not a proven normal Mapbox failure |
| Guarded publisher + known maneuver/optional hardening proposals | 49 passed | 19 original + 22 native-message diagnostics + 8 helper guards |
| PR native runtime/packing isolated contracts | 5 passed | Typed offsets, byte roundtrip, fused mapping, shape rejection; confirms PR accepts zero/stale source frames |
| Actual two public signatures/parser synthetic outputs | 2 passed | Native dtype/broadcast/state-pair and existing parser compatibility |
| MVL prediction-consumption synthetic probes | 5 passed | Existing interfaces/mode gates/clamp consumption, not learned inference |
| PR duplicate import | F811 reproduced; isolated one-line correction passes | Concrete lint defect and minimal repair |
| Isolated merge-tree reproduction | One upstream conflict; 19 custom-base conflicts | Exact source-history integration risk |

Tests from the authorized experimental branch include the real public-signature gate/parser fixture suite (**6 methods, 24 subtests passed**) and independent offline/publication-boundary suite (**9 methods, 54 subtests passed**). The latter checks actual native Params default false, all tested AGNOS permissions rejecting, unchanged ordinary-model admission and main guards before publisher construction. These per-suite counts are not additive to an eventual aggregate run; the completion record must provide final aggregate results after all source changes.

### Experimental implementation and validation

Corrected base: `mvl-honda-eps-selector21-chestnut-v28` at `4851eacac8fce2e875a0ef7703af5f5025fc0c15`. The new branch is [`mvl-honda-eps-nav-model-testing`](https://github.com/ZakariaSaad/ZakariaSaad/tree/mvl-honda-eps-nav-model-testing).

| Commit | Purpose |
| --- | --- |
| [`9ad4a310ab6e3d79b750b69242b1dde961ece75d`](https://github.com/ZakariaSaad/ZakariaSaad/commit/9ad4a310ab6e3d79b750b69242b1dde961ece75d) | Maneuver publisher/helper, typed inputs in both runners, optional freshness reader, offline admission/publication guards, contract/regression tests and isolated comparison tool |
| [`f52f0ad0e80d14176e6403812fa106d4c4fcdd3d`](https://github.com/ZakariaSaad/ZakariaSaad/commit/f52f0ad0e80d14176e6403812fa106d4c4fcdd3d) | Explicit scratch CPU warp for offline ModelState, target/camera/model contract checks; original default asset paths preserved |

The first implementation commit is a direct child of the requested base. Documentation is recorded in a subsequent commit on this same branch; its final SHA is supplied in the completion record rather than embedded self-referentially here.

| Automated verification | Final result |
| --- | --- |
| Normal Linux `scons -j4` build | Passed; native bindings, daemons, tools and firmware build artifacts compiled; nothing flashed |
| Navigation services/UI, both new runner suites, RPC, messaging, updater migrations and offline harness | **501 passed, 109 subtests passed**, zero failures |
| Honda EPS, existing controls, models/selector, fallback/recovery, camera offsets, parser/combined loader, Chestnut and Params | **581 passed, 2 skipped, 8 subtests passed**, zero failures |
| CPU-warp override plus combined loader checks | **73 passed, 6 subtests passed**; overlaps targeted suites and is not an additional aggregate total |
| Final changed Python Ruff checks and whitespace | Passed |
| Type checking on new code and other changed paths | Passed; **four pre-existing diagnostics** remain in stock modeld/native adapter and reproduce identically on untouched base source |
| Protected source/pin/timing review | Both control trees, hardware, updater, selector/catalog, all model assets, message schemas, dependency locks, launch configuration and workflows unchanged; all seven submodule pins identical; executable action functions and firmware/action timing ASTs identical in both runners |

The two skipped broad regressions are recorded as skips, not successes. One explicitly requires optional external-network integration. The type diagnostics are three existing stock-modeld shape/device types plus the same existing Tensor-shape diagnostic in the native adapter; unrelated code was not changed to suppress them. Initial UI failures were caused by LFS image/font pointer files in the isolated checkout; exact SHA-verified baseline assets resolved them, with no tracked asset changes. Fork-from-multithread warnings are inherited test-environment deprecations.

These results are local Ubuntu/CPU/Xvfb checks. They are separate from the upstream PR's GitHub Actions results. No new Actions workflow or pull request was created, and no GPU, macOS/AGNOS release build or full process replay was rerun on this experimental branch. The upstream PR replay's 66 passing cases do not establish navigation-model replay coverage.

### Actual learned Bigv2 ONNX CPU comparison

Two learned inferences completed using the exact SHA-verified Bigv2 ONNX and pinned tinygrad `69fd4d34`, with `DEV=CPU:LLVM`. Both arms used identical deterministic **synthetic** cameras, traffic/action timing, and independently zeroed recurrent inputs. The conditioned arm changed only the raw 0..255 right-turn map and the 47-value right-turn-at-100m vector. No public compiled pickle, publisher, CAN interface or controller was used.

All six model outputs and all parsed heads were finite. Total experiment time was 93.7 seconds; the neutral call took 58.6 seconds including compilation, the conditioned call 30.9 seconds, and peak process RSS was about 2.08 GiB. These are CPU research timings, not a real-time or Chestnut benchmark.

| Compared raw head | Maximum absolute change |
| --- | ---: |
| Plan | 1.7422409 |
| Lead | 9.4061584 |
| Lead-probability logits | 1.3134278 |
| Desire state | 3.2466488 |
| Action, including log-spread entries | 0.1503906 |
| Camera/vision heads and hidden state | 0, bit-identical |

The action means changed from `[0.04157054, -0.05556089]` to `[0.04489577, -0.03165460]`: the acceleration-mean component changed by about `+0.0239063`. Plan velocity and acceleration components also changed. This confirms numerical navigation influence on longitudinal predictions through the learned graph. It does **not** demonstrate appropriate braking, route following or a safe turn. Raw mixed-head deltas are not all in the same physical unit. Both map and vector changed together, so this test does not isolate which input caused each effect.

The native SPTG/compiler and ModelState experiment is recorded separately below; direct ONNX execution alone is not proof of that integration. Reproduction scripts, exact commands, hashes, logs, synthetic output arrays and result JSON are included in the evidence bundle.

### Actual native CPU SPTG, camera warp and ModelState comparison

The existing native `compile_modeld_v2.py` compiled the verified Bigv2 ONNX on pinned tinygrad into a CPU SPTG artifact: **671 kernels, 810.36 MB, 159.1 seconds**, including its PlainJit verification. The separate CPU warp compiled in **8.8 seconds, seven kernels**. An initial warp CLI import collision was resolved by placing the pinned tinygrad path first in that command's PYTHONPATH; no dependency or compiler source change was made.

The only artifact metadata adaptation rebuilt the already safely decoded output-slice dictionary from primitive JSON into locally generated metadata. No downloaded metadata pickle was executed. Kernel binaries, constants/check payloads, normalized kernel headers and input/output specs were verified unchanged. The resulting trusted local driving artifact SHA256 is `ce90e3de271c94289933c44c05a29f4606ce6ec1e827748c804464c1130c9bcf`; the CPU warp SHA256 is `556cdca4c1740434000e52a0c57f9d0027359d62502deb0c539735af82dc0337`. Neither artifact is committed, installed or added to a catalog.

The actual branch `ModelState` harness then completed **two independent one-frame CPU arms in 120.0 seconds**, including separate warmup/reset, fused typed upload, camera warp, recurrent-state allocation, learned inference and parsing. Both used the same explicitly **synthetic** prepared NV12 inputs, desire, transforms and action timing. The conditioned arm supplied the raw right-turn map and maneuver vector. No GPS fix, production route or recorded road scene was claimed. The prepared NPZ SHA256 is `c5634ac1e044027c1f05c10e8713029830f3661762c52124e41be6ece6088164`.

| Parsed output | Maximum absolute change |
| --- | ---: |
| Plan | 2.8503723 |
| Action | 0.01542127 |
| Lead | 11.6580353 |
| Camera/pose/meta/hidden-state heads | 0 |

All parsed outputs in both arms were finite; no timeout, OOM, publisher, CAN interface, controller or vehicle process occurred. This confirms that the **implemented native CPU path** can deliver navigation inputs through the compiled graph. It does not establish CPU-vs-GPU numerical parity, performance on Chestnut, useful driving behaviour, recorded-route timing, or multi-frame route-state correctness. Direct-ONNX and ModelState runs used different synthetic camera representations, so their differing deltas are not a reference-parity comparison.

The runtime discovery also produced a concrete fix: existing tracked warps target QCOM/AMD, so a CPU driving artifact alone is insufficient for this harness. The optional `warp_path` override now accepts a scratch CPU warp without replacing tracked assets. The tool requires both artifacts, rejects non-CPU/NPY targets, and checks camera copied extent, transform contract and model-image output shape/dtype. Normal callers still use the original paths and signatures.

The original run's generic scope string said “recorded”; the actual input metadata and commands identify it as synthetic. The final tool now says “prepared-input inference” to avoid that misleading label. This wording correction does not change numerical results.

Runtime publication at `f52f0ad0e80d14176e6403812fa106d4c4fcdd3d` was verified independently through Git and GitHub's commit API. All **249 pre-existing remote branches** still matched their original SHAs; the sole added branch was the authorized experiment. The final documentation tip is verified separately in the completion record.


### Modified and added files

The branch changes only these source/test paths plus the two documentation files. No weights or compiled artifacts are committed.

- `openpilot/common/params_keys.h`
- `openpilot/selfdrive/modeld/modeld.py`
- `openpilot/selfdrive/modeld/tests/test_navigation_inputs.py`
- `openpilot/selfdrive/navigation/helpers.py`
- `openpilot/selfdrive/navigation/model_inputs.py`
- `openpilot/selfdrive/navigation/navd.py`
- `openpilot/selfdrive/navigation/tests/fixtures/nav_models.json`
- `openpilot/selfdrive/navigation/tests/test_experimental_boundaries.py`
- `openpilot/selfdrive/navigation/tests/test_maneuver_contract.py`
- `openpilot/selfdrive/navigation/tests/test_model_input_admission.py`
- `openpilot/selfdrive/navigation/tests/test_optional_helper_hardening.py`
- `openpilot/selfdrive/navigation/tests/test_public_model_contracts.py`
- `openpilot/sunnypilot/modeld_v2/frame_fusion.py`
- `openpilot/sunnypilot/modeld_v2/model_adapters.py`
- `openpilot/sunnypilot/modeld_v2/modeld.py`
- `openpilot/sunnypilot/modeld_v2/tests/test_navigation_inputs.py`
- `tools/navigation_model_offline.py`
- `tools/tests/test_navigation_model_offline.py`
- `docs/honda-eps/navigation-model-testing.md`
- `docs/honda-eps/navigation-model-investigation.md`


### Evidence still missing

The initial artifact/graph investigation was non-executing; the later synthetic CPU inference evidence is explicitly separated above. AMD/Chestnut/QCOM execution, numerical target parity, actual GPU model compilation, production route/backend requests, private training-data comparison, vehicle CAN actuation, firmware flashing and road experiments remain outside this evidence. Synthetic CPU inputs are not recorded-route or vehicle validation. Public graph/checkpoint/source evidence is strong enough to design tests; it is not a replacement for those tests.

Necessary later evidence includes: exact compiler/runtime parity for each artifact; controlled offline causal comparisons of valid left/right route, map-only, vector-only, zero, stale and inconsistent route inputs across all consumed heads; temporal state/route-change policy; replay clock alignment and generation identity; model fallback and permission revocation; full latency/drop/thermal/USB profiling with navigation and Chestnut together; and only then separately authorized stationary and controlled hardware validation. Existing road experience with the non-nav model branch does not validate new learned navigation behaviour.

## I. Recommended minimal development plan

1. **Keep the current working branch as rollback baseline.** Continue only in the separately authorized experimental branch based on `4851eacac`; do not merge PR #2057 wholesale, change catalog models or modify controls/firmware.
2. **Retain the completed data-contract work.** The experiment now includes the official geometric maneuver publisher, tested helper, declared dtype/shape packing and contained freshness reader. Keep the original Alpha UI/API compatibility and both EPS timing hooks.
3. **Retain offline-only admission and publication guards.** The display toggle is not model permission. The default is off, AGNOS is denied and production modeld main cannot publish nav-model predictions even on an opted-in PC.
4. **Define unresolved temporal semantics.** Add route-generation association or demonstrate an adequate synchronization policy; choose validated behaviour for stale data, toggles, destination changes and all recurrent states. Zero input alone is not a proven trained fallback.
5. **Choose one explicit model generation and artifact path.** Use its full signature/state/output metadata and pinned compiler target. Big v1 and Bigv2 have different map dtype and history protocols. Do not use release name or input name alone as capability metadata.
6. **Extend the completed CPU compile/differential proof.** Initial direct ONNX and native SPTG/ModelState comparisons passed for synthetic inputs. Next compare equivalent recorded inputs across reference/native runtimes, both model generations and AMD/QCOM, without publishing to controls. Validate plan/action/leads and temporal outputs, not only a turn arrow or curvature value.
7. **Measure integrated timing and resource load.** Preserve action-time calculations; profile complete preparation→output latency, dropped frames, CPU/RAM/GPU, temperature, USB power/link stability and all fallback combinations.
8. **Review results before any broader enablement.** Only after evidence resolves the above should a separate decision consider supported catalog entries, a hardware diagnostic harness or controlled vehicle validation. No road-readiness claim follows from this prototype.

No exact engineering-hour estimate is justified yet. The source plumbing is a small, identifiable set of navigation/model-input files; completing reliable production behaviour is materially larger because compiler parity, semantic invalid-input policy, recurrent state and hardware/trained-model evaluation remain open.

## J. Direct answers to the eight requested conclusions

1. **Why has PR #2057 not been merged?** It is still draft, has a reproduced LFS workflow merge conflict, a failing static check and unresolved review threads. Its current nav path also omits the maneuver producer and lacks adequate validity/admission hardening. The complete human decision history is not publicly documented, so those observed blockers should not be turned into speculation about author intent.
2. **Does comma.ai have a more complete implementation?** Yes, on official public `nav-model-integration` at `afbf2e54`, including the publisher, stock feed and experimental artifacts. Inspected master does not feed navigation into driving inference. The experimental branch is not evidence of production deployment or complete reliability.
3. **What exactly is broken or missing?** Empty `allManeuvers`→zero vector; source validity/freshness gaps; route-generation ambiguity; recurrent-state/disable semantics; uncontained prep failures; explicit artifact/native compiler contract validation; and actual learned/GPU/performance evidence. A trivial duplicate import and unrelated workflow conflict additionally block merge hygiene.
4. **Can these issues be fixed?** The publisher/helper/packing/gating defects have bounded source-level fixes and CPU tests. Input transport can be made more robust. Learned behaviour, source-map/training parity, compiler binary parity and route-state policy still require evidence rather than a guessed patch.
5. **Can fixes work with Chestnut v28?** The existing native adapter architecture can represent the verified dtypes/state pairs, and the pinned ONNX reader supports the graph operators. The successful CPU native compilation and ModelState comparison support an offline integration path without replacing v28 or firmware. They do not establish successful AMD compilation, timing or hardware readiness. V28 is a catalog version, not a guarantee that arbitrary ONNX files are supported catalog artifacts.
6. **Are compatible files publicly available?** Yes: the two downloaded ONNXs are genuine nav-input graphs and exact matches to official experimental LFS objects. Their generations differ. They still require correct compilation/conversion and validation before use in the selector/native runtime; no model was installed or added to the catalog.
7. **How much must the existing branch change?** A scoped publisher/helper/input-reader/typed-packing/runtime-boundary/test set can be added without rewriting EPS, MVL, hardware, updater, selector or submodules. A wholesale merge would introduce 19 custom-base conflicts and unnecessary risk. Production support additionally needs a deliberate artifact/evaluation process; small code diff is not equivalent to small behavioural risk.
8. **What is required before road testing?** Verified artifact/target compiler parity; actual differential inference and route/state failure tests; source/route-generation consistency; unchanged ordinary-model/fallback timing; full GPU/CPU/thermal/USB load and latency measurement; clear rollback/permission boundaries; and separate authorization after reviewing those results. This investigation and offline branch do not establish or authorize learned navigation driving.

## Appendix 1. All 44 PR-reported commits

The subjects below are the captured GitHub commit list in order. Scope labels are a history guide based on the inspected net feature and meaningful changes; they are not assertions that all 44 represent separate nav requirements. Navigation's main change is #36, camera adapter correction #37, and unrelated workflow #44; generic runtime precursors and merges largely overlap existing upstream work.

| # | Commit | Subject | Scope guide |
|---:|---|---|---|
| 1 | [`75b5926f4f`](https://github.com/sunnypilot/sunnypilot/commit/75b5926f4fa0b066054dea7624b119893f10aca3) | models: fix qcom float16 overflow | Generic model/compiler/runtime precursor |
| 2 | [`1344824746`](https://github.com/sunnypilot/sunnypilot/commit/13448247464d483b36e25d1c9b657a643bab4325) | modeld_v2: shes deeeeeep | Generic model/compiler/runtime precursor |
| 3 | [`d7c1827134`](https://github.com/sunnypilot/sunnypilot/commit/d7c1827134c15fb618443bd46c38b1b93f3b85c4) | new revolution | Generic model/compiler/runtime precursor |
| 4 | [`bd27f3b54f`](https://github.com/sunnypilot/sunnypilot/commit/bd27f3b54fb9f5e596026893406d5b2d66a894fa) | offroad proc for cache warmup | Generic model/compiler/runtime precursor |
| 5 | [`f7c65f7d6a`](https://github.com/sunnypilot/sunnypilot/commit/f7c65f7d6a79bdab4d371da7594b26a528a3304a) | lint | Generic model/compiler/runtime precursor |
| 6 | [`be51bc793d`](https://github.com/sunnypilot/sunnypilot/commit/be51bc793dab9893b0ce4d360c40beeec7f56b5e) | 5ms speed up on qcom | Generic model/compiler/runtime precursor |
| 7 | [`bc064e892f`](https://github.com/sunnypilot/sunnypilot/commit/bc064e892f7c4d7ca876fa3a27a82e58c19accd5) | standin for chestnut linear_compile | Generic model/compiler/runtime precursor |
| 8 | [`04098893d9`](https://github.com/sunnypilot/sunnypilot/commit/04098893d9dec52a04b552172068caeedef84faa) | run as bash: for when i wake up: | Generic model/compiler/runtime precursor |
| 9 | [`b4c3a19db1`](https://github.com/sunnypilot/sunnypilot/commit/b4c3a19db1fb8ca698b6485004c9326305526fd7) | skip out on op modules | Generic model/compiler/runtime precursor |
| 10 | [`9f3a2607ee`](https://github.com/sunnypilot/sunnypilot/commit/9f3a2607ee7f5809a06c286a18766a247daf57c9) | dedup chestnutty | Generic model/compiler/runtime precursor |
| 11 | [`f3f7296776`](https://github.com/sunnypilot/sunnypilot/commit/f3f729677633ab9b56e4aded63928dbca88598d3) | stream chunks for big models to save oom | Generic model/compiler/runtime precursor |
| 12 | [`6220734b2d`](https://github.com/sunnypilot/sunnypilot/commit/6220734b2d07724f6f2d6dd25e3211c65dea7767) | parallel | Generic model/compiler/runtime precursor |
| 13 | [`a9ccd8fedf`](https://github.com/sunnypilot/sunnypilot/commit/a9ccd8fedf5a59451e448315804f29f80ef0b59c) | zed compress | Generic model/compiler/runtime precursor |
| 14 | [`b7d06aa440`](https://github.com/sunnypilot/sunnypilot/commit/b7d06aa440d1a947f7ed85cc253daf384d425ab9) | publish faster | Generic model/compiler/runtime precursor |
| 15 | [`e6e5b1c165`](https://github.com/sunnypilot/sunnypilot/commit/e6e5b1c1658338f906e937b17d0970ab4697bfa9) | bump tg to upstream tg/tg:master. no rebuild needed | Generic model/compiler/runtime precursor |
| 16 | [`2bf8f88978`](https://github.com/sunnypilot/sunnypilot/commit/2bf8f8897893469be9fa6d83c7f022dc66ae392d) | bump | Generic model/compiler/runtime precursor |
| 17 | [`5acb9d2c8e`](https://github.com/sunnypilot/sunnypilot/commit/5acb9d2c8eb9d7fb91d741fde8c03959a08fe58e) | whoops | Generic model/compiler/runtime precursor |
| 18 | [`0414843e2f`](https://github.com/sunnypilot/sunnypilot/commit/0414843e2f00bbcb037933ddaa3a54cf06335e02) | method() persistent_load | Generic model/compiler/runtime precursor |
| 19 | [`2f6e2560ea`](https://github.com/sunnypilot/sunnypilot/commit/2f6e2560ea09a6c34c1495ca61b70adfb1da3c9a) | modeld_v2: create a compiled pkl -> gpu kernel artifact converter | Generic model/compiler/runtime precursor |
| 20 | [`f4996a0529`](https://github.com/sunnypilot/sunnypilot/commit/f4996a0529d7247978d9b70a5cacfb5ea3a09ddc) | lint | Generic model/compiler/runtime precursor |
| 21 | [`07f755a4c7`](https://github.com/sunnypilot/sunnypilot/commit/07f755a4c743fc8aa6d0eb98b12cdad5be16fd1e) | Merge branch 'rebuild-deep-rl' into comma-pkl-converter | History/runtime/master integration |
| 22 | [`374659bf43`](https://github.com/sunnypilot/sunnypilot/commit/374659bf43fe51e2a7a859839f615bad467af220) | exec | Generic model/compiler/runtime precursor |
| 23 | [`023ef9419a`](https://github.com/sunnypilot/sunnypilot/commit/023ef9419a7260b770f1c4cab99811b0ab52d9a8) | one packed contiguous buffer submit | Generic model/compiler/runtime precursor |
| 24 | [`5823cf4f84`](https://github.com/sunnypilot/sunnypilot/commit/5823cf4f84edb0a4c0d3f2a11fc611a31dcd6860) | lowers chestnut loading to 3ms faster than stock | Generic model/compiler/runtime precursor |
| 25 | [`43cbdc65cd`](https://github.com/sunnypilot/sunnypilot/commit/43cbdc65cd35fe9d7cafdc8ec14599790176dfce) | more | Generic model/compiler/runtime precursor |
| 26 | [`f6e2fb1c33`](https://github.com/sunnypilot/sunnypilot/commit/f6e2fb1c339ad1f24dc5a223f8b1c59d7f9eb214) | Merge remote-tracking branch 'origin/master' into rebuild-deep-rl | History/runtime/master integration |
| 27 | [`0042016791`](https://github.com/sunnypilot/sunnypilot/commit/004201679122e44aca409f878a925783501d2ed8) | conflict res | Generic model/compiler/runtime precursor |
| 28 | [`32460cd5b1`](https://github.com/sunnypilot/sunnypilot/commit/32460cd5b1769c46a2a6deced803bf6e935aa7de) | Merge remote-tracking branch 'origin/rebuild-deep-rl' into comma-pkl-converter | History/runtime/master integration |
| 29 | [`033ba0d973`](https://github.com/sunnypilot/sunnypilot/commit/033ba0d973a07e64972bfd508f924c575a2f455a) | pkl warmup | Generic model/compiler/runtime precursor |
| 30 | [`eb7a6fc0f8`](https://github.com/sunnypilot/sunnypilot/commit/eb7a6fc0f8c01b8f67af537b9fa756d9fa9344d4) | converter fix | Generic model/compiler/runtime precursor |
| 31 | [`86f2e7b255`](https://github.com/sunnypilot/sunnypilot/commit/86f2e7b2553a075b793776636aa862dee8c35643) | fw fetch to kernel | Generic model/compiler/runtime precursor |
| 32 | [`45bfd88e7e`](https://github.com/sunnypilot/sunnypilot/commit/45bfd88e7eeddfd1add094e2979688ade3fb1a6b) | no more fw fetching just use decompress | Generic model/compiler/runtime precursor |
| 33 | [`e6d035b639`](https://github.com/sunnypilot/sunnypilot/commit/e6d035b6393f1c6d1960dc1a916334fa42dcc2b0) | fix ci | Generic model/compiler/runtime precursor |
| 34 | [`05099ae19c`](https://github.com/sunnypilot/sunnypilot/commit/05099ae19c822024901e24071e4e59116b761576) | Merge remote-tracking branch 'origin/master' into rebuild-deep-rl | History/runtime/master integration |
| 35 | [`04a8fb81ab`](https://github.com/sunnypilot/sunnypilot/commit/04a8fb81abe032485ea5b81b33720adcf174ff93) | Merge remote-tracking branch 'origin/rebuild-deep-rl' into comma-pkl-converter | History/runtime/master integration |
| 36 | [`2a84e1481c`](https://github.com/sunnypilot/sunnypilot/commit/2a84e1481c8ce347bd69e6a3eca1714fb2a584af) | modeld_v2: prep for nav models | Main navigation feature |
| 37 | [`84fda467b5`](https://github.com/sunnypilot/sunnypilot/commit/84fda467b5552ab08de9fffb987628c178d8c6bb) | caught on new icm comparison tool | Native camera-key correction |
| 38 | [`3163577bc9`](https://github.com/sunnypilot/sunnypilot/commit/3163577bc9f1247477906a4456004afd4a22293b) | Merge remote-tracking branch 'origin/rebuild-deep-rl' into comma-pkl-converter | History/runtime/master integration |
| 39 | [`ddf83be52b`](https://github.com/sunnypilot/sunnypilot/commit/ddf83be52bced56d8c2c24941ec66d6a971f25b5) | Merge remote-tracking branch 'origin/comma-pkl-converter' into reimplement-nav | History/runtime/master integration |
| 40 | [`b24f546a0f`](https://github.com/sunnypilot/sunnypilot/commit/b24f546a0fdb50611ab19dba82d644f54a8d39f2) | Merge branch 'master' into rebuild-deep-rl | History/runtime/master integration |
| 41 | [`723ebf374e`](https://github.com/sunnypilot/sunnypilot/commit/723ebf374ebff8aa0b6babb53348b23a162fdc03) | Merge branch 'rebuild-deep-rl' into comma-pkl-converter | History/runtime/master integration |
| 42 | [`6433886a65`](https://github.com/sunnypilot/sunnypilot/commit/6433886a65850c8d7998ff635c770444ab5a67de) | Merge branch 'comma-pkl-converter' into reimplement-nav | History/runtime/master integration |
| 43 | [`5d7fb82291`](https://github.com/sunnypilot/sunnypilot/commit/5d7fb822916686c2809d59e81c830bd18a2b660c) | Merge branch 'master' into reimplement-nav | History/runtime/master integration |
| 44 | [`f7f9a3efae`](https://github.com/sunnypilot/sunnypilot/commit/f7f9a3efae3a6552068201bd9e4e4dbe66aae577) | ci: lfs pointers survive gc | Unrelated LFS workflow |

## Appendix 2. Supporting evidence inventory

- PR/API state, all changed-file patches, 44-commit list, five inline comments/review threads, CI check/job snapshots and merge-tree results.
- Official master/development/historical source snapshots and five-commit model lineage.
- `compact-signatures.json`: exact graph contracts, provenance hashes, output slices and static per-head dependencies, without executable model data.
- `proposed-navd-maneuver-publisher.patch`, `proposed-helper-fail-closed.patch`, `duplicate-import.diff`: standalone research proposals with their test evidence.
- Native packing/parser tests, synthetic MVL interface tests and experimental boundary verification.
- Final original-branch/remote-ref preservation and new-branch publication record in the accompanying completion record.

Model downloads, raw metadata, inspection environments and large source archives remain intermediate research material. No untrusted pickle was executed. The portable report/fixtures/patches contain data and source only. ONNX graph inspection and any explicitly reported bounded CPU graph experiment are distinct from loading a compiled executable model PKL.
