# MVL Cinque Terre V3: review and validation

This change is based on MVL commit `73dca34cece29339f43835f1c9296a5809586f4b`
from `sp-honda-dev-202608`, on the existing `mvl-cinque-v3` branch. The MVL
opendbc submodule remains at `c40f4612d00db3485a7860e8ec3e89e5b5fa4064`.
Honda CAN, safety, longitudinal tuning, MADS, gas interceptor, and lane-path
code/settings are preserved. The change adds a model adapter and recovery
guard; it does not replace Honda controls or select V3 by default.

Do not merge or deploy for assisted driving until review, live-model behavior,
and fallback transitions have been validated. Synthetic CPU tests and a single
GPU warmup do not establish driving readiness.

## Source and artifact identity

- Official model commit: `c6d13eb723e0f6eee47ea7d56e47a700fba21380`.
- Official stateful interface: `64f9b47b6ea7b622967acd7042148e7a92f0ccc4`.
- Original export: `f78ed37d-afad-4dbc-8050-40ea885eedde/12864/big_driving_supercombo.onnx`.
- [Original ONNX](https://huggingface.co/commaai/openpilot_driving_models/resolve/main/f78ed37d-afad-4dbc-8050-40ea885eedde/12864/big_driving_supercombo.onnx).
- ONNX SHA-256: `404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4`.
- ONNX size: `766354845` bytes.
- Pinned MVL Tinygrad: `f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae`.

The downloaded original ONNX was verified against that size/checksum during
development. Its inputs are `new_img` `(2,6,128,256)`, `desire` `(8,)`,
`traffic_convention` and `action_t` `(1,2)`, and three recurrent state buffers:
`state_img_q` `(2,5,6,128,256)`, `state_desire_q` `(132,1,8)`, and
`state_feat_q` `(128,1,16384)`. Each state has a matching `next_state_*` output.
Predictions have shape `(1,18452)`.

The adapter supports official explicit-buffer exports
(`input_specs`, `output_specs`, `run`, `metadata`) and the integrated format
introduced by `64f9b47b` (`input_shapes`, `state_pairs`, `run_model`,
`input_devices`, `metadata`). It uses strict unpickling on the pinned Tinygrad
revision, without remapping incompatible enums or classes. An old single-output
TinyJit that discarded next-state outputs must be recompiled.

The existing `/data/media/0/cinque_v3_local_tinygrad.pkl` remains a separate
artifact. Its format, checksum, and behavior have not been checked on this
cloud machine. The compiled pickle's checksum is different from the ONNX
checksum and must be independently recorded on the source device. Only install
trusted compiled pickles: pickle loading executes code, and a digest verifies
identity, not provenance or driving quality.

## Separate installation and startup checks

These are instructions for later reviewed, offroad validation, not instructions
to deploy this branch now. Keep a downloaded, previously working small legacy
Tinygrad bundle selected in the QCOM model slot. Its artifacts must retain their
catalog SHA-256 values. Do not substitute a `COMBINED_MODEL_PKL` override.
The recovery model uses the built-in device, independently of a failed AMD GPU.
An unavailable or invalid fallback prevents V3 initialization.

With the MVL virtual environment activated, record the compiled artifact's
SHA-256 using `sha256sum /data/media/0/cinque_v3_local_tinygrad.pkl`. Then, from
the checkout root, after review:

```sh
python tools/cinque_v3.py --install /data/media/0/cinque_v3_local_tinygrad.pkl \
  --sha256 COMPILED_ARTIFACT_SHA256 \
  --tinygrad-ref f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae
python tools/cinque_v3.py --check --camera-size 1344x760
```

Replace `COMPILED_ARTIFACT_SHA256` with the recorded 64-character digest.
The installer copies to `Paths.model_root()/cinque-v3-<digest>.pkl`, verifies the
copy before atomically installing it, and stores `CinqueV3Manifest`. It leaves
the source pickle, working bundle, and Honda parameters intact. Installation
and checks keep V3 disabled. The tool requires `IsOffroad` to be true.
Repeat the check with `1928x1208` if that camera configuration is used.

Checks verify both artifacts, the pinned Tinygrad revision, input/state/output
contracts, recurrent warmup, reset/replay determinism, message serialization,
and fallback warmup. Camera warps for explicit-buffer artifacts are generated on
the model device with MVL's NV12 code; an external warp file is not required.

If recompilation is needed, compile on the intended Chestnut device using the
MVL environment and original ONNX above:

```sh
DEV=USB+AMD python -m openpilot.sunnypilot.modeld_v2.compile_cinque \
  --onnx /path/to/big_driving_supercombo.onnx \
  --output /data/media/0/cinque_v3_rebuilt.pkl
```

Use the same Tinygrad device/interface configuration that worked for the local
compilation. The compiler verifies the official ONNX size/checksum, retains all
named next-state outputs, checks recurrence and a pickle roundtrip, and refuses
to overwrite an existing output. Full target-GPU compilation was not run in
this CPU-only environment.

## Fault handling and persistence

`CinqueV3Enabled` is false by default. `CinqueV3Manifest` and `CinqueV3Fault`
persist across modeld/manager restarts and reboots. `CinqueV3Active` is runtime
status. Installation creates a fault/review latch; successful offroad checks do
not automatically clear it or enable the model.

Before V3 starts, its selected legacy fallback is checksum-verified and warmed.
Initialization is journaled with a persistent interruption latch; a crash or
reboot during loading cannot repeatedly retry the same artifact. V3 becomes
active only after startup checks succeed. Legacy model selection is unchanged.
While V3 is enabled or fault-latched, runner selection uses the selected small
QCOM bundle even if Chestnut settings/cache normally choose another model.
Stored model selections are retained. Clearing a fault is an explicit validation
action, not part of reboot or automatic model selection.

On initialization, inference, parsing, publication, or output-validation errors:

1. Persist the error in `CinqueV3Fault`, clear `CinqueV3Enabled` and
   `CinqueV3Active`, and disable `ExperimentalMode` with blocking writes.
2. Discard the failing frame; do not publish its predictions.
3. Reset/warm the verified fallback and reset action smoothing, lane-change
   state, road-edge state, frame-drop filtering, and publication history before
   processing a fresh frame. Use fallback-specific delays and metadata.
4. If the fallback fails, stop modeld; do not resume V3 or invent predictions.

V3 initialization has a 60-second watchdog; live inference has a 250-ms hard
deadline, including output synchronization/validation. The native GPU wait is
also bounded to 3 seconds. A wedged call cannot safely be abandoned in a Python
thread: the watchdog records the fault and exits modeld. The manager restarts
with V3 latched off and loads the fallback. Existing missing-model/control
health checks cover the interruption; seamless or uninterrupted assisted
control is not claimed. Validate this transition on the target system.

An upgrade reruns startup checks. A Tinygrad revision mismatch or corrupt
artifact disables V3 before unpickling. Never clear the fault latch automatically
on boot, upgrade, model download, or ignition transitions.

## Required validation before assisted driving

Run the automated suite below first. After code review, use a non-actuating
bench/replay harness to exercise the real AMD pickle over recorded frames.
Explicitly clearing `CinqueV3Fault` and setting `CinqueV3Enabled` is a separate
operator action for that harness; never add it to boot scripts or run it while
controls can actuate. Keep the verified fallback selected and available.

Record real model outputs, camera alignment, desire edges, temporal continuity,
finite and correctly shaped publication, inference latency, and reset/replay
behavior. Inject initialization failures, invalid outputs, GPU exceptions,
timeouts, process interruption, and reboot/upgrade during initialization.
Verify that failed frames are discarded, the fault survives restart, V3 is not
retried, and fallback frames/actions/messages use clean state. Validate the
existing vehicle health/disengagement response during watchdog restarts.
Finish with Honda regression/replay checks for the intended car and a reviewed
validation report. Passing the checks in this document does not itself authorize
assisted-driving deployment.

## Rollback

While offroad, run:

```sh
python tools/cinque_v3.py --rollback
```

This persistently disables V3 and Experimental Mode while retaining the existing
legacy model selection, artifact files, and Honda tuning. The next modeld startup
verifies and warms that fallback. Inspect `CinqueV3Fault` and verify ordinary
legacy-model operation before any later use. If the fallback is invalid, restore
the previously working bundle/artifacts while offroad and rerun its checks;
do not clear the V3 fault to bypass the failure.

For a code rollback, first preserve/commit local work and verify the checkout is
clean. In the isolated checkout, switch to the exact baseline and restore its
submodules:

```sh
git switch --detach 73dca34cece29339f43835f1c9296a5809586f4b
git submodule update --init --recursive
uv sync --frozen --all-extras
```

Rebuild the baseline using its normal build instructions and verify the working
model. Do not reset, clean, overwrite artifacts, or alter Honda settings as part
of rollback. This code rollback is a review procedure, not a deployment command.

## Automated checks

The dedicated CI workflow builds native prerequisites and runs the new safety
tests plus existing Honda fingerprint, lane-path, Nidec speed-lead, and legacy
model compiler/parser tests.
From the existing checkout (no additional worktree is needed):

```sh
# Ubuntu build prerequisites:
sudo apt-get update
sudo apt-get install -y --no-install-recommends clang capnproto libcapnp-dev
uv sync --frozen --all-extras
source .venv/bin/activate
scons --minimal -j4 \
  openpilot/common/libparams_c.so \
  msgq_repo/msgq/ipc_pyx.so \
  msgq_repo/msgq/visionipc/visionipc_pyx.so
DEV=CPU python -m unittest \
  openpilot.sunnypilot.modeld_v2.tests.test_cinque_guard \
  openpilot.sunnypilot.modeld_v2.tests.test_stateful \
  openpilot.sunnypilot.modeld_v2.tests.test_cinque_adapter \
  openpilot.sunnypilot.modeld_v2.tests.test_compile_cinque \
  opendbc.car.honda.tests.test_honda \
  opendbc.car.honda.tests.test_lane_path \
  opendbc.car.honda.tests.test_nidec_speed_lead -v
```

Tinygrad CPU tests require clang. The tests cover default/explicit selection,
startup rejection, recurrence, output rejection without state commit, state
reset, real pickle roundtrips, camera warps/message publication, both official
artifact formats, fallback errors, hard watchdog termination in a subprocess,
fault persistence across reopened Params stores, interrupted startup, corrupt
installation, repeat installation, offroad gating, and Tinygrad upgrades.

## Local validation result

The compatibility/recovery/compiler/adapter suite and existing Honda tests
passed: **40 tests**. The existing legacy model compiler/parser suites passed
**40 tests**, with **one skipped**. Ruff and `git diff --check` passed. The native
Params prerequisite was rebuilt with the new keys. The original ONNX's digest,
size, typed input/output contract, and all three state pairs were verified.
Honda control/safety code, dependency pins/lockfile, and submodule revisions
remain identical to the exact MVL baseline.

These results use synthetic CPU inference, not the device's 788-MB AMD pickle.
Full real-model GPU compilation/inference, target-camera replay, assisted-driving
behavior, and live hardware fallback/watchdog transitions remain unvalidated.
The implementation is published for review on `mvl-cinque-v3`; consult GitHub
Actions for the separate remote CI result. Publication does not enable V3 or
authorize vehicle deployment. No merge or vehicle deployment was performed.
