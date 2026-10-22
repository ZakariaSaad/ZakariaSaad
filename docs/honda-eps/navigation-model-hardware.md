# Non-actuating Chestnut hardware checks

This package runs local file-input diagnostics in a separate checkout. It does not install software, switch a selected model, publish normal model predictions, start manager/CAN/vehicle controllers, alter engagement, or flash Chestnut/EPS firmware. Physical Chestnut results have **not** been obtained in the development environment. A completed numerical check is not a driving-quality or vehicle-safety verdict.

The initial candidate is Bigv2 ONNX SHA256 `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`. The package uses the branch-pinned tinygrad commit and native `SPTGPLN1` compiler; Selector21 and catalogv28 are separate contracts. It never edits SunnyPilot's public catalogs or normal model cache.

## What detection establishes

`inspect` reads Git, `/sys` and `/proc`. It records USB VID/PID, product/firmware string, USB speed, descriptor revision, an irreversible serial hash, source/component hashes, submodule identities and process ownership. It opens no USB/GPU device. USB descriptors identify the bridge; the GPU ISA is explicitly unknown when the USB GPU is absent from host PCI sysfs.

The current branch profile expects `custom 8516b768-CLEAN` (`openpilot/common/hardware/usb.py`). A different product, ROM/bootloader mode or ambiguous USB enumeration blocks execution. This is a conservative branch-profile check, **not evidence of a Bigv2-specific firmware upgrade requirement**. The historical `custom ed4e39b7-CLEAN` string does not establish your device's present firmware. Send a mismatch report for compatibility review; this package provides no flashing/recovery command.

For admitted compute, each AMD worker uses `DEV=USB+AMD:LLVM` without forcing an architecture. It then checks the device-discovered `arch`, the actual `USBIface` and renderer target. Only physical AMD LLVM `gfx1200` USB is accepted. Hardcoding `gfx1200` in `DEV` would not independently establish the installed GPU architecture.

Normal GPU initialization claims/configures USB, can detach/reset a USB driver, powers PCIe and initializes the GPU, including loading volatile AMD GPU microcode. These ordinary runtime operations are distinct from persistent Chestnut bridge/EPS flashing. The package never imports the persistent flasher or `hardwared`; a running `hardwared` is rejected because its normal off-road loop can flash a mismatched bridge. No flag can eliminate all GPU initialization while still performing GPU inference.

## Preparation on your comma 4

Use a parked, ignition-off/bench setup with stable comma4/Chestnut power and adequate cooling. Do not drive during these checks. Plan for at least12GiB free after preparing the checkout and inputs. Every worker requires at least8GiB still free when it starts; compilation products and caches consume space, so8GiB initial free space alone is insufficient. Hardware workers run one at a time, with a4GiB resident-memory watchdog and stage deadlines. Physical4GiB sufficiency is unverified; a budget failure is reported separately from model/firmware incompatibility. Large virtual GPU mappings are not confused with resident RAM usage.

First obtain a separate checkout of the published target branch. These are preparation commands for you to run over SSH; no remote installation or service operation has been performed for you:

```sh
NAV_CHECKOUT=/data/navigation-model-src
NAV_WORKSPACE=/data/navigation-model-results/run-001
NAV_PYTHON=/data/openpilot/.venv/bin/python

git clone --recurse-submodules --branch mvl-honda-eps-selector21-chestnut-v28 https://github.com/ZakariaSaad/ZakariaSaad.git "$NAV_CHECKOUT"
cd "$NAV_CHECKOUT"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$NAV_PYTHON"):$PATH"
export PYTHONPATH="$NAV_CHECKOUT:$NAV_CHECKOUT/tinygrad_repo:$NAV_CHECKOUT/msgq_repo:$NAV_CHECKOUT/opendbc_repo:$NAV_CHECKOUT/rednose_repo:$NAV_CHECKOUT/panda"
```

Native bindings are built only after the stopped-service inspection below. Do not run a default/full SCons build on the comma: its default targets can initialize QCOM and compile learned models/warps outside this package. The existing interpreter is reused read-only; the separate checkout takes precedence in `PYTHONPATH`. Do not run its `launch*.sh`, manager or firmware updater. Do not move `/data/openpilot`, change its repository/branch, or copy experimental artifacts into its model cache. If the installed interpreter path differs, set `NAV_PYTHON` to its actual Python3.12 interpreter. Missing dependencies/native bindings are preparation failures, not a reason to start the production launcher.

Obtain the public ONNX explicitly in a separate input directory; the package itself never downloads a model:

```sh
mkdir -p /data/navigation-model-inputs
curl -fL https://github.com/happymaj11r/openpilot-models/releases/download/big-nav-model-integration-1d18c42e/big_driving_supercombo.onnx -o /data/navigation-model-inputs/big_driving_supercombo.onnx
sha256sum /data/navigation-model-inputs/big_driving_supercombo.onnx
```

The digest must match the pinned SHA above. Native AMD compilation can additionally need the runtime's pinned LLVM/volatile GPU-microcode dependencies through normal tinygrad fetching; failures are retained in the log. No bridge firmware migration is attempted.

Prepare file inputs using `tools/navigation_recording.py`. Prefer at least164 actual paired camera captures, with physical SOF timestamps and the recorded calibration/telemetry requested by that tool:

```sh
"$NAV_PYTHON" tools/navigation_recording.py --rlog /data/navigation-model-inputs/rlog --narrow-camera /data/navigation-model-inputs/fcamera.hevc --wide-camera /data/navigation-model-inputs/ecamera.hevc --segment 43 --start-index 0 --count 164 --scenario recorded-camera-clip --output /data/navigation-model-inputs/cameras164.npz --manifest /data/navigation-model-inputs/cameras164.json
```

Substitute the real segment and encoded indices. This reconstructs experimental action timing/neutral desire; it does not claim original model-process replay or recorded navigation. If only a short recorded clip is available, the explicit synthetic sequence adapter may repeat its cameras:

```sh
"$NAV_PYTHON" tools/navigation_model_sequence.py --recording /data/navigation-model-inputs/cameras8.npz --output /data/navigation-model-inputs/cameras164.npz --frames 164 --allow-repeated-cameras
```

Repeated cameras and generated timestamps are synthetic temporal fixtures, not164 distinct recorded captures. Use `--allow-repeated-cameras` below for that path. Generated map/route overlays always require `--synthetic-navigation`; real prepared navigation must supply independent route/frame/GPS metadata instead.

## Stop normal inference through your existing service controls

The package refuses active `manager`, `modeld`, `camerad`, `hardwared`, vehicle-control daemons and any competing `/dev/kfd`, `/dev/dri/*` or selected Chestnut USB owner. Being off-road is insufficient: manager can respawn GPU inference and hardwared can update firmware.

You control this maintenance transition. On an AGNOS installation using the normal `comma` systemd service, after parking and switching ignition off:

```sh
sudo systemctl stop comma
sudo systemctl status comma --no-pager
```

These are the existing service controls used by `tools/op.sh`; the diagnostic package never invokes them or kills a production process. If your device uses a different supervisor, use its normal maintenance procedure and verify the process inspection before proceeding. Do not spoof `IsOnroad`, engagement or other Params. If the service keeps restarting, stop and resolve the supervisor state rather than racing it.

## Inspect, review the plan, then run

Use sufficient privileges to inspect all process FDs. Inspection fails closed when ownership cannot be established. The workspace must be new, outside both the checkout and production/settings/model-cache paths.

```sh
sudo --preserve-env=PYTHONPATH,PYTHONDONTWRITEBYTECODE "$NAV_PYTHON" -m tools.navigation_hardware inspect --checkout "$NAV_CHECKOUT" --workspace "$NAV_WORKSPACE" --output /data/navigation-model-inputs/hardware-inspection.json

"$NAV_PYTHON" -m SCons --minimal -u -j2 openpilot/common/libparams_c.so msgq_repo/msgq/ipc_pyx.so msgq_repo/msgq/visionipc/visionipc_pyx.so

"$NAV_PYTHON" -m tools.navigation_hardware plan --checkout "$NAV_CHECKOUT" --workspace "$NAV_WORKSPACE" --onnx /data/navigation-model-inputs/big_driving_supercombo.onnx --recording /data/navigation-model-inputs/cameras164.npz --synthetic-navigation

sudo --preserve-env=PYTHONPATH,PYTHONDONTWRITEBYTECODE "$NAV_PYTHON" -m tools.navigation_hardware run --non-actuating --checkout "$NAV_CHECKOUT" --workspace "$NAV_WORKSPACE" --onnx /data/navigation-model-inputs/big_driving_supercombo.onnx --recording /data/navigation-model-inputs/cameras164.npz --synthetic-navigation
```

For the repeated-camera adapter, append `--allow-repeated-cameras` to both plan and run. Width/height default to1344×760; supply the actual prepared dimensions when different. The explicit three-target build only prepares Params/Msgq/VisionIPC bindings; it does not request default model or warp targets. Keep the shown interpreter directory on `PATH` because build actions can invoke `python3`. Unknown firmware, architecture, ownership or artifact compatibility is an explicit failure. Neither CPU results nor emulation can produce an accepted AMD manifest.

The bootstrap sequence is deliberate:

1. Compile CPU and physical AMD native artifacts from the same pinned ONNX, with independent verification seeds42/43, finite outputs and source-to-reloaded-artifact comparisons.
2. Compile matching CPU/AMD camera warps from the requested NV12 copied extent to the actual model image contract.
3. Use the existing data-only exporter to bind the successful physical AMD compiler report, source/artifact/compiler/runtime hashes, target and tensor metadata into a new local manifest.
4. Run the bounded GPU smoke test, then independent CPU/GPU comparisons on the same prepared camera bytes, transforms, navigation contexts and recurrent-reset schedules.
5. Run164 GPU frames per arm. The synthetic long sequence first fills at least132 consecutive valid recurrent frames (the longest Bigv2 desire queue;144valid frames precede the synthetic defects), then exercises route/GPS/permission/staleness/frame-association transitions and recovery. Invalid contexts may be explicitly skipped under diagnostic policy; this does not validate a driving fallback.

The runner records the exact subprocess arguments/environment, full source GitSHA, component hashes, discovered GPU identity, timing, errors, peak resident RAM, logical tinygrad allocation measurements from the diagnostic, and available sysfs USB/power/temperature telemetry. Bridge-specific voltage/temperature absent from sysfs stays marked unavailable; no production monitoring publisher is started.

CPU/GPU comparison initially uses explicit `atol=1e-3, rtol=1e-3`, finite-only (`equal_nan=False`) checks on every matched parsed tensor and all captured recurrent-state queues. The comparison stages explicitly capture state; the164-frame run omits full state dumps to avoid gigabytes of output. Optional `--comparison-atol` and `--comparison-rtol` may tighten these defaults; looser values are rejected. These conservative engineering tolerances identify implementation/numerical differences, not driving acceptability. A mismatch is a failed comparison to investigate, never a reason to silently relax tolerances. Compiler source/reload consistency is a separate check. No claim is made that this reconstruction equals the latest official compiled n16 binary.

## Cleanup and results

Ctrl-C, a deadline, excess resident memory or a newly competing production process stops only the package's own worker process group. Existing production processes are never signalled. No automatic service restart occurs. The package leaves active selected models/settings unchanged and retains its separate artifacts for investigation.

At completion or a stage failure, collect:

```sh
"$NAV_PYTHON" -m tools.navigation_hardware archive --workspace "$NAV_WORKSPACE"
sha256sum "$NAV_WORKSPACE/results.tar.gz"
```

Send `results.tar.gz` and its SHA back privately for review. The recurrent comparison NPZs include `_state.state_img_q`, containing warped/packed camera history, so this archive includes derived camera pixels. Do not publish it to GitHub or another public location. The archive contains source/artifact identities, compiler reports, commands, telemetry, per-frame diagnostic results, reset/recovery events and CPU/GPU comparison details. It excludes weights, caches and raw camera recordings. Keep those inputs locally if a failure needs reproduction. An interrupted process before the archive step can use the explicit archive command once the worker has exited.

When your maintenance session is finished, restore the normal application using your existing service control:

```sh
sudo systemctl start comma
sudo systemctl status comma --no-pager
```

That restarts the already-installed software; it does not install this checkout or select the candidate. To remove scratch data later, first retain the results archive, then manually remove only the explicit `/data/navigation-model-results/run-001` workspace and optional separate checkout/input directory. The package has no broad cleanup/delete command.

Remaining hardware-dependent acceptance includes actual firmware/product compatibility, successful AMD source/reload verification, memory/power/temperature/USB stability, sustained timing, numerical comparison and the recorded long-transition results. The package does not validate simultaneous AMD navigation and ordinary QCOM/small-model execution, its shared memory/power/timing budget, or the continuously warmed fallback transition path; those remain a specific candidate-activation gate. Vehicle-control validation, learned navigation quality and safe fallback behavior remain separate gates; this package cannot authorize activation by itself.
