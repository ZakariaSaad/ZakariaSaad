# Short Chestnut compatibility check from Windows

Use your existing 2022 Insight, installed comma 4, USB-connected Chestnut and 180 W
accessory outlet. No bench, external supply, AMD workstation, camera recordings,
manifests or hand-edited settings are needed. Navigation Model Driving stays OFF.
This checks software/device compatibility; a smoke pass does not approve driving.

First update the already-installed `mvl-honda-eps-selector21-chestnut-v28` through
Software settings and restart. There is no reinstall, installation-URL change or
branch switch. Use the IP in Network settings and your existing SSH setup.

Keep the car parked. The comma must genuinely report **off-road**, with controls
inactive. Physical ignition ON is not by itself a rejection: the test records Panda
ignition flags but uses software off-road/control state for admission. Do not spoof
Params or defeat normal on-road operation to make the test run. Use your existing
power arrangement; do not leave the car in ACCESSORY to drain its 12 V battery for a
long compilation. If normal operation makes the comma on-road, the command refuses
and reports that state. Do not drive while the normal comma service is stopped.

## Run the short check

In Windows PowerShell:

```powershell
$CommaIP = Read-Host 'Enter the comma 4 IP address shown in Network settings'
ssh "comma@$CommaIP"
```

At the comma SSH prompt:

```sh
cd /data/openpilot
sudo .venv/bin/python -m tools.navigation_hardware.smoke
```

The default command **does not download models or runtime dependencies, or compile
Bigv2**. It:

1. Checks the installed branch, pinned tinygrad/runtime and Chestnut USB identity.
2. Requires fresh `deviceState`, `pandaStates` and `managerState`, `IsOffroad=true`,
   `deviceState.started=false`, `IsEngaged=false`, and stopped vehicle-control
   processes. Fresh engagement/control messages must be valid and inactive;
   absent or stale messages are recorded as unavailable, never treated as current
   evidence. The fresh manager report independently establishes that all three
   control daemons are stopped. Admission repeats immediately before maintenance.
3. Temporarily stops the normal `comma.service` in a supervised maintenance job,
   checks exclusive device access, reads basic power/memory/storage information,
   discovers the actual GPU architecture and runs a small GPU arithmetic probe.
4. If a verified compatible Bigv2 AMD artifact and its compiler proof are already
   cached, runs eight built-in synthetic predictions: two each for neutral,
   map-only, maneuver-only and combined inputs. Otherwise reports **Bigv2 not
   tested: compatible AMD artifact missing**. It does not start preparing one.
5. Restores the original service and saves a results archive.

These are artificial, prewarped images and map/maneuver arrays. No cameras, rlogs,
CPU reference compilation, warp compilation, 164-frame sequence or user-prepared
fixtures are required. The existing runtime download cache is reused read-only;
its compiler database and selected driving models are untouched. A missing runtime
dependency is reported rather than downloaded unexpectedly.

GPU initialization can take time. The probe has a 60-second limit, optional model
loading/inference a 90-second limit, and the maintenance job a four-minute limit,
followed by supervised cleanup. These are upper bounds, not requested test durations.
There is no 30-minute compiler stage in this command.

## What remains protected during maintenance

Fresh software telemetry is checked **before** stopping its publishers; the tool
does not pretend those messages continue updating afterward. During maintenance it
rereads off-road/engagement Params, checks that vehicle-control processes remain
stopped, rejects competing GPU/SPI owners and reads internal Panda health. Panda
must remain in silent/noOutput safety with controls disallowed. Ignition flags are
recorded without being substituted for software off-road state. No off-road status
is written or spoofed.

The GPU worker cannot publish normal model predictions or access vehicle streams.
It sends no CAN, runs no controllers, changes no selected model, and performs no
persistent firmware flashing. GPU initialization uses the existing runtime's normal
USB/PCIe setup and volatile GPU microcode loading. The internal Panda is inspected
through read-only SPI health requests, without a `Panda()` constructor, heartbeat or
safety-mode writes.

The source profile expects `custom 8516b768-CLEAN` and discovered AMD LLVM `gfx1200`
over USB. A mismatch is reported rather than flashed. USB loss/re-enumeration,
power faults, insufficient resources, competing owners or active controls abort
the test. Existing conservative memory/power/temperature watchdogs bound this
maintenance command; they are not thermal certification or changed driving tuning.

The service is supervised independently of SSH. Its `ExecStopPost` cleanup runs
after success, worker failure, cancellation, timeout or OOM. An SSH disconnect or
Ctrl-C stops the viewer, not the cleanup. A lingering owner blocks service restart;
a restoration failure is explicitly reported. Do not drive until the original
service is restored.

## If the compatible AMD model artifact is missing

There is currently no verified downloadable native SunnyPilot AMD artifact for the
selected Bigv2 checkpoint. Official comma PKLs are a different runtime format;
renaming them does not make them compatible. Existing ordinary Chestnut big models
do not substitute for this checkpoint.

The missing software preparation is a native `SPTGPLN1` Bigv2 artifact for pinned
tinygrad `69fd4d34d4430036378aba2b43f9d339cdd6ea73`, with complete compiler/output/state
verification for real AMD LLVM `gfx1200`. Its source is the 814,591,106-byte ONNX,
SHA256 `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
The source checkpoint alone is not a runnable native AMD artifact. Loading a verified
artifact does not require keeping that 814 MB source file on the comma: its proof
must match the expected source, artifact, compiler and runtime identities. If the
ONNX is present, its bytes are also rehashed. The report distinguishes that local
source-byte check from retained compiler proof; neither is a signed attestation.

Send the probe report first. Do not run a long preparation job in ACCESSORY simply
to pass this check. The existing `prepare` action and
[comprehensive development tools](navigation-model-hardware.md) remain optional
for deliberate artifact development with a suitable power/resource budget. They
are not part of this beginner workflow. The short command will reuse a correctly
verified artifact when available; it never fabricates compiler success or accepts
an incompatible public PKL.

## Send the report

After the command finishes, type `exit` to return to the same PowerShell window:

```powershell
scp "comma@${CommaIP}:/data/navigation-smoke/latest-results.tar.gz" .
```

Send `latest-results.tar.gz` here. It contains `report.json`, device/runtime and
artifact identities, timings, any failure, monitoring samples and logs. Model
weights, private route recordings, raw camera pixels and credentials are excluded.
The report distinguishes GPU probe success, actual Bigv2 smoke success and failure;
neither success status enables navigation driving. Diagnostic caches remain under
`/data/navigation-smoke/cache`, separate from normal driving selection.

For an admission-only check without stopping the service:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke check
```

## Reconnect, cancel or restore

After reconnecting, return to `/data/openpilot`. Check the job with:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke status
```

To cancel it and invoke its normal cleanup:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke cancel
sudo systemctl status comma.service --no-pager
```

If service restoration was explicitly reported failed, retain the report and first
confirm that the diagnostic unit/workers have stopped. Do not restart over a hung
device owner. If the report shows a changed/ROM Chestnut product, disconnect Chestnut
USB before restarting; otherwise normal startup could invoke its existing firmware
handling. Keep it disconnected pending review. Then restore the original service:

```sh
sudo systemctl start comma.service
sudo systemctl status comma.service --no-pager
```

A device/OS hang or actual power loss cannot be repaired by cleanup. The normal
service's updater, models and firmware behavior remain unchanged. The
[activation workflow](navigation-model-activation-workflow.md) separates remaining
software delivery work from optional longer testing and future live authorization.
