# Parked-car Chestnut smoke test from Windows

Use your existing2022 Insight, installed comma4, USB-connected Chestnut and180W
accessory outlet. No bench, external supply, AMD workstation, source checkout,
camera recordings or hand-edited configuration is required. Normal navigation-model
driving remainsOFF. A smoke pass is not approval for navigation-conditioned driving.

First update the already-installed `mvl-honda-eps-selector21-chestnut-v28` normally
through Software settings and restart. These new commands arrive through that update;
no reinstall or installation-URL change is needed. Enable SSH using your existing
setup and use the IP shown in Network settings. Keep the car parked and ignitionOFF
throughout maintenance. Do not drive while the normal comma screen/service is stopped.

## First run: one short command

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

The default run downloads **no model or runtime dependency**. It verifies fresh
real off-road telemetry, physical ignition flags, installed branch/runtime and the
expected Chestnut USB product. It then uses a supervised maintenance job to stop the
original `comma.service`, obtain exclusive device access, read supply voltage/current
and physical ignition, discover the actual AMD architecture and check GPU arithmetic.
If a verified Bigv2 artifact already exists in the separate diagnostic cache, it also
runs eight built-in synthetic predictions across neutral/map/maneuver/combined inputs.
They are prewarped artificial pixels, not real cameras or navigation replay. No warp,
CPU reference model,164-frame preparation or manifests are needed for this smoke test.

The first GPU initialization can take time; the probe/inference stages each have a
three-minute limit, and the entire short maintenance unit has a ten-minute limit.
A missing cached LLVM/volatile-microcode dependency is an explicit failure rather than
an unexpected download. The existing installed runtime download cache is reused
through individual read links; its compiler database and selected models are untouched.

The normal comma service is restored before the final result is printed. The
supervised job continues after an SSH disconnect or Ctrl-C, with an independent
`ExecStopPost` cleanup registered before normal processes are stopped. The job has
memory/time limits and restores after success, worker failure, cancellation or OOM.
A lingering device owner blocks restoration instead of starting inference over it.
A service-restoration failure is explicitly reported; do not drive until resolved.

**Power/off-road compatibility comes first.** The socket's180W rating does not prove
available power when the car is genuinely off-road. If the outlet loses power with
ignitionOFF, or ACC mode makes the real panda report ignitionON, the command refuses.
It does not fake Params, suppress ignition or suggest an ignition workaround. Send
the report; that is a real power/state blocker with the present setup.

## If Bigv2 is not prepared yet

A successful small GPU probe is reported as **Bigv2 not tested** when its artifact is
missing. Once ready to allow the required download/compilation, run this single command:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke prepare
```

This first repeats admission/probe checks. It reuses a matching cached source when
available; otherwise it downloads only the official-matching Bigv2 ONNX,814,591,106
bytes, SHA256
`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
There is currently no verified downloadable AMD artifact for this checkpoint.
The source and native compiled artifact themselves are each roughly0.8GB; temporary
compiler verification and caches make the minimum **4GiB free storage and5GiB
MemAvailable** intentional. No CPU reference artifact is compiled. Installed production
model binaries/cache and selections are not replaced.

The unchanged supported native compiler runs on the actual Chestnut with independent
verification seeds42/43 and source-to-reloaded checks for every output/state. Only a
successful verified artifact is used for the eight synthetic predictions. Compilation
has a30-minute limit, download a15-minute limit, and the full prepare unit a55-minute
limit. Package RSS is watched against4GiB, with an additional systemd5GiB ceiling;
insufficient available RAM, disk, cooling or power stops the job. Physical sufficiency
has not been established here. A failed artifact is never approved or silently used;
keep the returned report for review instead of increasing limits or editing files.

The ordinary smoke needs256MiB disk/1GiB available RAM for a probe; model inference
requires512MiB disk/5GiB available RAM. Small-probe RSS is bounded to512MiB with a1GiB
unit ceiling. The service uses at most two CPU cores and disables swap for its
maintenance cgroup. The monitor refuses low internal-panda voltage, supply faults/loss,
re-enumeration, competing owners and unavailable monitoring. It uses the branch's
existing5V powered threshold, and aborts at162W measured bridge power (a conservative
90% of your stated180W outlet rating). It does not change GPU power limits. Sampling
cannot certify transient headroom or the outlet's rating. Device thermal sensors
reaching85°C abort maintenance; these are diagnostic limits, not changed driving tuning.

The source profile expects `custom 8516b768-CLEAN`; the actual GPU must discover as
AMD LLVM `gfx1200` over USB. A mismatch is reported, never flashed or hidden. GPU
initialization normally configures USB/PCIe and loads volatile GPU microcode. This
is distinct from persistent Chestnut/EPS firmware flashing, which the tool forbids.
The internal panda is read through SPI health requests only: no `Panda()` constructor,
CAN transmission, safety-mode/heartbeat changes, firmware operations or controllers.

## Collect the report

After the SSH command finishes, type `exit` to return to the same PowerShell window:

```powershell
scp "comma@${CommaIP}:/data/navigation-smoke/latest-results.tar.gz" .
```

Send `latest-results.tar.gz` here. It contains a concise `report.json`, device/runtime
and artifact hashes, inference timings, compiler proof when applicable, monitoring
samples, logs and a list of remaining activation checks. It contains no model weights,
private route recordings, raw camera pixels or credentials. Cache/artifacts remain
under `/data/navigation-smoke/cache` for reuse. Nothing is installed into the driving
model selection. Report statuses distinguish probe-only, Bigv2 smoke success and failure.

For a read-only check without stopping the service:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke check
```

## Reconnect, cancel or recover

After reconnecting, return to `/data/openpilot` and use:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke status
```

To cancel the supervised job and invoke its normal cleanup:

```sh
sudo .venv/bin/python -m tools.navigation_hardware.smoke cancel
sudo systemctl status comma.service --no-pager
```

If restoration was explicitly reported failed, first confirm the diagnostic unit and
its GPU workers have stopped. Retain the report. Then restore the original service:

```sh
sudo systemctl start comma.service
sudo systemctl status comma.service --no-pager
```

Do not start a competing service over a hung device owner. A device/OS hang or actual
power loss cannot be repaired by software cleanup; let the report identify that
failure. The normal service's existing startup behavior is preserved, including its
existing firmware checks; this tool neither invokes a flasher nor changes those checks.
No automatic scratch-data deletion is performed.

The [existing comprehensive hardware package](navigation-model-hardware.md) remains
available as an optional advanced workflow. Initial smoke testing does not require it.
The [activation workflow](navigation-model-activation-workflow.md) explains what results
are sufficient and how a future authorized update will avoid manual JSON/Params edits.
