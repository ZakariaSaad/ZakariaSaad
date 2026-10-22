# Navigation-model software compatibility correction

This continues the existing implementation on
`mvl-honda-eps-selector21-chestnut-v28`, starting at rollback commit
`0843c50307bda103a7e6884ea761ea7ef7ad60da`. It supersedes the older reports'
mandatory extended hardware procedure and their clock/extra-big-model limitations.
Navigation Model Driving remains disabled and unapproved.

## Software changes

- **Camera/navigation clocks:** live approved navigation measures the local
  BOOTTIME/MONOTONIC relationship and converts camera SOF into the Python GPS/map
  clock domain at the reader boundary. A changed offset clears held maps and
  recurrent eligibility; GPS, instructions and maps must be produced in the new
  epoch. Delayed clock samples, stale/future cameras and bad source ages fail
  closed. Exact map IPC frame signatures and existing freshness limits are retained.
- **Model startup:** approved navigation loads the configured native small fallback
  and navigation candidate before ordinary Chestnut model startup. It avoids keeping
  a third, unused selected big model allocated. OFF, rejected and failed candidate
  paths retain ordinary model startup and model selection.
- **Genuine off-road admission:** fresh `deviceState`, `managerState` and
  `pandaStates`, real Params and available engagement/control messages determine
  admission. Physical ignition ON alone does not reject genuinely off-road software.
  Fresh manager data must show `controlsd`, `selfdrived` and `card` stopped. During
  exclusive maintenance the publishers are stopped: the tool monitors real Params,
  process/device owners and read-only Panda control enforcement instead of claiming
  stale messages remain fresh. It requires silent/noOutput safety and disallowed
  controls without sending CAN, heartbeats or safety-mode commands.
- **Short default diagnostic:** no model/runtime downloads, Bigv2 compilation, CPU
  reference artifacts or recordings. It runs a GPU arithmetic probe, then eight
  synthetic predictions only if a verified native AMD artifact is already cached.
  Compiler proof must bind the exact source/model/compiler/runtime and all output
  and state contracts. Keeping the 814 MB ONNX is optional for this smoke; if present
  its bytes are rehashed. Retained compiler reports are not signed attestations.
  A missing artifact is reported as a software preparation requirement. Long tools
  remain optional and never substitute for fixing software defects.

## Parity and artifact availability

The existing [compatibility audit](navigation-model-compatibility.md) establishes
the raw float32 0..255 map, 47-value maneuver vector, typed model signatures, five
recurrent queues, native prediction parser and 20 Hz action-unit contract. These
are implemented already; they were not recreated in this correction.

The [bounded upstream recheck](evidence/navigation-model-artifact-availability.json)
still finds official reference `afbf2e54` and SunnyPilot PR #2057 draft `f7f9a3ef`.
The Bigv2 release contains only ONNX SHA256
`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
The current v28 catalog has 17 ordinary models and no navigation candidate.
No compatible public native AMD `SPTGPLN1` package was found in those sources.
Official TinyJit/OOB PKLs cannot be renamed into this format; the latest n16 PKL
also declares a different checkpoint from the public Bigv2 ONNX.

The pinned compiler performs actual source/reloaded execution, not just static
code generation. This cloud workspace has no AMD/USB device and cannot produce a
truthful successful physical-AMD verification report. The earlier mock-backend
bitcast failure is an emulation/software limitation, not evidence of defective
Chestnut hardware. A maintainer build environment with the matching AMD LLVM
gfx1200 USB/Chestnut target
could prepare one verified package and publish its exact bytes/provenance/hashes,
leaving only a short loading/inference check for the user's car. No such published
package has been established. An arbitrary AMD PCI workstation build is not
automatically admitted as a USB-target artifact. Vehicle-side long compilation is optional developer
work, not the default recommendation or an approval requirement.

## Remaining before activation

1. Deliver the exact native AMD Bigv2 artifact, compatible camera warp and complete
   metadata/provenance; review and pin the hashes. Verify native loaded outputs,
   recurrent behavior and equivalent-input CPU/GPU results for those exact bytes.
2. Complete automatic off-road Settings preparation/manifest registration, including
   interrupted/failed preparation recovery. The current selection helper exposes
   the candidate but does not install/register its package. The user should not
   edit JSON, source or hidden Params.
3. Check the delivered package's camera upload/timestamps and candidate plus warm
   fallback loading, prediction parsing, handover and inference timing using a
   short non-actuating production-runner check. Reproducible fixtures cover route,
   stale/missing GPS/map, permission and reset transitions. Any observed discrepancy
   is a specific software issue to resolve. Extended endurance, thermal certification
   or a user-assembled 164-frame recording are not blanket prerequisites.
4. Review those results and obtain explicit live-integration authorization before
   changing approval flags or enabling driving. A smoke pass alone cannot do so.

The Honda EPS and MVL control implementations, safety limits, ordinary action
conversion, existing model catalog/selector, tinygrad/submodule pins, firmware and
normal updater are preserved. No device deployment or vehicle actuation was run.
See the [short SSH commands](navigation-model-ssh-smoke.md) and
[approval/update workflow](navigation-model-activation-workflow.md).

## Verification

- Navigation, map/rendering and native daemon lifecycle, clock epochs, maneuver
  encoding, model admission/state/parser/fallback, compiler proof, Selector UI,
  short/advanced diagnostics, messaging, Athena and updater regressions:
  **721 passed, 355 subtests passed**.
- Honda EPS and existing control tests, MVL DEC, ordinary model loader/parser,
  fallback/recovery, Chestnut integration and native Params:
  **589 passed, 8 subtests passed, 2 skipped**. The skips are an abstract test
  base and an explicit opt-in live model-URL check, not failed controller tests.
- Ruff passed for all changed Python files; scoped type checks passed for the
  diagnostic and clock changes. Source preservation checks retain all seven
  submodule revisions and ordinary action conversion.

These tests use CPU/native bindings and mocks for unavailable device dependencies.
They include real Msgq/VisionIPC and reviewed-permit admission/revocation; mocked
approval fixtures do not approve any real model. Existing fork deprecation warnings
remain. No new physical AMD inference or comma update was performed in this cloud
environment, and no new AMD artifact or GPU performance result is claimed.
