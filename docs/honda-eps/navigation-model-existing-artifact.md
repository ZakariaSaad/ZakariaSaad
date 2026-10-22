# Register the already compiled Bigv2 model

The existing `/data/nav-bigv2/bigv2-amd.sptg` is the starting point. Do not rerun
the Chestnut smoke diagnostic, download Bigv2 again or recompile it for this step.
The new registration action performs file validation only. It opens no GPU,
Panda or camera, stops no service, changes no firmware and never enables driving.

## Compiler report reviewed

The uploaded `bigv2-amd.verification.json` records:

- Bigv2 source SHA256 `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
- Clean pinned tinygrad `69fd4d34d4430036378aba2b43f9d339cdd6ea73` and physical AMD
  LLVM `gfx1200` over USB.
- Exact source/reload agreement for all six prediction/state outputs on seeds 42
  and 43, exact source repeatability, finite results and changing predictions.
- Artifact SHA256 `feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4`.
- Compiler SHA256 `d55aa36999ce05fa545d77dc1cc28e1e46fc28e60620c0b0db816e698bf79e7a`,
  matching the existing branch compiler. No compiler recovery or recompilation
  is needed.

The report's structure and numerical evidence pass software review. The attachment
does not contain the model bytes/header or inference timings. Registration checks
the actual local model bytes/header against that report; it does not fabricate a
new compiler result or treat the report as a signed attestation.

## Existing camera warp

The branch already includes the native OS04 1344×760 Chestnut warp, SHA256
`f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b`.
It uses stride 1408, aligned Y/UV heights 768/384 and 1,622,016 copied bytes per
camera, within a 2,428,928-byte NV12 allocation. Two frames and two float32 3×3
matrices produce packed uint8 `[2,6,128,256]`, matching Bigv2.

The validator checks the complete warp hash, native kernel target, tensor/slot
bindings, reviewed layout/recipe sources and saved capture. An independent NumPy
warp/packing calculation matches all 393,216 captured output bytes exactly. This
is a saved-capture software check, not a new physical GPU test. No warp compilation
is required. The same check also covers the existing 1928×1208 warp.

`mici` alone does not identify a sensor: this source can probe OS04 or OX03. The
registered default is explicitly the OS04 profile. Candidate startup compares
actual camera width, height, stride and aligned extents with the registered
profile, rejecting a mismatch before loading the model. Matching buffer length
alone is insufficient. No connected-camera geometry is inferred from the compiler
report.

## One registration action

First update/restart the same installed branch normally. While parked and genuinely
off-road, choose **Models → Navigation models → Register existing Bigv2** on comma 4.
The file checks run in the background so the interface remains responsive.

Alternatively, at your normal `comma` SSH prompt, run **without sudo**:

```sh
cd /data/openpilot
.venv/bin/python -m openpilot.selfdrive.navigation.preparation
```

The action discovers either existing report filename:
`bigv2-amd.verification.json` or `bigv2-amd.sptg.verification.json`. It hashes the
existing model, validates its data-only native header and complete compiler proof,
and reuses the shipped camera warp. It copies only the small warp, report and
matching compiler source into a separate package under `/data/navigation-models`.
The compiler snapshot is never executed. Model weights stay in their original
location; the ONNX source is neither opened nor required to remain installed.

Portable metadata paths survive atomic package publication. A valid retry reuses
the package. Missing, changed or conflicting inputs fail without overwriting a
working manifest or ordinary model selection. Off-road/disengagement is checked
again before registration. `NavigationModelDrivingEnabled` stays false, and the
new manifest is published only after full validation. Existing approval flags
remain false independently of registration success.

The result is saved here:

```text
/data/navigation-models/latest-registration.json
```

After leaving SSH, retrieve it from the same Windows PowerShell window where
`$CommaIP` is your comma address:

```powershell
scp "comma@${CommaIP}:/data/navigation-models/latest-registration.json" .
```

Send that small JSON receipt. It supplies the actual artifact, warp and metadata
hashes and the exact prepared manifest path; no camera frames, model upload,
manual JSON edits or further smoke diagnostic are needed for registration.

## Remaining software review and activation

Automated tests exercise registration, tampering, portable provenance, interrupted
publication, OFF-mode startup, typed model/state inputs, map/maneuver handling,
clock/reset transitions, prediction parsing, fallback and the Models interface.
The existing EPS/MVL controllers, timing hooks, model slots, runtime/submodule pins,
firmware and updater are preserved.

The receipt is still needed to bind the actual device-held model/header and final
metadata bytes. The report supplies no latency measurement, and this file-only
action does not run live camera fusion or candidate/fallback inference together.
Those observations are not claimed complete. Review the prepared package and
remaining integration evidence before a separate activation decision. No endurance
or thermal-certification requirement is added here. Navigation driving remains
disabled until explicit authorization and a reviewed branch approval update;
registration alone cannot enable it.

Automated validation for this change: **753 tests and 411 subtests passed** in the
navigation/integration suite; **589 tests and 8 subtests passed** in the existing
control/model preservation suite. Six conditional tests were skipped (four
root-only Git cases in the unprivileged suite, an abstract test base and an opt-in
live model-URL check). Ruff and scoped type checks passed. The uploaded report and
actual shipped warp checks are recorded in the
[evidence summary](evidence/navigation-model-existing-artifact.json).
