# Selector 21 / Chestnut v28 testing branch

This branch starts at the unchanged Honda EPS branch commit
`cfe7445e4f3ec7661be1129bd9a75fbe74f09e3c`, whose original MVL baseline is
`cb15c04e918b759688cb354b1bbc9b3c5a47a65c`. Do not merge this testing PR until
builds, regression checks, and the owner's comma 4 / Chestnut checks pass.

## Why selective upstream commits

Pinned SunnyPilot master: `922513f3128b00611d695ce465989a9bc8af9d59`.
The common ancestor is `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56`.
A complete merge brings 269 upstream commits and six conflicts, including an
opendbc revision and unrelated UI/launcher/updater changes. Of 49 commits
that touch model/runtime paths, eight model/build commits plus one firmware
commit provide this stack. The selected model/runtime paths have zero merge
conflicts. The two broad upstream sync merges have five and 51 raw conflicts
outside the selected scope; those changes were excluded and MVL retained.

The nine source commits were applied with their original authors, messages,
and cherry-pick provenance. Broad sync merges use their first-parent delta
scoped to the model system. See `selector21-upstream-provenance.json` for
source/result SHAs and excluded files. Supporting upstream USB detection,
telemetry and LFS changes are included separately; no model implementation
was recreated by hand.

## Models and runtime

The official URLs remain live (future catalog contents may change):

- Standard: `https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_v25.json`
- Chestnut: `https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v28.json`

The checked snapshots contain 77 standard and all 17 Chestnut big models.
Their catalog SHA256 values, refs, generations and rates are recorded in
`selector21-catalog-validation.json`. Both model slots and upstream fallback
behavior are retained; there is no local shortlist of big models.

Big models include Lebowski, Tee Time, TTTTFBRLM, BMRLNAP v1/v2/v3/v4/v6,
I Desire, Happy Birthday, Sad, Cinque Terre v1/v2/v3, Mountain Dew v1/v3 and
Train Track. Inspect the JSON for the exact official display names.

The runtime submodule is upstream's `69fd4d34d4430036378aba2b43f9d339cdd6ea73`
(tinygrad 0.14.0). The catalog's `5d186bdb43c2bcc6c56b6ececf427c34bada0de9`
is compilation provenance. Selector 21 uses portable plain GPU artifacts
(`SPTGPLN1`), replacing Python/tinygrad-object pickles in the catalog.
The bundled stock default uses upstream's separate out-of-band loader; its
800 MB artifact also loaded successfully on CPU with the pinned runtime. The upstream
compatibility workflow downloads and checksum-verifies every catalog artifact
and loads it with QCOM/AMD mapped to CPU. That verifies format/runtime loading;
it does not execute those GPU kernels or establish vehicle compatibility.

The old MVL `default_chestnut.py` downloader and its tests target the former
artifact format and have been replaced by the upstream default-model path.
Updated default driving/driver-monitoring ONNX files and the default big
Mountain Dew v3 compiled artifact are intentional model-system changes.
The LFS endpoint now matches upstream Hugging Face. All 275 pointer objects
were confirmed available there through read-only LFS batch requests.

## Preserved behavior and limits

The Honda firmware controller, corrected TXM-A040 calibration, steering unit
conversion, driver override, default-off setting, admission checks, original
lateral fallback and both model-runner timing hooks are preserved. The
longitudinal planner, braking tuning, controller implementations, opendbc,
Panda and all other submodule pins except tinygrad remain unchanged.

Changing models can change their predictions and therefore vehicle braking
and steering behavior even with identical planner/controller source. This is
not a claim of identical closed-loop behavior. Hardware timing, GPU stability,
power, thermals and driver override still require controlled vehicle checks.

## Chestnut firmware authorization

The owner reports `custom ed4e39b7-CLEAN` and explicitly authorizes the normal
upstream firmware update on this testing branch. The bundled image and USB
requirement are `custom 8516b768-CLEAN`, taken from upstream commit
`a742df61827f8b34eee1af99bbb881201163e472`.

No additional approval parameter or flashing blocker is present. The inherited
automatic updater may flash an attached mismatched Chestnut while offroad.
It retains its onroad restriction, retry limit and image/config verification.
Tests mock the worker; no USB device was accessed, no firmware was flashed,
and nothing was deployed during this integration. Honda EPS firmware is
unaffected by this Chestnut update.

## Basic comma 4 / Chestnut verification (owner performs manually)

1. Keep the current working installation and settings recoverable. Install
   only the testing branch when ready; keep the Honda firmware controller
   disabled initially. Do not merge the testing PR yet.
2. While parked/offroad with stable power and a reliable Chestnut connection,
   allow the normal updater to complete. Avoid unplugging or removing power
   while it programs. Verify the reported USB product becomes
   `custom 8516b768-CLEAN`; save any update error and stop if verification fails.
3. Read-only USB check over SSH:
   `for f in /sys/bus/usb/devices/*/product; do cat "$f"; done`.
   This command does not flash firmware. A cold Chestnut power cycle may be
   needed to activate a completed image if the device remained powered.
4. Confirm the model selector presents the standard catalog and the Chestnut
   catalog, including all 17 big models. Download one standard model and one
   big model; verify completion, persistence across restart, active model/ref,
   Chestnut detection, GPU telemetry and no repeated modeld restarts.
5. Validate the supported standard-model fallback with Chestnut disconnected
   before startup (offroad). Check startup, camera streams and driver
   monitoring with the stock lateral controller before testing Honda EPS.
6. Observe model frequency, frame drops, execution time, prediction delay,
   Chestnut temperature/power and supply faults during a parked soak and a
   controlled drive. Check both 20 Hz and other supported model rates.
   Save logs of underruns, USB/PCIe link faults or fallback events.
7. Only after basic model/hardware stability, separately verify the existing
   Honda firmware controller's admission, disengagement and driver override
   under the project's controlled validation procedure. No automatic enable
   or EPS firmware flashing is part of this branch.
8. Report installed commit, USB product, selected model/ref, model timing,
   fallback behavior and logs in this PR. Hardware results are a merge gate.

## Automated evidence

Targeted integration command:

```
python -m pytest -q openpilot/system/hardware/tests/test_chestnut_integration.py \
  openpilot/sunnypilot/selfdrive/controls/lib/honda_eps/tests \
  openpilot/sunnypilot/models/tests openpilot/sunnypilot/modeld_v2/tests
```

Result: **549 passed, 5 skipped**. The 94-entry official catalog check passed.
Baseline-comparison helper: **13 passed**. It remains strict by default;
the explicit `--selector21` mode permits only the declared tinygrad revision
and package version, removal of the 20 explicitly listed obsolete model-system tests, and newly
passing/skipped tests. All shared outcomes, inherited failures, other
submodules, fixture hashes and replay numerical differences must still match.

The complete Linux build and static analysis passed. All **94 catalog GPU
artifacts**, including all 17 big models, passed the checksum and CPU-mapped
loading jobs in Actions run `39283547910` on integration commit `23642a207`.
The full CI unit suite reported **1500 passed, 50 skipped, 1 xfailed and
3 failed**; the failures are the same calibration auto-reset, consistent gyro
spike and consistent timing spike assertions previously verified on exact MVL.

Release CI exposed an 800 MB raw-PKL packaging regression. The release scripts
from the same upstream sync (`2149006ec`) now exclude bundled big artifacts from
ordinary stripped releases and use LFS when `INCLUDE_BIG_MODEL=1`. Both packaging
variants are checked separately; neither publishes or deploys a release.

The moving upstream replay artifact branch no longer carries MVL's old route
references. The CI harness now stages all 66 original reference logs from
immutable artifact commit `cb59dbea61fd15d4b5a43b05c22108e96a58e476`, reference
`27c132984df60e9ef42ebd1a805738b8c851356f`. Every log was downloaded and checked
against its previously recorded hash. The complete 83-file original fixture
manifest is preserved in `replay-fixtures-original-mvl.json`. Both baseline and
candidate use the same references via the replay CLI's normal local-file path;
assertions, tolerances, process selection and tracked MVL source stay unchanged.
Local Azure input-log access is blocked by the environment proxy (HTTP 403),
reproduced with the untouched MVL downloader. The local broad unit evidence collector rejected duplicate records and did
not produce a valid report; the completed CI unit execution above is the
usable evidence. Final paired replay and Actions results
are reported in the testing PR. A skipped hardware test is not evidence of GPU inference
or vehicle validation. The known original MVL calibration failures and replay
differences remain documented in the Honda EPS baseline reports.
