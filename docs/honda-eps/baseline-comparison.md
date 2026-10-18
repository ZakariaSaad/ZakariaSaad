# Exact MVL baseline comparison

Baseline: `cb15c04e918b759688cb354b1bbc9b3c5a47a65c`, the requested
`mvl-boston/openpilot` / `sp-honda-dev-202608` foundation. The controller work
retains this commit's ancestry and all seven submodule revisions. The original
publication backup remains at `backup/mvl-honda-eps-before-clean-4d6accd8`.

## Evidence collection

The paired workflow checks out the baseline and candidate into separate,
unmodified source directories on Ubuntu 24.04, installs each frozen dependency
set and builds each with SCons. An external harness observes the original unit
runner's final records without changing its assertions, collection, execution,
warnings or exit status. Both full suites use two workers.

The original full process-replay CLI also runs with two workers. It retains all
66 comparisons, its original ignore fields/tolerances, original inputs and
upstream references. Fingerprints cover the original replay input/reference
files and the location-scenario fixture. The comparator requires identical
fixture bytes, reference commit, shared dependency versions and gitlinks.

Every unit outcome and normalized failure traceback must match. Full replay
differences must match byte for byte after removing only `ref:` and `new:` file
paths, which contain the checkout location and current commit. Numerical values,
field names, process names and segment names remain in the comparison. Collection
errors, execution errors, partial replay, missing evidence and differing outcomes
fail the comparison. Existing test failures remain failures in normal CI.

## Results

The executed paired results, exact candidate SHA, fixture fingerprints, numeric
diff digest and full inherited-failure traces are recorded in
[verification PR #3](https://github.com/ZakariaSaad/ZakariaSaad/pull/3) and its
`Honda EPS baseline comparison` workflow artifacts. Publication requires that
comparison to pass; source inspection alone does not establish baseline parity.

The standard workflow at `c7e77772` and `faf507dd` reports:

| Check | Measured result |
| --- | --- |
| Dedicated firmware pytest | 389 passed |
| Broad original unit runner | 1,511 passed, 47 skipped, 1 expected failure, 3 failed; no collection errors |
| Original full replay against `27c132984df60e9ef42ebd1a805738b8c851356f` | 61 passed, 5 changed, 0 execution errors |
| Release/macOS builds, static analysis and UI | Passed |

The unit failures are calibration auto-reset (`test_calibrationd.py:102`),
consistent gyro spikes (`test_locationd_scenarios.py:171`) and consistent timing
spikes (`test_locationd_scenarios.py:212`). The five replay differences are
Hyundai/Toyota `selfdrived` startup alert text, Honda `controlsd`, and Honda/Honda2
`card`. They compare MVL behavior with SunnyPilot's older upstream replay
references. The paired comparison determines whether their complete numerical
differences and assertion traces are inherited from the exact MVL foundation;
these tests are not renamed, skipped, retuned or reported as passing.

## Integration scope and limits

The verification follow-up changes tooling and documentation only. The controller
implementation remains the one published at `c7e77772`. Its targeted regression
suite passes 405 tests, including native Params, Honda transport and LongControl;
the dedicated firmware-only CI step passes 389 tests. The helper has ten tests,
including rejection of changed numeric replay data, fixture bytes, test results,
assertion traces, dependencies, submodules and reference commits, plus the
relative artifact-output path regression found during initial evidence upload.

Longitudinal planner/MPC, LongControl, Honda acceleration/braking tuning, Panda
and model weights/inference implementation are unchanged from MVL. Both model
runners select the same lateral timing schedule when the firmware controller is
admitted. Enabled mode changes lateral timing inputs; offline core replay does
not prove identical joint neural-model outputs or closed-loop braking behavior.

The controller remains disabled by default, with original MVL fallback. Admission
requires independent confirmation of the installed corrected TXM-A040 build;
the diagnostic part number alone cannot distinguish the old and corrected images.
The supplied route does not cover speeds above 46.7 mph and contains truncated
tails in the final rlog/qlog. Tight-turn geometry differs unless the explicit
James geometry is selected. Model/geometry/response combinations, driver handover,
EPS saturation and real-time vehicle behavior still require controlled hardware
validation. No software is deployed, no firmware is flashed, and no controller
setting is enabled by this publication.
