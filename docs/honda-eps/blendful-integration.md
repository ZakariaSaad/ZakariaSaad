# Blendful (Alpha) integration

Target: `ZakariaSaad/ZakariaSaad`, `mvl-sp-steer-eps-nav`.
Starting commit: `84a3ff66139e0eb044076fba679ed5c4cfdaf8f8`.

## Upstream sources

- Feature: SunnyPilot [PR #2073](https://github.com/sunnypilot/sunnypilot/pull/2073),
  `97135965effb3630ed55993903449edb73468d50`.
- Unavailable-model explanation: [PR #2081](https://github.com/sunnypilot/sunnypilot/pull/2081),
  `24f531b83c4b3c43163e31654451afec8bece6de`.
- SunnyLink capability classification: [PR #2082](https://github.com/sunnypilot/sunnypilot/pull/2082),
  `52600eccbf234f206a65e236addcaf7c94f3b7de`.
- Inspected master: `6f7a2c94c10dffcf9a83b120ab1e6985e4496868`.
  No later changes to the Blendful algorithm or its upstream tests were present.

Only the feature and these relevant follow-ups were ported. No upstream branch
merge, dependency update or model artifact change was required.

## Behavior and compatibility adaptations

`sunnypilot/modeld_v2/blendful.py` and its nine upstream tests are byte-identical
to the feature commit. The filter and model constants it uses already match
upstream. Lane fitting, trust, smoothing, low-speed fading, lane-change handling,
line-loss behavior and obstacle yielding retain their upstream mathematics and
tuning. The visualization implementation is also copied unchanged.

The tinygrad runner reads `Blendful` every 60 frames, beginning with its runtime
flag OFF. It passes the upstream correction callback once to action calculation,
before the existing lateral smoothing/low-speed hold. Desire and lane-change
conditions are the upstream conditions. Switching OFF invokes the upstream
disabled update/reset behavior; no new reset or activation policy is added.

Required adaptations to this fork:

1. **Native Bigv2 action helper.** Bigv2 retains its official action interpretation
   through the shared stock helper. That helper gains the same optional callback
   at the corresponding point, and the native subclass forwards it. The stock
   model runner never supplies the callback, so unsupported Default operation is
   unchanged. This avoids assigning Bigv2 an ordinary model generation or
   replacing its action/stop behavior.
2. **Exact OFF behavior.** The callback still updates/resets upstream Blendful
   state, but a zero correction is not added to curvature. This preserves the
   baseline representation, including negative zero. Nonzero corrections follow
   the upstream addition and smoothing order.
3. **Parameter polling.** The unchanged 60-frame polling interval is placed
   outside this fork's Bigv2-specific camera/timing branch so both native Bigv2
   and ordinary tinygrad models consume the setting.
4. **Existing settings layouts.** Upstream controls are inserted into the two
   existing customized Models layouts without replacing Bigv2 download/selection
   or EPS controls. The new SunnyLink car-capability list omits upstream's
   `toyota_tss2`, which this branch does not define; unrelated Toyota functionality
   is not imported.

`ModelDataV2SP` adds upstream fields at unused ordinals 3–9. Existing fields,
schema identifiers and navigation metadata are unchanged. The service receives
upstream's qlog decimation setting. These fields report correction/trust for
logging and visualization; they do not apply a second correction.

The existing downstream sp-steer lane-centering option remains a separate,
unchanged feature. Blendful is called only by model action calculation. The
ordinary common curvature limits, existing maneuver override, custom EPS law,
prediction schedule, command delay, torque/rate limits and Panda safety remain
in place. Longitudinal acceleration/stop calculation is unchanged, including
when Blendful is enabled. Navigation inputs, FP32 map correction, Light Speed's
separate recurrent buffers, ordinary aliases, model loading and fallback are
unchanged.

## Settings and SunnyLink path

Location: **SunnyLink → Models → Model Behavior → Blendful (Alpha)**. The same
feature is available in the on-device Models settings with the upstream overlay.

`Blendful` is registered as `PERSISTENT | BACKUP`, Boolean, default `0`. Missing
settings therefore mean OFF on fresh installs and upgrades. An explicit setting
survives reboots and ordinary on/off-road lifecycle clears. Nothing in this port
enables the setting on the owner's device.

Eligibility uses `ModelRunnerTypeCache`, the manager's actual runner decision,
rather than a highlighted UI item. Chestnut's selected big slot participates in
that decision. The stock Default runner is unsupported; a big model whose
display name includes “Default” is not automatically excluded. Upstream's
in-process small fallback remains in the tinygrad runner and retains its runner
eligibility. An unavailable toggle does not erase the saved preference.

The device serves compiled `settings_ui.json` through
`getParamsMetadata()` → `generate_schema()`, adding live capabilities and their
car-dependent classification. The real `saveParams()` handler writes the same
typed `Blendful` parameter that modeld reads; no parallel or display-only setting
was introduced.

Official SunnyLink frontend source
`26795b7766413369f03c105fda6cd4f1e53eae1f` supports this contract:

- `dashboard/+layout.svelte` collects every panel/section setting key and
  prefetches and decodes its value, including dynamically added settings.
- The Models page renders the device's Models `SchemaPanel`.
- `SchemaItemRenderer` applies capability rules and uses its Boolean toggle.
- `batchPush` → `setDeviceParams` → the settings endpoint sends the ordinary
  parameter RPC consumed by the device.

No separate frontend patch or deployment is required by that implementation.
Device schema, compressed/uncompressed RPC readback and setting writes were
tested locally. An authenticated browser against the live hosted service and
the owner's comma was not exercised; no vehicle connection was made.

## Verification

- **14 core tests passed:** all nine unchanged upstream tests plus five focused
  source/OFF/reset/fading/line-loss cases.
- **28 runtime tests and 17 subtests passed:** action functions pinned to the
  starting commit and upstream; direct, plan, PlanPlus and native Bigv2 paths;
  smoothing/speed thresholds/signed zero; ON exactly once; OFF reset; actual
  122-frame modeld parameter refresh/publication and lane-change sequences.
- **444 preserved integration tests passed:** EPS/timing/transport, sp-steer
  longitudinal consumption, official Bigv2 behavior and Light Speed buffers.
- **151 SunnyLink/RPC tests passed.** Nine initial host skips were resolved
  separately: the JSON schema validator passed using installed system
  `jsonschema`, and all eight EPS rendering tests passed under Xvfb.
- **24 rendered UI tests and 78 subtests passed** under network-disabled Xvfb,
  including the two Blendful layouts, Bigv2/navigation and the EPS tests above.
- Native Params shared library, C++ cereal schemas/library and service header
  rebuilt successfully. Generated settings JSON matches its YAML; schema
  validation, changed Python syntax/Ruff and Git whitespace checks pass.

Test groups overlap and are not summed. OFF comparisons use representative
synthetic predictions and exact serialized action values from the recorded
starting source, rather than claiming recorded `modelV2` messages reproduce
neural inference. The main-loop tests verify unchanged EPS `action_t` bytes.
No new test failures remain in the exercised groups. No hardware/road validation,
model recompilation, firmware flashing or installation occurred.

## Normal update

Update `mvl-sp-steer-eps-nav` through Software settings and restart normally.
Refresh SunnyLink afterward. Blendful stays OFF unless a saved preference already
enables it; use the Models toggle to choose it explicitly on an eligible runner.
Changes are consumed at the existing 60-frame parameter refresh (approximately
three seconds at 20 Hz). Other model and EPS selections remain unchanged.
