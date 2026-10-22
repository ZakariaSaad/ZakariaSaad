# Navigation-model compatibility and checkpoint audit

The [latest software status](navigation-model-software-status.md) records the measured camera clock bridge, candidate-first startup, short SSH workflow and a fresh bounded artifact-availability check. Extended physical characterization is optional; exact artifact/runtime integration remains required before activation.

This continues the Stage 2 investigation from experimental commit
`036fd8ca9afb5240fa163c377bcfa2b961a38fe0`. The working selector/navigation
baseline is `4851eacac8fce2e875a0ef7703af5f5025fc0c15`. This report identifies
the model contract and hardware requirements; final integration and regression
results belong in the [software integration report](navigation-model-integration.md).

**Bigv2's software interpretation is established from its exact public ONNX and
the official experimental consumer. Its physical Chestnut compatibility remains
unverified.** There is no evidence that this model needs a newer Chestnut bridge
firmware than the existing branch already expects. The latest official compiled
"n16" binary declares a different checkpoint from the public Bigv2 ONNX.

## Evidence labels and scope

- **Enforced:** an exact source check, tensor declaration or admission rule.
- **Tested:** executed in this development environment, with backend stated.
- **Source-compatible:** inspected code agrees; physical execution is untested.
- **Unknown:** neither public metadata nor observed hardware establishes it.

The [machine-readable evidence](evidence/navigation-model-compatibility.json)
records source refs, object identities, bounded retrievals and comparisons. No
public model pickle was executed. The downloaded ONNX was inspected using
protobuf and executed through the pinned tinygrad ONNX frontend in isolated CPU
experiments. The compiled official PKLs were inspected with `pickletools` and an
inert literal-only decoder of their leading metadata dictionaries. Four 4 MiB
prefixes were sufficient to contain those headers; full PKL files were not
downloaded or hashed locally. Their full SHA256 identities below are authoritative
git-LFS OIDs, not newly recomputed full-file hashes.

Primary upstream snapshots are commaai/openpilot
[`nav-model-integration` at afbf2e54](https://github.com/commaai/openpilot/tree/afbf2e547e9319e2eb9f515e1e4330af63705cec),
its five relevant commits, and SunnyPilot
[#2057 at f7f9a3ef](https://github.com/sunnypilot/sunnypilot/pull/2057/changes/f7f9a3efae3a6552068201bd9e4e4dbe66aae577).
These are experimental sources. They do not establish deployed model quality,
turn compliance, red-light behavior or a private training specification.

## Compatibility matrix

| Model / path | Exact source / artifact | Target and firmware | Runtime / compiler | Selector / catalog and status |
| --- | --- | --- | --- | --- |
| **Prioritized Nav Model Integration Bigv2** | Full ONNX SHA256 `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`, 814,591,106 bytes; official a03bb1d4 | Intended existing Chestnut path: AMD, LLVM, `gfx1200`, USB. Actual connected device not observed. Existing exact runtime product is `custom 8516b768-CLEAN`; no additional model-specific firmware dependency found | Branch tinygrad `69fd4d34d4430036378aba2b43f9d339cdd6ea73`; native `SPTGPLN1` compiler. Full CPU source/reload checks passed; no usable AMD artifact produced here | Separate branch-controlled native action contract; numeric SunnyPilot generation is **null**, not invented. Displayed through a dedicated opt-in extension; activation remains gated on reviewed physical results and exact model/warp/compiler hashes |
| Official Bigv2 compiled binary | PKL LFS OID `20cae5d01d0c59352fa6bf23cb0bf2a9c6c7257f6b9997652f6ef64669e194f8`, a03bb1d4; checkpoint matches the ONNX | Official source selects `USB+AMD:LLVM` for Chestnut; complete binary kernels/target not decoded in this audit | Official tinygrad `d3f09c9bbd542fbfbe68c8569a1173550c408969`; TinyJit/OOB pickle format | Not a SunnyPilot native artifact. Cannot be renamed into `SPTGPLN1` or admitted based on the file extension |
| Official latest "n16 nav-wm" binary | PKL LFS OID `52205044e136efb64a6a841c9b016cf35b1db81186dd0f941b62eee5ebdcde24`, afbf2e54; **different checkpoint** | Same checked-in Chestnut USB/product expectation | Same official pinned runtime snapshot; actual binary behavior untested here | Neither its title nor matching shapes proves SunnyPilot generation 16. No matching public ONNX was established; recompiling Bigv2 does not reproduce this checkpoint |
| Earlier navigation model v1 / overflow update | ONNX SHA256 `6ea13501cd6682321ee303c2fe425667988fa43a8aa326d35845602e22e4460d`; compiled OIDs b964531e / 17c06f92 | Same experimental runtime family; not physically tested | Source recipe compatible in principle; input/state contract differs from Bigv2 | Research reference, not the current admitted candidate; do not reuse v2 shapes blindly |
| Small non-navigation CPU reference | Unchanged baseline ONNX `65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce`; local safe CPU SPTG `53ddb5500429688d9b1d0fe256bee0cfc879b74926dde10ff00e8e580282b11f` | CPU/LLVM reference only; no AMD artifact or physical test | Actual clean pinned 69fd; two seeds, all four outputs finite and source/reload-exact | File-only diagnostic fallback/reference. No numeric generation assigned from this audit; not a replacement for the user's ordinary installed fallback |
| Existing ordinary Chestnut models | Existing v28 bundles and their own hashes remain the source of identity | Existing product/driver/warp behavior preserved | Existing pinned runtime and artifact loading preserved | Selector **21** and `driving_models_chestnut_v28.json` retained. A catalog version check alone is insufficient to approve the nav candidate |

`gfx1200` is derived from the JSON target descriptors of the already-tracked
working Chestnut warp, not from the navigation model's name. That warp's SHA256
is `f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b`.
The hardware package must still detect the actual attached device and compare
its target. No `/dev/dri`, `/dev/kfd` or USB device passthrough was available in
this environment.

## Exact checkpoint lineage and official compiled variants

The official branch adds the following five commits. The wiring commit changes
the consumer; the others also show why model identity cannot be inferred from a
filename or a later commit title.

| Official commit | Change and compiled artifact identity | Declared checkpoint |
| --- | --- | --- |
| [a52113ee, nav driving model](https://github.com/commaai/openpilot/commit/a52113eea9a81c79c9309dbc379f0ce25ffa943d) | v1 ONNX `6ea13501...`; PKL `b964531e23471f2c18f31a5636901e56edebe39d48248a3b7bfda9d58580df70`, 806,702,824 bytes | `cf8f6120-49ee-48f8-b2d7-d663a4708271/56320` |
| [8b8ce13b, modeld wired](https://github.com/commaai/openpilot/commit/8b8ce13b407ddae14f2e415afad9c16f603bdb81) | Model input/packing/nav wiring; no new compiled-model pointer | Consumer implementation, not another model checkpoint |
| [3ec09aed, fix overflow inputs in map model](https://github.com/commaai/openpilot/commit/3ec09aed2ea589cbd3234137ae6fb74aecde3c4c) | Only PKL pointer changes: `17c06f9208ab9d1529b52589662af85734425a482e5972d342f676e5d6597f56`, 806,704,311 bytes | Same as v1; **entire leading metadata dictionary identical** |
| [a03bb1d4, rl nav driving model (80eedcdd/12864)](https://github.com/commaai/openpilot/commit/a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9) | Prioritized Bigv2 ONNX `1d18c42e...`; PKL `20cae5d01d0c59352fa6bf23cb0bf2a9c6c7257f6b9997652f6ef64669e194f8`, 822,465,506 bytes | `cf8f6120-49ee-48f8-b2d7-d663a4708271/56320/80eedcdd-f97b-46d0-9b94-780685292acf/12864` |
| [afbf2e54, n16 nav-wm model](https://github.com/commaai/openpilot/commit/afbf2e547e9319e2eb9f515e1e4330af63705cec) | Only PKL pointer changes: `52205044e136efb64a6a841c9b016cf35b1db81186dd0f941b62eee5ebdcde24`, 822,465,506 bytes; ONNX pointer stays `1d18c42e...` | **`2931d3d9-7c90-4988-a6af-654ed71e0af5/56320/96244b1f-6c4e-4adc-848d-eccd0b44b182/12800`** |

For the overflow fix, same declared checkpoint and shapes are supported; the
precise overflow mechanism or compiler flags are not explained by its pointer-only
diff. For the n16 update, the embedded checkpoint explicitly changes. Its input
and output shapes and slice metadata remain identical to Bigv2, so matching those
alone cannot establish equal weights or behavior. There is no numeric SunnyPilot
generation in any of the four inspected metadata dictionaries.

The repository's
[LFS configuration](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/.lfsconfig)
uses `https://huggingface.co/commaai/openpilot-lfs.git/info/lfs`. A GitHub media
404 is insufficient evidence that an official compiled object is unavailable.
The bounded prefix inspection used that configured service, with HTTP 206
responses and total object sizes matching the pinned LFS pointers. It establishes
header identities, not full-file content verification or numerical equivalence.

## Complete Bigv2 tensor and parser contract

The downloaded public release's full digest matches the
[official a03 ONNX LFS pointer](https://github.com/commaai/openpilot/blob/a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx).
The static signature, checkpoint and complete slices are retained in
[`nav_models.json`](../../openpilot/selfdrive/navigation/tests/fixtures/nav_models.json).
ONNX IR is 10, standard opset 20 and `org.tinygrad` opset 1; the custom op is
`Contiguous`. These describe the frontend contract, not GPU certification.

| Input | Dtype | Exact shape | Interpretation |
| --- | --- | --- | --- |
| `new_img` | uint8 | `[2,6,128,256]` | Two cameras, each six packed-YUV planes; the leading 2 is not two historical timesteps |
| `desire` | float32 | `[8]` | Rising-edge desire pulse, handled by the existing consumer |
| `traffic_convention` | float32 | `[1,2]` | Left/right traffic convention |
| `action_t` | float32 | `[1,2]` | Lateral and longitudinal prediction horizons, seconds |
| `nav_map` | float32 | `[1,1,256,256]` | Raw 0..255 grayscale road/route raster; **no host `/255`** |
| `navigation` | float32 | `[1,47]` | 46-way Valhalla maneuver code plus distance/100 |
| `state_img_q` | uint8 | `[2,5,6,128,256]` | Persistent image history |
| `state_desire_q` | float32 | `[132,1,8]` | Persistent desire history |
| `state_feat_q` | float32 | `[128,1,16384]` | Persistent vision features |
| `state_map_features_q` | float32 | `[128,1,64]` | Persistent encoded map features |
| `state_navigation_features_q` | float32 | `[128,1,64]` | Persistent encoded maneuver features |

Each of the five `state_*` inputs has a same-shaped, same-dtype `next_state_*`
output. The primary output is float32 `[1,18580]`. The graph drops the first queue
entry and appends the current value on every inference invocation: image queue
axis 1, all other queues axis 0. At 20 invocations/second this corresponds to
0.25 seconds of previous image entries, 6.6 seconds of desire entries and
6.4 seconds of feature entries. This is an execution-contract calculation, not
an independently recovered training cadence. **132 accepted frames** are needed
to replace all longest-queue entries after a complete reset; 128 is insufficient
for the desire queue.

The graph itself divides `nav_map` by 255 before its encoder. The upstream
renderer uses a 256×256 grayscale route-aligned raster: background 0, route 128,
roads 255, ego at `(128,192)`, one meter/pixel. Its helper's navigation vector is
46 one-hot Valhalla command slots plus remaining distance divided by 100 meters,
using the next eligible forward maneuver. This is more specific than arbitrary
RGB map screenshots, turn text or a normalized distance clamped to 0..1.
See the pinned upstream
[renderer](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/navigation/renderer.py#L1)
and [navigation helper](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/navigation/helpers.py#L15).
The implementation is evidence of the intended raster/input interface; actual
private training data and route adherence were not verified.

| Flat output slice | Bounds | Consumer meaning |
| --- | --- | --- |
| `lane_lines` / `lane_lines_prob` | `0:528` / `528:536` | Lane geometry and probabilities |
| `road_edges` / `meta` / `desire_pred` | `536:800` / `800:855` / `855:887` | Road edges and ancillary heads |
| `pose` / `wide_from_device_euler` / `road_transform` | `887:899` / `899:905` / `905:917` | Motion/calibration heads |
| `plan` | `917:1907` | 33×15 means plus corresponding uncertainty values |
| `lead` / `lead_prob` / `desire_state` | `1907:2051` / `2051:2054` / `2054:2062` | Leads and desire probabilities |
| `action` | **`2062:2066`** | Two action means, then two uncertainty/log-scale values |
| `hidden_state` | `2066:18450` | 16,384 vision features |
| `map_features` / `navigation_features` | `18450:18514` / `18514:18578` | 64 map and 64 maneuver features |
| `pad` | `18578:18580` | Padding |

The official parser leaves the raw four-element `action` slice available; its
consumer reads the first two entries. SunnyPilot's native parser MDN-decodes
that slice to those same two means. Geometry and action interpretations must
be compared after their appropriate parser, not by assuming every raw slice is
already a normalized prediction. The plan uses 33 quadratically spaced times
from 0 to 10 seconds and 15 channels: position, velocity, acceleration,
orientation and orientation rate, each three components.

## Generation, action units and timing: resolved without an invented number

SunnyPilot's ordinary catalog supplies an integer `generation`; its current
consumer uses it for legacy feature/plan conversion and low-speed smoothing
decisions. Public Bigv2 metadata contains a checkpoint and output slices, but no
such integer. Catalog v28, Selector 21, model names and "n16" are separate labels.
Assigning 12 or 16 would fabricate a requirement.

The branch's separate candidate registry therefore records
`sunnypilot_generation: null` and an explicit native contract:

```text
comma-nav-v2-20hz-lateral-acceleration-longitudinal-acceleration
```

The source-backed contract is tied to exact ONNX SHA `1d18c42e...` and upstream
a03/afbf consumer behavior, instead of weakening the ordinary catalog's generation
checks. The [branch registry](../../openpilot/selfdrive/navigation/driving_models.json)
marks software review separately from physical review, activation approval and
compiled model/warp/compiler-metadata hashes. A local diagnostic manifest or
checkbox cannot approve those missing branch-controlled identities.

The official [action conversion, lines 47–78](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/modeld/modeld.py#L47)
explicitly uses:

```python
desired_acceleration = action[1]
desired_curvature = action[0] / max(1.0, v_ego)**2
```

The consumer thus interprets action 0 as lateral acceleration in m/s², action 1
as longitudinal acceleration in m/s², and desired curvature as 1/m. These units
are established by the public consumer's conversion and messages; this is not
independent evidence that the learned model is physically calibrated on this
Honda. Official smoothing is lateral 0.0 seconds, longitudinal 0.3 seconds;
previous curvature is held at speeds at or below 0.3 m/s. The explicit native
contract must preserve that hold even though numeric generation is null.

Official [model constants, lines 8–18](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/modeld/constants.py#L8)
declare `MODEL_RUN_FREQ=20`, while `MODEL_CONTEXT_FREQ=5` is annotated
`model_trained_fps`. These are distinct. The ONNX metadata has no independent
training-FPS declaration. The runtime must invoke this native graph at 20 Hz;
its internal queue graph owns history updates, without adding legacy external
queue decimation. Physical sustained 20 Hz performance remains unverified.

Official [action horizons, lines 403–410](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/modeld/modeld.py#L403)
add one `DT_MDL` plus half `DT_MDL` to the respective lateral/longitudinal delay.
`DT_MDL` is 0.05 seconds. The existing Honda firmware prediction-delay hook
continues to supply its lateral component; there is no reason to replace EPS
gains, torque conversion or the MVL planner to consume this contract. Mode-specific
longitudinal use is documented in the [final audit](navigation-model-final-audit.md):
a model acceleration is not an unconditional braking command in every ACC/DEC mode.

## Firmware, USB protocol and architecture

The exact branch and official nav snapshot both expect:

- Bridge firmware identifier **`8516b768`**; product **`custom 8516b768-CLEAN`**.
- Runtime USB IDs `add1:0001` or `3801:0001`; recovery/ROM IDs `174c:2464`
  or `174c:2463` are a different mode, not a successful runtime connection.
- Identical bundled `firmware_wrapped.bin`, 9,235 bytes, SHA256
  **`6fdb1ffc2ee9206ba2bc95a384d88b3346dac47a0524d54a34a5a14cb50132fb`**.

Sources: exact baseline
[`common/hardware/usb.py:4–7`](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/common/hardware/usb.py#L4),
[`chestnut_present():22–31`](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/selfdrive/modeld/helpers.py#L22),
and the corresponding [official USB constants](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/common/hardware/usb.py#L4).
`chestnut_present()` requires both an accepted runtime USB ID and the exact
product string. This is the strict application-level firmware identity check.

The lower-level tinygrad USB driver accepts product prefixes `custom` or
`AS2462`, claims/configures the USB interface, and uses control writes request
`0x40`, control reads `0xc0`, bulk out endpoint `0x02` and bulk in `0x81`.
Merely opening that driver is **device I/O**, not a read-only identification
probe. Use sysfs for initial identity collection. The driver source files
`runtime/support/usb.py`, `runtime/support/system.py`, `runtime/support/am/amdev.py`
and AMD firmware structures `runtime/autogen/am/fw.py` are byte-identical between
official d3f09 and pinned 69fd; evidence includes their hashes. Pinned69fd's
`ops_amd.py` additionally CPU-maps small aligned USB allocations (up to 64 KiB),
so the entire AMD runtime is not byte-identical.

Pinned runtime source:
[`usb.py:45–83`](https://github.com/tinygrad/tinygrad/blob/69fd4d34d4430036378aba2b43f9d339cdd6ea73/tinygrad/runtime/support/usb.py#L45),
[`ops_amd.py:792–814`](https://github.com/tinygrad/tinygrad/blob/69fd4d34d4430036378aba2b43f9d339cdd6ea73/tinygrad/runtime/ops_amd.py#L792),
and [`helpers.py:514–526`](https://github.com/tinygrad/tinygrad/blob/69fd4d34d4430036378aba2b43f9d339cdd6ea73/tinygrad/helpers.py#L514).
AMD GPU microcode loaded by the runtime is distinct from persistent ASM2464
USB-bridge firmware. Its existing linux-firmware fallback is pinned to
`0a6871b19abf5d6e024b5d208b101ae53e7fa0de` with validation; this model audit does
not introduce another microcode or bridge-firmware requirement.

The earlier user-reported `custom ed4e39b7-CLEAN` is historical information.
Subsequent successful big-model driving does not reveal today's literal USB
product string; that must be measured from the actual device. No navigation-model
specific protocol/version dependency or reproducible hardware result justifies
changing firmware here.

The baseline already has an offroad mismatched-product firmware recovery path
in [`hardwared.py:43–83`](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/system/hardware/hardwared.py#L43).
It launches the existing flasher with up to three attempts, 20 seconds apart.
This project does not change that behavior or invoke it. File-only diagnostics
must not start manager, hardwared or the flasher. No firmware was flashed.

## Runtime, compiler, warp and artifact preparation

The actual branch tinygrad pin remains **69fd4d34**; official nav uses **d3f09c9b**.
The exact `examples/openpilot/compile_onnx.py`, `helpers.py` and `compile_warp.py`
files at those revisions are byte-identical. Their hashes are respectively
`512571a84dc7931a07713a81b342e5087c41b173edea2d9a3b209fb83c5ab340`,
`65bb91045ce78a29d0e240815629da0f3a04eda03729e8ff2c33b214cf064bd9`, and
`97a6814fde7f16e1ce265c6b6388cd5799c8e2f8969ac78cfc4fd5a929b1921d`.
This removes a missing frontend recipe as the presumed blocker. Lower-level
optimization, compiler and allocator changes still prevent a claim of complete
runtime numerical parity.

The official recipe uses `OPENPILOT_HACKS=1`; its AMD setup sets `TC_OPT=2` and
`TC_MIN_GLOBALS=32`. The native compiler's sidecar records actual environment,
targets, runtime commit/dirty state, source digest, deterministic seeds,
tolerances and per-output finite/difference results. Changing precision or
backend flags invalidates the claim that a previously reviewed artifact was
tested. See the pinned
[official compile recipe](https://github.com/tinygrad/tinygrad/blob/d3f09c9bbd542fbfbe68c8569a1173550c408969/examples/openpilot/compile_onnx.py)
and [AMD recipe helper](https://github.com/tinygrad/tinygrad/blob/d3f09c9bbd542fbfbe68c8569a1173550c408969/examples/openpilot/helpers.py).

SunnyPilot's existing native compiler was introduced by
[#2050 / 6d51c892](https://github.com/sunnypilot/sunnypilot/commit/6d51c8926baa32769f4d5c74d9727bdcd3c327c8).
PR2057 adds model navigation feeds and dtype-aware packing; it does not supply a
new nav-specific compiler frontend. The old generic `compile_modeld.py` assumes
rank-three desire and fails on Bigv2's rank-one `[8]` input. The correct path is
`sunnypilot/modeld_v2/compile_modeld_v2.py`, not a manual reshape to satisfy the
legacy compiler.

The native compiler produces **`SPTGPLN1`**, with JSON target/kernel descriptors
and raw blobs. Official PKLs contain executable TinyJit pickle/OOB objects.
The existing upstream-PKL converter executes its input and is not a safe metadata
inspection tool; it was not used on downloaded public objects. Diagnostic nested
slice metadata uses a restricted slices-only decoder and checks all bounds before
allocation. Renaming a PKL does not convert formats.

Bigv2 `new_img=[2,6,128,256]` requires a matching two-camera warp to 512×256
before packed YUV. For comma4's 1344×760 NV12 input, the copied source extent is
`1408 × (768+384) = 1,622,016` bytes per camera. The larger allocation extent
is not the copied frame extent. The matching warp must use the actual model
backend/architecture/interface; a CPU warp pass is not a GPU warp pass.
Typed packed inputs retain declared dtypes and 128-byte alignment, and all
`next_state_*` outputs must alias the intended persistent state buffers.

### What actually passed and what did not

The actual full Bigv2 CPU compile at clean pinned69fd completed in 375.21 seconds.
Independent source and reloaded native-artifact execution used seeds 42 and 43,
full-range uint8 inputs, raw-scale maps, `rtol=atol=1e-3`, `equal_nan=False`.
All six output/state tensors were finite and had maximum absolute difference 0;
same-input source repeatability was exact and prediction changed across seeds.
This is consistency of one frontend/backend across serialization, not an
independent ONNX-runtime oracle or CPU-versus-GPU comparison.

The original certified CPU SPTG SHA is
`df458e202b29deede75c3cc4be8aa9b4907a592b95b7cebd25d1714364a539aa`.
A metadata-only safe-slices derivative has SHA
`a280f8873a0391685bf4ea115504203b7448ce5a034b401788cc1e2921384790`;
its kernels, constants and tensor specs were preserved, but the original
certificate's artifact hash must not be relabeled as the derivative's.
These large scratch artifacts are not committed or installed.

An AMD `gfx1200` LLVM mock/emulator attempt completed a small arithmetic
preflight. On the full graph, it stopped at kernel 6 after its CPU-lifted emulator
generated a clang `__builtin_bit_cast` between unsigned-short and float of unequal
size. No complete AMD SPTG or usable learned-model GPU result was produced.
That is an **emulator/compiler-path limitation**, not evidence of physical
Chestnut incompatibility or a model-specific firmware requirement. Native
hardware compilation, source/reloaded checks, repeated inference, thermal/power
behavior and CPU-reference parity remain outstanding physical tests.

## Catalog, selector and fallback are separate contracts

The existing selector enforces **equality** with
`REQUIRED_JSON_VERSION=21` against each bundle's `minimumSelectorVersion`; it is
not a greater-than-or-equal comparison. The fetcher separately parses `generation`,
`runner`, `is_20hz` and overrides. Chestnut's URL remains
`driving_models_chestnut_v28.json`; ordinary QCOM uses v25.

Sources on the untouched working baseline:
[`helpers.py:18–55`](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/models/helpers.py#L18),
[`fetcher.py:53–73`](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/models/fetcher.py#L53),
and [`fetcher.py:115–116`](https://github.com/ZakariaSaad/ZakariaSaad/blob/4851eacac8fce2e875a0ef7703af5f5025fc0c15/openpilot/sunnypilot/models/fetcher.py#L115).

The retained [v28 catalog fixture](../../openpilot/selfdrive/navigation/tests/fixtures/chestnut_v28_baseline.json)
declares top-level `tinygrad_ref=5d186bdb43c2bcc6c56b6ececf427c34bada0de9`,
while this branch actually runs69fd. The fetcher does not enforce that top-level
field. Existing generation12 entries describe other named models. Neither the
catalog's filename, an ignored runtime hint, nor another entry's generation
certifies this new candidate. No public external catalog was edited.

The scoped branch extension separates Navigation Alpha display, diagnostic
selection and optional navigation-conditioned driving. It leaves ordinary
model bundle keys, downloads and persistence intact. Default-off updates do
not select or enable Bigv2. Source integration can be complete while this
specific candidate is ineligible: branch-controlled physical review/activation
flags and model/warp/compiler-metadata hashes remain unset until actual results
are reviewed. A locally generated diagnostic report cannot promote itself.

A usable normal fallback is an independently configured, installed ordinary
non-navigation model on the appropriate backend with its own camera/outputs
contract. No evidence requires it to share an invented Bigv2 generation number.
The actual diagnostic CPU fallback source is the unchanged baseline small ONNX,
checkpoint `6a7d09ad-bcc9-43bc-916d-29287e60cee2/200/a27b3122-733e-4a65-938b-acfebebbe5e8/100`.
Its clean pinned69fd CPU compile took49.74 seconds; both seeds/all four outputs
were finite and source/reload-exact. This establishes a CPU reference, not
physical GPU recovery or safe vehicle transitions. Zero nav inputs or finite
outputs after clearing history are not themselves validated driving fallbacks.

## Outstanding physical acceptance

Before this exact candidate may be approved for ordinary opt-in driving, the
file-only hardware package must record the actual USB product and GPU target,
compile native model and matching warp on the actual pinned runtime, verify
independent source/reloaded outputs, run bounded and full-history sequences,
measure timing/memory/USB/power/temperature, and compare equivalent prepared
camera/state/navigation inputs with the CPU reference using declared tolerances.
Transition/reset/fallback behavior also requires review; finite arrays and a
nonzero A/B delta are insufficient acceptance criteria.

The required package must run in a separate diagnostic checkout, offroad,
with production GPU inference stopped and file inputs only. It must not start
manager, CAN, controllers, prediction publishers or firmware operations. No
physical-device result was supplied during this audit. The user's previous
successful ordinary big-model drive does not validate this new checkpoint,
its different recurrent queues, map inputs or AMD compile.

No firmware, submodule pin, public catalog, steering gain, longitudinal tuning
or vehicle setting was changed by this compatibility audit.
