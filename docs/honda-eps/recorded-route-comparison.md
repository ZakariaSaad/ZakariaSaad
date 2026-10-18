# Recorded Insight controller input replay

Route: [`f133facb1b9b7420|00000032--212d34d24d`](https://stable.konik.ai/f133facb1b9b7420/00000032--212d34d24d).
After the completed upload, the documented Konik route-files API provided
**all 58 rlogs and 58 qlogs, segments 0–57**. All rlogs downloaded successfully,
537,389,320 bytes in total. Only logs were
downloaded; no cameras were requested. Local raw logs, extracted inputs and
signed access URLs remain outside the repository. Aggregate results and every
downloaded log's SHA-256 are in [recorded-route-comparison.json](recorded-route-comparison.json).
This complete-route analysis supersedes the earlier four-rlog analysis. Its
original per-segment results reproduce exactly, but its zero integer-mismatch
claim does not hold across the larger dataset.

## Recorded source and owner settings

The logged clean source commit is **`f3a6a124bdad3151f875f4f29c8d7396c7617702`**,
branch `ns-bosch-radar-testing`, exactly the corrected Insight calibration
commit requested for this port. CarParams identify `HONDA_INSIGHT`, the EPS
identifier `39990-TXM,A040`, the source PID gain tables and the `[0,3840]`
transport map. All recorded EPS debug messages identify `insight_txm_a040`.
The part number still cannot prove which firmware revision is installed.
An independent header scan found matching source/settings and EPS calibration in
all 58 rlogs and matching CarParams in segments 0–56. Truncated segment 57 contains
no CarParams event; it has all replay signal types but only 322 inactive controls
frames, so its CarParams cannot be independently verified.

The original source at this exact commit was downloaded for the comparison;
the report records its core, PID, filter, rack and firmware-angle-table hashes.
The relevant Insight calibration and algorithms agree with the later `bfcbbf24`
reference used by the deterministic comparison. Its changes to other vehicle
calibrations and startup provenance logging do not change the Insight law.

Whitelisted `initData` settings establish:

| Setting | Recorded value |
| --- | --- |
| Firmware controller | Enabled |
| Output LPF | Enabled, **0.1 s in all three speed bands** |
| Desired-angle rate limit | 300 degrees/s |
| Driver override fade-down / fade-up | 0.1 s / 1.5 s |
| Driver override torque scale | 0 |
| Raw driver override threshold | **2400** |
| Learned angle offset / steering ratio / stiffness | Enabled |
| Automatic steering delay | Enabled |

The `route32` response preset explicitly selects the recorded 0.1 s output
filtering and 0.1 s override fade-down. The `source` preset retains the source
defaults, 0.07 / 0.05 / 0.01 s output filtering and zero fade-down. No gains or
calibration constants were adjusted to fit this route. The response preset does
not change geometry or make MVL's vehicle interface identical to James's.

## Executed replay and reconstruction

The 58 rlogs contain **340,869 recorded controls frames**, including 313,504
active frames before replay exclusions. Each segment is kept
separate; gaps between segments are never interpolated. At the first active
frame after an excluded five-second opening interval, each core receives a
single initialization of its observable integral, feedforward, join ramp and
output-filter states from the recorded state. Subsequent validation frames
receive **no further logged-state injections**. The short final segment is
inactive; no missing or truncated messages are synthesized.

The source code, its actual PID and filter, and the port consume reconstructed
100 Hz inputs: desired wheel angle, measured wheel angle, speed, angle offset,
roll, recorded raw driver press and inferred delivery limitation. CarParams'
serialized float32 gains are used by the source; the port keeps its explicit
source gain constants. The resulting tiny precision difference is reported.

The delivery-limit proxy must use the **previous tick's request and delivery
snapshot**: original controlsd calculates this proxy after the controller update
and consumes it on the next tick. Applying the current delivery snapshot froze
the wrong integration frames. The replay now preserves the source ordering,
with a regression test for the one-tick delay.

After initialization, **287,710 active frames** are compared. The reconstructed
original source is close to the commands actually recorded by that source:

| Coverage | Compared active frames | Source reconstruction vs logged core, RMS E4 | Maximum absolute E4 error |
| --- | --- | --- | --- |
| **All 58 rlogs** | **287,710** | **0.07018** | **2.94302** |

Per-segment summaries and hashes are retained in the JSON report. The combined
95th percentile absolute error is **0.12731 E4 counts**. The maximum reconstruction
error is below three E4 counts. This is a comparison against actual
logged controller requests, not just a claim that two implementations execute.
Remaining differences reflect float32 logs and approximate SubMaster snapshots;
this is not a bit-exact replay of the whole source process.

## Physical command comparison

All values below use equivalent **physical E4 counts before carcontroller
delivery**. The port's complete canonical output is converted by 3840/4096;
MVL's transport map and safety limits remain unchanged.

| Implementation / response | RMS difference from reconstructed original source E4 | Maximum absolute difference E4 |
| --- | --- | --- |
| Port, explicit `route32` response | **0.00002543** | **0.00324710** |
| Port, default `source` response | 33.45521 | 634.53727 |
| Original NRDR source defaults | 48.69446 | 796.91591 |

The explicit `route32` core has **one truncated integer E4 request mismatch in
287,710 frames**, in segment 47. This replaces the earlier subset's zero-mismatch
claim. The continuous difference remains below 0.00325 E4 counts. Its numerical remainder
comes from the serialized source gain precision. The `source` response differs
because the owner recorded nondefault output filtering; that difference is
preserved and exposed as a setting.

Coverage by speed:

| Speed band | Active full-rate frames | `route32` vs source RMS E4 | Default `source` vs source RMS E4 |
| --- | --- | --- | --- |
| Below 25 mph | 124,500 | 0.00003038 | 43.98213 |
| 25–50 mph | 163,210 | 0.00002088 | 22.30285 |
| Above 50 mph | **0** | Not replayed | Not replayed |

The compared full-rate frames include **207 filtered driver override frames**
in segments 4 and 23, and zero logged saturation flags. Across the recorded
active frames, output reaches normalized magnitude 1.0 and 192 frames have
magnitude at least 0.999: zero diagnostic saturation flags does not mean the
request never approached its rail. All rlogs contain 4,049 raw `steeringPressed`
carState samples; raw presses are not equivalent to filtered overrides.
The maximum recorded speed is **20.897 m/s**, about 46.7 mph, so the highway gain
regime is still not covered by this route. Highway behavior remains covered by
separate deterministic/unit tests. The final rlog and qlog have incomplete
Cap'n Proto tails; complete preceding messages were retained and both warnings
are recorded. The qlogs remain summary data and are not upsampled for replay.

## Surrounding architecture and limits

The comparison supplies the **recorded desired wheel angle**. It isolates the
firmware inversion and residual feedback while keeping geometry, model inference
and command timing outside the core comparison. At the same reconstructed path,
MVL's conversion using this route's recorded learned ratio and stiffness differs
from James's rack by **6.582 degrees RMS**, **74.419 degrees maximum**, and
0.459 degrees at the 95th percentile. The large differences occur in slow,
large-angle turns: the maximum is at **1.696 m/s**, with James requesting
**−339.616 degrees** and MVL requesting **−414.035 degrees** for the same path.
At 25–50 mph, the difference is much smaller: **0.087 degrees RMS** and
**0.506 degrees maximum**. For desired-angle magnitude above 150 degrees, RMS
difference is 40.056 degrees. These outliers were absent from the earlier four
segments. Default `mvl` geometry therefore does not reproduce the source's
tight-turn conversion; the optional `james` mode selects the original rack.
Core-command parity with recorded desired angles does not establish geometry
parity. Either mode still requires controlled vehicle validation.

James's logged command delay ranges from 0 to 0.15 s. The port retains NRDR's
additional 25 ms SunnyPilot interpolation compensation, using 0.025 to 0.175 s.
The source EPS debug `epsModelDelay` ranges approximately 0.180–0.309 s and
includes controlsd's lateral smoothing interval in addition to its scheduled
model delay. Matching core commands here does not validate that surrounding
prediction timing in the vehicle.

The recorded raw driver threshold is 2400. At the owner's request the admitted
firmware mode now uses that same 2400-count threshold for published CarState and
actuator state; disabled fallback retains MVL's original 1200 threshold. This
replay consumes recorded booleans, so separate packed-CAN boundary tests validate
the live threshold hook. The newly available override frames exercise the core's
override/integrator behavior, but this core-input replay does not independently
validate the entire delivered fade-down sequence. CAN command and Panda
restrictions remain unchanged.

Logged delivered CAN commands are reported separately from core requests. They
include the original carcontroller's rate limiting, override delivery and sample
timing. The replay's integer request parity does not imply identical delivered
CAN in MVL's surrounding architecture. It does not execute full controlsd/modeld,
run a closed-loop plant, measure perception feedback, validate firmware in the
vehicle, or establish port road validation. Controlled vehicle validation is
still required before ordinary use.

## Reproduce

Use the original source checkout at `f3a6a124` and the NRDR PR #22 reference. Put
the downloaded logs in `<route-dir>/<segment>/rlog.zst` and `qlog.zst`; the tool
performs no downloading and never uses signed URLs. Keep extracted inputs in a
private workspace, outside the repository.

```sh
python -m openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.route_replay \
  --route-dir /path/to/private-route-logs \
  --james-root /path/to/james-at-f3a6a124 \
  --nrdr-root /path/to/nrdr-at-5777cf9 \
  --extracted /path/to/private-work/extracted-inputs.json \
  > recorded-route-comparison.json
```

`--schema-root` can point to a separate complete James checkout when the
comparison root contains only the pinned pure source files. `--refresh-extraction`
decodes the local logs again. Reports publish only aggregate measurements,
whitelisted controller settings and content hashes.
