# Insight stock-ACC Bosch-A observation recording

This is a separately selected, receive-only logging configuration. It does not
select the longitudinal integration or the stop-command guard. Both observation
and longitudinal integration default OFF. No car deployment, parameter change,
CAN transmission or actual parked recording was performed during preparation.

Use main testing branch **`insight-bosch-a-observation`** and its exact pinned
dependency on **`insight-bosch-a-observation-opendbc`**. Initialize submodules at
the parent gitlinks, without `--remote`. The observation branch descends from
main `2a6cedd77ea3338ec6e21a3bb2d8c219f12e6163` and dependency
`0651ffc04fc463899568158ee56932ef5d11873f`; the completed observation commit IDs
are reported separately with the final verification receipt.

The stop-command proposal is preserved independently on local branches
`insight-bosch-a-stop-guard-proposal` (`a34f06d291ef1bbdd0ae4af40612eb7bb7041b58`)
and `insight-bosch-a-stop-guard-proposal-opendbc`
(`b6ebf9e9c0d2f749fff17d8a8cdc470cfe278280`). Its controller changes and parameter
are absent from this observation branch. Do not merge or select it for this task.

## Parser revision and newer changes

The decoder, DBC and wrapper are byte-identical to NRDR opendbc parser PR #3
`461f81cddfa96be4fbf2b9c01e821790f8110f27`, PR #4
`427f1986ed241f111d106cc72cdefbda5efe5f7f`, and the current development head
`f3e0e2904f388bdf17d7779e7f9ad6191062e2ad`, verified on 2026-11-10. The newer
dependency commits concern Civic longitudinal/PCM/controller/safety behavior;
they are not parser prerequisites and were not imported.

| File | SHA-256 recorded by the observer |
|---|---|
|`bosch_a_radar.py`|`50547fc509fb80be95e410d32d9dd57c9591accbb249f1975ade0588498da5be`|
|`honda_bosch_a_radar.dbc`|`36646473a8058056decaa65a198536f81f8e82754d71ee79a427a4b93ad1e107`|

This is a statement about those source files, not the entire opendbc tree. The
target retains its baseline generic CAN-parser API. NRDR openpilot development
`622a1479bd27598a949d5ca601e486380b0369a0` pins the newer dependency; its parked
Honda tuning change affects UI/cached configuration, not this logging task.

## Exact configuration, before any future parked check

These settings are a reviewable preset, not settings applied by this work. Apply
them only offroad after choosing the observation source revision, then restart
normally so startup-latched configuration is rebuilt. An old prebuilt `loggerd`
does not know the new service: a fresh normal build must include cereal schemas,
generated services, `loggerd`, and the new Params key. Do not copy cloud x86
binaries onto the car; use the repository's normal device build procedure.

| Parameter | Value | Purpose |
|---|---:|---|
|`BoschARadarObservationEnabled`|1|Select the isolated observation publisher|
|`BoschALongitudinalEnabled`|0|Keep the integration/planner path off|
|`AlphaLongitudinalEnabled`|0|Keep Honda factory ACC authority and normal stock safety configuration|
|`HondaEnforceStockLongitudinal`|1|Explicit stock selection|
|`IntelligentCruiseButtonManagement`|0|Keep stock cruise-speed authority; ICBM must not create longitudinal override|
|`OpenpilotEnabledToggle`|1|Run the normal stock-ACC/lateral software configuration; engagement is unnecessary while parked|
|`DisableLogging`|0|Do not suppress normal logging; this development key also clears at normal transitions|

No stop-command guard setting is needed: its implementation is absent. Cabin
video/audio and upload preferences need not change. Existing forward qcamera
recording and complete rlog logging are the required evidence.

The strict gate requires Insight/Honda, observation bit128, stock flag16, no
Bosch-A integration bit32, openpilot longitudinal false, `pcmCruise=true`,
`pcmCruiseSpeed=true`, no gas interceptor, no Honda `BOSCH_LONG` safety bit2,
normal HondaBosch safety, and no passive/dashcam/notCar configuration. It does
not silently repair inconsistent settings.

Two mistakes are specifically rejected: alpha-long ON followed by forced stock
can leave `pcmCruise=false` and the long safety bit set; ICBM can independently
set `pcmCruiseSpeed=false`. Merely seeing openpilotLongitudinalControl=false is
therefore insufficient. Final eligibility is rechecked after late setup.

After startup, read the effective configuration, not only saved preferences:

```sh
python - <<'PY'
from openpilot.common.params import Params
from openpilot.cereal import custom
from opendbc.car.structs import car
from opendbc.car.honda.bosch_a_observation import observation_enabled

p = Params()
raw, raw_sp = p.get('CarParams'), p.get('CarParamsSP')
if raw is None or raw_sp is None:
  raise SystemExit('Not ready: card has not published both effective configurations')
with car.CarParams.from_bytes(raw) as cp, custom.CarParamsSP.from_bytes(raw_sp) as sp:
  print('fingerprint:', cp.carFingerprint)
  print('observation_selected:', observation_enabled(cp, sp))
  print('openpilotLongitudinalControl:', cp.openpilotLongitudinalControl)
  print('pcmCruise / pcmCruiseSpeed:', cp.pcmCruise, sp.pcmCruiseSpeed)
  print('radarUnavailable:', cp.radarUnavailable)
  print('flagsSP:', int(sp.flags))
  print('safety:', [(str(c.safetyModel), int(c.safetyParam)) for c in cp.safetyConfigs])
  assert observation_enabled(cp, sp), 'Ineligible preset: do not proceed to a route'
PY
```

Expected: observation selected, openpilot longitudinal false, both stock cruise
fields true, and **radarUnavailable true**. The last value is intentional: normal
RadarInterface/RadarD remain on their stock path. Other unrelated Honda flag bits
may exist; do not require flagsSP to equal exactly 144.

## Parked checks before requesting another route

1. Keep the car secured and stationary with its normal ignition/software active;
   do not engage ACC/openpilot to prove recording. Check the effective startup
   configuration above. The observer is receive-only and sends no diagnostic
   wake-up or radar-disable commands.
2. Allow a complete normal log segment to close—roughly one minute, with useful
   steady capture after startup. `carParams`/`carParamsSP` are periodically logged
   at approximately 50-second intervals, so retain the segment containing them.
   A visible stationary target can make a measured-point check useful, but an
   empty scene is not automatically a decoder failure. Do not infer metric range
   truth from a visually apparent bumper.
3. Preserve **full rlog plus matching qcamera.ts** and their segment numbers.
   qlog alone omits modelV2 and heavily decimates raw CAN; it cannot validate the
   associations. The observation service is full-rlog only. Existing camera
   state and encode-index messages supply frame ID/EOF/ordinal pairing.
4. Run the local checker from the same observation source checkout using its
   normal Python environment. The files can be copied locally for this read-only
   check; the tool never contacts the car or a cloud route service:

```sh
python -m openpilot.tools.bosch_a.check_observation_log \
  /absolute/path/to/segment0/rlog.zst \
  --qcamera 0=/absolute/path/to/segment0/qcamera.ts \
  --require-measured-points \
  --output /absolute/path/to/parked-observation-check.json
```

Repeat positional rlog paths and `--qcamera N=FILE` for additional complete
segments from the same startup. Drop `--require-measured-points` only when no
known target is expected; that permits an empty but correctly recorded stream.
The video check uses local `ffprobe` to count decodable frames. It verifies that
the referenced model/camera index maps into a supplied file; it does not identify
the target or certify picture quality by itself. Inspect the corresponding
images as well.

5. Inspect the JSON result, counts, errors, timing gaps and scene. By default
   the checker requires a **contiguous 10-second healthy parked window** with
   valid decoded samples, exact carState joins reporting standstill, healthy
   card model reception, valid raw CAN and camera/index envelopes, exact
   camera/index/video pairing, and inter-sample
   gaps no larger than the decoder's 0.20-second stale boundary. Ten seconds is
   an explicit diagnostic capture criterion, not a calibration or vehicle-safety
   threshold. `--min-healthy-seconds` can change it for tool tests; retain the
   default for the actual parked check. `pass`/exit0 means those recording
   checks succeeded; it is not approval to drive or proof of uninterrupted
   healthy operation outside the reported window. `fail`/exit1 means a
   structural/configuration/authority contradiction. `incomplete`/exit2 means
   required evidence is absent, insufficient or unreadable. Two valid samples
   followed by prolonged loss cannot satisfy the default check. A latched
   observer exception, missing sample sequence, incomplete video index or missing
   source join is also incomplete. Review reported decoder-call duration and
   carState cumLagMs for timing burden, without interpreting them as physical
   sensor latency. Resolve material errors/gaps and verify useful stationary
   data before requesting another route.

The checker verifies selected strict stock parameters; no longActive;
unchanged empty normal radarTracks/control-status carrier; observed parser/DBC
hashes; positive sample identities and timestamps; exact sample/status pairing;
measured versus coasted counts (the known-target option requires valid measured
records); exact carState publication and ego speed used; recomputed staleness;
raw CAN/trigger references on the camera-side
bus; exact model publication/frame/EOF matches; camera-state and qcamera index
matches; and absence of outgoing `ACC_CONTROL`, `ACC_CONTROL_ON` or Honda radar
CommunicationControl-disable intents in the supplied recording. Received stock
ACC CAN, normal button messages and identification reads are not mistaken for
openpilot longitudinal actuation. Absence in one recording is not cross-run byte
equivalence, physical bus acknowledgement or proof of ECU application.

The decoder's nominal 14.35 Hz and 0.20 s stale boundary are source policy, not
an independent Insight hardware specification. The checker reports measured
cadence rather than imposing an invented tight frequency tolerance. `aRel` and
`yvRel` are unknown NaNs in the pinned output and are not rejected; measured is
the decoder's `points[].deprecated.measured` acceptance flag, not ground truth.

## New evidence and remaining blockers

The new `boschARadarObservation` service records the **actual online decoder
result and its receiver context**, alongside unchanged raw CAN and camera logs:

- decoded points, measured/coasted flags, error bits, sample validity and exact
  status snapshot pairing, including discarded stale status IDs;
- source CAN batch and trigger timestamps, host receipt/decode/publication times,
  sample sequence, heartbeat/staleness and observer exceptions, plus the exact
  carState publication and ego speed supplied to the decoder;
- exact parser/DBC hashes and selected parser bus;
- the model publication/frame/EOF seen by card, with card's actual receipt time,
  valid/alive/frequency flags, rather than offline delivery assumptions;
- logged stock-control configuration and command intents for authority checks.

The observer filters cached statuses to the IDs actually present in each output;
the unchanged upstream stale-clear path can otherwise leave old status IDs.
Coasted points can legitimately retain an older F0 status: pairing identifies
the decoder snapshot, not a newly measured status for every physical object.

This will let a later review compare recorded online output with offline
decoding of the same raw bank and examine runtime batching, startup, dropped
samples and receiver health. Existing offline decoding could not supply those
actual online results or original receipt clocks. Model receiver health here
belongs to **card**, not a future radard subscriber; it does not establish that
future longitudinal model-path qualification would pass.

Actual Insight parked capture is still missing. If the radar sleeps or the
camera-side object bank is absent while parked, an invalid heartbeat proves
only that the observer is running. Inspect the raw camera-side CAN evidence and
the stock radar's operating conditions; do not enable alpha-long or request a
drive as a workaround. A stale old loggerd binary, rejected preset, missing
full rlog/video, parser exception or sustained validity errors also block a
useful-data claim.

No sensor-origin offset, calibrated target distance, physical object identity,
ECU-applied force or closed-loop benefit is established. Host CAN timestamps are
not independent physical radar acquisition times, and camera EOF is a camera
pipeline timestamp. The inherited Insight range-origin uncertainty remains.
