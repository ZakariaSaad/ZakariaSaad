# Bigv2 numerical failure during navigation driving

This investigation covers the three supplied rlog segments 5–7 from the clean
`1db06a2ec277446e7f545751ed37a8da41f455f9` installation. They contain 3,600
camera frames and 3,579 model predictions: 1,519 Bigv2 and 2,060 ordinary small.
No device access, model compilation, model replacement or hardware diagnostic
was performed.

## Confirmed failure and limits

At **16:06:57.651050 UTC**, `ModelState.run()` rejected its raw primary output
with `RuntimeError: model output not finite`. Camera frame **7523** is inferred
from the missing frame between the last big and first small predictions. The
finiteness check precedes parsing and action conversion, so those downstream
operations did not create this exception. The preceding GPU error-state check
did not report a device error.

The rlogs contain no raw prediction bytes, consumed camera/map pixels, packed
host tensors, or recurrent queues. Consequently, they cannot identify the bad
output section, its NaN/Inf count, or the first invalid internal tensor. A
numerical failure in Bigv2/native execution is confirmed; compiled arithmetic
overflow, recurrent-state propagation, an unrecorded consumed input anomaly and
a device numerical fault cannot be distinguished. No specific corrective change
to the artifact, recurrent state or input values is justified by these records.

The recorded sources show no preceding anomaly:

- GPS fixes and location/sensor status were healthy before failure, with no GPS
  timestamp rewind or camera timestamp regression.
- Navigation instructions and map metadata were valid, with no reroute across
  the supplied segments. Their validity does not prove the actual map pixels.
- Car speed, calibration and lateral-delay fields were finite. All 1,436 camera
  matrices reconstructed from logged calibration and the recorded zero camera
  offset were finite. Reconstructed action horizons were also finite; these are
  reconstructions rather than captures of the consumed tensors.
- No preceding AMD, USB, OOM, timeout, supply-fault or reset event was logged.
  USB error counters did not increase. GPU temperature was 61°C and memory
  temperature 59–60°C in the preceding window.

Only Bigv2 failed in this recording. The generic finite-output guard is shared
with other Chestnut supercombo models, but these logs do not establish a defect
affecting ordinary big models. Published actions remained finite. Existing NaN
entries in lane/edge `.t` arrays occur in both models when
`plan_x_idxs_helper()` leaves timestamps beyond the predicted horizon unset;
they are separate from this raw-network rejection.

## Fallback interruption and locationd

| Recorded measurement | Value |
| --- | ---: |
| Last Bigv2 frame | 7522 |
| Last Bigv2 execution time | 33.494 ms |
| First ordinary small frame | 7524 |
| First small execution time | 1,075.122 ms |
| Last-big to first-small publication gap | 1.141559 s |
| First small camera EOF to publication age | 1.100053 s |
| Following small frame | 7545 |
| Following small execution time | 41.730 ms |

`modeld_tinygrad` remained running with the same PID. The switch discarded the
failed frame; 20 additional camera frames passed during the first small-model
call. Camera delivery continued normally. Subsequent small-model calls returned
to approximately 35–43 ms. The logged execution interval includes host-side
preparation and model execution; it does not separately profile allocation,
upload and GPU kernels. The one-time first-use cost is consistent with deferred
initialization, but its internal component is not recorded.

The branch follows official comma navigation integration at
[`afbf2e54`](https://github.com/commaai/openpilot/blob/afbf2e547e9319e2eb9f515e1e4330af63705cec/openpilot/selfdrive/modeld/modeld.py):
initialize the ordinary fallback without running it continuously, discard the
failed big-model frame, and run the small model on the next camera iteration.
The decision was correct, but the measured handover was not seamless. No
separate restart, reload or fallback-selection bug was found. This change does
not add dual execution or a priming requirement.

The single relevant `locationd` warning was logged at **16:06:58.757592 UTC**,
**1.10654 seconds after** Bigv2 failed, following the first delayed fallback
observation. This is consistent with a consequence of the stale observation.
There is no supporting evidence that `locationd` caused the preceding Bigv2
failure; the inherited location filter is unchanged.

## Upstream comparison and published change

Native host packing, five recurrent input/output aliases, warmup cleanup and
synchronized output readback match the official implementation. The raw output
is copied before parsing. No state-layout or parsing incompatibility was found.
Official commit
[`3ec09aed`](https://github.com/commaai/openpilot/commit/3ec09aed2ea589cbd3234137ae6fb74aecde3c4c),
“fix overflow inputs in map model”, changes a compiled-model pointer only. It
does not expose a source correction or establish that the registered Bigv2
artifact is defective, so no binary or model hash was changed.

The published improvement is **failure-only diagnostic logging**, not a claimed
repair of the unresolved numerical cause. A `model_output_nonfinite` event
records the already-copied primary output's finite/NaN/+Inf/−Inf counts, bounded
invalid indices and named output sections. It summarizes existing CPU input
arrays, action timings, camera frame/timestamps, inference time and navigation
freshness reasons. It records no image/map pixels, route coordinates, route
identities, maneuver contents or raw prediction values. Recurrent finiteness is
explicitly marked unobserved; no extra GPU reads delay fallback. Logging errors
cannot replace the original exception.

The finite-output guard, single-active-model fallback, registered artifacts,
model selection, Honda EPS, MVL controls, Panda safety, firmware and updater are
unchanged. Offline rlog input checks and synthetic non-finite regression tests
can verify these boundaries, but cannot reproduce the failing AMD graph without
the missing internal tensor sequence. No new on-road testing is required to
complete this investigation.

Focused CPU/NPY regressions passed **43 tests and 29 subtests**, including an
injected NaN through the actual native model/publication loop. The invalid
prediction is rejected before parsing, the failed frame is discarded, and the
following frame uses the ordinary small model. Tests also verify report privacy,
unchanged buffers, no recurrent readbacks and preservation of the original
exception when logging fails. Ruff and scoped type checks passed.
