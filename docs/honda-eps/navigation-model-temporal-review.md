# Navigation temporal review

This continues the Stage 2 implementation at experimental starting commit
`036fd8ca9afb5240fa163c377bcfa2b961a38fe0`. The focused results below tested
that commit **plus uncommitted changes**, not the commit alone. Exact commands,
SHA256 source snapshots, the tracked diff, JUnit and raw output are retained in
`/workspace/work/nav-model-final/temporal/`. Final integration must retest the
actual committed result; later source changes invalidate the corresponding
earlier claim.

The review found and fixed these producer defects:

| Defect | Resulting behavior |
| --- | --- |
| A delayed old `navRoute` could replace a newer route to the same destination. | MapsD retains a generation and event-time high-water mark. Lower generations and older events cannot roll the renderer back. One generation cannot change geometry or destination. |
| An old route could be replayed after a clear message. | A clear retains the high-water mark as a tombstone. Reacquisition requires a newly allocated generation. A normal navd restart allocates that new identity. |
| A freshly delivered GPS packet with an old or foreign-clock source timestamp looked current. | Model sources require both fresh receipt and physical source time within the existing two-second past / 200 ms future bounds. No global tolerance was increased. |
| A blocking same-destination route request left its old model instruction usable. | navd sends a model-invalid snapshot before HTTP begins. Existing UI instructions remain visible; model GPS timestamp and maneuver vector are withdrawn until the request completes. |
| A model instruction could stamp one GPS source while projecting the retained display position from another source. | Maneuver projection uses the coordinates from the exact source associated with `gpsSourceMonoTime`. An invalid model source cannot advance the model route tracker. |
| Retained display coordinates could advance route steps or remove the destination during GPS loss. | Route transitions and arrival removal require an admissible physical GPS source. Display state remains visible while source validity is withdrawn. |
| A destination could change between completing a map render and sending metadata. | The live destination is checked again when constructing the metadata message. A completed old image cannot receive valid current metadata. |

The original display-only positioning policy remains available when a legacy log
has old timestamps. It cannot authorize a new model route or model snapshot.
Standalone `MapsD(None)` retains legacy viewing compatibility; the production
service injects live Params and enforces source-age and destination checks.
Default settings, steering/longitudinal code, model timing, firmware, normal model
selection and updater configuration were not modified by these producer fixes.

Route-generation tags and exact VisionIPC frame/SOF/EOF matching are additional
transport safeguards beyond the official experimental reference at
[`afbf2e547e9319e2eb9f515e1e4330af63705cec`](https://github.com/commaai/openpilot/tree/afbf2e547e9319e2eb9f515e1e4330af63705cec).
The upstream route projection, snapped heading and receipt-age pose extrapolation
remain in use. These tags describe route identity, not trained model classes.

Python GPS publishers and map rendering use monotonic time; native camera capture
can use Linux boottime. Suspend can separate the clocks. Unaligned contexts are
rejected explicitly; this review neither infers an offset nor changes global
clocks. Actual device clock-domain agreement and suspend/resume recovery must be
measured before candidate activation. A `permission_valid=False` prepared context
records synthetic permission withdrawal; real runtime capabilities must still be
checked independently on every frame.

The producer/UI/lifecycle Xvfb run passed **110 tests and 27 subtests**, with no
failures or skips and five existing Python fork deprecation warnings. It includes
the existing UI, actual child daemon lifecycle, native map metadata/frame
association, HTTP destination-race checks and source-clock rejection. The final
GPS-loss transition guard was added afterward. A subsequent Xvfb/native run of
that guard, producers, long native sequence, selection and new rendered model
settings passed **79 tests and 9 subtests**, with no failures or skips and one
existing fork warning. Ruff and scoped ty checks passed. All 18 source/test/config
files fingerprinted in `ui-tests-final.sources.json` remained unchanged during
this later run; neither run is a final committed-tree claim.

Five new real-raylib tests instantiate the installed ModelsLayoutMici and parent
navigation-model button, open the actual NavScroller and BigDialog callbacks,
render both comma 4 and comma 3 layouts, and inspect their actual native Params.
They cover the default/unreviewed disabled toggle, visible candidate-specific
hardware gate, off-road restriction, unknown/tampered checkbox rejection,
candidate preparation persistence and preservation of both ordinary bundle slots,
runner cache and Navigation Alpha display flag. Only the positive enable/disable
case substitutes a synthetic eligible status; it does not approve any hardware
artifact. Status wording distinguishes a request from actual model operation:
`Requested on`, `Blocked`, or `Off`, with `requested` on the comma 4 parent button.

`test_native_temporal_sequence.py` sends native Msgq messages and VisionIPC frames
through the production RouteEngine, MapsD and NavigationInputs implementations.
It makes 300 reader decisions, including **140 stable admitted frames before a
same-destination reroute and 140 afterward**. Both windows exceed the 128-frame
feature/map/navigation queues and the 132-entry desire queue. It also covers
UInt32 frame-ID rollover, delayed old route events, GPS loss/recovery, mapsd
restart/route resend, and navd restart/new identity. Only HTTP and tile rendering
are fixtures; route coordinates, pixels and the 20 Hz virtual timestamps are
explicitly synthetic. This is transport/state-boundary validation, not a learned
model history experiment or real navigation replay.

A broader navigation run taken while other final integration edits were still
in progress recorded 154 passed tests and 185 passed subtests, plus three selector
failures awaiting the new native Params build and six publication-guard harness
subfailures from its extracted namespace omitting a newly added dependency. Those
results are preserved in `owned-tests.log`; they are not reported as a passing
full integration suite. The final reviewed tree requires its own complete run.

Reader-side reordered instruction/frame handling, runtime recurrent-buffer
aliasing, production fallback/admission and learned long-sequence inference are
reviewed separately. Zero navigation inputs, finite outputs, clearing history and
passing these transport tests do not establish a safe driving fallback or model
quality. No vehicle, firmware or production prediction endpoint was used.
