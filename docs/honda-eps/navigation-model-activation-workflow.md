# Navigation Model Driving activation record

Use the [software compatibility workflow](navigation-model-software-compatibility.md)
for the one-command file check. The model is already compiled and registered.
No standalone navigation hardware diagnostic remains, and no production service
is stopped by compatibility validation.

The states are separate:

| State | Meaning |
| --- | --- |
| Registered | The local receipt points to the prepared model, warp, compiler proof and metadata. The supplied receipt confirms this. |
| Artifact evidence reviewed | The exact checkpoint and native AMD artifact are bound to successful physical source/reload compiler evidence. This is completed evidence, not an outstanding compilation request. |
| Software compatibility | Source-specific regression tests and local file validation agree with the native input/output and fallback contracts. A file report identifies exact failed checks. |
| Runtime observations | Camera/GPU scheduling, concurrent warm small-model execution and real handover have or have not been observed. The supplied compiler report contains no timing measurement. |
| Eligible | Required identities, software evidence, fallback and explicit branch activation approval all satisfy admission. Reasons for a block are shown in Models settings. |
| Enabled | The owner separately enables Navigation Model Driving in Settings. Default is OFF. |

The current change does **not** approve navigation-conditioned vehicle control.
Branch approval stays false independently of registration success. The user
setting cannot approve a model, and a local manifest cannot replace the
branch-reviewed identities.

The model uses the Bigv2-specific native 20 Hz action contract, not an invented
SunnyPilot numeric generation. Navigation map and maneuver tensors condition
predictions. Existing Honda EPS timing and model-to-controller interfaces remain
in place. The MVL planner consumes model acceleration directly only in its
existing allowed ExperimentalMode/DEC modes; leads and trajectory can still
influence other modes indirectly.

Before a future enablement update, review the local compatibility report and
source-specific integration results against the supplied compiler and
registration evidence. Remaining runtime evidence must address the concrete
camera-to-AMD/20 Hz path with continuously warm QCOM fallback and actual fallback
handover. A source/reload compiler pass alone makes no claim about those paths.
No extended battery, endurance or thermal-certification procedure is required by
this workflow.

If the evidence supports activation and the owner explicitly authorizes it, a
reviewed commit on the same branch can update activation approval with the exact
artifact, warp and metadata hashes and evidence record. After a normal Software
update/restart, the owner can use the existing Models setting to enable the
feature while off-road. No manual JSON edits, source edits, hidden parameters,
model recompilation or fresh installation are part of that eventual process.
