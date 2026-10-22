# Navigation-model software status

The current installed-branch workflow is
[Nav Bigv2 software compatibility](navigation-model-software-compatibility.md).
The standalone navigation hardware framework and service orchestration have been
removed. All model compilation and existing-package registration requested from
the owner are complete; no repeat is required.

The earlier software corrections remain integrated: explicit camera BOOTTIME to
navigation MONOTONIC conversion; clock-epoch map-cache invalidation; candidate-first startup;
unchanged ordinary startup when disabled; and owner-aware read-only Git children
that preserve original stderr without changing Git configuration or permissions.
The complete model loader, typed inputs/states, native output contract,
single-model inference and initialized ordinary fallback are exercised by the
software suite. The custom activation frame counter, shadow execution and
continuous QCOM fallback inference are removed.

Physical source/reload compiler verification, the supplied registration receipt
and the owner's seven-check device compatibility report at `f18366ef` are
completed evidence. They are distinguished from the unmeasured camera-to-Bigv2
20 Hz prediction path and the normal inference-failure handover. Navigation Model Driving
remains OFF and branch activation remains unapproved pending the explicit
[activation decision](navigation-model-activation-workflow.md).

Historical test counts and missing-artifact findings in earlier reports describe
their recorded revisions. The final source-specific report records the tests for
the current change; a file-only check never claims GPU performance or safe driving.
