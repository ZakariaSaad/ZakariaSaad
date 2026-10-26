# Navigation-model software status

> **Historical / superseded:** This document records the earlier original-Bigv2 software investigation.
> Its original-model selection/activation commands are no longer supported and
> must not be used as current instructions. Use [SunnyLink model and EPS controls](sunnylink-model-and-eps-controls.md)
> and [corrected artifact evidence](navigation-model-corrected-integration.md).
> The original `big-nav-v2` is retired; useful evidence and technical findings below are retained.


The then-current installed-branch workflow was
[Nav Bigv2 selection and activation](navigation-model-activation-workflow.md).
The existing [software compatibility report](navigation-model-software-compatibility.md)
remains available for inspecting files; the seven successful device checks do
not need to be repeated for model selection.
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
remains OFF by default. The owner explicitly authorized the one-line branch
approval change for experimental Bigv2 activation. The device still requires
manual opt-in through Models or SSH after updating, and every admission check
remains in force. This approval does not claim measured live inference
performance. The Models interface and SSH command share the same admission checks.

Historical test counts and missing-artifact findings in earlier reports describe
their recorded revisions. The final source-specific report records the tests for
the current change; a file-only check never claims GPU performance or safe driving.
