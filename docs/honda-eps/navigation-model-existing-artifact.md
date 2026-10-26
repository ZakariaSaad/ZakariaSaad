# Existing Bigv2 registration: completed evidence

> **Historical / superseded:** This document records the original Bigv2 registration evidence.
> Its original-model selection/activation commands are no longer supported and
> must not be used as current instructions. Use [SunnyLink model and EPS controls](sunnylink-model-and-eps-controls.md)
> and [corrected artifact evidence](navigation-model-corrected-integration.md).
> The original `big-nav-v2` is retired; useful evidence and technical findings below are retained.


The physical model compilation and existing-artifact registration are complete.
The [software compatibility workflow](navigation-model-software-compatibility.md)
was the following investigation stage. It is retained as historical evidence.

The supplied `latest-registration.json` records `prepared: true`,
`registered: true`, `status: registered_disabled` and `activation_allowed: false`.
It binds:

- Model SHA256 `feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4`.
- Warp SHA256 `f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b`.
- Metadata SHA256 `2f15fb98990afff3da873ada6180f8328e3af51baa771114ff4c9c6ff428fbf8`.
- Source ONNX SHA256 `1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.

The manifest is:

```text
/data/navigation-models/big-nav-v2-7356adbbb1025695ae42a44462107d938059103c5fb446e8839cee34e9aac99c/manifest.json
```

The compiler report records physical AMD LLVM gfx1200/USB execution and exact
source/reloaded agreement for all prediction/state outputs on seeds 42 and 43.
Its compiler source SHA256 is
`d55aa36999ce05fa545d77dc1cc28e1e46fc28e60620c0b0db816e698bf79e7a`.
The subsequent registration intentionally performs file checks only; its
`gpu_execution_performed: false` does not invalidate that earlier evidence.

The registered OS04 warp accepts two 1344×760 NV12 cameras and produces uint8
`[2,6,128,256]`. Its source/layout/hash and saved-capture check passed, with all
393,216 output bytes matching. Candidate startup still compares actual geometry
before GPU loading. The model remains at `/data/nav-bigv2/bigv2-amd.sptg`; weights
were neither copied into the package nor recompiled during registration.

Registration is atomic, retries preserve a matching package, and failed or
interrupted preparation cannot replace the previous manifest. These behaviors
remain regression-tested. Registration keeps driving disabled and preserves
ordinary model selection. The UI registration action remains available for
future supported existing packages, but repeating it is unnecessary for this
successfully registered package.

The [earlier registration evidence](evidence/navigation-model-existing-artifact.json)
records the software checks before the device receipt was available. The receipt
and current compatibility report supply later evidence with their own scope.
