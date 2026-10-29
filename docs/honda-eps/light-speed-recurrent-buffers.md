# Light Speed recurrent-buffer correction

Branch: `mvl-sp-steer-eps-nav`. Starting commit:
`228648b3dd3e645da99e75d3cceef2ce05af4125`.

Reference: comma.ai/openpilot [PR #39173](https://github.com/commaai/openpilot/pull/39173),
specifically [b20a8e4017525febac7d8810a964e7b505f9861d](https://github.com/commaai/openpilot/commit/b20a8e4017525febac7d8810a964e7b505f9861d),
“Keep model history inputs separate from outputs.” The port follows that commit,
without merging the PR's other changes or updating the model artifact.

## Scope and execution order

Only the selected Chestnut bundle whose `ref` is exactly
`e4acf619d4eb8b726fc405c264109310ac2951df` enables separate recurrent buffers.
Names, generation numbers and other model metadata do not enable the change.
Corrected Bigv2's independent native registration has no ordinary bundle and
continues using its existing aliases, as do other native models. Legacy adapters
are unchanged.

`modeld.py` passes this internal allocation option to `NativeTinygradAdapter`.
For Light Speed, the adapter retains independently allocated, zero-initialized
state inputs and `next_state` outputs. Its existing runner receives those
current tensors on every call. `ModelState.run()` advances the recurrent state
at the official boundary:

1. Execute the compiled model.
2. Check the Chestnut error state and run the existing callback.
3. Read the primary output and apply the existing finite-output rejection.
4. Swap each state input with its corresponding next-state output.
5. Parse the model output.

Execution, device, callback, readback and non-finite-output failures do not swap
the recurrent buffers. A parser failure occurs after the swap, exactly as in the
official implementation; the existing failure/fallback path remains in force.
No fabricated outputs, additional rejection policy or retry state machine is
introduced.

The ordinary successful warmup performs the same swap, then clears current input
history and packed inputs. The other bank still contains the initial zero inputs.
Explicit reset already clears both banks and desire history. No extra copies or
recurrent advancement are added between frames.

## Existing download is reusable

The Chestnut v28 catalog identifies the existing Light Speed artifact as
`driving_light_speed_model_october_29_2026_tinygrad.pkl`, SHA256:

`6c84bb8c9536c21e0761addac3d58eb651c25eaea405427e57ceb6f76f59c0f3`

The alias was introduced by Python's `input_view` assignment in the adapter.
The native artifact describes state inputs and output buffers separately.
`frame_fusion.frame_part()` retains those as independent dynamic slots, and
`PlainJit._Jit.__call__()` resolves current tensor bindings on every invocation.
The boot-cache wrapper uses the same call path. Swapping references therefore
requires neither new kernels nor new model weights, artifact hashes, catalog
entries or cache invalidation.

No Light Speed artifact was downloaded, retrained, recompiled or replaced for
this change. Synthetic CPU kernels used by tests are separate temporary fixtures.

## Verification and limits

Focused tests compare the recurrent sequence and failure boundary against the
exact corrected upstream source, exercise dynamic PlainJit bindings, and cover
allocation, warmup/reset, multiple successful frames and failures. Existing
Bigv2, ordinary-model, fallback and EPS timing regressions are also checked.

- **16 Light Speed tests passed:** eight consecutive frames match upstream
  exactly with both fresh `PlainJit` and boot-cached `_Jit`, using independent
  FP32/FP16 recurrent pairs. Initial/warmup/reset states, each failure boundary,
  exact-ref selection and all five Bigv2 state aliases are covered.
- **65 existing runtime/Bigv2/fallback/EPS timing tests and 32 subtests passed.**
- **80 ordinary adapter/native artifact/reset/fallback tests and 17 subtests
  passed**, including real CPU native-artifact serialization, loading, boot-cache
  reuse and warp fusion with small synthetic fixtures.
- Changed Python passes Ruff and syntax checks; Git whitespace checks pass.

The production diff is confined to `model_adapters.py` and five added lines in
`modeld.py`. EPS prediction/command timing, controller law/calibration, sp-steer
longitudinal control, Panda safety, downloads, selection and fallback policy are
unchanged. Corrected Bigv2's pinned artifact and alias behavior are unchanged.

These are host CPU and synthetic software tests. They establish the adapter and
binding behavior, not a physical Chestnut performance measurement or full Light
Speed prediction comparison. Update this same branch normally through Software
settings and restart when prompted; an already downloaded Light Speed model can
be reused without recompilation or selection changes.
