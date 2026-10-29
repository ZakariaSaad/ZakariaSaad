# Blendful action-consumer references

These are source function excerpts, with class indentation removed where needed.
Function bodies are unchanged. Tests verify their SHA256 before execution.

- `blendful_baseline_ordinary_action.txt`: `ModelState.get_action_from_model`
  from this fork's `openpilot/sunnypilot/modeld_v2/modeld.py` at
  `84a3ff66139e0eb044076fba679ed5c4cfdaf8f8`; SHA256
  `271a50690746eb6c52214864b846d91b7f1487721f9786d0ffdd9bf6c7d591f1`.
- `blendful_baseline_native_action.txt`: `get_action_from_model` from this fork's
  `openpilot/selfdrive/modeld/modeld.py` at that same baseline; SHA256
  `b93be2b60917a9685450522e16b7d1e980ba5308839f188e785e918a0476a149`.
- `blendful_upstream_ordinary_action.txt`: `ModelState.get_action_from_model`
  from Sunnypilot's `openpilot/sunnypilot/modeld_v2/modeld.py` at merged commit
  `97135965effb3630ed55993903449edb73468d50`; SHA256
  `aa907fd0647d148f533439566e61bebcdd24d120bc842c38d9dcf23c89c72424`.

Sunnypilot source is Copyright (c) 2021-, Haibin Wen, sunnypilot, and contributors,
under the repository's MIT license (`LICENSE.md`). The native action helper is
derived from comma.ai/openpilot under MIT; the adjacent
`light_speed_b20_LICENSE` retains that notice.

Synthetic predictions exercise the actual action consumers, including smoothing,
speed thresholds and stop/acceleration outputs. These comparisons do not run model
weights or establish vehicle behavior equivalence.
