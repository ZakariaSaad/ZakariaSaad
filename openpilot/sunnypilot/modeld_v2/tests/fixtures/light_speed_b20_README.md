# Official Light Speed state reference

`light_speed_b20_modelstate.txt` contains the unchanged `ModelState` class from
[comma.ai/openpilot b20a8e4017525febac7d8810a964e7b505f9861d](https://github.com/commaai/openpilot/blob/b20a8e4017525febac7d8810a964e7b505f9861d/openpilot/selfdrive/modeld/modeld.py).
It is covered by the adjacent `light_speed_b20_LICENSE` (upstream MIT notice).

- Original full-file Git blob: `42861548f1b3b188f523ccbf4aa9e4c6557a994b`
- Original full-file SHA256: `d41da7feef30dd81c8d5caf427d09e37d0421d989e82e96fbfc79615a91ca88a`
- Unmodified class excerpt SHA256: `347654ee8a68a0c1a032027a685c2b4f23b802e14ea9831e281553be960e2edd`

The focused test checks the excerpt hash and executes its allocation, packing,
run and warmup code with an independently allocated synthetic CPU graph. It does
not execute upstream pickle loading, vehicle processes or the Light Speed model
weights. CPU results establish buffer-binding/state-sequence behavior, not AMD
numerical or vehicle-performance equivalence.
