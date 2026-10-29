# Official Bigv2 differential reference

These are byte-for-byte source files from comma.ai/openpilot commit
`a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9`, under the adjacent upstream MIT
license. `manifest.json` records their original paths, SHA256 and Git blob IDs.
Names are shortened only on disk; file contents are unmodified.

The corresponding ONNX Git LFS pointer names
`1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e`.
No model weights, pickles, routes or private captures are included.

`../../official_reference.py` checks the hashes, extracts original AST definitions
and allows tests to inject clocks, messaging and other dependencies. The inline
modeld map/instruction consumer is executed from its exact original source block.
This keeps the oracle independent of production implementations and available in
an offline checkout without requiring Git object retention or network access.

Do not reformat these fixtures: formatting changes invalidate their source hashes.
They are reference inputs, not executable vehicle processes or current production
modules. Selecting definitions for a test does not prove full neural inference
equivalence; each test reports what was actually executed.
