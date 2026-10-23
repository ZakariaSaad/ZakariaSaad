# Synthetic map-overflow pair

`navigation_map_numerics.npz` contains two `uint8[1,1,256,256]` rasters generated
entirely from NumPy RNG seeds. No route, location, camera or device recording is
included. The reproducer converts these raw 0–255 pixels to float32, as required
by the exact Bigv2 input contract; normalization remains inside its ONNX graph.

The working map produces finite values throughout the exposed map encoder. The
second map changes one rectangle and causes a float16 MatMul overflow on the
registered source checkpoint. This is an input-domain numerical reproducer, not
a claim that the failing drive consumed this synthetic raster.

The search used seed 42 for bounded synthetic road stripes, then seed 12345 for
rectangular perturbations. The following deterministic recipe reconstructs the
stored arrays without performing inference. Accepted perturbation indices are
fixed evidence from the offline search, not runtime navigation logic.

```python
import numpy as np

y, x = np.indices((256, 256))
rng = np.random.default_rng(42)
for _ in range(91):
  raster = np.zeros((256, 256), dtype=np.uint8)
  for _ in range(int(rng.integers(1, 10))):
    angle = float(rng.uniform(0, np.pi))
    offset = float(rng.uniform(-90, 90))
    width = float(rng.uniform(1, 45))
    mask = np.abs((x - 128) * np.cos(angle) + (y - 128) * np.sin(angle) - offset) < width
    raster[mask] = int(rng.choice([32, 64, 128, 192, 255]))

accepted = {1, 6, 12, 13, 17, 19, 20, 21, 24, 36, 38, 41, 43, 46, 47,
            50, 51, 52, 55, 56, 58, 67, 76, 87, 89, 91, 92, 109, 117, 121, 122}
rng = np.random.default_rng(12345)
for step in range(1, 123):
  a, b = rng.integers(0, 256, 2)
  size = int(rng.integers(2, 64))
  value = int(rng.choice([0, 32, 64, 128, 192, 255]))
  if step == 122:
    working = raster.copy().reshape(1, 1, 256, 256)
  if step in accepted:
    raster[a:min(a + size, 256), b:min(b + size, 256)] = value

overflow = raster.reshape(1, 1, 256, 256)
```

SHA256 of raw uint8 bytes:

- Working: `6647562b678d42e16dd5b82e7f77b5c491e3bba2b21ddf56cfebb49d2356f52f`
- Overflow: `cd62957449abf0fe5c0a79999ef7e19f2cf7dca6f7be85e59380348379951cd6`

The archive is 3,597 bytes. It includes no model weights. The developer command
requires the separately available, exact source ONNX and optional `onnx` and
`onnxruntime` packages. It never downloads or compiles a model, changes the
registered artifact, accesses a GPU, or publishes vehicle messages.

Adding `--check-precision-repair` compares an **in-memory CPU candidate** that
keeps the specific FC2 MatMul, bias addition and existing channel-scale product
in float32, then casts to float16 before the unchanged residual addition. It
checks the exact source digest and guards the affected graph structure, shapes,
dtypes and consumers. It does not save, compile, register or activate the
candidate. The report records the numerical change on the finite comparison
map as well as the failing map.

Intermediate tracing changes materialization and fusion opportunities. CPU
backends can retain wider temporaries internally even for FP16 ONNX operators.
Passing this optional experiment establishes a candidate precision correction
for the reproducer, not that the registered AMD artifact has been repaired or
that its outputs and performance match this CPU execution.
