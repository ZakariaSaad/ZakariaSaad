"""Compile the original, verified Cinque V3 ONNX with MVL's pinned Tinygrad.

Run on the target GPU; compiled artifacts live outside Git and are installed
with tools/cinque_v3.py. Do not use old recompiled26/recompiled27 URLs.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from tinygrad import Tensor, Device
from tinygrad.nn.onnx import OnnxRunner

from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
from openpilot.sunnypilot.modeld_v2.cinque_guard import ONNX_SHA256, ONNX_SIZE
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.stateful import StatefulRunner, make_stateful_jit
from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref


def compile_model(onnx_path, output_path):
  with open(onnx_path, 'rb') as f:
    if Path(onnx_path).stat().st_size != ONNX_SIZE or hashlib.file_digest(f, 'sha256').hexdigest() != ONNX_SHA256:
      raise ValueError('Cinque V3 ONNX size or checksum mismatch')
  ref = get_tinygrad_ref()
  if ref is None:
    raise ValueError('Cannot establish the pinned Tinygrad revision')
  model = OnnxRunner(onnx_path)
  metadata = make_metadata_dict(onnx_path)
  def specs(items):
    return {k: (tuple(s.shape), np.dtype(s.dtype.fmt).name, Device.DEFAULT) for k, s in items.items()}
  input_specs = specs(model.graph_inputs)
  output_specs = {}
  for name in model.graph_outputs:
    dtype = 'float32' if name == 'outputs' else input_specs[name.removeprefix('next_')][1]
    output_specs[name] = (tuple(metadata['output_shapes'][name]), dtype, Device.DEFAULT)
  artifact = {'format': 'cinque-stateful-v1', 'tinygrad_ref': ref, 'metadata': metadata,
              'input_specs': input_specs, 'output_specs': output_specs, 'run': make_stateful_jit(model)}
  # The controller/publication contract uses float32 predictions.
  artifact['output_specs']['outputs'] = (artifact['output_specs']['outputs'][0], 'float32', Device.DEFAULT)
  runner = StatefulRunner(artifact, Parser())
  image_shape, image_dtype, dev = runner.input_specs['new_img']
  image = Tensor(np.zeros(image_shape, dtype=image_dtype), device=dev).realize()
  first = None
  for _ in range(3):
    predictions = runner.run(image, np.zeros(8, dtype='float32'), np.array([[1., 0.]], dtype='float32'), np.zeros((1, 2), dtype='float32'))
    if first is None:
      first = {k: v.copy() for k, v in predictions.items()}
  runner.reset()
  with open(output_path, 'xb') as f:
    dump_oob(artifact, f)
  with open(output_path, 'rb') as f:
    restored = StatefulRunner(load_oob(f), Parser())
  restored.reset()
  predictions = restored.run(image, np.zeros(8, dtype='float32'), np.array([[1., 0.]], dtype='float32'), np.zeros((1, 2), dtype='float32'))
  for name in first:
    np.testing.assert_allclose(predictions[name], first[name], rtol=1e-5, atol=1e-6, err_msg=f'V3 pickle roundtrip: {name}')
  with open(output_path, 'rb') as f:
    print(f'Compiled artifact SHA-256: {hashlib.file_digest(f, "sha256").hexdigest()}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--onnx', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  compile_model(args.onnx, args.output)
