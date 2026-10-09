import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import dtypes

from openpilot.sunnypilot.modeld_v2 import compile_cinque
from openpilot.sunnypilot.modeld_v2.tests.test_stateful import SyntheticModel, artifact


class TestCinqueCompiler(unittest.TestCase):
  def test_wrong_original_onnx_rejected_before_model_load(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / 'wrong.onnx'
      path.write_bytes(b'wrong original')
      with patch.object(compile_cinque, 'OnnxRunner') as load:
        with self.assertRaisesRegex(ValueError, 'size or checksum'):
          compile_cinque.compile_model(path, Path(root) / 'model.pkl')
        load.assert_not_called()

  def test_compiler_keeps_next_state_and_roundtrips_jit(self):
    data = artifact()
    model = SyntheticModel()
    model.graph_inputs = {k: SimpleNamespace(shape=shape, dtype=dtypes.uint8 if dtype == 'uint8' else dtypes.float32)
                          for k, (shape, dtype, _) in data['input_specs'].items()}
    model.graph_outputs = tuple(data['output_specs'])
    metadata = data['metadata'] | {'output_shapes': {k: s[0] for k, s in data['output_specs'].items()}}
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / 'synthetic.onnx'
      path.write_bytes(b'test export')
      output = Path(root) / 'model.pkl'
      with patch.object(compile_cinque, 'ONNX_SIZE', path.stat().st_size), \
           patch.object(compile_cinque, 'ONNX_SHA256', hashlib.sha256(path.read_bytes()).hexdigest()), \
           patch.object(compile_cinque, 'OnnxRunner', return_value=model), \
           patch.object(compile_cinque, 'make_metadata_dict', return_value=metadata):
        compile_cinque.compile_model(path, output)
      self.assertGreater(output.stat().st_size, 0)
