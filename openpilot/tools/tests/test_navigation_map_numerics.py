import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.navigation_map_numerics import check_source, first_nonfinite, matmul_reference, synthetic_maps


def test_synthetic_pair_is_bounded_and_distinct():
  maps = synthetic_maps()
  assert set(maps) == {'working', 'overflow'}
  assert not np.array_equal(maps['working'], maps['overflow'])
  for array in maps.values():
    assert array.dtype == np.float32 and array.shape == (1, 1, 256, 256)
    assert np.isfinite(array).all() and array.min() == 0 and array.max() == 255


def test_wrong_checkpoint_is_rejected_before_model_loading(tmp_path):
  path = tmp_path / 'different.onnx'
  path.write_bytes(b'not the reviewed graph')
  with pytest.raises(ValueError, match='exact registered Bigv2'):
    check_source(path)


def test_first_bad_node_uses_execution_order_and_json_is_finite():
  nodes = [SimpleNamespace(name='first', op_type='MatMul', output=['early']), SimpleNamespace(name='later', op_type='Add', output=['late'])]
  values = {'late': np.array([np.nan], np.float16), 'early': np.array([-np.inf], np.float16)}
  failure = first_nonfinite(nodes, values)
  assert failure is not None
  assert failure['node'] == 'first' and failure['summary']['neginf'] == 1
  json.dumps(failure, allow_nan=False)
  assert first_nonfinite(nodes, {'early': np.array([1.0]), 'late': np.array([2.0])}) is None


def test_finite_matmul_inputs_can_overflow_declared_float16_output():
  left, right = np.array([[256., 256.]], np.float16), np.array([[-128.], [-128.]], np.float16)
  with np.errstate(over='ignore'):
    observed = left @ right
  report = matmul_reference(left, right, observed)
  assert report['input_finite'] and report['dot'] == -65536.0 and report['outside_float16_range']
  assert np.isneginf(observed).all()
  json.dumps(report, allow_nan=False)


def test_matmul_reference_does_not_misclassify_invalid_input():
  result = matmul_reference(np.array([[np.nan]], np.float16), np.ones((1, 1), np.float16), np.array([[np.nan]], np.float16))
  assert result == {'coordinate': [0, 0], 'input_finite': False}


@pytest.fixture
def map_precision_span():
  onnx = pytest.importorskip('onnx')
  helper, tensor = onnx.helper, onnx.TensorProto
  prefix = 'model.off_policy.temporal_summarizer.map_encoder.stages.3.blocks.0'
  weights = np.zeros((2048, 512), dtype=np.float16)
  weights[:2, 0] = -128
  initializers = [onnx.numpy_helper.from_array(weights, 'val_710'),
                  onnx.numpy_helper.from_array(np.zeros(512, dtype=np.float16), prefix + '.mlp.fc2.bias'),
                  onnx.numpy_helper.from_array(np.full((1, 512, 1, 1), .218, dtype=np.float16), 'view_53')]
  nodes = [helper.make_node('MatMul', ['gelu_46', 'val_710'], ['val_711'], name='node_MatMul_599'),
           helper.make_node('Add', ['val_711', prefix + '.mlp.fc2.bias'], ['linear_93'], name='node_linear_93'),
           helper.make_node('Transpose', ['linear_93'], ['permute_112'], name='node_permute_112', perm=[0, 3, 1, 2]),
           helper.make_node('Mul', ['permute_112', 'view_53'], ['mul_46'], name='node_mul_46'),
           helper.make_node('Add', ['mul_46', 'conv2d_54'], ['add_46'], name='node_add_46')]
  inputs = [helper.make_tensor_value_info('gelu_46', tensor.FLOAT16, [1, 8, 8, 2048]),
            helper.make_tensor_value_info('conv2d_54', tensor.FLOAT16, [1, 512, 8, 8])]
  info = [helper.make_tensor_value_info(name, tensor.FLOAT16, shape) for name, shape in
          [('val_711', [1, 8, 8, 512]), ('linear_93', [1, 8, 8, 512]), ('permute_112', [1, 512, 8, 8]), ('mul_46', [1, 512, 8, 8])]]
  # Expose the FP16 boundary, matching the actual map-intermediate trace. ORT's
  # CPU lowering can otherwise retain FP32 temporaries between FP16 operators.
  outputs = [helper.make_tensor_value_info('add_46', tensor.FLOAT16, [1, 512, 8, 8]),
             helper.make_tensor_value_info('val_711', tensor.FLOAT16, [1, 8, 8, 512])]
  return helper.make_model(helper.make_graph(nodes, 'precision-test', inputs, outputs, initializers, value_info=info),
                           opset_imports=[helper.make_opsetid('', 20)], ir_version=10)


def test_precision_candidate_preserves_source_and_finite_residual_contract(map_precision_span):
  ort = pytest.importorskip('onnxruntime')
  from openpilot.tools.navigation_map_numerics import precision_repair_experiment
  original_bytes = map_precision_span.SerializeToString()
  candidate = precision_repair_experiment(map_precision_span)
  assert map_precision_span.SerializeToString() == original_bytes
  assert [value.SerializeToString() for value in candidate.graph.initializer] == [value.SerializeToString() for value in map_precision_span.graph.initializer]
  assert candidate.graph.output[0].SerializeToString() == map_precision_span.graph.output[0].SerializeToString()
  inputs = {'gelu_46': np.zeros((1, 8, 8, 2048), dtype=np.float16), 'conv2d_54': np.full((1, 512, 8, 8), 100, dtype=np.float16)}
  inputs['gelu_46'][0, 1, 5, :2] = 256
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  original = ort.InferenceSession(original_bytes, options, providers=['CPUExecutionProvider']).run(None, inputs)
  repaired = ort.InferenceSession(candidate.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
  assert np.isneginf(original[1][0, 1, 5, 0]) and np.isfinite(repaired).all()
  expected_scaled = np.float16(-65536. * float(np.float16(.218)))
  assert repaired[0, 0, 1, 5] == np.float16(expected_scaled + np.float16(100))
  assert repaired.dtype == original[0].dtype == np.float16
  assert np.all(repaired[0, 1:] == 100)  # Unaffected channels retain the actual residual.


@pytest.mark.parametrize('mutation', ['operation', 'transpose', 'shape', 'initializer', 'additional_consumer'])
def test_precision_candidate_rejects_changed_graph(map_precision_span, mutation):
  import onnx
  from openpilot.tools.navigation_map_numerics import precision_repair_experiment
  if mutation == 'operation':
    map_precision_span.graph.node[0].op_type = 'Mul'
  elif mutation == 'transpose':
    map_precision_span.graph.node[2].attribute[0].ints[:] = [0, 1, 2, 3]
  elif mutation == 'shape':
    map_precision_span.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 2047
  elif mutation == 'initializer':
    map_precision_span.graph.initializer[0].data_type = onnx.TensorProto.FLOAT
  else:
    map_precision_span.graph.node.append(onnx.helper.make_node('Identity', ['val_711'], ['extra'], name='extra_consumer'))
  with pytest.raises(ValueError, match='Precision experiment'):
    precision_repair_experiment(map_precision_span)
