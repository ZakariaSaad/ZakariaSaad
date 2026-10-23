"""Opt-in, CPU-only numerical reproducer for the exact Nav Bigv2 map encoder.

No model downloads, compilation, vehicle messaging, parameters or GPU access.
Requires developer dependencies onnx and onnxruntime. The small built-in rasters
are synthetic; they contain no recorded route or camera data. This is a map-only
source-model reproducer, not an AMD artifact or recurrent driving replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SOURCE_SHA256 = '1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e'
FIXTURES = Path(__file__).parent / 'tests/fixtures/navigation_map_numerics.npz'
FIXTURE_SHA256 = {
  'working': '6647562b678d42e16dd5b82e7f77b5c491e3bba2b21ddf56cfebb49d2356f52f',
  'overflow': 'cd62957449abf0fe5c0a79999ef7e19f2cf7dca6f7be85e59380348379951cd6',
}


def check_source(path: Path) -> None:
  with path.open('rb') as file:
    if hashlib.file_digest(file, 'sha256').hexdigest() != SOURCE_SHA256:
      raise ValueError('This reproducer requires the exact registered Bigv2 source ONNX SHA256')


def synthetic_maps() -> dict[str, np.ndarray]:
  with np.load(FIXTURES, allow_pickle=False) as bundle:
    if set(bundle.files) != set(FIXTURE_SHA256):
      raise ValueError('Unexpected synthetic fixture keys')
    result = {}
    for name, digest in FIXTURE_SHA256.items():
      array = bundle[name]
      if array.dtype != np.uint8 or array.shape != (1, 1, 256, 256) or hashlib.sha256(array.tobytes()).hexdigest() != digest:
        raise ValueError(f'Synthetic fixture integrity failed: {name}')
      result[name] = array.astype(np.float32)
    return result


def extract_map_graph(source):
  # ONNX protobuf parsing only. Never load the upstream pickle metadata.
  import onnx
  model = onnx.utils.Extractor(source).extract_model(['nav_map'], ['select_11'])
  values = {value.name: value for value in (*source.graph.value_info, *source.graph.output, *source.graph.input)}
  for node in model.graph.node:
    for name in node.output:
      if name != 'select_11' and name in values:
        model.graph.output.append(values[name])
  if any(node.domain not in ('', 'ai.onnx') for node in model.graph.node):
    raise ValueError('Unexpected custom operator in the map-only graph')
  return model


def precision_repair_experiment(source):
  """Return an in-memory FP32-span candidate; never save or register a model.

  The CLI hashes the complete source first. Exact span/shape guards additionally
  prevent applying this experiment to a different operation or broadcast layout.
  The experiment preserves the existing weights, bias, scale, residual and FP16
  output contract. It changes intermediate rounding, which is measured below.
  """
  import onnx
  model = onnx.ModelProto()
  model.CopyFrom(source)
  nodes = {node.name: node for node in model.graph.node}
  prefix = 'model.off_policy.temporal_summarizer.map_encoder.stages.3.blocks.0'
  span = {
    'node_MatMul_599': ('MatMul', ['gelu_46', 'val_710'], ['val_711']),
    'node_linear_93': ('Add', ['val_711', prefix + '.mlp.fc2.bias'], ['linear_93']),
    'node_permute_112': ('Transpose', ['linear_93'], ['permute_112']),
    'node_mul_46': ('Mul', ['permute_112', 'view_53'], ['mul_46']),
    'node_add_46': ('Add', ['mul_46', 'conv2d_54'], ['add_46']),
  }
  for name, (operation, inputs, outputs) in span.items():
    node = nodes.get(name)
    if node is None or (node.op_type, list(node.input), list(node.output), node.domain) != (operation, inputs, outputs, ''):
      raise ValueError(f'Precision experiment graph mismatch: {name}')
  perm = nodes['node_permute_112'].attribute
  if len(perm) != 1 or perm[0].name != 'perm' or list(perm[0].ints) != [0, 3, 1, 2]:
    raise ValueError('Precision experiment transpose differs')
  info = {value.name: value for value in (*model.graph.value_info, *model.graph.output, *model.graph.input)}
  tensors = {'gelu_46': [1, 8, 8, 2048], 'val_711': [1, 8, 8, 512], 'linear_93': [1, 8, 8, 512],
             'permute_112': [1, 512, 8, 8], 'mul_46': [1, 512, 8, 8]}
  for name, shape in tensors.items():
    tensor = info[name].type.tensor_type if name in info else None
    if tensor is None or tensor.elem_type != onnx.TensorProto.FLOAT16 or [dim.dim_value for dim in tensor.shape.dim] != shape:
      raise ValueError(f'Precision experiment tensor contract differs: {name}')
  initializers = {value.name: value for value in model.graph.initializer}
  for name, shape in {'val_710': [2048, 512], prefix + '.mlp.fc2.bias': [512], 'view_53': [1, 512, 1, 1]}.items():
    value = initializers.get(name)
    if value is None or value.data_type != onnx.TensorProto.FLOAT16 or list(value.dims) != shape:
      raise ValueError(f'Precision experiment initializer differs: {name}')
  for output, consumer in [('val_711', 'node_linear_93'), ('linear_93', 'node_permute_112'), ('permute_112', 'node_mul_46'),
                           ('mul_46', 'node_add_46')]:
    if [node.name for node in model.graph.node if output in node.input] != [consumer]:
      raise ValueError(f'Precision experiment has an additional consumer: {output}')
  if any(name.endswith('_map_f32') for name in {*info, *initializers, *(out for node in model.graph.node for out in node.output)}):
    raise ValueError('Precision experiment temporary name collision')

  def cast(name, dtype):
    return onnx.helper.make_node('Cast', [name], [name + '_map_f32'], name='experiment_cast_' + name, to=dtype)

  changed = []
  for node in model.graph.node:
    if node.name == 'node_MatMul_599':
      changed.extend([cast('gelu_46', onnx.TensorProto.FLOAT), cast('val_710', onnx.TensorProto.FLOAT)])
      node.input[:] = ['gelu_46_map_f32', 'val_710_map_f32']
    elif node.name == 'node_linear_93':
      changed.append(cast(prefix + '.mlp.fc2.bias', onnx.TensorProto.FLOAT))
      node.input[1] += '_map_f32'
    elif node.name == 'node_mul_46':
      changed.append(cast('view_53', onnx.TensorProto.FLOAT))
      node.input[1] += '_map_f32'
      node.output[0] = 'mul_46_map_f32'
    changed.append(node)
    if node.name == 'node_mul_46':
      changed.append(onnx.helper.make_node('Cast', ['mul_46_map_f32'], ['mul_46'], name='experiment_cast_back_after_scale',
                                          to=onnx.TensorProto.FLOAT16))
  del model.graph.node[:]
  model.graph.node.extend(changed)
  for value in (*model.graph.value_info, *model.graph.output):
    if value.name in ('val_711', 'linear_93', 'permute_112'):
      value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
  return model


def array_summary(array: np.ndarray) -> dict:
  finite = np.isfinite(array)
  values = array[finite]
  return {'shape': list(array.shape), 'dtype': str(array.dtype), 'total': array.size, 'finite': int(finite.sum()),
          'nan': int(np.isnan(array).sum()), 'posinf': int(np.isposinf(array).sum()), 'neginf': int(np.isneginf(array).sum()),
          'min_finite': float(values.min()) if values.size else None, 'max_finite': float(values.max()) if values.size else None}


def first_nonfinite(nodes: list, values: dict[str, np.ndarray]) -> dict | None:
  for node in nodes:
    for name in node.output:
      if name in values and not np.isfinite(values[name]).all():
        return {'node': node.name, 'operation': node.op_type, 'output': name, 'summary': array_summary(values[name])}
  return None


def matmul_reference(left: np.ndarray, right: np.ndarray, observed: np.ndarray) -> dict:
  # A single independent dot product at the first invalid coordinate; no complete
  # high-precision model and no replacement prediction is constructed here.
  if right.ndim != 2 or left.shape[-1] != right.shape[0] or observed.shape != (*left.shape[:-1], right.shape[1]):
    raise ValueError('Unsupported MatMul reference shapes')
  invalid = np.argwhere(~np.isfinite(observed))
  if not len(invalid):
    raise ValueError('MatMul reference requires an observed nonfinite output')
  index = tuple(int(value) for value in invalid[0])
  lhs, rhs = left[index[:-1]], right[:, index[-1]]
  if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
    return {'coordinate': list(index), 'input_finite': False}
  result = float(np.dot(lhs.astype(np.float64), rhs.astype(np.float64)))
  return {'coordinate': list(index), 'input_finite': True, 'accumulation': 'numpy_float64_dot', 'dot': result,
          'float16_max': float(np.finfo(np.float16).max), 'outside_float16_range': abs(result) > float(np.finfo(np.float16).max)}


def run_probe(source: Path, *, check_precision_repair: bool = False) -> dict:
  check_source(source)
  import onnx
  import onnxruntime as ort
  model = extract_map_graph(onnx.load(source, load_external_data=False))
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  options.inter_op_num_threads = 1
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
  repaired_model = precision_repair_experiment(model) if check_precision_repair else None
  repaired_session = (ort.InferenceSession(repaired_model.SerializeToString(), options, providers=['CPUExecutionProvider'])
                      if repaired_model is not None else None)
  names = [output.name for output in session.get_outputs()]
  initializers = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
  report = {'scope': 'Synthetic stateless map encoder; not route replay or AMD equivalence', 'source_sha256': SOURCE_SHA256,
            'backend': 'ONNXRuntime CPUExecutionProvider', 'onnxruntime_version': ort.__version__,
            'graph_optimization': 'disabled', 'all_intermediate_outputs_exposed': True, 'source_file_changed': False,
            'registered_artifact_changed': False,
            'nodes': len(model.graph.node), 'cases': []}
  if check_precision_repair:
    report['precision_repair'] = {'scope': 'In-memory CPU candidate only; not AMD-verified or deployed',
                                  'source_or_artifact_files_modified': False, 'intermediate_rounding_changed': True,
                                  'change': 'FP32 MatMul, bias and existing channel scale; cast back to FP16 after scale'}
  for name, nav_map in synthetic_maps().items():
    started = time.perf_counter()
    values = dict(zip(names, session.run(names, {'nav_map': nav_map}), strict=True))
    failure = first_nonfinite(list(model.graph.node), values)
    case = {'fixture': name, 'input_sha256_float32': hashlib.sha256(nav_map.tobytes()).hexdigest(),
            'input': array_summary(nav_map), 'elapsed_s': time.perf_counter() - started,
            'first_nonfinite': failure, 'map_features': array_summary(values['select_11']),
            'invalid_intermediates': {key: array_summary(value) for key, value in values.items() if not np.isfinite(value).all()}}
    if failure is not None:
      node = next(node for node in model.graph.node if node.name == failure['node'])
      if node.op_type == 'MatMul' and node.input[0] in values and node.input[1] in initializers:
        case['independent_dot_reference'] = matmul_reference(values[node.input[0]], initializers[node.input[1]], values[node.output[0]])
    if repaired_session is not None and repaired_model is not None:
      repaired = dict(zip(names, repaired_session.run(names, {'nav_map': nav_map}), strict=True))
      before, after = values['select_11'].astype(np.float64), repaired['select_11'].astype(np.float64)
      delta = np.abs(before - after)
      case['precision_repair'] = {'first_nonfinite': first_nonfinite(list(repaired_model.graph.node), repaired),
                                  'map_features': array_summary(repaired['select_11']),
                                  'original_finite_comparison': bool(np.isfinite(delta).all()),
                                  'max_abs_feature_difference': float(delta.max()) if np.isfinite(delta).all() else None}
    report['cases'].append(case)
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-onnx', required=True, type=Path, help='Existing exact Bigv2 ONNX; never downloaded or modified')
  parser.add_argument('--output', required=True, type=Path, help='New JSON report path')
  parser.add_argument('--check-precision-repair', action='store_true',
                      help='Compare an in-memory FP32-span candidate on CPU; never write or register the candidate')
  args = parser.parse_args()
  if args.output.exists() or args.output.resolve() == args.source_onnx.resolve():
    parser.error('Output must be a new file, separate from the source model')
  try:
    report = run_probe(args.source_onnx, check_precision_repair=args.check_precision_repair)
  except ImportError as error:
    parser.error(f'Optional developer dependency missing: {error}. Run on a development host with onnx and onnxruntime installed.')
  with args.output.open('x') as file:
    json.dump(report, file, indent=2, allow_nan=False)
    file.write('\n')
  for case in report['cases']:
    failure = case['first_nonfinite']
    print(f"{case['fixture']}: {failure['node'] if failure else 'all map intermediates finite'}")
  print(f'Report: {args.output}')


if __name__ == '__main__':
  main()
