"""Opt-in compiler lowering for the reviewed Bigv2 map-MLP overflow.

This does not modify ONNX bytes, runtime admission, registration or activation.
The original FP16 operands/weights and residual interface stay intact. Only the
known overflowing dot, bias and existing gamma scale retain FP32 intermediates.
"""
import dataclasses
import hashlib
from pathlib import Path
import re

SOURCE_SHA256 = '1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e'
CORRECTION_ID = 'bigv2-map-fc2-fp32-through-gamma-v1'
PREFIX = 'model.off_policy.temporal_summarizer.map_encoder.stages.3.blocks.0'
BIAS = PREFIX + '.mlp.fc2.bias'
MATMUL_OP = 'Bigv2ReviewedMapMatMul'
SCALED_CAST_OP = 'Bigv2ReviewedMapScaledCast'
SCALED = 'mul_46_reviewed_fp32'
SPAN = (
  ('MatMul', ('gelu_46', 'val_710'), ('val_711',), {}),
  ('Add', ('val_711', BIAS), ('linear_93',), {}),
  ('Transpose', ('linear_93',), ('permute_112',), {'perm': (0, 3, 1, 2)}),
  ('Mul', ('permute_112', 'view_53'), ('mul_46',), {}),
  ('Add', ('mul_46', 'conv2d_54'), ('add_46',), {}),
)


def _map_matmul(left, right):
  from tinygrad import dtypes
  if left.shape != (1, 8, 8, 2048) or right.shape != (2048, 512) or left.dtype != dtypes.float16 or right.dtype != dtypes.float16:
    raise ValueError('Reviewed map MatMul input shape/dtype differs')
  # dtype= controls the reduction/output, while FP16 tensor-core operands remain
  # unchanged. Materialize before the epilogue; a cast alone is not the proof.
  return left.matmul(right, dtype=dtypes.float32).contiguous()


def _scaled_cast(value):
  from tinygrad import dtypes
  if value.shape != (1, 512, 8, 8) or value.dtype != dtypes.float32:
    raise ValueError('Reviewed scaled map intermediate shape/dtype differs')
  # The second boundary prevents the optimizer from folding FP32 gamma into
  # the residual add and silently skipping the declared FP16 residual input.
  return value.contiguous().cast(dtypes.float16).contiguous()


def apply_nav_map_precision_fix(runner, source_sha256: str) -> dict:
  from tinygrad import dtypes
  from tinygrad.nn.onnx import Domain, OnnxNode
  if source_sha256 != SOURCE_SHA256:
    raise ValueError('Map precision correction requires the exact reviewed Bigv2 source SHA256')
  all_names = {name for node in runner.graph_nodes for name in (*node.inputs, *node.outputs)} | set(runner.graph_values)
  if SCALED in all_names or MATMUL_OP in runner.onnx_ops or SCALED_CAST_OP in runner.onnx_ops:
    raise ValueError('Map precision correction is already applied or has a name collision')
  selected = {}
  for op, inputs, outputs, options in SPAN:
    matches = [node for node in runner.graph_nodes if node.outputs == outputs]
    if len(matches) != 1:
      raise ValueError(f'Map precision correction graph output differs: {outputs[0]}')
    node = matches[0]
    normalized = {key: tuple(value) if isinstance(value, list) else value for key, value in node.opts.items()}
    if (node.op, node.inputs, normalized, node.opset_id.domain) != (op, inputs, options, Domain.ONNX):
      raise ValueError(f'Map precision correction graph wiring differs: {outputs[0]}')
    selected[outputs[0]] = node
  for (_, _, outputs, _), (_, inputs, _, _) in zip(SPAN[:-1], SPAN[1:], strict=True):
    consumers = [node for node in runner.graph_nodes if outputs[0] in node.inputs]
    if len(consumers) != 1 or consumers[0].inputs != inputs:
      raise ValueError(f'Map precision correction has unexpected consumers: {outputs[0]}')
  if any(name in runner.graph_outputs for name in ('val_711', 'linear_93', 'permute_112', SCALED)):
    raise ValueError('Map precision correction must not change an exported output contract')
  initializers = {}
  for name, shape in {'val_710': (2048, 512), BIAS: (512,), 'view_53': (1, 512, 1, 1)}.items():
    tensor = runner.graph_values.get(name)
    if tensor is None or tensor.dtype != dtypes.float16 or tensor.shape != shape:
      raise ValueError(f'Map precision correction initializer shape/dtype differs: {name}')
    raw = tensor.numpy().tobytes()
    initializers[name] = {'shape': list(shape), 'dtype': 'float16', 'nbytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
  changed = []
  for node in runner.graph_nodes:
    if node is selected['val_711']:
      node = dataclasses.replace(node, op=MATMUL_OP)
    elif node is selected['mul_46']:
      node = dataclasses.replace(node, outputs=(SCALED,))
    changed.append(node)
    if node.outputs == (SCALED,):
      changed.append(OnnxNode(SCALED_CAST_OP, node.opset_id, (SCALED,), ('mul_46',), {}))
  runner.graph_nodes = tuple(changed)
  # Never mutate tinygrad's process-global operator table or ordinary models.
  runner.onnx_ops = dict(runner.onnx_ops) | {MATMUL_OP: _map_matmul, SCALED_CAST_OP: _scaled_cast}
  return {'id': CORRECTION_ID, 'source_onnx_sha256': source_sha256,
          'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Compile-only in-memory lowering; source ONNX bytes and weights unchanged',
          'original_matmul': 'node_MatMul_599', 'operand_dtype': 'float16', 'dot_bias_gamma_dtype': 'float32',
          'residual_input_dtype': 'float16', 'materialized_fp32_shape': [1, 512, 8, 8],
          'initializers': initializers, 'requires_separate_registration_and_approval': True}


def verify_compiled_precision(part, kernel_linear, correction: dict) -> dict:
  """Fail closed unless actual serialized kernels retain both FP32 boundaries.

  Identify weights/bias/gamma by their bytes, not kernel names or output labels.
  Inspect the compiled kernel IR as well as buffer types: an FP32 output alone
  cannot exclude an earlier FP16 narrowing (as the original AMD export shows).
  """
  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops
  header, blob = part
  buffers, calls = header['buffers'], header['calls']

  def initializer_buffers(name):
    expected = correction['initializers'][name]
    return {index for index, buffer in enumerate(buffers)
            if buffer.get('nbytes') == expected['nbytes'] and 'init' in buffer
            and hashlib.sha256(blob[slice(*buffer['init'])]).hexdigest() == expected['sha256']}

  weights, bias, gamma = (initializer_buffers(name) for name in ('val_710', BIAS, 'view_53'))
  if not weights or not bias or not gamma:
    raise ValueError('Cannot identify unchanged reviewed map initializer bytes in native plan')

  def location(argument):
    return (argument['buffer'], argument.get('offset', 0), argument['nbytes']) if 'buffer' in argument else None

  def overlaps(a, b):
    return a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]

  def fp32_output(call):
    outputs = [call['args'][i] for i in call.get('outs', ())]
    if len(outputs) != 1 or outputs[0].get('dtype') != repr(dtypes.float32) or outputs[0].get('nbytes') != 1 * 512 * 8 * 8 * 4:
      raise ValueError('Reviewed map kernel must materialize its exact FP32 intermediate')
    return location(outputs[0])

  def inspect(index):
    program = kernel_linear.src[index].src[0]
    if program.op is not Ops.PROGRAM:
      raise ValueError('Reviewed map computation is not a compiled program')
    sinks = [src for src in program.src if src.op is Ops.SINK]
    if len(sinks) != 1:
      raise ValueError('Compiled map kernel IR is unavailable for precision verification')
    uops = list(sinks[0].toposort())
    arithmetic = [op for op in uops if op.op in (Ops.ADD, Ops.MUL, Ops.WMMA) and dtypes.is_float(op.dtype)]
    if not arithmetic or any(op.dtype != dtypes.float32 for op in arithmetic):
      raise ValueError('Compiled map kernel contains non-FP32 arithmetic')
    if any(op.op is Ops.CAST and op.dtype in (dtypes.float16, dtypes.bfloat16) for op in uops):
      raise ValueError('Compiled map kernel narrows before the reviewed gamma boundary')
    sources = [src.arg for src in program.src if src.op is Ops.SOURCE]
    if len(sources) != 1 or not sources[0].lstrip().startswith('define '):
      raise ValueError('Reviewed precision correction requires inspectable generated LLVM source')
    source = sources[0]
    if re.search(r'\bfptrunc\b[^\n]*\bto\s+(?:<\d+\s+x\s+)?(?:half|bfloat)\b', source):
      raise ValueError('Generated LLVM narrows before the reviewed gamma boundary')
    if re.search(r'\bf(?:add|sub|mul|div|rem|neg)\b[^\n]*\b(?:half|bfloat)\b', source):
      raise ValueError('Generated LLVM uses half arithmetic before the reviewed gamma boundary')
    return {'call': index, 'name': calls[index]['name'], 'target': calls[index]['target'],
            'generated_llvm_sha256': hashlib.sha256(source.encode()).hexdigest(), 'llvm_early_half_narrowing': False,
            'kernel_sha256': hashlib.sha256(blob[slice(*calls[index]['lib'])]).hexdigest(),
            'floating_arithmetic': ['float32'], 'early_half_casts': 0,
            'wmma_count': sum(op.op is Ops.WMMA for op in arithmetic)}

  # Memory planning may pack initializers and both FP32 temporaries into shared
  # arenas. Follow exact byte ranges through copies and invalidate overwritten
  # ranges; matching only a root-buffer index would misidentify other tensors.
  labels = {}
  for name, indexes in (('weight', weights), ('bias', bias), ('gamma', gamma)):
    for index in indexes:
      labels.setdefault((index, 0, buffers[index]['nbytes']), set()).add(name)
  dots, scales, casts, residuals = [], [], [], []
  for index, call in enumerate(calls):
    args = call['args']
    reads = (1,) if call['kind'] == 'copy' else call['ins']
    writes = (0,) if call['kind'] == 'copy' else call['outs']
    consumed = set().union(*(labels.get(location(args[i]), set()) for i in reads))
    new_labels = {}
    if call['kind'] == 'copy':
      source, destination = location(args[1]), location(args[0])
      if source is not None and destination is not None and source[2] == destination[2]:
        for old, names in labels.items():
          if old[0] == source[0] and source[1] <= old[1] and old[1] + old[2] <= source[1] + source[2]:
            new_labels[(destination[0], destination[1] + old[1] - source[1], old[2])] = names.copy()
    elif 'weight' in consumed:
      dots.append(index)
      new_labels[fp32_output(call)] = {'dot'}
    elif {'dot', 'bias', 'gamma'} <= consumed:
      scales.append(index)
      new_labels[fp32_output(call)] = {'scale'}
    elif 'scale' in consumed:
      outputs = [args[i] for i in writes]
      if len(outputs) != 1 or outputs[0].get('dtype') != repr(dtypes.float16) or outputs[0].get('nbytes') != 1 * 512 * 8 * 8 * 2:
        raise ValueError('Scaled map must materialize FP16 before the unchanged residual')
      casts.append(index)
      new_labels[location(outputs[0])] = {'residual_half'}
    elif 'residual_half' in consumed:
      if any(args[i].get('dtype') != repr(dtypes.float16) for i in reads if 'residual_half' in labels.get(location(args[i]), set())):
        raise ValueError('Residual kernel does not consume the materialized FP16 map')
      residuals.append(index)
    for i in writes:
      written = location(args[i])
      if written is not None:
        labels = {old: names for old, names in labels.items() if not overlaps(old, written)}
    for loc, names in new_labels.items():
      labels.setdefault(loc, set()).update(names)
  if any(len(stage) != 1 for stage in (dots, scales, casts)) or not residuals or not dots[0] < scales[0] < casts[0] < min(residuals):
    raise ValueError('Cannot uniquely identify the reviewed dot, gamma, FP16 conversion and residual kernels')
  return {'status': 'passed_compiled_ir_and_native_buffer_checks', 'dot': inspect(dots[0]), 'bias_gamma': inspect(scales[0]),
          'residual_boundary': {'conversion_call': casts[0], 'consumer_calls': residuals, 'materialized_dtype': 'float16',
                                'nbytes': 1 * 512 * 8 * 8 * 2},
          'scope': 'Post-optimization tinygrad IR, generated LLVM and native plan; final ISA execution checked by source/reload tests'}
