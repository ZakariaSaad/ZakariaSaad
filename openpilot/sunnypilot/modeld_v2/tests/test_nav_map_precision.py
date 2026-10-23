"""Focused guards/precision tests; synthetic weights, never a driving artifact."""
import dataclasses
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.sunnypilot.modeld_v2 import compile_modeld_v2 as compiler
from openpilot.sunnypilot.modeld_v2.nav_map_precision import (
  BIAS, MATMUL_OP, SCALED, SOURCE_SHA256, SPAN, apply_nav_map_precision_fix, verify_compiled_precision,
)


def runner():
  from tinygrad import Tensor, dtypes
  from tinygrad.nn.onnx import Domain, OnnxNode, OnnxRunner, OnnxValue, OpSetId
  spec = OpSetId(Domain.ONNX, 18)
  values = {'val_710': np.full((2048, 512), 32, np.float16), BIAS: np.full(512, 0.5, np.float16),
            'view_53': np.full((1, 512, 1, 1), 0.25, np.float16)}
  graph = {'name': '', 'node': [{'parsed_node': OnnxNode(op, spec, inputs, outputs, dict(opts))} for op, inputs, outputs, opts in SPAN],
           'initializer': [{'name': name, 'parsed_tensor': Tensor(value, device='CPU')} for name, value in values.items()],
           'input': [{'name': 'gelu_46', 'parsed_type': OnnxValue((1, 8, 8, 2048), dtypes.float16, False, False)},
                     {'name': 'conv2d_54', 'parsed_type': OnnxValue((1, 512, 8, 8), dtypes.float16, False, False)}],
           'output': [{'name': 'add_46'}]}
  return OnnxRunner._from_subgraph(graph)


def test_changes_only_reviewed_span_preserves_global_ops_weights_and_io():
  from tinygrad import Tensor, dtypes
  model = runner()
  before_nodes, before_ops = model.graph_nodes, model.onnx_ops
  before_values = dict(model.graph_values)
  before_input, before_output = dict(model.graph_inputs), model.graph_outputs
  correction = apply_nav_map_precision_fix(model, SOURCE_SHA256)
  assert model.graph_nodes[0].op == MATMUL_OP
  assert model.graph_nodes[1:3] == before_nodes[1:3]
  assert model.graph_nodes[-1] == before_nodes[-1]
  assert model.graph_nodes[3].outputs == (SCALED,)
  assert all(model.graph_values[key] is value for key, value in before_values.items())
  assert model.graph_inputs == before_input and model.graph_outputs == before_output
  assert MATMUL_OP not in before_ops
  assert correction['requires_separate_registration_and_approval']
  inputs = {'gelu_46': Tensor.zeros(1, 8, 8, 2048, dtype=dtypes.float16, device='CPU'),
            'conv2d_54': Tensor.zeros(1, 512, 8, 8, dtype=dtypes.float16, device='CPU')}
  result = model(inputs)
  assert result['add_46'].dtype == dtypes.float16 and result['add_46'].shape == (1, 512, 8, 8)
  for name in ('val_711', 'linear_93', 'permute_112', SCALED):
    assert model.graph_values[name].dtype == dtypes.float32
  assert model.graph_values['mul_46'].dtype == dtypes.float16


def test_source_hash_mismatch_leaves_runner_unchanged():
  model = runner()
  before = model.graph_nodes
  with pytest.raises(ValueError, match='source SHA256'):
    apply_nav_map_precision_fix(model, '0' * 64)
  assert model.graph_nodes is before


@pytest.mark.parametrize('mutation', ['consumer', 'duplicate', 'bias', 'transpose', 'exported', 'collision'])
def test_changed_graph_is_rejected_before_mutation(mutation):
  from tinygrad import Tensor
  model = runner()
  if mutation == 'consumer':
    model.graph_nodes += (dataclasses.replace(model.graph_nodes[1], outputs=('unrelated',)),)
  elif mutation == 'duplicate':
    model.graph_nodes += (model.graph_nodes[0],)
  elif mutation == 'bias':
    model.graph_values[BIAS] = Tensor(np.zeros(512, np.float32), device='CPU')
  elif mutation == 'transpose':
    model.graph_nodes = (*model.graph_nodes[:2], dataclasses.replace(model.graph_nodes[2], opts={'perm': (0, 1, 2, 3)}), *model.graph_nodes[3:])
  elif mutation == 'exported':
    model.graph_outputs += ('val_711',)
  else:
    model.graph_values[SCALED] = None
  before = model.graph_nodes
  with pytest.raises(ValueError, match='precision correction'):
    apply_nav_map_precision_fix(model, SOURCE_SHA256)
  assert model.graph_nodes is before


def test_reapplying_correction_is_rejected():
  model = runner()
  apply_nav_map_precision_fix(model, SOURCE_SHA256)
  with pytest.raises(ValueError, match='already applied'):
    apply_nav_map_precision_fix(model, SOURCE_SHA256)


def test_compiled_precision_refuses_labels_without_initializer_evidence():
  with pytest.raises(ValueError, match='initializer bytes'):
    verify_compiled_precision(({'buffers': [], 'calls': []}, memoryview(b'')), None,
                              {'initializers': {name: {'nbytes': 4, 'sha256': '0' * 64} for name in ('val_710', BIAS, 'view_53')}})


@pytest.mark.parametrize('existing', ['model', 'report'])
def test_correction_cli_never_replaces_existing_paths(tmp_path, existing):
  source, output = tmp_path / 'source.onnx', tmp_path / 'candidate.sptg'
  report = tmp_path / 'candidate.sptg.verification.json'
  source.write_bytes(b'original source')
  selected = output if existing == 'model' else report
  selected.write_bytes(b'preserve this')
  with patch('sys.argv', ['compiler', str(source), str(output), '--nav-map-precision-fix']), pytest.raises(SystemExit):
    compiler.main()
  assert selected.read_bytes() == b'preserve this'
  assert source.read_bytes() == b'original source'


def test_map_fixture_verification_uses_identical_inputs_and_state():
  from openpilot.sunnypilot.modeld_v2.tests.test_native_compiler_verification import allocate_arrays, benchmark_arrays
  inputs = {'nav_map': ((1, 1, 256, 256), '<f4', 'CPU'), 'state_q': ((8,), '|u1', 'CPU')}
  outputs = {'outputs': ((1,), '<f4', 'CPU'), 'next_state_q': ((8,), '|u1', 'CPU')}
  calls = []

  def run(output_buffers, nav_map, state_q):
    calls.append((nav_map.value.copy(), state_q.value.copy(), id(nav_map), id(state_q)))
    output_buffers['outputs'].value[:] = nav_map.value.mean()
    output_buffers['next_state_q'].value[:] = state_q.value

  result = compiler.verify_map_fixtures(run, run, inputs, outputs, allocate_arrays, benchmark_arrays)
  assert set(result['cases']) == {'working', 'overflow'}
  assert len(calls) == 4
  for a, b in zip(calls[::2], calls[1::2], strict=True):
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2:] != b[2:]
  assert not np.array_equal(calls[0][0], calls[2][0])


def test_real_cpu_compiler_preserves_fp32_span_and_native_reload(tmp_path):
  """Default regression exercises the actual compiler/verifier, without ONNX."""
  from tinygrad import Tensor, TinyJit
  from tinygrad.helpers import Context
  from openpilot.sunnypilot.modeld_v2.helpers import load_model, write_bundle
  from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit
  with Context(DEV='CPU:LLVM'):
    values = {'gelu_46': np.ones((1, 8, 8, 2048), np.float16), 'conv2d_54': np.zeros((1, 512, 8, 8), np.float16)}

    def inputs():
      return {name: Tensor(value.copy(), device='CPU').realize() for name, value in values.items()}

    original = runner()
    assert np.isinf(original(inputs())['add_46'].numpy()).all()
    model = runner()
    correction = apply_nav_map_precision_fix(model, SOURCE_SHA256)

    @TinyJit
    def run(gelu_46, conv2d_54, output_buffers):
      output_buffers['add_46'].assign(model({'gelu_46': gelu_46, 'conv2d_54': conv2d_54})['add_46']).realize()

    def arguments():
      return inputs() | {'output_buffers': {'add_46': Tensor(np.zeros((1, 512, 8, 8), np.float16), device='CPU').realize()}}

    args = arguments()
    with record_kernel_linear() as recorded:
      for _ in range(3):
        run(**args)
    check, _ = capture_frame(run, args, 'output_buffers')
    part = serialize_jit(run, recorded[0], check)
    proof = verify_compiled_precision(part, recorded[0], correction)
    assert proof['dot']['early_half_casts'] == proof['bias_gamma']['early_half_casts'] == 0
    assert proof['dot']['generated_llvm_sha256'] and proof['bias_gamma']['generated_llvm_sha256']
    path = tmp_path / 'synthetic-span.sptg'
    write_bundle(str(path), {'run': part})
    reloaded = load_model(str(path), cache=False, verify=True)['run']
    native_args = arguments()
    reloaded(**native_args)
    expected, actual = args['output_buffers']['add_46'].numpy(), native_args['output_buffers']['add_46'].numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    # Identical native buffer labels are insufficient: inspect the compiled IR.
    from tinygrad.uop.ops import Ops
    dot_index = proof['dot']['call']
    calls = list(recorded[0].src)
    program = calls[dot_index].src[0]
    source = next(src for src in program.src if src.op is Ops.SOURCE)
    bad_source = source.replace(arg=source.arg + '\n%bad = fptrunc float %x to half\n')
    bad_program = program.replace(src=tuple(bad_source if src is source else src for src in program.src))
    calls[dot_index] = calls[dot_index].replace(src=(bad_program, *calls[dot_index].src[1:]))
    with pytest.raises(ValueError, match='Generated LLVM narrows'):
      verify_compiled_precision(part, recorded[0].replace(src=tuple(calls)), correction)
