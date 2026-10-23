"""Optional exact-checkpoint numerical regressions; no downloads or device access.

Set NAV_BIGV2_SOURCE_ONNX to the existing reviewed ONNX. These tests execute CPU
LLVM only and compile a 2 MB span and the 22 MB stateless map encoder, never the
complete AMD model. The separate
NAV_BIGV2_FULL_REFERENCE=1 test additionally needs about 6 GB RAM for ONNX Runtime.
Synthetic map fixtures contain no recorded route data. No generated artifacts
leave pytest's temporary directory or enter registration/activation.
"""
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from openpilot.tools.navigation_map_numerics import check_source, extract_map_graph, precision_repair_experiment, synthetic_maps


@pytest.fixture(scope='module')
def source_path():
  path = os.environ.get('NAV_BIGV2_SOURCE_ONNX')
  if not path:
    pytest.skip('Set NAV_BIGV2_SOURCE_ONNX to the existing exact Bigv2 ONNX; no download is performed')
  check_source(Path(path))
  return Path(path)


def _session(model):
  ort = pytest.importorskip('onnxruntime')
  options = ort.SessionOptions()
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  return ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])


@pytest.fixture
def cpu_runtime():
  pytest.importorskip('tinygrad')
  from tinygrad.helpers import Context
  with Context(DEV='CPU:LLVM'):
    yield


@pytest.mark.parametrize('packed_constants', [False, True])
def test_actual_source_lowering_survives_native_compile_and_reload(source_path, tmp_path, cpu_runtime, packed_constants):
  onnx = pytest.importorskip('onnx')
  pytest.importorskip('tinygrad')
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.nn.onnx import OnnxRunner
  from openpilot.sunnypilot.modeld_v2.helpers import load_model, write_bundle
  from openpilot.sunnypilot.modeld_v2.nav_map_precision import SOURCE_SHA256, apply_nav_map_precision_fix, verify_compiled_precision
  from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit

  # Exposing intermediates preserves the original declared FP16 boundary in ORT.
  model = extract_map_graph(onnx.load(source_path, load_external_data=False))
  session = _session(model)
  names = ['gelu_46', 'conv2d_54', 'val_711']
  inputs = {}
  for name, raster in synthetic_maps().items():
    gelu, residual, dot = session.run(names, {'nav_map': raster})
    assert np.isfinite(gelu).all() and np.isfinite(residual).all()
    assert np.isfinite(dot).all() == (name == 'working')
    inputs[name] = {'gelu_46': gelu, 'conv2d_54': residual}
  del session
  span = onnx.utils.Extractor(model).extract_model(['gelu_46', 'conv2d_54'], ['add_46'])
  path = tmp_path / 'original-map-span.onnx'
  onnx.save(span, path)
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  results = {}
  device = 'CPU:LLVM' if packed_constants else 'CPU'

  def make_run(runner):
    @TinyJit
    def run(gelu_46, conv2d_54, output_buffers):
      values = runner({'gelu_46': gelu_46, 'conv2d_54': conv2d_54})
      Tensor.realize(*(output.assign(values[name]) for name, output in output_buffers.items()))
    return run

  for corrected in (False, True):
    runner = OnnxRunner(path)
    if packed_constants:
      runner.to(device)
    correction = apply_nav_map_precision_fix(runner, SOURCE_SHA256) if corrected else None
    run = make_run(runner)

    def arguments(values):
      return {**{name: Tensor(value.copy(), device=device).realize() for name, value in values.items()},
              'output_buffers': {name: Tensor.zeros(1, 512, 8, 8, dtype=dtypes.float16, device=device).realize()
                                     for name in ('add_46',)}}

    kwargs = arguments(inputs['working'])
    with record_kernel_linear() as recorded:
      for _ in range(3):
        run(**kwargs)
    assert len(recorded) == 1
    check, _ = capture_frame(run, kwargs, 'output_buffers')
    part = serialize_jit(run, recorded[0], check)
    if correction:
      verification = verify_compiled_precision(part, recorded[0], correction)
      assert verification['status'] == 'passed_compiled_ir_and_native_buffer_checks'
      assert verification['dot']['early_half_casts'] == verification['bias_gamma']['early_half_casts'] == 0
      assert verification['dot']['target']['device'] == verification['bias_gamma']['target']['device'] == 'CPU'
    artifact = tmp_path / f'candidate-{corrected}.sptg'
    write_bundle(str(artifact), {'run': part})
    native = load_model(str(artifact), cache=False, verify=True)['run']
    results[corrected] = {}
    for fixture, values in inputs.items():
      outputs = []
      for execute in (run, native):
        kwargs = arguments(values)
        execute(**kwargs)
        outputs.append({name: value.numpy().copy() for name, value in kwargs['output_buffers'].items()})
        for name, value in values.items():
          np.testing.assert_array_equal(kwargs[name].numpy(), value)  # No aliased input mutation.
      for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
      results[corrected][fixture] = outputs[1]
      # Inspect the scaled source tensor separately. Exporting it from the JIT
      # would turn the internal residual boundary into an external output buffer
      # and change the memory plan being verified above.
      source_runner = OnnxRunner(path)
      if packed_constants:
        source_runner.to(device)
      if corrected:
        apply_nav_map_precision_fix(source_runner, SOURCE_SHA256)
      source_runner({name: Tensor(value.copy(), device=device) for name, value in values.items()})
      results[corrected][fixture]['mul_46'] = source_runner.graph_values['mul_46'].numpy().copy()
  assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
  for fixture in inputs:
    for name in ('add_46', 'mul_46'):
      original, corrected = results[False][fixture][name], results[True][fixture][name]
      assert original.shape == corrected.shape == (1, 512, 8, 8)
      assert original.dtype == corrected.dtype == np.float16
      assert np.isfinite(corrected).all()
      finite = np.isfinite(original)
      assert finite.all() == (fixture == 'working')
      if name == 'mul_46':
        np.testing.assert_array_equal(corrected[finite], original[finite])
    # The old fused half epilogue may round differently at the residual. Check
    # the promised materialized FP16 interface exactly, not a wide tolerance.
    expected_residual = np.add(results[True][fixture]['mul_46'], inputs[fixture]['conv2d_54'], dtype=np.float16)
    np.testing.assert_array_equal(results[True][fixture]['add_46'], expected_residual)
  assert np.count_nonzero(~np.isfinite(results[False]['overflow']['mul_46'])) == 1
  assert results[True]['overflow']['mul_46'][0, 374, 1, 5] == np.float16(-14288)


def test_full_map_encoder_native_candidate(source_path, tmp_path, cpu_runtime):
  """Exercise real downstream fusion/packing and all 64 map-feature outputs."""
  onnx = pytest.importorskip('onnx')
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.nn.onnx import OnnxRunner
  from openpilot.sunnypilot.modeld_v2.helpers import load_model, write_bundle
  from openpilot.sunnypilot.modeld_v2.nav_map_precision import SOURCE_SHA256, apply_nav_map_precision_fix, verify_compiled_precision
  from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit

  source = onnx.load(source_path, load_external_data=False)
  model = onnx.utils.Extractor(source).extract_model(['nav_map'], ['select_11'])
  del source
  path = tmp_path / 'map.onnx'
  onnx.save(model, path)
  maps = synthetic_maps()

  def make_run(runner):
    @TinyJit
    def run(nav_map, output_buffers):
      Tensor.realize(output_buffers['select_11'].assign(runner({'nav_map': nav_map})['select_11']))
    return run

  def arguments(raster):
    return {'nav_map': Tensor(raster.copy(), device='CPU').realize(),
            'output_buffers': {'select_11': Tensor.zeros(1, 64, dtype=dtypes.float16, device='CPU').realize()}}

  for corrected in (False, True):
    runner = OnnxRunner(path)
    correction = apply_nav_map_precision_fix(runner, SOURCE_SHA256) if corrected else None
    run = make_run(runner)
    kwargs = arguments(maps['working'])
    with record_kernel_linear() as recorded:
      for _ in range(3):
        run(**kwargs)
    assert len(recorded) == 1
    check, _ = capture_frame(run, kwargs, 'output_buffers')
    part = serialize_jit(run, recorded[0], check)
    if correction:
      assert verify_compiled_precision(part, recorded[0], correction)['status'] == 'passed_compiled_ir_and_native_buffer_checks'
    artifact = tmp_path / f'map-{corrected}.sptg'
    write_bundle(str(artifact), {'run': part})
    native = load_model(str(artifact), cache=False, verify=True)['run']
    for fixture, raster in maps.items():
      results = []
      for execute in (run, native):
        kwargs = arguments(raster)
        execute(**kwargs)
        results.append(kwargs['output_buffers']['select_11'].numpy().copy())
        np.testing.assert_array_equal(kwargs['nav_map'].numpy(), raster)
      np.testing.assert_array_equal(*results)
      assert results[0].dtype == np.float16 and results[0].shape == (1, 64)
      assert int(np.isfinite(results[0]).sum()) == (64 if corrected or fixture == 'working' else 0)


@pytest.mark.skipif(os.environ.get('NAV_BIGV2_FULL_REFERENCE') != '1', reason='Optional full CPU reference needs about 6 GB RAM')
def test_full_reference_same_inputs_and_recurrent_state(source_path, tmp_path):
  """Independent ONNX arithmetic reference, not execution of the AMD candidate.

  Four continuous reference frames start from zero state. Every corrected call
  receives the exact original pre-state, including after the overflow raster.
  ORT may keep FP32 intermediates even for the original FP16 graph; equality here
  establishes preserved full-model contracts, not original AMD numerical safety.
  """
  onnx = pytest.importorskip('onnx')
  from openpilot.sunnypilot.modeld_v2.model_adapters import diagnostic_output_slices
  maps = synthetic_maps()
  for corrected in (False, True):
    model = onnx.load(source_path, load_external_data=False)
    signature = [value.SerializeToString() for value in (*model.graph.input, *model.graph.output)]
    weights = {value.name: hashlib.sha256(value.SerializeToString()).hexdigest() for value in model.graph.initializer}
    metadata = {value.key: value.value for value in model.metadata_props}
    if corrected:
      model = precision_repair_experiment(model)
      assert signature == [value.SerializeToString() for value in (*model.graph.input, *model.graph.output)]
      assert weights == {value.name: hashlib.sha256(value.SerializeToString()).hexdigest() for value in model.graph.initializer}
      assert metadata == {value.key: value.value for value in model.metadata_props}
    custom = [node for node in model.graph.node if node.domain]
    assert len(custom) == 1
    node = custom[0]
    assert node.domain == 'org.tinygrad' and node.op_type == 'Contiguous' and len(node.input) == len(node.output) == 1 and not node.attribute
    node.domain, node.op_type = '', 'Identity'  # Contiguity is not an ONNX Runtime value transformation.
    session = _session(model)
    del model
    gc.collect()
    types = {'tensor(uint8)': np.uint8, 'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(int64)': np.int64}
    values = {spec.name: np.zeros(spec.shape, types[spec.type]) for spec in session.get_inputs()}
    values['action_t'][:] = [.2595362067222595, .625]
    values['traffic_convention'][:] = [1, 0]
    state_names = [name for name in values if name.startswith('state_')]
    assert len(state_names) == 5
    names = [output.name for output in session.get_outputs()]
    slices = diagnostic_output_slices(metadata['output_slices'], 18580)
    for index, fixture in enumerate(('working', 'overflow', 'working', 'overflow')):
      before_path, after_path = tmp_path / f'pre-{index}.npz', tmp_path / f'out-{index}.npz'
      if not corrected:
        values['nav_map'][:] = maps[fixture]
        np.savez(before_path, **values)
      with np.load(before_path, allow_pickle=False) as recorded:
        inputs = {name: recorded[name] for name in recorded.files}
      result = dict(zip(names, session.run(names, inputs), strict=True))
      assert all(np.isfinite(value).all() for value in result.values())
      if not corrected:
        np.savez(after_path, **result)
      with np.load(after_path, allow_pickle=False) as original:
        for name, value in result.items():
          assert value.shape == original[name].shape and value.dtype == original[name].dtype
          np.testing.assert_array_equal(value, original[name])
        for span in slices.values():
          np.testing.assert_array_equal(result['outputs'].reshape(-1)[span], original['outputs'].reshape(-1)[span])
      for name in state_names:
        values[name] = result['next_' + name].copy()
    del session, values, result
    gc.collect()
