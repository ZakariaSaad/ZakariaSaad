"""Source/reloaded compiler checks, including a real CPU SPTG round trip."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.sunnypilot.modeld_v2 import compile_modeld_v2 as compiler


class Array:
  def __init__(self, value):
    self.value = value

  def numpy(self):
    return self.value


def allocate_arrays(specs, initialize=None):
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in specs.items()}
  if initialize is not None:
    initialize(arrays)
  return {name: Array(value) for name, value in arrays.items()}


INPUTS = {'x': ((16,), 'float32', 'CPU'), 'state_q': ((16,), 'uint8', 'CPU')}
OUTPUTS = {'outputs': ((16,), 'float32', 'CPU'), 'next_state_q': ((16,), 'uint8', 'CPU')}


class SyntheticRunner:
  def __init__(self):
    from tinygrad import dtypes
    self.graph_inputs = {'x': SimpleNamespace(shape=(16,), dtype=dtypes.float32),
                         'state_q': SimpleNamespace(shape=(16,), dtype=dtypes.uint8)}

  def __call__(self, inputs):
    return {'outputs': inputs['x'] * 2, 'next_state_q': inputs['state_q']}


def source_arrays(output_buffers, x, state_q):
  output_buffers['outputs'].value[:] = x.value * 2
  output_buffers['next_state_q'].value[:] = state_q.value


def benchmark_arrays(run, **inputs):
  run(**inputs)


def test_full_raw_pixel_range_and_reproducible_seeds():
  def inputs():
    return {'camera': np.zeros(65536, np.uint8), 'nav_map': np.zeros((1, 1, 256, 256), np.float32), 'navigation': np.zeros(47, np.float32)}
  before, repeat, different = inputs(), inputs(), inputs()
  compiler.initialize_check_inputs(before, seed=42)
  compiler.initialize_check_inputs(repeat, seed=42)
  compiler.initialize_check_inputs(different, seed=43)
  for name in before:
    np.testing.assert_array_equal(before[name], repeat[name])
    assert not np.array_equal(before[name], different[name])
  for name in ('camera', 'nav_map'):
    assert before[name].min() == 0 and before[name].max() == 255
    np.testing.assert_array_equal(before[name], np.floor(before[name]))


def test_each_replay_gets_independent_identical_seeded_inputs_and_outputs():
  calls = []

  def observe(output_buffers, **inputs):
    calls.append((inputs, output_buffers))
    source_arrays(output_buffers, **inputs)

  result = compiler.verify_reloaded(observe, observe, INPUTS, OUTPUTS, allocate_arrays, benchmark_arrays, [42, 43])
  assert len(calls) == 5  # source42, repeat42, reload42, source43, reload43
  for name in INPUTS:
    for index in (1, 2):
      np.testing.assert_array_equal(calls[0][0][name].value, calls[index][0][name].value)
    assert len({id(inputs[name].value) for inputs, _ in calls}) == 5
  for name in OUTPUTS:
    assert len({id(outputs[name].value) for _, outputs in calls}) == 5
  assert result['seeds'] == [42, 43] and result['equal_nan'] is False
  assert all(set(run['outputs']) == set(OUTPUTS) for run in result['runs'])
  assert all(stats['max_abs_difference'] == 0 for run in result['runs'] for stats in run['outputs'].values())


def test_recurrent_output_corruption_is_rejected():
  def corrupted(output_buffers, **inputs):
    source_arrays(output_buffers, **inputs)
    output_buffers['next_state_q'].value[0] ^= 128

  with pytest.raises(ValueError, match='next_state_q differs'):
    compiler.verify_reloaded(source_arrays, corrupted, INPUTS, OUTPUTS, allocate_arrays, benchmark_arrays, [42, 43])


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_matching_nonfinite_outputs_never_pass(invalid):
  values = {'outputs': np.array([invalid], np.float32)}
  with pytest.raises(ValueError, match='nonfinite'):
    compiler.compare_outputs(values, values)


@pytest.mark.parametrize('seeds', [[42], [42, 42]])
def test_two_distinct_seeds_are_required(seeds):
  with pytest.raises(ValueError, match='two independent'):
    compiler.verify_reloaded(source_arrays, source_arrays, INPUTS, OUTPUTS, allocate_arrays, benchmark_arrays, seeds)


def test_zero_execution_cannot_receive_a_passed_verification():
  with pytest.raises(ValueError, match='no-op'):
    compiler.verify_reloaded(lambda **_: None, lambda **_: None, INPUTS, OUTPUTS, allocate_arrays, benchmark_arrays, [42, 43])


def test_backend_provenance_rejects_no_code_and_labels_emulation():
  with patch.dict('os.environ', {'NO_HIPCC': '0'}):
    with pytest.raises(ValueError, match='No-op'):
      compiler.backend_provenance([{'device': 'NULL'}])
    mock = compiler.backend_provenance([{'device': 'AMD', 'interface': 'MOCKKFD', 'arch': 'gfx1200'}])
    physical = compiler.backend_provenance([{'device': 'AMD', 'interface': 'USB', 'arch': 'gfx1200'}])
    assert mock['emulated'] is True and mock['kind'] == 'emulated'
    assert physical['emulated'] is False and physical['kind'] == 'physical_device'
    assert not mock['hardware_validation'] and not physical['hardware_validation']
  with patch.dict('os.environ', {'NO_HIPCC': '1'}):
    with pytest.raises(ValueError, match='No-op'):
      compiler.backend_provenance([{'device': 'AMD'}])


def test_failed_verification_reports_error_and_preserves_existing_artifact(tmp_path):
  source, output = tmp_path/'source.onnx', tmp_path/'model.sptg'
  source.write_bytes(b'synthetic input for provenance only')
  output.write_bytes(b'previous verified artifact')
  with patch('sys.argv', ['compile_modeld_v2.py', str(source), str(output)]), \
       patch.object(compiler, 'compile_and_verify', side_effect=ValueError('next_state_q differs after reload')):
    with pytest.raises(ValueError, match='next_state_q'):
      compiler.main()
  report = json.loads(Path(str(output) + '.verification.json').read_text())
  assert report['status'] == 'failed' and report['artifact_promoted'] is False
  assert report['source_onnx_sha256'] == compiler.file_sha256(source)
  assert report['compiler_sha256'] == compiler.file_sha256(Path(compiler.__file__))
  assert report['artifact'] == str(output) and report['benchmark_context'] == {'OPENPILOT_HACKS': 1}
  assert report['tinygrad']['revision'] and isinstance(report['tinygrad']['dirty'], bool)
  assert output.read_bytes() == b'previous verified artifact'


def test_report_replace_failure_preserves_previous_complete_json(tmp_path):
  path = tmp_path/'verification.json'
  path.write_text('{"previous":true}\n')
  with patch.object(compiler.os, 'replace', side_effect=OSError('read only destination')):
    with pytest.raises(OSError, match='read only'):
      compiler.write_verification_report(path, {'new': True})
  assert json.loads(path.read_text()) == {'previous': True}
  assert list(tmp_path.iterdir()) == [path]


def test_actual_reload_mismatch_cleans_staged_artifact_and_preserves_previous(tmp_path):
  source, output = tmp_path/'source.onnx', tmp_path/'model.sptg'
  source.write_bytes(b'synthetic runner stand-in; no downloaded pickle')
  output.write_bytes(b'previous verified artifact')
  actual_load = compiler.load_model

  def corrupted_load(*args, **kwargs):
    loaded = actual_load(*args, **kwargs)
    run = loaded['run']

    def corrupt(output_buffers, **inputs):
      run(output_buffers=output_buffers, **inputs)
      output_buffers['next_state_q'].assign(output_buffers['next_state_q'] + 1).realize()
    loaded['run'] = corrupt
    return loaded

  with patch('sys.argv', ['compile_modeld_v2.py', str(source), str(output), '--benchmark-runs', '0', '--device-input', '*']), \
       patch('tinygrad.nn.onnx.OnnxRunner', return_value=SyntheticRunner()), \
       patch('examples.openpilot.compile_onnx.onnx_metadata', return_value=({}, {'outputs': (16,), 'next_state_q': (16,)})), \
       patch.object(compiler, 'load_model', side_effect=corrupted_load):
    with pytest.raises(ValueError, match='next_state_q differs'):
      compiler.main()
  report = json.loads(Path(str(output) + '.verification.json').read_text())
  assert report['status'] == 'failed' and report['artifact_promoted'] is False
  assert output.read_bytes() == b'previous verified artifact'
  assert not list(tmp_path.glob('model.sptg.verification-*'))


def test_successful_cli_promotes_only_verified_artifact_with_reproducibility_json(tmp_path):
  source, output = tmp_path/'source.onnx', tmp_path/'model.sptg'
  source.write_bytes(b'synthetic runner stand-in; no downloaded pickle')
  output.write_bytes(b'previous artifact')
  with patch('sys.argv', ['compile_modeld_v2.py', str(source), str(output), '--benchmark-runs', '0', '--device-input', '*']), \
       patch('tinygrad.nn.onnx.OnnxRunner', return_value=SyntheticRunner()), \
       patch('examples.openpilot.compile_onnx.onnx_metadata', return_value=({}, {'outputs': (16,), 'next_state_q': (16,)})):
    compiler.main()
  report = json.loads(Path(str(output) + '.verification.json').read_text())
  assert report['status'] == 'passed_source_reload_consistency' and report['artifact_promoted'] is True
  assert report['artifact_sha256'] == compiler.file_sha256(output)
  assert report['backend']['kind'] == 'host_cpu' and report['backend']['hardware_validation'] is False
  assert report['verification']['seeds'] == [42, 43]
  assert all(set(run['outputs']) == set(OUTPUTS) for run in report['verification']['runs'])
  assert output.read_bytes().startswith(b'SPTGPLN1')
  assert not list(tmp_path.glob('model.sptg.verification-*'))


def test_real_cpu_source_to_reloaded_sptg_checks_every_output(tmp_path):
  from examples.openpilot.helpers import allocate_inputs, benchmark
  from tinygrad import Tensor, TinyJit
  from openpilot.sunnypilot.modeld_v2.helpers import load_model, write_bundle
  from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit

  @TinyJit(prune=True)
  def source(output_buffers, x, state_q):
    Tensor.realize(output_buffers['outputs'].assign(x * 2), output_buffers['next_state_q'].assign(state_q))

  inputs = allocate_inputs(INPUTS, compiler.initialize_check_inputs) | {'output_buffers': allocate_inputs(OUTPUTS)}
  with record_kernel_linear() as recorded:
    for _ in range(2):
      benchmark(source, **inputs)
  assert len(recorded) == 1
  check, _ = capture_frame(source, inputs, 'output_buffers')
  part = serialize_jit(source, recorded[0], check)
  path = tmp_path/'synthetic-cpu.sptg'
  write_bundle(str(path), {'run': part}, {'input_specs': INPUTS, 'output_specs': OUTPUTS})
  reloaded = load_model(str(path), cache=False, verify=False)['run']
  result = compiler.verify_reloaded(source, reloaded, INPUTS, OUTPUTS, allocate_inputs, benchmark, [42, 43])
  assert result['primary_output_changes_with_seed']
  assert all(stats['passed'] and stats['source_finite'] and stats['reloaded_finite']
             for run in result['runs'] for stats in run['outputs'].values())
  assert json.loads(json.dumps(result)) == result
