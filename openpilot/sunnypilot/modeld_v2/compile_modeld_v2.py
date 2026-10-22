#!/usr/bin/env python3
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from openpilot.sunnypilot.modeld_v2.helpers import PlainJit, load_model, write_bundle
from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit


VERIFICATION_RTOL = VERIFICATION_ATOL = 1e-3


def initialize_check_inputs(arrays: dict, seed: int = 7):
  rng = np.random.default_rng(seed)
  for name, value in arrays.items():
    if value.dtype == np.uint8 or name == 'nav_map':
      # Float32 map inputs still carry raw pixels; normalization belongs to the graph.
      value[...] = rng.integers(0, 256, value.shape, dtype=np.uint8)
    else:
      value[...] = rng.standard_normal(value.shape) if value.dtype.kind == 'f' else rng.integers(0, 2 if value.dtype == np.bool_ else 16, value.shape)


def file_sha256(path: Path) -> str:
  with path.open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def write_verification_report(path: Path, report: dict):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', delete=False) as file:
      temporary = Path(file.name)
      json.dump(report, file, indent=2)
      file.write('\n')
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def tinygrad_provenance(root: Path) -> dict:
  def git(*args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None
  revision, status = git('rev-parse', 'HEAD'), git('status', '--porcelain', '--untracked-files=all')
  return {'root': str(root), 'revision': revision, 'dirty': bool(status) if status is not None else None, 'git_status': status}


def backend_provenance(targets: list[dict]) -> dict:
  no_op = any(target['device'] == 'NULL' for target in targets) or os.environ.get('NO_HIPCC', '0') not in ('', '0')
  if no_op:
    raise ValueError('No-op backends cannot produce a source/reload verification report')
  emulated = any(target.get('interface', '').startswith('MOCK') or target['device'] == 'PYTHON' for target in targets)
  kind = 'emulated' if emulated else 'host_cpu' if all(target['device'] == 'CPU' for target in targets) else 'physical_device'
  return {'kind': kind, 'emulated': emulated, 'targets': targets, 'hardware_validation': False,
          'scope': 'Source/reloaded artifact consistency only; no independent hardware, driving or model-quality certification'}


def compare_outputs(expected: dict, actual: dict, *, rtol=VERIFICATION_RTOL, atol=VERIFICATION_ATOL) -> dict:
  if expected.keys() != actual.keys():
    raise ValueError('Source/reloaded output names differ')
  results = {}
  for name, before in expected.items():
    after = actual[name]
    if before.shape != after.shape or before.dtype != after.dtype:
      raise ValueError(f'Output {name} shape/dtype differs after reload')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
      raise ValueError(f'Output {name} contains nonfinite source/reloaded values')
    delta = np.abs(before.astype(np.float64) - after.astype(np.float64))
    passed = bool(np.allclose(before, after, rtol=rtol, atol=atol, equal_nan=False))
    results[name] = {'shape': list(before.shape), 'dtype': str(before.dtype), 'source_finite': True, 'reloaded_finite': True,
                     'max_abs_difference': float(delta.max()) if delta.size else 0.0, 'passed': passed,
                     'source_sha256': hashlib.sha256(np.ascontiguousarray(before).tobytes()).hexdigest(),
                     'reloaded_sha256': hashlib.sha256(np.ascontiguousarray(after).tobytes()).hexdigest()}
    if not passed:
      raise ValueError(f'Output {name} differs after reload: max_abs_difference={results[name]["max_abs_difference"]}')
  return results


def verify_reloaded(source, reloaded, input_specs, output_specs, allocate_inputs, benchmark, seeds: list[int]) -> dict:
  if len(seeds) < 2 or len(set(seeds)) != len(seeds):
    raise ValueError('Verification needs at least two independent deterministic seeds')
  if not output_specs:
    raise ValueError('Verification needs named model outputs')

  def evaluate(run, seed):
    # Allocate all inputs, recurrent histories and output buffers independently for every call.
    inputs = allocate_inputs(input_specs, partial(initialize_check_inputs, seed=seed))
    outputs = allocate_inputs(output_specs)
    benchmark(run, output_buffers=outputs, **inputs)
    return {name: tensor.numpy().copy() for name, tensor in outputs.items()}

  results, first, last = [], None, {}
  for seed in seeds:
    expected = evaluate(source, seed)
    compare_outputs(expected, expected)  # Reject nonfinite source values before replaying the artifact.
    repeatability = compare_outputs(expected, evaluate(source, seed), rtol=0, atol=0) if first is None else None
    actual = evaluate(reloaded, seed)
    results.append({'seed': seed, 'source_repeatability': repeatability, 'outputs': compare_outputs(expected, actual)})
    if first is None:
      first = expected
    last = expected
  assert first is not None
  primary = 'outputs' if 'outputs' in first else next(iter(first))
  if np.array_equal(first[primary], last[primary]):
    raise ValueError(f'Primary output {primary} did not change across seeds; cannot exclude a no-op execution')
  return {'seeds': seeds, 'rtol': VERIFICATION_RTOL, 'atol': VERIFICATION_ATOL, 'equal_nan': False,
          'independent_buffers': True, 'primary_output_changes_with_seed': True, 'runs': results}


def main():
  sys.path.insert(0, os.environ.get("TINYGRAD_PATH", "tinygrad_repo"))
  from tinygrad import Device

  parser = argparse.ArgumentParser(description="Compile a native-architecture ONNX model into a plain tinygrad artifact")
  parser.add_argument('onnx')
  parser.add_argument('output')
  parser.add_argument('--device-input', action='append', default=[])
  parser.add_argument('--benchmark-runs', type=int, default=20)
  parser.add_argument('--verification-runs', type=int, default=2, help='Independent deterministic seeds, at least two')
  parser.add_argument('--verification-output', type=Path, help='Reproducibility JSON; defaults to OUTPUT.verification.json')
  args = parser.parse_args()
  if args.verification_runs < 2 or args.benchmark_runs < 0:
    parser.error('verification-runs must be at least two and benchmark-runs must be nonnegative')
  verification_path = args.verification_output or Path(args.output + '.verification.json')
  args.verification_output = verification_path
  if verification_path.resolve() in (Path(args.onnx).resolve(), Path(args.output).resolve()):
    parser.error('Verification JSON must be separate from the source and compiled artifact')
  if Path(args.onnx).resolve() == Path(args.output).resolve():
    parser.error('Compiled output must not replace the source ONNX')

  import tinygrad
  from tinygrad.helpers import DEV
  configured_target = vars(DEV.target(Device.DEFAULT))
  report = {'status': 'compiling', 'source_onnx': str(Path(args.onnx).resolve()), 'source_onnx_sha256': file_sha256(Path(args.onnx)),
            'compiler_sha256': file_sha256(Path(__file__)), 'artifact': str(Path(args.output).resolve()),
            'tinygrad': tinygrad_provenance(Path(tinygrad.__file__).resolve().parent.parent),
            'backend': {'targets': [configured_target], 'hardware_validation': False}, 'artifact_promoted': False,
            'environment': {name: os.environ.get(name) for name in ('DEV', 'BEAM', 'NOOPT', 'TC', 'TC_OPT', 'TC_MIN_GLOBALS',
                            'OPENPILOT_HACKS', 'ASM_CALL', 'ASM_CALL_BACKEND', 'NO_HIPCC', 'LLVMOPT', 'GMMU', 'AMD_AQL',
                            'AMD_DISABLE_SDMA', 'GRAPH_ONE_KERNEL')},
            'benchmark_context': {'OPENPILOT_HACKS': 1} | ({'TC_OPT': 2, 'TC_MIN_GLOBALS': 32} if Device.DEFAULT == 'AMD' else {}),
            'verification': {'seeds': list(range(42, 42 + args.verification_runs)), 'rtol': VERIFICATION_RTOL,
                             'atol': VERIFICATION_ATOL, 'equal_nan': False, 'status': 'pending'}}
  write_verification_report(verification_path, report)
  try:
    report['backend'] = backend_provenance([configured_target])
    compile_and_verify(args, report)
  except BaseException as error:
    report.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
  finally:
    # A failed compilation never promotes an unchecked artifact over an existing model.
    write_verification_report(verification_path, report)


def compile_and_verify(args, report):
  from examples.openpilot.compile_onnx import onnx_metadata
  from examples.openpilot.helpers import allocate_inputs, benchmark
  from tinygrad import Device, Tensor, TinyJit
  from tinygrad.dtype import _to_np_dtype
  from tinygrad.nn.onnx import OnnxRunner

  runner, (properties, output_shapes) = OnnxRunner(args.onnx), onnx_metadata(args.onnx)
  device_inputs = runner.graph_inputs.keys() if '*' in args.device_input else args.device_input
  metadata = {'metadata': properties} | {f'{kind}_shapes': {name: tuple(d if isinstance(d, int) else 0 for d in shape) for name, shape in shapes.items()}
                                         for kind, shapes in [('input', {name: spec.shape for name, spec in runner.graph_inputs.items()}),
                                                              ('output', output_shapes)]}
  input_specs = {name: (tuple(s if isinstance(s, int) else 1 for s in spec.shape), np.dtype(_to_np_dtype(spec.dtype)).str,
                        Device.DEFAULT if name in device_inputs else 'NPY') for name, spec in runner.graph_inputs.items()}

  def model(inputs):
    return {name: value.contiguous() for name, value in runner({name: value.to(Device.DEFAULT) for name, value in inputs.items()}).items()}
  output_specs = {name: (value.shape, np.dtype(_to_np_dtype(value.dtype)).name, Device.DEFAULT) for name, value in model(allocate_inputs(input_specs)).items()}

  @TinyJit(prune=True)
  def jit(output_buffers, **inputs):
    Tensor.realize(*(output_buffers[name].assign(value) for name, value in model(inputs).items()))

  check_inputs = allocate_inputs(input_specs, initialize_check_inputs) | {'output_buffers': allocate_inputs(output_specs)}
  with record_kernel_linear() as recorded:
    for i in range(2 + args.benchmark_runs):
      benchmark(jit, cb=lambda t, i=i: print(f"  [{i + 1}/{2 + args.benchmark_runs}] {t * 1e3:.2f} ms"), **check_inputs)
  assert len(recorded) == 1, f"expected one captured kernel list, got {len(recorded)}"

  frame, _ = capture_frame(jit, check_inputs, 'output_buffers')
  for index, output in enumerate(frame['outputs']):
    if not np.isfinite(output).all():
      raise ValueError(f'Compile-time output {index} contains nonfinite values')
  part = serialize_jit(jit, recorded[0], frame)
  PlainJit(part[0], memoryview(part[1]))
  targets = list({json.dumps(call['target'], sort_keys=True): call['target'] for call in part[0]['calls'] if call['kind'] == 'program'}.values())
  report['backend'] = backend_provenance(targets)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(dir=Path(args.output).resolve().parent, prefix=Path(args.output).name + '.verification-', delete=False) as file:
      temporary = Path(file.name)
    write_bundle(str(temporary), {'run': part}, {'metadata': metadata, 'input_specs': input_specs, 'output_specs': output_specs})
    loaded = load_model(str(temporary), cache=False, verify=False)
    seeds = list(range(42, 42 + args.verification_runs))
    report['verification'] = {'seeds': seeds, 'rtol': VERIFICATION_RTOL, 'atol': VERIFICATION_ATOL, 'equal_nan': False, 'status': 'running'}
    report['verification'] = verify_reloaded(jit, loaded['run'], input_specs, output_specs, allocate_inputs, benchmark, seeds)
    report['artifact_sha256'] = file_sha256(temporary)
    # Persist the complete successful comparison before promoting the artifact.
    # A later I/O failure can leave an explicit unpromoted report, never a stale pass.
    report['status'] = 'verified_pending_promotion'
    write_verification_report(args.verification_output, report)
    os.replace(temporary, args.output)
    report.update(status='passed_source_reload_consistency', artifact_promoted=True)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)
  print(f"Saved model to {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB, {len(part[0]['calls'])} kernels)")


if __name__ == '__main__':
  main()
