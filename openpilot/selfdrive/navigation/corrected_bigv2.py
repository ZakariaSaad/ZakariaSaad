"""Reviewed corrected-artifact identity and data-only precision evidence checks.

These bindings describe the user's existing compiled file. They grant no driving
approval and never execute a kernel, unpickle extras or alter the source model.
"""
import hashlib
from pathlib import Path
import struct

from openpilot.selfdrive.navigation.experimental_catalog import MAX_HEADER_BYTES, SOURCE_ONNX_SHA256, TINYGRAD_REF, _json, _require

MODEL_ID = 'big-nav-v2-map-fp32'
DISPLAY_NAME = 'Nav Model Integration Bigv2 — FP32 map correction'
ARTIFACT_SHA256 = '4f4c215db8ad33f2fa074fb6075e2a48ae5c8d766e3c2f9842acf8094c1a919c'
REPORT_SHA256 = '3f776eafc180c87fc85333bd029c4ee2bddf206b00ce7df695eba54e306af591'
COMPILER_SHA256 = '0a1c5174d7a27faf03b0304defd2dac6fa1b1cec036165b7dfd42ef0e6a06ce0'
COMPILER_REVISION = 'b9aba01d3a25f0a6c86707fe3aa52b29dcb918de'
IMPLEMENTATION_SHA256 = '696d9c3b9949618d051ed6fafcfe69c7cd8bc0e2ae17ae0d94d7eeea8e174c4a'
# Deterministic portable sidecar from this report, the existing reviewed camera
# profile and native all-AMD contract. Registration must produce these bytes.
METADATA_SHA256 = 'cff885b4ef1af332582f6df847351b8d69c2da6e70979718a03e1931f7dbad95'
CORRECTION_ID = 'bigv2-map-fc2-fp32-through-gamma-v1'
FIXTURES = {'working': '6647562b678d42e16dd5b82e7f77b5c491e3bba2b21ddf56cfebb49d2356f52f',
            'overflow': 'cd62957449abf0fe5c0a79999ef7e19f2cf7dca6f7be85e59380348379951cd6'}


def validate_precision_report(report, contract, output_comparison):
  """Check the reviewed correction and both already executed map fixtures."""
  _require(report.get('artifact_sha256') == ARTIFACT_SHA256 and report.get('compiler_sha256') == COMPILER_SHA256
           and report.get('compiler_revision') == COMPILER_REVISION, 'Corrected artifact/compiler identity differs from the reviewed report')
  _require(report.get('candidate_activation_approved') is False, 'Compiler verification must not grant candidate activation approval')
  correction = report.get('map_precision_correction', {})
  _require(isinstance(correction, dict) and correction.get('id') == CORRECTION_ID
           and correction.get('implementation_sha256') == IMPLEMENTATION_SHA256
           and correction.get('source_onnx_sha256') == SOURCE_ONNX_SHA256
           and correction.get('compiler_sha256') == COMPILER_SHA256 and correction.get('compiler_revision') == COMPILER_REVISION
           and correction.get('tinygrad_revision') == TINYGRAD_REF, 'Corrected map precision provenance mismatch')
  _require(correction.get('operand_dtype') == 'float16' and correction.get('dot_bias_gamma_dtype') == 'float32'
           and correction.get('residual_input_dtype') == 'float16' and correction.get('materialized_fp32_shape') == [1, 512, 8, 8]
           and correction.get('original_matmul') == 'node_MatMul_599'
           and correction.get('requires_separate_registration_and_approval') is True, 'Corrected map precision contract mismatch')
  compiled = correction.get('compiled_precision', {})
  _require(compiled.get('status') == 'passed_compiled_ir_and_native_buffer_checks', 'Corrected compiled precision checks did not pass')
  for key in ('dot', 'bias_gamma'):
    evidence = compiled.get(key, {})
    _require(evidence.get('early_half_casts') == 0 and evidence.get('llvm_early_half_narrowing') is False
             and evidence.get('floating_arithmetic') == ['float32'], 'Corrected compiled map arithmetic narrows prematurely')
  boundary = compiled.get('residual_boundary', {})
  _require(boundary.get('materialized_dtype') == 'float16' and boundary.get('nbytes') == 65536,
           'Corrected residual interface differs from the original float16 contract')
  cases = report.get('map_fixture_verification', {}).get('cases', {})
  _require(isinstance(cases, dict) and set(cases) == set(FIXTURES), 'Both reviewed synthetic map cases are required')
  for name, digest in FIXTURES.items():
    case = cases[name]
    _require(case.get('synthetic_uint8_map_sha256') == digest and case.get('other_inputs_seed') == 42
             and case.get('independent_buffers') is True, 'Corrected map fixture/input-state identity mismatch')
    output_comparison(case.get('outputs'), contract)


def validate_precision_artifact(path: Path, header: dict, correction: dict) -> None:
  """Bind reported protected kernels and saved initializer bytes to the SPTG.

  The caller has already checked the full artifact hash and bounded native header.
  Only JSON-described byte ranges are read; pickle extras are never inspected.
  """
  with path.open('rb') as source:
    source.seek(8)
    size = struct.unpack('<Q', source.read(8))[0]
    _require(0 < size <= MAX_HEADER_BYTES, 'Corrected SPTG header exceeds bounds')
    container = _json(source.read(size))
    part = container['parts'][0]
    base = 16 + size + -(16 + size) % 256 + part['blob'][0]
    part_size = part['blob'][1] - part['blob'][0]

    def digest(bounds):
      _require(isinstance(bounds, list) and len(bounds) == 2 and all(type(value) is int for value in bounds)
               and 0 <= bounds[0] <= bounds[1] <= part_size, 'Corrected kernel/initializer range is invalid')
      source.seek(base + bounds[0])
      hasher, remaining = hashlib.sha256(), bounds[1] - bounds[0]
      while remaining:
        data = source.read(min(remaining, 1 << 20))
        _require(bool(data), 'Truncated corrected kernel/initializer bytes')
        hasher.update(data)
        remaining -= len(data)
      return hasher.hexdigest()

    calls = header['calls']
    compiled = correction['compiled_precision']
    for key in ('dot', 'bias_gamma'):
      evidence = compiled[key]
      index = evidence.get('call')
      _require(type(index) is int and 0 <= index < len(calls), 'Corrected map kernel index is invalid')
      call = calls[index]
      _require(call.get('kind') == 'program' and call.get('name') == evidence.get('name')
               and call.get('target') == evidence.get('target') and digest(call.get('lib')) == evidence.get('kernel_sha256'),
               'Corrected map kernel bytes/target differ from the reviewed compiler proof')
      outputs = [call['args'][slot] for slot in call['outs']]
      _require(len(outputs) == 1 and outputs[0].get('dtype') in ('dtypes.f32', 'dtypes.float32')
               and outputs[0].get('nbytes') == 131072, 'Corrected map kernel does not materialize the reported float32 buffer')
    for name, expected in correction['initializers'].items():
      matches = [buffer for buffer in header['buffers'] if buffer.get('nbytes') == expected['nbytes'] and 'init' in buffer
                 and digest(buffer['init']) == expected['sha256']]
      _require(bool(matches), 'Corrected model initializer bytes differ: ' + name)
