"""Opt-in gfx1200 cross-compilation; no device, model weights, or inference.

Run with NAV_BIGV2_AMD_CROSSCOMPILE=1 on a host with a gfx1200-capable LLVM.
This checks generated AMD code, not AMD numerical execution or performance.
"""
import os
import struct

import pytest


@pytest.mark.skipif(os.environ.get('NAV_BIGV2_AMD_CROSSCOMPILE') != '1', reason='Opt-in offline AMD cross-compilation')
def test_native_gfx1200_codegen_preserves_fp32_through_gamma(monkeypatch):
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import do_to_program
  from tinygrad.helpers import Context, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.uop.ops import Ops
  from openpilot.sunnypilot.modeld_v2.nav_map_precision import _map_matmul, _scaled_cast

  opened_before = set(Device._opened_devices)

  def refuse_device_access(_self, _device):
    raise AssertionError('Offline cross-compilation must not open a device')

  monkeypatch.setattr(type(Device), '__getitem__', refuse_device_access)
  with Context(DEV='CPU', BEAM=0):
    # Empty lazy tensors carry only the real span's shapes/dtypes. Never realize
    # them: this test has no input values, weights, GPU allocation or execution.
    left = Tensor.empty(1, 8, 8, 2048, device='CPU', dtype=dtypes.float16)
    weight = Tensor.empty(2048, 512, device='CPU', dtype=dtypes.float16)
    bias = Tensor.empty(512, device='CPU', dtype=dtypes.float16)
    gamma = Tensor.empty(1, 512, 1, 1, device='CPU', dtype=dtypes.float16)
    residual = Tensor.empty(1, 512, 8, 8, device='CPU', dtype=dtypes.float16)
    scaled = (_map_matmul(left, weight) + bias).permute(0, 3, 1, 2) * gamma
    calls = (_scaled_cast(scaled) + residual).schedule_linear().src
    assert len(calls) == 4
    assert all(call.src[0].op is Ops.SINK for call in calls)
    assert [tuple(argument.dtype for argument in call.src[1:]) for call in calls] == [
      (dtypes.float32, dtypes.float16, dtypes.float16),
      (dtypes.float32, dtypes.float32, dtypes.float16, dtypes.float16),
      (dtypes.float16, dtypes.float32),
      (dtypes.float16, dtypes.float16, dtypes.float16),
    ]
    renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200', 'USB'))
    sources = []
    for call in calls:
      program = do_to_program(call.src[0], renderer)
      assert program.arg.target == renderer.target
      source = [entry.arg for entry in program.src if entry.op is Ops.SOURCE]
      binary = [entry.arg for entry in program.src if entry.op is Ops.BINARY]
      assert len(source) == len(binary) == 1
      # Verify real AMD ELF output, without loading or executing the binary.
      assert binary[0][:6] == b'\x7fELF\x02\x01'
      assert struct.unpack_from('<H', binary[0], 18)[0] == 224  # EM_AMDGPU
      assert b'gfx1200' in binary[0]
      sources.append(source[0])

    dot, bias_gamma, cast, add_residual = sources
    assert 'llvm.amdgcn.wmma' in dot and 'float* noalias %data0' in dot
    assert 'half* noalias %data1' in dot and 'half* noalias %data2' in dot
    for source in (dot, bias_gamma):
      assert 'fptrunc' not in source, 'Half narrowing before gamma can restore the original overflow'
    assert 'fadd' in bias_gamma and 'fmul' in bias_gamma and 'fpext half' in bias_gamma
    assert 'float* noalias %data0' in bias_gamma
    assert 'fptrunc float' in cast and 'to half' in cast
    assert 'half* noalias %data0' in cast
    assert 'fadd' in add_residual and ' half ' in add_residual
    assert 'float' not in add_residual and 'fptrunc' not in add_residual
  assert set(Device._opened_devices) == opened_before
