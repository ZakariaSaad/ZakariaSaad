"""CPU/NPY-only timing instrumentation tests, not QCOM/Chestnut benchmarks."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tinygrad.engine import jit, realize
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import Ops, UOp

from openpilot.sunnypilot.modeld_v2.helpers import _Jit
from openpilot.sunnypilot.modeld_v2.model_adapters import NativeTinygradAdapter
from openpilot.sunnypilot.modeld_v2.tests.test_driving_navigation import fixture_data
from openpilot.tools.replay.nav_fallback_profile import OfflineInferenceProfile


class TestOfflineInferenceProfile(unittest.TestCase):
  def test_actual_native_adapter_preserves_output_and_aliased_state(self):
    jits, _ = fixture_data()

    def graph(host, output_buffers, **states):
      output = output_buffers['outputs']
      np.frombuffer(output._buffer().as_memoryview(allow_zero_copy=True), np.float32)[:] = 7
      for name, tensor in states.items():
        state = np.frombuffer(tensor._buffer().as_memoryview(allow_zero_copy=True), tensor.numpy().dtype)
        state[:] += 1
        self.assertEqual(tensor._buffer().base, output_buffers[f'next_{name}']._buffer().base)

    jits['run'] = graph
    adapter = NativeTinygradAdapter(jits, 1344, 760, 'NPY', 'NPY', 'NPY', diagnostic=True)
    frames = {'img': bytes(1622016), 'big_img': bytes(1622016)}
    original_run = adapter.run
    barrier = Mock()
    with OfflineInferenceProfile(adapter, synchronize=barrier) as profile:
      adapter.copy_frames(frames)
      actual = adapter.run().numpy()
    self.assertEqual(barrier.call_count, 2)
    self.assertTrue((actual == 7).all())
    self.assertEqual(adapter.run, original_run)
    for state in adapter.input_queues.values():
      self.assertTrue((state.numpy() == 1).all())
    report = profile.report()
    self.assertEqual(report['configured_adapter_devices'], ['NPY'])
    self.assertFalse(report['physical_hardware_identity_verified'])
    self.assertFalse(report['kernel_and_transfer_costs_separated'])
    self.assertFalse(report['failed'])
    for name in ('camera_host_copy_or_binding', 'fused_host_submission', 'adapter_host_submission',
                 'device_completion_wait', 'pre_invocation_wait', 'output_numpy_wait_and_copy'):
      self.assertGreater(report['stages'][name]['host_wall_ms'], 0)
    json.dumps(report, allow_nan=False)

  def test_real_captured_jit_links_only_on_first_invocation(self):
    # Same lazy cached_property used by cached ordinary models; no device kernels.
    captured = jit.CapturedJit(None, UOp(Ops.LINEAR, src=()), [], [])
    runner = _Jit(captured, [], [])
    with OfflineInferenceProfile() as first:
      runner()
    with OfflineInferenceProfile() as second:
      runner()
    self.assertEqual(first.report()['stages']['first_use_link']['calls'], 1)
    self.assertNotIn('first_use_link', second.report()['stages'])

  def test_runtime_creation_and_cache_hit_are_distinguished(self):
    ast = SimpleNamespace(key=b'isolated-test-key')
    cache = {}

    def runtime(device, program, cache=True):
      return realize.runtime_cache.setdefault((program.key, device), object())

    with patch.object(realize, 'runtime_cache', cache), patch.object(realize, 'get_runtime', runtime):
      with OfflineInferenceProfile() as profile:
        first = realize.get_runtime('NPY', ast)
        second = realize.get_runtime('NPY', ast)
    self.assertIs(first, second)
    self.assertEqual(profile.report()['stages']['runtime_creation']['calls'], 1)
    self.assertEqual(profile.report()['stages']['runtime_cache_hit']['calls'], 1)

  def test_failure_propagates_and_every_wrapper_is_restored(self):
    failure = RuntimeError('original model output not finite')
    adapter = SimpleNamespace(run=Mock(side_effect=failure), copy_frames=Mock())
    before = (adapter.run, adapter.copy_frames, Tensor.numpy, jit.link_linear, realize.get_runtime)
    profile = OfflineInferenceProfile(adapter)
    with self.assertRaises(RuntimeError) as caught, profile:
      adapter.run()
    self.assertIs(caught.exception, failure)
    self.assertEqual(before, (adapter.run, adapter.copy_frames, Tensor.numpy, jit.link_linear, realize.get_runtime))
    self.assertTrue(profile.report()['failed'])
    self.assertGreater(profile.report()['invocation_wall_ms'], 0)

  def test_failed_initial_barrier_restores_wrappers_without_inference(self):
    adapter = SimpleNamespace(run=Mock())
    original_numpy = Tensor.numpy
    with self.assertRaisesRegex(RuntimeError, 'lost device'):
      with OfflineInferenceProfile(adapter, synchronize=Mock(side_effect=RuntimeError('lost device'))):
        self.fail('An initial barrier failure must prevent inference')
    self.assertIs(Tensor.numpy, original_numpy)
    adapter.run.assert_not_called()


if __name__ == '__main__':
  unittest.main()
