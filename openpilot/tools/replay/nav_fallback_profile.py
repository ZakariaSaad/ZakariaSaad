"""Optional host-side profiling for a single isolated model invocation.

Use around an already constructed ModelState in offline replay. This module does
not initialize models/devices, publish messages, warm models, or change fallback.
Nested durations are inclusive and must not be added together. GPU transfers
inside a fused schedule cannot be separated by these host timers.
"""
from contextlib import ExitStack
from functools import wraps
import threading
import time
from unittest.mock import patch


class OfflineInferenceProfile:
  """Time existing calls; an optional caller-owned barrier measures completion.

  ``with OfflineInferenceProfile(model.adapter, synchronize=barrier) as profile:``
  surrounds ``model.run(...)``. The barrier, when supplied, runs before the
  invocation and after adapter.run. It must synchronize only the replay's devices.
  No additional barrier is inserted between warp/policy kernels. Without it,
  submission timings include any existing waits but do not prove device latency.
  The original inference exception always propagates and the report remains
  available after exiting the context.
  """
  def __init__(self, adapter=None, *, synchronize=None):
    self.adapter = adapter
    self.synchronize = synchronize
    self.stages: dict[str, dict] = {}
    self.elapsed_ms = 0.0
    self.failed = False
    self._stack = ExitStack()
    self._thread = None

  def _measure(self, name, function, *args, **kwargs):
    if threading.get_ident() != self._thread:
      return function(*args, **kwargs)
    start = time.perf_counter()
    try:
      return function(*args, **kwargs)
    finally:
      entry = self.stages.setdefault(name, {'calls': 0, 'host_wall_ms': 0.0})
      entry['calls'] += 1
      entry['host_wall_ms'] += (time.perf_counter() - start) * 1000

  def _wrap(self, obj, attribute, name):
    original = getattr(obj, attribute)

    @wraps(original)
    def timed(*args, **kwargs):
      return self._measure(name, original, *args, **kwargs)

    self._stack.enter_context(patch.object(obj, attribute, timed))

  def __enter__(self):
    from tinygrad.engine import jit, realize
    from tinygrad.tensor import Tensor
    if self._thread is not None:
      raise RuntimeError('Use a new offline profile for each invocation')
    self._thread = threading.get_ident()
    try:
      self._wrap(jit, 'link_linear', 'first_use_link')
      self._wrap(Tensor, 'numpy', 'output_numpy_wait_and_copy')
      original_runtime = realize.get_runtime

      def runtime(device, ast, cache=True):
        category = 'runtime_cache_hit' if (ast.key, device) in realize.runtime_cache else 'runtime_creation'
        return self._measure(category, original_runtime, device, ast, cache=cache)

      self._stack.enter_context(patch.object(realize, 'get_runtime', runtime))
      if self.adapter is not None:
        for attribute, name in (('copy_frames', 'camera_host_copy_or_binding'), ('run_warp', 'warp_host_submission'),
                                ('run_policy', 'policy_host_submission'), ('run_model', 'fused_host_submission')):
          if hasattr(self.adapter, attribute):
            self._wrap(self.adapter, attribute, name)
        original_run = self.adapter.run

        def run(*args, **kwargs):
          result = self._measure('adapter_host_submission', original_run, *args, **kwargs)
          if self.synchronize is not None:
            self._measure('device_completion_wait', self.synchronize)
          return result

        self._stack.enter_context(patch.object(self.adapter, 'run', run))
      if self.synchronize is not None:
        self._measure('pre_invocation_wait', self.synchronize)
      self._start = time.perf_counter()
      return self
    except BaseException:
      self._stack.close()
      raise

  def __exit__(self, exc_type, exc_value, traceback):
    self.elapsed_ms = (time.perf_counter() - self._start) * 1000
    self.failed = exc_type is not None
    self._stack.close()
    return False

  def report(self) -> dict:
    adapter = self.adapter
    devices = sorted({getattr(adapter, name) for name in ('DEV', 'QUEUE_DEV', 'WARP_DEV')
                      if isinstance(getattr(adapter, name, None), str)}) if adapter is not None else []
    return {
      'scope': 'isolated_invocation_host_wall_times',
      'configured_adapter_devices': devices,
      'physical_hardware_identity_verified': False,
      'invocation_wall_ms': self.elapsed_ms,
      'failed': self.failed,
      'stages': self.stages.copy(),
      'completion_barrier_added': self.synchronize is not None,
      'durations_are_inclusive': True,
      'kernel_and_transfer_costs_separated': False,
      'limitation': 'Fused device transfers and kernel execution overlap; host stages are not individual GPU timings.',
    }
