"""Focused process/cached-firmware checks; no GPU, vehicle publishers or services."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openpilot.sunnypilot.modeld_v2 import compile_offroad as guard


class TestOffroadCompiler(unittest.TestCase):
  def test_normal_host_compiler_is_unchanged(self):
    with patch.dict(os.environ, {}, clear=True), patch.object(guard.subprocess, 'Popen') as start:
      self.assertIsNone(guard.offroad_compile(['source.onnx', 'new.sptg']))
    start.assert_not_called()

  def test_agnos_correction_requires_current_monitored_process(self):
    with patch.object(guard.Path, 'is_file', return_value=True), patch.object(guard, '_monitored_pid', None):
      with self.assertRaisesRegex(guard.OffroadCompileError, '--offroad'):
        guard.require_compile_monitor()
    with patch.object(guard.Path, 'is_file', return_value=True), patch.object(guard, '_monitored_pid', os.getpid()):
      guard.require_compile_monitor()

  def test_child_accepts_inherited_pipe_and_installs_parent_death_protection(self):
    read_fd, write_fd = os.pipe()
    try:
      os.write(write_fd, json.dumps({'parent_pid': os.getppid()}).encode())
      libc = SimpleNamespace(prctl=Mock(return_value=0))
      with patch.dict(os.environ, {guard.MONITOR_FD: str(read_fd)}), patch.object(guard.ctypes, 'CDLL', return_value=libc), \
           patch.object(guard.signal, 'signal') as signals, patch.object(guard, '_monitored_pid', None):
        self.assertIsNone(guard.offroad_compile(['--offroad', '--nav-map-precision-fix']))
        self.assertEqual(guard._monitored_pid, os.getpid())
        self.assertNotIn(guard.MONITOR_FD, os.environ)
      libc.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0)
      self.assertEqual({call.args[0] for call in signals.call_args_list}, {signal.SIGTERM, signal.SIGHUP})
      with self.assertRaises(OSError):
        os.fstat(read_fd)
    finally:
      os.close(write_fd)

  def test_forged_or_disappearing_parent_is_refused(self):
    for parent_reads in ([42], [42, 43]):
      read_fd, write_fd = os.pipe()
      try:
        os.write(write_fd, json.dumps({'parent_pid': 42 if len(parent_reads) == 2 else 41}).encode())
        libc = SimpleNamespace(prctl=Mock(return_value=0))
        with patch.dict(os.environ, {guard.MONITOR_FD: str(read_fd)}), patch.object(guard.ctypes, 'CDLL', return_value=libc), \
             patch.object(guard.os, 'getppid', side_effect=parent_reads), self.assertRaises(guard.OffroadCompileError):
          guard._child_admission()
      finally:
        os.close(write_fd)

  @contextmanager
  def state_modules(self):
    params = Mock(get_bool=Mock(return_value=True))
    sm = SimpleNamespace(seen={'deviceState': True, 'managerState': True}, update=Mock())
    messaging = SimpleNamespace(SubMaster=Mock(return_value=sm))
    modules = {'openpilot.common.params': SimpleNamespace(Params=Mock(return_value=params)),
               'openpilot.cereal': SimpleNamespace(messaging=messaging)}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
      yield params, sm, messaging

  def test_admission_failure_never_starts_compiler(self):
    with self.state_modules(), patch.object(guard, 'offroad_reason', return_value='deviceState.started is true'), \
         patch.object(guard.subprocess, 'Popen') as start, self.assertRaisesRegex(guard.OffroadCompileError, 'deviceState.started'):
      guard.offroad_compile(['--offroad', '--nav-map-precision-fix'])
    start.assert_not_called()

  def test_success_keeps_service_and_params_unchanged(self):
    child = Mock(pid=456, returncode=0)
    child.poll.side_effect = [None, 0, 0]
    with self.state_modules() as (params, sm, messaging), patch.object(guard, 'offroad_reason', return_value='') as admission, \
         patch.object(guard.subprocess, 'Popen', return_value=child) as start, patch.object(guard.os, 'killpg') as kill:
      self.assertEqual(guard.offroad_compile(['source.onnx', 'new.sptg', '--offroad', '--nav-map-precision-fix']), 0)
    command = start.call_args.args[0]
    self.assertEqual(command[:4], [sys.executable, '-B', '-m', 'openpilot.sunnypilot.modeld_v2.compile_modeld_v2'])
    self.assertIn('--offroad', command)
    self.assertTrue(start.call_args.kwargs['start_new_session'])
    self.assertEqual(len(start.call_args.kwargs['pass_fds']), 1)
    self.assertEqual(admission.call_count, 3)
    self.assertEqual(sm.update.call_count, 3)
    messaging.SubMaster.assert_called_once_with(list(guard.SERVICES), poll='deviceState')
    params.put.assert_not_called()
    params.put_bool.assert_not_called()
    kill.assert_not_called()
    child.wait.assert_called_once()

  def test_withdrawal_stops_and_reaps_whole_compiler_group(self):
    child = Mock(pid=456, returncode=None)
    child.poll.return_value = None
    with self.state_modules(), patch.object(guard, 'offroad_reason', side_effect=['', 'managerState is stale']), \
         patch.object(guard.subprocess, 'Popen', return_value=child), patch.object(guard.os, 'killpg') as kill, \
         self.assertRaisesRegex(guard.OffroadCompileError, 'managerState is stale'):
      guard.offroad_compile(['--offroad', '--nav-map-precision-fix'])
    kill.assert_called_once_with(456, signal.SIGTERM)
    self.assertEqual(child.wait.call_count, 2)

  def test_successful_child_exit_does_not_hide_final_admission_loss(self):
    child = Mock(pid=456, returncode=0)
    child.poll.return_value = 0
    with self.state_modules(), patch.object(guard, 'offroad_reason', side_effect=['', 'deviceState.started is true']), \
         patch.object(guard.subprocess, 'Popen', return_value=child), \
         self.assertRaisesRegex(guard.OffroadCompileError, 'admission withdrawn'):
      guard.offroad_compile(['--offroad', '--nav-map-precision-fix'])
    child.wait.assert_called_once()

  def test_keyboard_interrupt_releases_child_before_return(self):
    child = Mock(pid=456, returncode=None)
    child.poll.return_value = None
    with self.state_modules(), patch.object(guard, 'offroad_reason', return_value=''), \
         patch.object(guard.subprocess, 'Popen', return_value=child), patch.object(guard, '_watch_child', side_effect=KeyboardInterrupt), \
         patch.object(guard.os, 'killpg') as kill, self.assertRaises(KeyboardInterrupt):
      guard.offroad_compile(['--offroad', '--nav-map-precision-fix'])
    kill.assert_called_once_with(456, signal.SIGTERM)
    child.wait.assert_called_with()

  def test_unresponsive_child_is_killed_then_reaped(self):
    child = Mock(pid=456)
    child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired('compiler', 2), -signal.SIGKILL]
    with patch.object(guard.os, 'killpg') as kill:
      guard._stop_child(child)
    self.assertEqual([call.args for call in kill.call_args_list], [(456, signal.SIGTERM), (456, signal.SIGKILL)])
    child.wait.assert_called_with()


class TestCachedMicrocode(unittest.TestCase):
  @contextmanager
  def fake_runtime(self, cache):
    helpers = ModuleType('tinygrad.helpers')
    helpers.cache_dir = str(cache)
    amdev = ModuleType('tinygrad.runtime.support.am.amdev')

    def fetch_fw(path, name, sha256):
      # Same pinned fetch_fw fallback contract; the undefined fetch must only
      # resolve to the context's cache-only dependency, never the network.
      return fetch(f'https://firmware.invalid/{path}/{name}', subdir='fw', sha256=sha256).read_bytes()  # noqa: F821

    amdev.fetch_fw = fetch_fw
    modules = {name: ModuleType(name) for name in ('tinygrad', 'tinygrad.runtime', 'tinygrad.runtime.support', 'tinygrad.runtime.support.am')}
    modules |= {'tinygrad.helpers': helpers, 'tinygrad.runtime.support.am.amdev': amdev}
    for name, module in modules.items():
      if '.' in name:
        parent, child = name.rsplit('.', 1)
        setattr(modules[parent], child, module)
    with patch.dict(sys.modules, modules), patch('openpilot.sunnypilot.modeld_v2.helpers.patch_fetch_fw') as zstd:
      yield amdev, fetch_fw, zstd

  def test_verified_cache_is_reused_and_hook_is_restored(self):
    with tempfile.TemporaryDirectory() as directory, self.fake_runtime(directory) as (amdev, original, zstd):
      contents = b'local GPU microcode'
      root = Path(directory) / 'downloads/fw'
      root.mkdir(parents=True)
      (root / hashlib.md5(b'https://firmware.invalid/amdgpu/test.bin').hexdigest()).write_bytes(contents)
      with guard.cached_gpu_firmware():
        self.assertEqual(amdev.fetch_fw('amdgpu', 'test.bin', hashlib.sha256(contents).hexdigest()), contents)
      self.assertIs(amdev.fetch_fw, original)
      zstd.assert_called_once()

  def test_missing_or_corrupt_cache_fails_without_download(self):
    for contents in (None, b'incorrect'):
      with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory, self.fake_runtime(directory) as (amdev, original, _):
        root = Path(directory) / 'downloads/fw'
        root.mkdir(parents=True)
        if contents is not None:
          (root / hashlib.md5(b'https://firmware.invalid/amdgpu/test.bin').hexdigest()).write_bytes(contents)
        with self.assertRaisesRegex(guard.OffroadCompileError, 'does not download firmware'):
          with guard.cached_gpu_firmware():
            amdev.fetch_fw('amdgpu', 'test.bin', hashlib.sha256(b'expected').hexdigest())
        self.assertIs(amdev.fetch_fw, original)


if __name__ == '__main__':
  unittest.main()
