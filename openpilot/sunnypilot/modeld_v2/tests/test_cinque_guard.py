import hashlib
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from openpilot.common.params import Params
from openpilot.cereal import custom
from openpilot.sunnypilot.models.helpers import get_active_model_runner
from openpilot.sunnypilot.modeld_v2.cinque_guard import CinqueGuard, verify_manifest
from tools.cinque_v3 import install_artifact, require_offroad


class TestCinqueGuard(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.params = Params(self.tmp.name)
    self.fallback, self.candidate = Mock(), Mock()
    self.fallback_factory, self.candidate_factory = Mock(return_value=self.fallback), Mock(return_value=self.candidate)

  def guard(self):
    return CinqueGuard(self.params, self.fallback_factory, self.candidate_factory)

  def test_default_retains_legacy_model(self):
    guard = self.guard()
    self.assertIsNone(guard.start())
    self.fallback_factory.assert_not_called()
    self.candidate_factory.assert_not_called()
    self.assertFalse(self.params.get_bool('CinqueV3Active'))

  def test_startup_requires_both_warmups(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    guard = self.guard()
    self.assertIs(guard.start(), self.candidate)
    self.fallback.warmup.assert_called_once()
    self.candidate.warmup.assert_called_once()
    self.assertTrue(self.params.get_bool('CinqueV3Active'))
    self.assertIsNone(self.params.get('CinqueV3Fault'))

  def test_explicit_request_bypasses_cache_but_fault_returns_to_normal(self):
    small = custom.ModelManagerSP.ModelBundle(runner='tinygrad', minimumSelectorVersion=19, ref='working-small')
    self.params.put('ModelManager_ActiveBundle', small.to_dict(), block=True)
    self.params.put('ModelRunnerTypeCache', int(custom.ModelManagerSP.Runner.stock), block=True)
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    with patch('openpilot.sunnypilot.models.helpers.get_active_bundle', return_value=None):
      self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.tinygrad)
      self.guard().disable('V3 failed')
      self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.stock)
    self.assertEqual(self.params.get('ModelManager_ActiveBundle')['ref'], 'working-small')

  def test_invalid_fallback_selection_disables_v3(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    with patch('openpilot.sunnypilot.models.helpers.get_active_bundle', return_value=None):
      self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.stock)
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertIn('small legacy', self.params.get('CinqueV3Fault'))

  def test_startup_error_disables_v3_and_experimental(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.params.put_bool('ExperimentalMode', True, block=True)
    self.candidate.warmup.side_effect = ValueError('bad output')
    self.assertIs(self.guard().start(), self.fallback)
    self.assertIn('bad output', self.params.get('CinqueV3Fault'))
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertFalse(self.params.get_bool('ExperimentalMode'))

  def test_fallback_error_prevents_v3_start(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.fallback.warmup.side_effect = RuntimeError('fallback missing')
    self.assertIsNone(self.guard().start())
    self.candidate_factory.assert_not_called()
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))

  def test_inference_error_resets_verified_fallback(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    guard = self.guard()
    guard.start()
    self.candidate.run.side_effect = RuntimeError('GPU fault')
    with self.assertRaisesRegex(RuntimeError, 'GPU fault') as caught:
      guard.run()
    self.assertIs(guard.recover(caught.exception), self.fallback)
    self.assertEqual(self.fallback.warmup.call_count, 2)
    self.assertFalse(guard.active)
    self.assertIn('GPU fault', self.params.get('CinqueV3Fault'))
    guard.run()
    self.fallback.run.assert_called_once()

  def test_recovery_error_stops_without_retrying_v3(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    guard = self.guard()
    guard.start()
    self.fallback.warmup.side_effect = ValueError('reset failed')
    with self.assertRaisesRegex(ValueError, 'reset failed'):
      guard.recover(ValueError('V3 bad'))
    self.assertFalse(guard.active)
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))

  def test_fault_survives_reboot_and_manual_enable_does_not_clear_it(self):
    self.guard().disable('latched error')
    self.params = Params(self.tmp.name)
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.assertIsNone(self.guard().start())
    self.fallback_factory.assert_not_called()
    self.candidate_factory.assert_not_called()
    self.assertEqual(self.params.get('CinqueV3Fault'), 'latched error')
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))

  def test_interrupted_initialization_is_not_retried(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.candidate_factory.side_effect = KeyboardInterrupt
    with self.assertRaises(KeyboardInterrupt):
      self.guard().start()
    self.candidate_factory.reset_mock(side_effect=True)
    self.guard().start()
    self.candidate_factory.assert_not_called()
    self.assertIn('initialization interrupted', self.params.get('CinqueV3Fault'))

  def test_hard_timeout_exits_and_persists_fallback_selection(self):
    script = """import sys, threading
from openpilot.common.params import Params
from openpilot.sunnypilot.modeld_v2.cinque_guard import CinqueGuard
p = Params(sys.argv[1])
p.put_bool('CinqueV3Enabled', True)
CinqueGuard(p, None, None).checked_call(lambda: threading.Event().wait(), .05, 'inference')
"""
    result = subprocess.run([sys.executable, '-c', script, self.tmp.name], timeout=5, capture_output=True)
    self.assertEqual(result.returncode, 1, result.stderr.decode())
    self.assertIn('timeout', self.params.get('CinqueV3Fault'))
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.guard().start()
    self.candidate_factory.assert_not_called()

  def test_returning_after_deadline_is_rejected(self):
    guard = CinqueGuard(self.params, None, None, terminate=Mock())
    with self.assertRaises(TimeoutError):
      guard.checked_call(lambda: time.sleep(.04), .01, 'inference')
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))

  def test_late_return_waits_for_fault_and_restart_writes(self):
    guard = CinqueGuard(self.params, None, None, terminate=Mock())
    writing, finished = threading.Event(), threading.Event()
    original_disable = guard.disable
    def slow_disable(reason):
      writing.set()
      time.sleep(.03)
      original_disable(reason)
      finished.set()
    guard.disable = slow_disable
    with self.assertRaises(TimeoutError):
      guard.checked_call(lambda: writing.wait(1), .01, 'inference')
    self.assertTrue(finished.is_set())
    self.assertIn('timeout', self.params.get('CinqueV3Fault'))
    self.assertTrue(self.params.get_bool('CinqueV3RecoveryPending'))

  def test_install_and_upgrade_leave_fallback_files_and_flags_unchanged(self):
    root = Path(self.tmp.name) / 'models'
    root.mkdir()
    (root / 'working.pkl').write_bytes(b'working model')
    source = Path(self.tmp.name) / 'v3.pkl'
    source.write_bytes(b'compiled V3')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = install_artifact(source, digest, 'revision-a', root)
    self.assertEqual(verify_manifest(manifest, root, 'revision-a').read_bytes(), b'compiled V3')
    self.assertEqual((root / 'working.pkl').read_bytes(), b'working model')
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    # A new Tinygrad revision must reject the old artifact before unpickling.
    with self.assertRaisesRegex(ValueError, 'pinned Tinygrad'):
      verify_manifest(manifest, root, 'revision-b')
    install_artifact(source, digest, 'revision-a', root)
    self.assertFalse(list(root.glob('.cinque-install-*')))

  def test_corrupt_artifact_is_not_installed(self):
    source = Path(self.tmp.name) / 'bad.pkl'
    source.write_bytes(b'corrupted')
    root = Path(self.tmp.name) / 'models'
    with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
      install_artifact(source, '0' * 64, 'revision', root)
    self.assertEqual(list(root.iterdir()), [])

  def test_offroad_required(self):
    self.params.put_bool('IsOffroad', False, block=True)
    with self.assertRaisesRegex(RuntimeError, 'offroad'):
      require_offroad(self.params)
    self.params.put_bool('IsOffroad', True, block=True)
    require_offroad(self.params)
