"""Exercise production startup policy; factories isolate unavailable hardware."""
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from openpilot.cereal import custom
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.sunnypilot.models.helpers import get_active_model_runner, validate_active_bundles
from openpilot.sunnypilot.modeld_v2.cinque_guard import MODEL_COMMIT, ONNX_SHA256, verify_manifest
from openpilot.sunnypilot.modeld_v2.startup import load_models, ModelStartupError
from openpilot.sunnypilot.modeld_v2.cinque import load_fallback
from openpilot.system.manager import process_config


class TestStartup(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.params = Params(self.tmp.name)
    self.small, self.big, self.candidate = (Mock(chestnut=chestnut) for chestnut in (False, True, True))
    self.legacy = Mock(side_effect=lambda **kw: self.big if kw['chestnut'] else self.small)
    self.fallback_factory = Mock(return_value=self.small)
    self.candidate_factory = Mock(return_value=self.candidate)
    for source, runner in (('ModelManager_ActiveBundle', 'tinygrad'), ('ModelManager_ActiveBundleChestnut', 'tinygrad')):
      self.params.put(source, custom.ModelManagerSP.ModelBundle(runner=runner, minimumSelectorVersion=19, ref=source).to_dict(), block=True)

  def load(self, chestnut=True):
    return load_models(self.params, 1344, 760, chestnut, self.legacy,
                       v3_factories=(self.fallback_factory, self.candidate_factory))

  def test_disabled_original_chestnut_and_small_selection_unchanged(self):
    startup = self.load()
    self.assertIs(startup.model, self.big)
    self.assertIs(startup.small_model, self.small)
    self.assertIsNone(startup.guard)
    self.fallback_factory.assert_not_called()
    self.candidate_factory.assert_not_called()
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertFalse(self.params.get_bool('ChestnutLoading'))

  def test_disabled_with_fault_does_not_require_any_v3_fallback(self):
    self.params.put('CinqueV3Fault', 'old GPU fault', block=True)
    self.params.remove('ModelManager_ActiveBundle')
    for _ in range(3):
      self.assertIs(self.load().model, self.big)
    self.fallback_factory.assert_not_called()
    self.candidate_factory.assert_not_called()
    self.assertEqual(self.params.get('CinqueV3Fault'), 'old GPU fault')

  def test_faulted_selection_ignores_old_v3_runner_cache_and_preserves_slots(self):
    self.params.put('CinqueV3Fault', 'persistent', block=True)
    self.params.put('ModelRunnerTypeCache', int(custom.ModelManagerSP.Runner.tinygrad), block=True)
    self.params.remove('ModelManager_ActiveBundleChestnut')
    with patch('openpilot.sunnypilot.models.helpers.chestnut_present', return_value=True):
      self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.stock)
      CP = Mock()
      self.assertTrue(process_config.managed_processes['modeld'].should_run(True, self.params, CP))
      self.assertFalse(process_config.managed_processes['modeld_tinygrad'].should_run(True, self.params, CP))
    self.assertIsNotNone(self.params.get('ModelManager_ActiveBundle'))

  def test_enabled_verified_artifact_and_fallback_initialize_once(self):
    pkl = Path(self.tmp.name) / 'candidate.pkl'
    pkl.write_bytes(b'trusted test fixture')
    self.params.put('CinqueV3Manifest', {'model': 'Cinque Terre V3', 'model_commit': MODEL_COMMIT,
                                     'source_onnx_sha256': ONNX_SHA256, 'sha256': hashlib.sha256(pkl.read_bytes()).hexdigest(),
                                     'file_name': pkl.name, 'tinygrad_ref': 'pinned'}, block=True)
    def checked_candidate():
      verify_manifest(self.params.get('CinqueV3Manifest'), self.tmp.name, 'pinned')
      # The journal is not a fault and must not make manager change runners.
      with patch('openpilot.sunnypilot.models.helpers.chestnut_present', return_value=True):
        self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.tinygrad)
        self.assertTrue(self.params.get_bool('CinqueV3Enabled'))
      return self.candidate
    self.candidate_factory.side_effect = checked_candidate
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.assertIs(self.load().model, self.candidate)
    self.candidate_factory.assert_called_once()
    self.small.warmup.assert_called_once()
    self.assertIsNone(self.params.get('CinqueV3Fault'))
    self.assertIsNone(self.params.get('CinqueV3InitJournal'))

  def test_missing_or_corrupt_manifest_latches_and_uses_verified_fallback(self):
    for manifest in (None, {}, {'model': 'wrong'}):
      with self.subTest(manifest=manifest):
        self.params.remove('CinqueV3Fault')  # Explicit test operator action.
        self.params.put_bool('CinqueV3Enabled', True, block=True)
        self.candidate_factory.side_effect = lambda manifest=manifest: verify_manifest(manifest, self.tmp.name, 'pinned')
        startup = self.load()
        self.assertIs(startup.model, self.small)
        self.assertFalse(startup.guard.active)
        self.assertTrue(self.params.get_bool('CinqueV3UseFallback'))
        self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
        self.assertIn('startup failed', self.params.get('CinqueV3Fault'))

  def test_request_withdrawn_or_fault_recorded_during_initialization_blocks_activation(self):
    for record_fault in (False, True):
      with self.subTest(record_fault=record_fault):
        self.params.remove('CinqueV3Fault')
        self.params.put_bool('CinqueV3Enabled', True, block=True)
        def candidate(record_fault=record_fault):
          if record_fault:
            self.params.put('CinqueV3Fault', 'operator safety stop', block=True)
          else:
            self.params.put_bool('CinqueV3Enabled', False, block=True)
          return self.candidate
        self.candidate_factory.side_effect = candidate
        startup = self.load()
        self.assertIs(startup.model, self.small)
        self.assertFalse(startup.guard.active)
        self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
        self.assertIsNotNone(self.params.get('CinqueV3Fault'))

  def test_missing_or_corrupt_fallback_returns_to_normal_mvl(self):
    for error in (FileNotFoundError('missing'), ValueError('checksum mismatch')):
      with self.subTest(error=error):
        self.params.remove('CinqueV3Fault')
        self.params.put_bool('CinqueV3Enabled', True, block=True)
        self.fallback_factory.side_effect = error
        startup = self.load()
        self.assertIs(startup.model, self.big)
        self.assertIsNone(startup.guard)
        self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
        self.assertFalse(self.params.get_bool('CinqueV3UseFallback'))
    self.candidate_factory.assert_not_called()

  def test_corrupt_manifest_json_is_an_explicit_startup_fault(self):
    Path(self.params.get_param_path('CinqueV3Manifest')).write_text('{invalid JSON')
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.candidate_factory.side_effect = lambda: self.params.get('CinqueV3Manifest')
    self.assertIs(self.load().model, self.small)
    self.assertIn('startup failed', self.params.get('CinqueV3Fault'))
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))

  def test_missing_fallback_can_handoff_to_stock_runner(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.params.remove('ModelManager_ActiveBundleChestnut')
    self.fallback_factory.side_effect = FileNotFoundError('small missing')
    with patch('openpilot.sunnypilot.models.helpers.chestnut_present', return_value=True):
      with self.assertRaisesRegex(ModelStartupError, 'stock modeld'):
        self.load()
      self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.stock)
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertFalse(self.params.get_bool('ChestnutLoading'))
    self.candidate_factory.assert_not_called()

  def test_chestnut_model_unavailable_or_disconnect_during_load_uses_small(self):
    for error in (FileNotFoundError('selected AMD pickle missing'), RuntimeError('USB disconnected')):
      with self.subTest(error=error):
        def legacy(error=error, **kwargs):
          if kwargs['chestnut']:
            raise error
          return self.small
        self.legacy.side_effect = legacy
        self.assertIs(self.load().model, self.small)
        self.assertTrue(self.params.get_bool('ChestnutModelError'))
        self.assertFalse(self.params.get_bool('ChestnutActive'))

  def test_chestnut_absent_never_initializes_v3_or_amd(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.assertIs(self.load(chestnut=False).model, self.small)
    self.candidate_factory.assert_not_called()
    self.fallback_factory.assert_not_called()
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertEqual(self.legacy.call_args.kwargs['chestnut'], False)

  def test_chestnut_warmup_error_uses_small(self):
    self.big.warmup.side_effect = RuntimeError('GPU warmup failed')
    self.assertIs(self.load().model, self.small)

  def test_chestnut_loader_timeout_does_not_select_late_model(self):
    release = threading.Event()
    self.addCleanup(release.set)
    self.big.warmup.side_effect = lambda: release.wait(2)
    startup = load_models(self.params, 1344, 760, True, self.legacy, big_timeout=.01)
    self.assertIs(startup.model, self.small)
    release.set()
    self.assertIs(startup.model, self.small)

  def test_working_chestnut_can_boot_without_small_but_has_no_fake_fallback(self):
    self.legacy.side_effect = lambda **kw: self.big if kw['chestnut'] else (_ for _ in ()).throw(FileNotFoundError('no small'))
    startup = self.load()
    self.assertIs(startup.model, self.big)
    self.assertIsNone(startup.small_model)

  def test_no_ordinary_model_is_an_explicit_fatal_error(self):
    self.legacy.side_effect = FileNotFoundError('no installed model')
    with self.assertRaisesRegex(ModelStartupError, 'no installed model'):
      self.load()
    self.assertFalse(self.params.get_bool('ChestnutLoading'))
    self.assertIsNone(self.params.get('CinqueV3Fault'))

  def test_initialization_exception_does_not_retry_on_modeld_restart(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.candidate_factory.side_effect = RuntimeError('AMD initialization failed')
    self.assertIs(self.load().model, self.small)
    for _ in range(3):
      self.assertIs(self.load().model, self.small)  # Same live-session recovery flag.
    self.candidate_factory.assert_called_once()
    self.assertIn('AMD initialization failed', self.params.get('CinqueV3Fault'))

  def test_reboot_and_upgrade_clear_runtime_recovery_but_keep_fault_and_disable(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.candidate_factory.side_effect = ValueError('V3 failed')
    self.load()
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
      self.params.clear_all(flag)
    self.params = Params(self.tmp.name)
    self.assertIs(self.load().model, self.big)
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertIn('V3 failed', self.params.get('CinqueV3Fault'))
    self.candidate_factory.assert_called_once()

  def test_stale_enable_with_fault_never_retries_candidate(self):
    self.params.put('CinqueV3Fault', 'latched error', block=True)
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.assertIs(self.load().model, self.big)
    self.candidate_factory.assert_not_called()
    self.fallback_factory.assert_not_called()
    self.assertEqual(self.params.get('CinqueV3Fault'), 'latched error')

  def test_fault_written_before_interrupted_journal_cleanup_keeps_original_error(self):
    self.params.put('CinqueV3Fault', 'original GPU diagnosis', block=True)
    self.params.put('CinqueV3InitJournal', 'candidate warmup', block=True)
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.assertIs(self.load().model, self.big)
    self.assertEqual(self.params.get('CinqueV3Fault'), 'original GPU diagnosis')
    self.candidate_factory.assert_not_called()

  def test_interrupted_fallback_verification_is_journaled_before_candidate(self):
    self.params.put_bool('CinqueV3Enabled', True, block=True)
    self.fallback_factory.side_effect = KeyboardInterrupt
    with self.assertRaises(KeyboardInterrupt):
      self.load()
    self.assertIsNotNone(self.params.get('CinqueV3InitJournal'))
    self.fallback_factory.side_effect = None
    self.assertIs(self.load().model, self.big)
    self.candidate_factory.assert_not_called()
    self.assertIn('initialization interrupted', self.params.get('CinqueV3Fault'))

  def test_live_recovery_fallback_missing_returns_to_normal_without_enabling_v3(self):
    self.params.put('CinqueV3Fault', 'previous failure', block=True)
    self.params.put_bool('CinqueV3UseFallback', True, block=True)
    self.fallback_factory.side_effect = FileNotFoundError('fallback removed')
    self.assertIs(self.load().model, self.big)
    self.assertFalse(self.params.get_bool('CinqueV3UseFallback'))
    self.candidate_factory.assert_not_called()
    self.assertIn('fallback removed', self.params.get('CinqueV3Fault'))

  def test_invalid_bundle_json_does_not_crash_manager_selection(self):
    Path(self.params.get_param_path('ModelManager_ActiveBundleChestnut')).write_text('{invalid JSON')
    with patch('openpilot.sunnypilot.models.helpers.chestnut_present', return_value=True):
      self.assertEqual(get_active_model_runner(self.params, force_check=True), custom.ModelManagerSP.Runner.stock)

  def test_invalid_bundle_json_does_not_crash_offroad_catalog_validation(self):
    Path(self.params.get_param_path('ModelManager_ActiveBundleChestnut')).write_text('{invalid JSON')
    validate_active_bundles(self.params, {'qcom': [], 'chestnut': []})
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertEqual(Path(self.params.get_param_path('ModelManager_ActiveBundleChestnut')).read_text(), '{invalid JSON')

  def test_corrupt_or_unusable_runner_cache_is_recomputed(self):
    for value in ('not-an-integer', '999', '0'):
      with self.subTest(value=value):
        Path(self.params.get_param_path('ModelRunnerTypeCache')).write_text(value)
        with patch('openpilot.sunnypilot.models.helpers.chestnut_present', return_value=True):
          self.assertEqual(get_active_model_runner(self.params), custom.ModelManagerSP.Runner.tinygrad)

  def test_chunked_fallback_requires_every_checksum(self):
    bundle = custom.ModelManagerSP.ModelBundle(runner='tinygrad', models=[{'artifact': {'fileName': 'legacy.pkl',
                                               'chunks': [{'sha256': '0' * 64}, {'sha256': ''}]}}])
    with patch('openpilot.sunnypilot.modeld_v2.cinque.get_active_bundle', return_value=bundle), \
         patch('openpilot.sunnypilot.modeld_v2.cinque._bundle_is_valid_locally') as check:
      with self.assertRaisesRegex(ValueError, 'complete verified checksums'):
        load_fallback(1344, 760)
      check.assert_not_called()

  def test_amd_warp_cannot_be_accepted_as_independent_recovery(self):
    bundle = custom.ModelManagerSP.ModelBundle(runner='tinygrad', models=[{'artifact': {'fileName': 'legacy.pkl',
                                               'downloadUri': {'sha256': '0' * 64}}}])
    model = Mock(chestnut=False, WARP_DEV='AMD', DEV='QCOM')
    with patch('openpilot.sunnypilot.modeld_v2.cinque.get_active_bundle', return_value=bundle), \
         patch('openpilot.sunnypilot.modeld_v2.cinque._bundle_is_valid_locally', return_value=True), \
         patch('openpilot.sunnypilot.modeld_v2.cinque.VerifiedFallback', return_value=model):
      with self.assertRaisesRegex(ValueError, 'built-in device'):
        load_fallback(1344, 760)

  def test_initialization_watchdog_persists_fault_and_requests_bounded_restart(self):
    script = '''import sys, threading
from unittest.mock import Mock, patch
from openpilot.common.params import Params
from openpilot.sunnypilot.modeld_v2.cinque_guard import CinqueGuard
p = Params(sys.argv[1])
p.put_bool('CinqueV3Enabled', True, block=True)
with patch('openpilot.sunnypilot.modeld_v2.cinque_guard.INIT_TIMEOUT', .05):
  CinqueGuard(p, lambda: Mock(), lambda: threading.Event().wait()).start()
'''
    child = subprocess.run([sys.executable, '-c', script, self.tmp.name], timeout=5, capture_output=True, env=os.environ.copy())
    self.assertEqual(child.returncode, 1, child.stderr.decode())
    self.assertIn('timeout', self.params.get('CinqueV3Fault'))
    self.assertTrue(self.params.get_bool('CinqueV3RecoveryPending'))
    self.assertIs(self.load().model, self.small)
    self.candidate_factory.assert_not_called()


if __name__ == '__main__':
  unittest.main()
