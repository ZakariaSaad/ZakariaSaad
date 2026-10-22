"""Existing-artifact registration with real data-only headers and compiler proof.

The binaries are inert admission fixtures, never model weights or executable
kernels. Only the unavailable physical warp profile and runtime Git pin are
substituted; model/header/proof hashes and atomic package loading are real.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation import export_experimental_model as exporter, preparation
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError, load_experimental_model
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub
from openpilot.selfdrive.navigation.tests import test_experimental_export as export_fixtures


class RecordingParams(ParamsStub):
  def __init__(self, **values):
    super().__init__(**values)
    self.writes = []

  def put(self, key, value, **kwargs):
    self.writes.append((key, value))
    if key == 'NavigationModelManifest':
      assert not self.get_bool('NavigationModelDrivingEnabled')
      assert Path(value).is_file()
      assert load_experimental_model(value, experimental=True, backend='AMD').compiler_verification_validated
    super().put(key, value, **kwargs)

  def put_bool(self, key, value, **kwargs):
    self.writes.append((key, bool(value)))
    super().put_bool(key, value, **kwargs)


class TestExistingModelPreparation(unittest.TestCase):
  def setUp(self):
    self.fixture = export_fixtures.TestExperimentalExport('runTest')
    self.fixture.setUp()
    self.addCleanup(self.fixture.doCleanups)
    self.root = self.fixture.root
    self.model_dir = self.root / 'existing-model'
    self.model_dir.mkdir()
    self.artifact = self.model_dir / 'bigv2-amd.sptg'
    shutil.copyfile(self.root / 'model.sptg', self.artifact)
    self.report = copy.deepcopy(self.fixture.report)
    self.report_path = self.model_dir / 'bigv2-amd.verification.json'
    self.report_path.write_text(json.dumps(self.report))
    self.package_root = self.root / 'packages'
    self.params = RecordingParams(IsOffroad=True, IsEngaged=False, NavigationModelDrivingEnabled=False,
                                  NavigationModelCandidate='old-candidate', NavigationModelManifest='previous-manifest.json',
                                  ModelManager_ActiveBundle={'keep': 'small'}, ModelManager_ActiveBundleChestnut={'keep': 'big'},
                                  ModelRunnerTypeCache=1, NavigationEnabled=True)
    self.preserved = {key: copy.deepcopy(value) for key, value in self.params.values.items() if not key.startswith('NavigationModel')}
    self.profile = {'artifact_sha256': self.fixture.fixture.entry['warp_artifact']['sha256'],
                    'width': 1344, 'height': 760, 'frame_size': 1024, 'captured_check_passed': True}
    self.enterContext(patch.object(preparation, 'COMPILER', self.fixture.compiler))
    self.enterContext(patch.object(preparation, 'DEFAULT_WARP', self.root / 'warp.sptg'))
    self.enterContext(patch.object(preparation, 'REPORTED_ARTIFACT_SHA256', self.report['artifact_sha256']))
    self.enterContext(patch.object(preparation, 'REVIEWED_COMPILER_SHA256', self.report['compiler_sha256']))
    self.enterContext(patch.object(preparation, 'validate_existing_warp', side_effect=lambda *_args, **_kwargs: copy.deepcopy(self.profile)))
    self.enterContext(patch.object(exporter, '_runtime_checkout'))
    # This synthetic warp is intentionally not a real camera profile. Its native
    # header remains checked; the separate real-asset tests cover profile binding.
    self.enterContext(patch('openpilot.selfdrive.navigation.warp_compatibility.validate_camera_profile', create=True))
    self.enterContext(patch('pickle.load', side_effect=AssertionError('Preparation must not unpickle models')))
    self.enterContext(patch('pickle.loads', side_effect=AssertionError('Preparation must not unpickle models')))
    self.enterContext(patch('requests.get', side_effect=AssertionError('Existing-artifact preparation must not download')))
    self.enterContext(patch('subprocess.run', side_effect=AssertionError('No diagnostic, compiler or service command')))
    self.enterContext(patch('subprocess.Popen', side_effect=AssertionError('No diagnostic, compiler or service process')))

  def prepare(self, **kwargs):
    return preparation.prepare_existing_model(self.params, is_offroad=True, model_dir=self.model_dir,
                                              package_root=self.package_root, **kwargs)

  def assert_preserved(self):
    self.assertEqual(self.preserved, {key: self.params.values[key] for key in self.preserved})

  def assert_old_manifest(self):
    self.assertEqual(self.params.get('NavigationModelManifest'), 'previous-manifest.json')
    self.assert_preserved()

  def test_success_publishes_real_validated_manifest_last_and_keeps_driving_off(self):
    before = self.artifact.read_bytes()
    receipt = self.prepare()
    self.assertTrue(receipt['prepared'] and receipt['registered'])
    self.assertFalse(receipt['activation_allowed'])
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(self.params.writes[-1], ('NavigationModelManifest', receipt['manifest_path']))
    self.assertEqual(self.params.writes[0], ('NavigationModelDrivingEnabled', False))
    entry = load_experimental_model(receipt['manifest_path'], experimental=True, backend='AMD')
    self.assertTrue(entry.compiler_verification_validated)
    self.assertFalse(entry.normal_activation_allowed)
    self.assertFalse(entry.metadata['hardware_driving_validation'])
    self.assertEqual(entry.artifact_path, self.artifact)
    self.assertEqual(before, self.artifact.read_bytes())
    for flag in ('gpu_execution_performed', 'model_recompiled', 'models_downloaded', 'normal_service_modified', 'source_onnx_bytes_rechecked'):
      self.assertIs(receipt[flag], False)
    self.assert_preserved()

  def test_portable_proof_survives_package_rename_and_unrelated_working_directory(self):
    receipt = self.prepare()
    manifest = Path(receipt['manifest_path'])
    package = manifest.parent
    with patch('pathlib.Path.cwd', return_value=self.root / 'unrelated-current-directory'):
      entry = load_experimental_model(manifest, experimental=True, backend='AMD')
    proof = entry.metadata['compiler_verification']
    self.assertEqual(proof['compiler']['path'], 'compiler-source.py')
    self.assertEqual(proof['report']['path'], 'compiler-report.json')
    self.assertFalse(proof['source_onnx']['bytes_retained'])
    self.assertEqual(hashlib.sha256((package / proof['compiler']['path']).read_bytes()).hexdigest(), proof['compiler']['sha256'])
    self.assertEqual(entry.artifact_path, self.artifact)
    self.assertFalse(list(self.package_root.glob('.prepare-existing-*')))

  def test_retry_reuses_complete_package_and_compiler_snapshot_survives_source_update(self):
    first = self.prepare()
    with patch.object(preparation, 'export_experimental_model', side_effect=AssertionError('Must reuse a verified complete package')):
      second = self.prepare()
    self.assertEqual(first['manifest_path'], second['manifest_path'])
    self.assertEqual(first['metadata_sha256'], second['metadata_sha256'])
    self.fixture.compiler.write_text('# A later software update; never executed.\n')
    # Existing manifest provenance uses its retained snapshot, not that mutable source.
    entry = load_experimental_model(first['manifest_path'], experimental=True, backend='AMD')
    self.assertTrue(entry.compiler_verification_validated)

  def test_default_report_filename_and_identical_duplicate_report_are_supported(self):
    standard = self.model_dir / 'bigv2-amd.sptg.verification.json'
    self.report_path.rename(standard)
    self.assertTrue(self.prepare()['registered'])
    self.report_path.write_bytes(standard.read_bytes())
    self.assertTrue(self.prepare()['registered'])

  def test_existing_onnx_is_never_opened_or_required_after_registration(self):
    source = self.model_dir / 'big_driving_supercombo.onnx'
    source.write_bytes(b'Unrelated retained source fixture; must not be read or hashed')
    original_open = Path.open
    def without_source(path, *args, **kwargs):
      if path == source:
        raise AssertionError('Existing-artifact registration must not open the ONNX source')
      return original_open(path, *args, **kwargs)
    with patch.object(Path, 'open', without_source), \
         patch.object(exporter, '_source_file', side_effect=AssertionError('No source-file dependency')):
      receipt = self.prepare()
      entry = load_experimental_model(receipt['manifest_path'], experimental=True, backend='AMD')
    self.assertFalse(receipt['source_onnx_bytes_rechecked'])
    self.assertEqual(entry.metadata['compiler_verification']['source_onnx'],
                     {'sha256': exporter.SOURCE_ONNX_SHA256, 'bytes_retained': False})
    source.unlink()
    self.assertTrue(load_experimental_model(receipt['manifest_path'], experimental=True, backend='AMD').compiler_verification_validated)

  def test_receipt_write_failure_after_manifest_commit_reports_success_with_warning(self):
    original_write = preparation._write_json
    def fail_after_commit(path, value):
      if value.get('status') == 'registered_disabled':
        self.assertTrue(value['registered'])
        self.assertEqual(self.params.get('NavigationModelManifest'), value['manifest_path'])
        raise OSError('Receipt disk became unavailable after commit')
      return original_write(path, value)
    with patch.object(preparation, '_write_json', side_effect=fail_after_commit):
      receipt = self.prepare()
    self.assertTrue(receipt['registered'])
    self.assertEqual(receipt['status'], 'registered_disabled')
    self.assertIn('after commit', receipt['receipt_warning'])
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertTrue(load_experimental_model(receipt['manifest_path'], experimental=True, backend='AMD').compiler_verification_validated)
    retained = json.loads((self.package_root / 'latest-registration.json').read_text())
    self.assertEqual(retained['status'], 'prepared_disabled')
    self.assertNotEqual(retained['status'], 'registration_failed')
    self.assert_preserved()

  def test_root_or_wrong_user_is_rejected_before_params_or_package_creation(self):
    owner_checkout = SimpleNamespace(stat=lambda: SimpleNamespace(st_uid=1000))
    for current_uid in (0, 1001):
      with self.subTest(current_uid=current_uid), patch.object(preparation, 'CHECKOUT', owner_checkout), \
           patch.object(preparation.os, 'geteuid', return_value=current_uid):
        self.params.writes.clear()
        with self.assertRaisesRegex(ExperimentalCatalogError, 'checkout owner.*without sudo'):
          self.prepare()
        self.assertFalse(self.params.writes)
        self.assertFalse(self.package_root.exists())
        self.assert_old_manifest()
  def test_conflicting_reports_require_explicit_existing_file_without_recompilation(self):
    duplicate = self.model_dir / 'bigv2-amd.sptg.verification.json'
    duplicate.write_text('{}')
    with self.assertRaisesRegex(ExperimentalCatalogError, 'Conflicting Bigv2 reports'):
      self.prepare()
    self.assert_old_manifest()
    self.assertTrue(self.prepare(verification=self.report_path)['registered'])

  def test_missing_or_changed_artifact_does_not_download_or_register(self):
    original = self.artifact.read_bytes()
    for value in (None, b'changed model'):
      with self.subTest(missing=value is None):
        if value is None:
          self.artifact.unlink()
        else:
          self.artifact.write_bytes(value)
        with self.assertRaisesRegex(ExperimentalCatalogError, 'missing|SHA256 mismatch'):
          self.prepare()
        self.assert_old_manifest()
        self.artifact.write_bytes(original)

  def test_wrong_or_failed_report_and_unmatched_warp_keep_old_manifest(self):
    mutations = [lambda r: r.update(artifact_sha256='0' * 64), lambda r: r.update(compiler_sha256='0' * 64),
                 lambda r: r.update(artifact_promoted=False), lambda r: r['verification']['runs'][1]['outputs']['outputs'].update(passed=False)]
    for mutation in mutations:
      with self.subTest(mutation=mutation):
        changed = copy.deepcopy(self.report)
        mutation(changed)
        self.report_path.write_text(json.dumps(changed))
        with self.assertRaises(ExperimentalCatalogError):
          self.prepare()
        self.assert_old_manifest()
    self.report_path.write_text(json.dumps(self.report))
    with patch.object(preparation, 'validate_existing_warp', side_effect=ExperimentalCatalogError('Wrong camera warp')):
      with self.assertRaisesRegex(ExperimentalCatalogError, 'Wrong camera warp'):
        self.prepare()
    self.assert_old_manifest()
    self.assertFalse(list(self.package_root.glob('.prepare-existing-*')))

  def test_interrupted_publication_cleans_staging_and_retains_original_model_and_manifest(self):
    original = self.artifact.read_bytes()
    with patch.object(preparation.os, 'rename', side_effect=OSError('Interrupted package publication')):
      with self.assertRaisesRegex(OSError, 'Interrupted package publication'):
        self.prepare()
    self.assertFalse(list(self.package_root.glob('.prepare-existing-*')))
    self.assertFalse(list(self.package_root.glob('big-nav-v2-*')))
    self.assertEqual(original, self.artifact.read_bytes())
    self.assert_old_manifest()
    latest = json.loads((self.package_root / 'latest-registration.json').read_text())
    self.assertEqual(latest['status'], 'registration_failed')
    self.assertFalse(latest['activation_allowed'])

  def test_genuine_offroad_and_disengagement_required_before_any_param_changes(self):
    for is_offroad, param_offroad, engaged in ((False, True, False), (True, False, False), (True, True, True)):
      with self.subTest(is_offroad=is_offroad, param_offroad=param_offroad, engaged=engaged):
        self.params.values.update(IsOffroad=param_offroad, IsEngaged=engaged)
        self.params.writes.clear()
        with self.assertRaisesRegex(ExperimentalCatalogError, 'genuinely off-road'):
          preparation.prepare_existing_model(self.params, is_offroad=is_offroad, model_dir=self.model_dir, package_root=self.package_root)
        self.assertFalse(self.params.writes)
        self.assertEqual(self.params.get('NavigationModelManifest'), 'previous-manifest.json')

  def test_offroad_revocation_before_commit_refuses_registration_then_allows_retry(self):
    package_receipt = preparation._package_receipt
    def revoked(*args, **kwargs):
      receipt = package_receipt(*args, **kwargs)
      self.params.values['IsOffroad'] = False
      return receipt
    with patch.object(preparation, '_package_receipt', side_effect=revoked):
      with self.assertRaisesRegex(ExperimentalCatalogError, 'genuinely off-road'):
        self.prepare()
    self.assertEqual(self.params.get('NavigationModelManifest'), 'previous-manifest.json')
    self.params.values['IsOffroad'] = True
    with patch.object(preparation, 'export_experimental_model', side_effect=AssertionError('Prepared package should survive interrupted registration')):
      self.assertTrue(self.prepare()['registered'])

  def test_exclusive_lock_prevents_concurrent_package_registration(self):
    self.package_root.mkdir()
    with preparation._preparation_lock(self.package_root):
      with self.assertRaisesRegex(ExperimentalCatalogError, 'already running'):
        self.prepare()
    self.assert_old_manifest()

  def test_self_consistent_package_tampering_cannot_change_reviewed_identities(self):
    receipt = self.prepare()
    package = Path(receipt['manifest_path']).parent
    snapshot = {path: path.read_bytes() for path in package.iterdir() if path.is_file()}
    metadata_path = package / 'manifest.json.metadata.json'
    report_path = package / 'compiler-report.json'
    for changed in ('profile', 'report', 'compiler'):
      with self.subTest(changed=changed):
        for path, value in snapshot.items():
          path.write_bytes(value)
        metadata = json.loads(metadata_path.read_text())
        report = json.loads(report_path.read_text())
        if changed == 'profile':
          metadata['camera_profile']['width'] += 1
        else:
          if changed == 'compiler':
            compiler_path = package / 'compiler-source.py'
            compiler_path.write_text('# Changed retained source; never executed\n')
            digest = hashlib.sha256(compiler_path.read_bytes()).hexdigest()
            report['compiler_sha256'] = digest
            metadata['compiler_verification']['compiler']['sha256'] = digest
          else:
            report['unreviewed_note'] = 'Different retained report identity'
          report_path.write_text(json.dumps(report))
          metadata['compiler_verification']['report']['sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
        manifest_path = package / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['entries'][0]['compiler_metadata']['sha256'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        # Internal hashes agree; registration must still reject a package whose
        # profile or retained evidence differs from the reviewed input identity.
        self.assertTrue(load_experimental_model(manifest_path, experimental=True, backend='AMD').compiler_verification_validated)
        before = {path: path.read_bytes() for path in snapshot}
        with patch.object(preparation, 'export_experimental_model', side_effect=AssertionError('Must not overwrite an existing package')):
          with self.assertRaisesRegex(ExperimentalCatalogError, 'reviewed warp/compiler/report identity'):
            self.prepare()
        self.assertEqual(before, {path: path.read_bytes() for path in snapshot})
        self.assertEqual(self.params.get('NavigationModelManifest'), receipt['manifest_path'])
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))


if __name__ == '__main__':
  unittest.main()
