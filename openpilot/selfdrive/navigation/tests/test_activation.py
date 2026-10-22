"""Driving permits require branch approval independently of display/diagnostic settings."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation import activation
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled


class ParamsStub:
  def __init__(self, **values):
    self.values = values

  def get(self, key, **kwargs):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value, **kwargs):
    self.values[key] = value

  def put_bool(self, key, value, **kwargs):
    self.values[key] = bool(value)

  def remove(self, key):
    self.values.pop(key, None)


@contextmanager
def reviewed_fixture(input_specs=None, metadata=None):
  """Simulate explicit branch authorization and the separately tested verifier.

  Dummy files are never executed. This is admission-unit evidence, not a claim
  that any real candidate has been authorized for driving.
  """
  with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    paths = [root / name for name in ('artifact.sptg', 'warp.sptg', 'manifest.json')]
    for path in paths[:2]:
      path.write_bytes(path.name.encode())
    metadata_path = root / 'metadata.json'
    metadata_path.write_text('{}')
    source = json.loads(activation.REGISTRY.read_text())
    record = source['candidates'][0]
    record.update(software_contract_reviewed=True, software_integration_reviewed=True,
                  artifact_evidence_reviewed=True, activation_approved=True)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths[:2]] + [hashlib.sha256(metadata_path.read_bytes()).hexdigest()]
    record.update(dict(zip(('artifact_sha256', 'warp_sha256', 'metadata_sha256'), hashes, strict=True)))
    paths[2].write_text(json.dumps({'entries': [{'id': 'big-nav-v2',
      'compiled_artifact': {'path': str(paths[0]), 'sha256': hashes[0]},
      'warp_artifact': {'path': str(paths[1]), 'sha256': hashes[1]},
      'compiler_metadata': {'path': str(metadata_path), 'sha256': hashes[2]}}]}))
    registry = root / 'registry.json'
    registry.write_text(json.dumps(source))
    params = ParamsStub(NavigationModelDrivingEnabled=True, NavigationModelCandidate='big-nav-v2', NavigationModelManifest=str(paths[2]))
    reviewed_metadata = dict(metadata or {})
    reviewed_metadata['compiler_verification'] = {'compiler': {'sha256': record['compiler_sha256']},
                                                'report': {'sha256': record['compiler_report_sha256']}}
    entry = SimpleNamespace(model_id='big-nav-v2', artifact_path=paths[0], warp_path=paths[1], input_specs=input_specs or {},
                            metadata=reviewed_metadata, warp_frame_size=1622016, compiler_verification_validated=True,
                            **dict(zip(('artifact_sha256', 'warp_sha256', 'metadata_sha256'), hashes, strict=True)))
    with patch.object(activation, 'REGISTRY', registry), patch.object(activation, 'Params', return_value=params), \
         patch.object(activation, 'ordinary_fallback_reason', return_value=''), \
         patch.object(activation, 'load_experimental_model', return_value=entry) as verifier:
      yield params, activation.authorize_runtime(params), verifier


class TestActivation(unittest.TestCase):
  def test_off_mode_does_not_open_candidate_registry_or_artifacts(self):
    params = ParamsStub(NavigationModelDrivingEnabled=False, NavigationEnabled=True, NavigationModelInputsEnabled=True)
    with patch.object(activation, '_candidate') as registry, patch.object(activation, 'load_experimental_model') as loader:
      self.assertIsNone(activation.authorize_runtime(params))
    registry.assert_not_called()
    loader.assert_not_called()

  def test_checked_in_candidate_cannot_be_enabled_by_any_checkbox(self):
    params = ParamsStub(NavigationModelDrivingEnabled=True, NavigationEnabled=True, NavigationModelInputsEnabled=True,
                        NavigationModelCandidate='big-nav-v2', NavigationModelManifest='/nonexistent')
    with patch.object(activation, 'ordinary_fallback_reason', return_value=''):
      status = activation.candidate_status(params)
      self.assertFalse(status['eligible'])
      self.assertIn('Explicit user authorization', status['reason'])
      self.assertTrue(status['artifact_evidence_reviewed'])
      self.assertTrue(status['software_compatible'])
      self.assertFalse(status['registered'])
      self.assertEqual(status['runtime_compatibility'], 'unverified')
      with self.assertRaises(NavigationModelDisabled):
        activation.authorize_runtime(params)

  def test_only_reviewed_hashes_can_issue_a_runtime_permit(self):
    with reviewed_fixture() as (params, permit, verifier):
      self.assertFalse(hasattr(permit, 'minimum_valid_frames'))
      self.assertEqual(permit.warp_frame_size, 1622016)
      verifier.assert_called_once()
      activation.validate_runtime_permit(permit, params=params)
      params.values['NavigationModelDrivingEnabled'] = False
      with self.assertRaises(NavigationModelDisabled):
        activation.validate_runtime_permit(permit, params=params)

  def test_local_artifact_mutation_and_candidate_change_revoke_admission(self):
    with reviewed_fixture() as (params, permit, _):
      params.values['NavigationModelCandidate'] = 'unreviewed-model'
      with self.assertRaises(NavigationModelDisabled):
        activation.validate_runtime_permit(permit, params=params)
      params.values['NavigationModelCandidate'] = 'big-nav-v2'
      permit.artifact_path.write_bytes(b'mutated')
      with self.assertRaises(NavigationModelDisabled):
        activation.validate_runtime_permit(permit, params=params)

  def test_native_specs_must_match_reviewed_shapes_dtypes_and_devices(self):
    specs = {'navigation': ((1, 47), 'float32', 'AMD')}
    with reviewed_fixture(specs) as (params, permit, _):
      activation.validate_runtime_permit(permit, specs, params)
      for wrong in ({'navigation': ((47,), 'float32', 'AMD')}, {'navigation': ((1, 47), 'float32', 'CPU')}):
        with self.assertRaises(NavigationModelDisabled):
          activation.validate_runtime_permit(permit, wrong, params)

  def test_compiler_sidecar_mutation_also_revokes_the_bound_permit(self):
    with reviewed_fixture() as (params, permit, _):
      permit.metadata_path.write_text('{"changed":true}')
      with self.assertRaises(NavigationModelDisabled):
        activation.validate_runtime_permit(permit, params=params)

  def test_registration_and_software_review_do_not_grant_driving_authorization(self):
    with reviewed_fixture() as (params, permit, _):
      registry = json.loads(activation.REGISTRY.read_text())
      registry['candidates'][0]['activation_approved'] = False
      activation.REGISTRY.write_text(json.dumps(registry))
      status = activation.candidate_status(params)
      self.assertTrue(status['registered'])
      self.assertTrue(status['software_compatible'])
      self.assertTrue(status['artifact_evidence_reviewed'])
      self.assertFalse(status['eligible'])
      self.assertIn('Explicit user authorization', status['reason'])
      self.assertNotIn('Physical Chestnut', status['reason'])
      with self.assertRaises(NavigationModelDisabled):
        activation.authorize_runtime(params)

  def test_manifest_declarations_are_checked_before_showing_registered_status(self):
    with reviewed_fixture() as (params, permit, _):
      manifest = json.loads(permit.manifest_path.read_text())
      manifest['entries'][0]['warp_artifact']['sha256'] = '0' * 64
      permit.manifest_path.write_text(json.dumps(manifest))
      status = activation.candidate_status(params)
      self.assertFalse(status['registered'])
      self.assertFalse(status['eligible'])
      self.assertIn('does not declare the reviewed', status['reason'])
      for entries in (None, {}, [None], [{'id': 'big-nav-v2', 'compiled_artifact': None}]):
        with self.subTest(entries=entries):
          permit.manifest_path.write_text(json.dumps({'entries': entries}))
          self.assertFalse(activation.candidate_status(params)['eligible'])

  def test_exact_compiler_and_report_evidence_is_required_even_with_matching_artifact_hashes(self):
    with reviewed_fixture() as (params, permit, verifier):
      for key in ('compiler', 'report'):
        with self.subTest(key=key):
          proof = verifier.return_value.metadata['compiler_verification']
          original = proof[key]['sha256']
          proof[key]['sha256'] = '0' * 64
          with self.assertRaisesRegex(NavigationModelDisabled, 'compiler source or verification report'):
            activation.authorize_runtime(params)
          proof[key]['sha256'] = original

  def test_each_software_and_artifact_review_remains_a_separate_gate(self):
    with reviewed_fixture() as (params, permit, _):
      original = json.loads(activation.REGISTRY.read_text())
      for flag in ('software_contract_reviewed', 'software_integration_reviewed', 'artifact_evidence_reviewed'):
        with self.subTest(flag=flag):
          registry = json.loads(json.dumps(original))
          registry['candidates'][0][flag] = False
          activation.REGISTRY.write_text(json.dumps(registry))
          self.assertFalse(activation.candidate_status(params)['eligible'])
          with self.assertRaises(NavigationModelDisabled):
            activation.authorize_runtime(params)
