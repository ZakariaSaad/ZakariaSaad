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
  """Simulate branch/hardware review and the separately tested native verifier.

  Dummy files are never executed. This is admission-unit evidence, not a claim
  that any real candidate has been hardware approved.
  """
  with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    paths = [root / name for name in ('artifact.sptg', 'warp.sptg', 'manifest.json')]
    for path in paths[:2]:
      path.write_bytes(path.name.encode())
    metadata_path = root / 'metadata.json'
    metadata_path.write_text('{}')
    paths[2].write_text(json.dumps({'entries': [{'id': 'big-nav-v2', 'compiler_metadata': {'path': str(metadata_path)}}]}))
    source = json.loads(activation.REGISTRY.read_text())
    record = source['candidates'][0]
    record.update(software_contract_reviewed=True, hardware_reviewed=True, activation_approved=True)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths[:2]] + [hashlib.sha256(metadata_path.read_bytes()).hexdigest()]
    record.update(dict(zip(('artifact_sha256', 'warp_sha256', 'metadata_sha256'), hashes, strict=True)))
    registry = root / 'registry.json'
    registry.write_text(json.dumps(source))
    params = ParamsStub(NavigationModelDrivingEnabled=True, NavigationModelCandidate='big-nav-v2', NavigationModelManifest=str(paths[2]))
    entry = SimpleNamespace(model_id='big-nav-v2', artifact_path=paths[0], warp_path=paths[1], input_specs=input_specs or {},
                            metadata=metadata or {}, warp_frame_size=1622016, compiler_verification_validated=True,
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
      self.assertIn('Physical Chestnut', status['reason'])
      with self.assertRaises(NavigationModelDisabled):
        activation.authorize_runtime(params)

  def test_only_reviewed_hashes_can_issue_a_runtime_permit(self):
    with reviewed_fixture() as (params, permit, verifier):
      self.assertEqual(permit.minimum_valid_frames, 132)
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
