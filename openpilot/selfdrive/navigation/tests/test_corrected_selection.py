"""Corrected registration/permit identity; retired files are never substitutes."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import activation, corrected_bigv2 as corrected
from openpilot.selfdrive.navigation.candidate_identity import RETIRED_MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub


@pytest.fixture
def registered(tmp_path):
  registry = json.loads(activation.REGISTRY.read_text())
  record = next(row for row in registry['candidates'] if row['id'] == corrected.MODEL_ID)
  paths = {}
  declarations = {}
  for field, key, name in (('compiled_artifact', 'artifact_sha256', 'corrected.sptg'),
                           ('warp_artifact', 'warp_sha256', 'warp.sptg'),
                           ('compiler_metadata', 'metadata_sha256', 'metadata.json')):
    path = tmp_path / name
    path.write_bytes(name.encode())
    record[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    paths[key] = path
    declarations[field] = {'path': str(path), 'sha256': record[key]}
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'entries': [{'id': corrected.MODEL_ID, **declarations}]}))
  retired = tmp_path / 'retired-manifest.json'
  retired.write_text(json.dumps({'entries': [{'id': RETIRED_MODEL_ID}]}))
  receipt = tmp_path / (corrected.MODEL_ID + '-registration.json')
  receipt.write_text(json.dumps({'registered': True, 'model_id': corrected.MODEL_ID, 'manifest_path': str(manifest),
                                 **{key: record[key] for key in paths}}))
  registry_path = tmp_path / 'registry.json'
  registry_path.write_text(json.dumps(registry))
  fallback = tmp_path / 'ordinary.sptg'
  fallback.write_bytes(b'SPTGPLN1-inert-ordinary')
  params = ParamsStub(NavigationModelCandidate=RETIRED_MODEL_ID, NavigationModelManifest=str(retired),
                      NavigationModelDrivingEnabled=False, IsOffroad=True, IsEngaged=False,
                      ModelManager_ActiveBundle={'minimumSelectorVersion': 21, 'runner': 'tinygrad',
                                                'models': [{'artifact': {'fileName': fallback.name}}]},
                      ModelManager_ActiveBundleChestnut={'keep': 'ordinary-big'}, ModelRunnerTypeCache=1)
  def load(path, *, model_id, **kwargs):
    assert Path(path) == manifest and model_id == corrected.MODEL_ID
    return SimpleNamespace(model_id=model_id, artifact_path=paths['artifact_sha256'], warp_path=paths['warp_sha256'],
                           input_specs={}, warp_frame_size=1622016, compiler_verification_validated=True,
                           metadata={'compiler_verification': {'compiler': {'sha256': record['compiler_sha256']},
                                                              'report': {'sha256': record['compiler_report_sha256']}}},
                           **{key: record[key] for key in paths})
  with ExitStack() as stack:
    for name, value in (('REGISTRY', registry_path), ('REGISTRATION_ROOT', tmp_path)):
      stack.enter_context(patch.object(activation, name, value))
    stack.enter_context(patch.object(activation, 'ordinary_fallback_reason', return_value=''))
    stack.enter_context(patch.object(activation, '_model_root', return_value=tmp_path))
    loader = stack.enter_context(patch.object(activation, 'load_experimental_model', side_effect=load))
    yield SimpleNamespace(params=params, manifest=manifest, retired=retired, receipt=receipt,
                          registry=registry_path, loader=loader, artifact=paths['artifact_sha256'])


def test_registered_corrected_candidate_admission_is_not_an_independent_selection(registered):
  before = copy.deepcopy(registered.params.values)
  status = activation.candidate_status(registered.params, candidate_id=corrected.MODEL_ID)
  assert status['registered'] and status['installed'] and not status['selected']
  permit = activation.admit_candidate(registered.params, candidate_id=corrected.MODEL_ID)
  assert permit.model_id == corrected.MODEL_ID
  assert registered.params.values == before
  assert activation.authorize_runtime(registered.params) is None


def test_corrected_receipt_cannot_redirect_to_retired_manifest(registered):
  receipt = json.loads(registered.receipt.read_text())
  receipt['manifest_path'] = str(registered.retired)
  registered.receipt.write_text(json.dumps(receipt))
  before = copy.deepcopy(registered.params.values)
  assert not activation.candidate_status(registered.params, candidate_id=corrected.MODEL_ID)['registered']
  with pytest.raises(NavigationModelDisabled, match='does not declare the reviewed'):
    activation.admit_candidate(registered.params, candidate_id=corrected.MODEL_ID)
  assert registered.params.values == before


def test_corrected_admission_binds_only_corrected_artifact_without_changing_selection(registered):
  before = copy.deepcopy(registered.params.values)
  permit = activation.admit_candidate(registered.params, candidate_id=corrected.MODEL_ID)
  assert registered.params.values == before
  assert permit.artifact_path == registered.artifact and permit.manifest_path == registered.manifest
  activation.validate_admission(permit, registered.params, allow_unselected_manifest=True)
  with pytest.raises(NavigationModelDisabled):
    activation.validate_runtime_permit(permit, params=registered.params)
  registered.params.put('NavigationModelCandidate', corrected.MODEL_ID)
  registered.params.put('NavigationModelManifest', str(registered.manifest))
  from openpilot.sunnypilot.models.navigation_bundle import navigation_bundle
  registered.params.put('ModelManager_ActiveBundleChestnut', navigation_bundle())
  activation.validate_runtime_permit(permit, params=registered.params)
  registered.params.put('NavigationModelManifest', str(registered.retired))
  with pytest.raises(NavigationModelDisabled):
    activation.validate_runtime_permit(permit, params=registered.params)


def test_configured_corrected_manifest_tampering_does_not_silently_use_other_receipt(registered):
  registered.params.put('NavigationModelManifest', str(registered.manifest))
  value = json.loads(registered.manifest.read_text())
  value['entries'][0]['compiled_artifact']['sha256'] = '0' * 64
  registered.manifest.write_text(json.dumps(value))
  assert not activation.candidate_status(registered.params, candidate_id=corrected.MODEL_ID)['eligible']


def test_branch_registry_has_only_corrected_candidate_with_unchanged_pins():
  registry = json.loads(activation.REGISTRY.read_text())
  assert len(registry['candidates']) == 1
  candidate = registry['candidates'][0]
  assert candidate['id'] == corrected.MODEL_ID and 'activation_approved' not in candidate
  assert candidate['artifact_sha256'] == corrected.ARTIFACT_SHA256
  assert candidate['metadata_sha256'] == corrected.METADATA_SHA256
  assert candidate['compiler_sha256'] == corrected.COMPILER_SHA256
  assert candidate['compiler_report_sha256'] == corrected.REPORT_SHA256
  assert candidate['map_precision_implementation_sha256'] == corrected.IMPLEMENTATION_SHA256
  assert candidate['compiler_revision'] == corrected.COMPILER_REVISION
  assert candidate['display_name'] == corrected.DISPLAY_NAME
  assert candidate['warp_sha256'] == 'f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b'
