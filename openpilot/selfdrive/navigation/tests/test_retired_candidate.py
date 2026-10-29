"""Retired identity cannot become loadable through saved settings or aliases."""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import activation, preparation
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID, RETIRED_MODEL_ID, RETIRED_ARTIFACT_SHA256
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError, load_experimental_model
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub, reviewed_fixture


def test_only_corrected_candidate_is_exposed_and_saved_retired_request_is_rejected():
  assert [entry['id'] for entry in activation.list_candidates()] == [MODEL_ID]
  params = ParamsStub(NavigationModelCandidate=RETIRED_MODEL_ID, NavigationModelDrivingEnabled=True,
                      NavigationModelManifest='/data/navigation-models/old/manifest.json')
  with patch.object(activation, 'load_experimental_model', side_effect=AssertionError('Retired model must never load')):
    assert activation.authorize_runtime(params) is None
  assert not activation.candidate_status(params)['selected']
  with pytest.raises(NavigationModelDisabled, match='Original Bigv2 is retired'):
    activation.admit_candidate(params, candidate_id=RETIRED_MODEL_ID)
  with pytest.raises(NavigationModelDisabled, match='Original Bigv2 is retired'):
    activation.candidate_manifest(params, RETIRED_MODEL_ID)


def test_renamed_registry_entry_cannot_admit_known_retired_artifact():
  with reviewed_fixture() as (params, permit, _):
    registry = json.loads(activation.REGISTRY.read_text())
    registry['candidates'][0]['artifact_sha256'] = RETIRED_ARTIFACT_SHA256
    activation.REGISTRY.write_text(json.dumps(registry))
    assert activation.list_candidates() == []
    with pytest.raises(NavigationModelDisabled, match='Original Bigv2 is retired'):
      activation.admit_candidate(params, candidate_id=MODEL_ID)


def test_retired_permit_identity_is_rejected_before_runtime_loading():
  with reviewed_fixture() as (params, permit, _):
    with pytest.raises(NavigationModelDisabled, match='Original Bigv2 is retired'):
      activation.validate_runtime_permit(replace(permit, model_id=RETIRED_MODEL_ID), params=params)


def test_retired_catalog_id_and_hash_are_rejected_before_artifact_read(tmp_path):
  with pytest.raises(ExperimentalCatalogError, match='Original Bigv2 is retired'):
    load_experimental_model(tmp_path / 'not-opened.json', experimental=True, backend='AMD', model_id=RETIRED_MODEL_ID)
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'schema_version': 1, 'scope': 'non_actuating_diagnostic_only',
                                  'normal_activation_allowed': False, 'entries': [
                                    {'id': MODEL_ID, 'compiled_artifact': {'sha256': RETIRED_ARTIFACT_SHA256}}]}))
  with pytest.raises(ExperimentalCatalogError, match='Original Bigv2 is retired'):
    load_experimental_model(manifest, experimental=True, backend='AMD')


def test_retired_registration_never_mutates_saved_configuration(tmp_path):
  params = ParamsStub(NavigationModelCandidate=RETIRED_MODEL_ID, NavigationModelDrivingEnabled=True)
  before = dict(params.values)
  with pytest.raises(ExperimentalCatalogError, match='Original Bigv2 is retired'):
    preparation.prepare_existing_model(params, is_offroad=True, model_dir=tmp_path, model_id=RETIRED_MODEL_ID)
  assert params.values == before
