"""Corrected candidate registration with inert native bytes and real file hashes.

Only the reviewed identities/profile/runtime are replaced with synthetic fixture
identities. No device, compiler, pickle reader, download or service is used.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import corrected_bigv2 as corrected, export_experimental_model as exporter, preparation
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError, load_experimental_model
from openpilot.selfdrive.navigation.tests import test_preparation as legacy


@pytest.fixture
def existing():
  fixture = legacy.TestExistingModelPreparation('runTest')
  fixture.setUp()
  yield fixture
  fixture.doCleanups()


@pytest.fixture
def candidate(existing):
  fixture = existing
  return fixture, fixture.artifact, fixture.report_path, fixture.report, fixture.fixture.fixture.model_header


def retired_package(fixture):
  """Retained historical files must survive cleanup but are never loaded."""
  package = fixture.package_root / 'big-nav-v2-historical'
  package.mkdir(parents=True, exist_ok=True)
  (package / 'artifact.sptg').write_bytes(b'inert historical artifact; never executable')
  manifest = package / 'manifest.json'
  manifest.write_text(json.dumps({'entries': [{'id': 'big-nav-v2'}]}))
  receipt = {'model_id': 'big-nav-v2', 'manifest_path': str(manifest), 'registered': True}
  (fixture.package_root / 'latest-registration.json').write_text(json.dumps(receipt))
  fixture.params.values.update(NavigationModelCandidate='big-nav-v2', NavigationModelManifest=str(manifest))
  return receipt


def prepare(candidate):
  fixture, *_ = candidate
  return fixture.prepare(model_id=corrected.MODEL_ID)


def test_corrected_registration_preserves_original_and_snapshots_exact_provenance(candidate):
  fixture, artifact, report_path, report, _ = candidate
  original = retired_package(fixture)
  original_manifest = Path(original['manifest_path'])
  original_files = {path: path.read_bytes() for path in original_manifest.parent.iterdir() if path.is_file()}
  original_receipt = (fixture.package_root / 'latest-registration.json').read_bytes()
  artifact_before = artifact.read_bytes()
  result = prepare(candidate)
  assert result['model_id'] == corrected.MODEL_ID and result['registered'] and not result['activation_allowed']
  assert (fixture.package_root / 'latest-registration.json').read_bytes() == original_receipt
  assert all(path.read_bytes() == data for path, data in original_files.items())
  assert artifact.read_bytes() == artifact_before
  assert fixture.params.get('NavigationModelCandidate') == corrected.MODEL_ID
  assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
  entry = load_experimental_model(result['manifest_path'], experimental=True, backend='AMD', model_id=corrected.MODEL_ID)
  assert entry.artifact_path == artifact and entry.metadata['map_precision_correction'] == report['map_precision_correction']
  assert entry.metadata['compiler_verification']['map_precision_implementation']['path'] == 'map-precision-source.py'
  package = Path(result['manifest_path']).parent
  assert (package / 'compiler-report.json').read_bytes() == report_path.read_bytes()
  assert json.loads((package / 'registration.json').read_text())['model_id'] == corrected.MODEL_ID
  fixture.assert_preserved()
  with pytest.raises(ExperimentalCatalogError, match='Original Bigv2 is retired'):
    load_experimental_model(result['manifest_path'], experimental=True, backend='AMD', model_id='big-nav-v2')
  with patch.object(preparation, 'export_experimental_model', side_effect=AssertionError('Reuse immutable package')):
    assert prepare(candidate)['manifest_path'] == result['manifest_path']


@pytest.mark.parametrize('mutation', [
  lambda r: r.update(candidate_activation_approved=True),
  lambda r: r.update(compiler_revision='0' * 40),
  lambda r: r['map_precision_correction'].update(implementation_sha256='0' * 64),
  lambda r: r['map_precision_correction']['compiled_precision']['dot'].update(early_half_casts=1),
  lambda r: r['map_precision_correction']['compiled_precision']['bias_gamma'].update(kernel_sha256='0' * 64),
  lambda r: r['map_fixture_verification']['cases']['overflow']['outputs']['next_state_feat_q'].update(source_finite=False),
  lambda r: r['map_fixture_verification']['cases'].pop('working'),
])
def test_corrected_report_or_native_kernel_discrepancy_refuses_registration(candidate, mutation):
  fixture, _, report_path, report, _ = candidate
  changed = copy.deepcopy(report)
  mutation(changed)
  report_path.write_text(json.dumps(changed))
  # Exercise semantic/kernel checks in addition to the reviewed whole-file pin.
  with patch.object(corrected, 'REPORT_SHA256', exporter._digest(report_path)), pytest.raises(ExperimentalCatalogError):
    prepare(candidate)
  assert fixture.params.get('NavigationModelManifest') == 'previous-manifest.json'
  assert fixture.params.get('NavigationModelCandidate') == 'old-candidate'
  fixture.assert_preserved()


def test_corrected_registration_rejects_unreviewed_report_bytes(candidate):
  fixture, _, report_path, *_ = candidate
  report_path.write_text(report_path.read_text() + '\n')
  with pytest.raises(ExperimentalCatalogError, match='report differs'):
    prepare(candidate)
  assert fixture.params.get('NavigationModelManifest') == 'previous-manifest.json'


def test_corrected_registration_requires_fresh_offroad_before_any_selection(candidate):
  fixture, *_ = candidate
  with patch.object(preparation, 'require_offroad', side_effect=ValueError('deviceState.started is true')):
    with pytest.raises(ValueError, match='deviceState.started'):
      prepare(candidate)
  assert fixture.params.get('NavigationModelManifest') == 'previous-manifest.json'
  assert fixture.params.writes == []


@pytest.mark.parametrize('failure', ['raise', 'ignore', 'write_then_raise'])
def test_failed_manifest_publication_restores_original_selection_and_keeps_driving_off(candidate, failure):
  fixture, *_ = candidate
  original = retired_package(fixture)
  original_receipt = (fixture.package_root / 'latest-registration.json').read_bytes()
  original_put = fixture.params.put
  failed = False

  def put(key, value, **kwargs):
    nonlocal failed
    if key == 'NavigationModelManifest' and not failed:
      failed = True
      if failure == 'ignore':
        return None
      if failure == 'write_then_raise':
        original_put(key, value, **kwargs)
      raise OSError('synthetic manifest storage error')
    return original_put(key, value, **kwargs)

  with patch.object(fixture.params, 'put', side_effect=put), pytest.raises((OSError, ExperimentalCatalogError)):
    prepare(candidate)
  assert failed
  assert fixture.params.get('NavigationModelCandidate') == 'big-nav-v2'
  assert fixture.params.get('NavigationModelManifest') == original['manifest_path']
  assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
  assert (fixture.package_root / 'latest-registration.json').read_bytes() == original_receipt
  assert json.loads((fixture.package_root / (corrected.MODEL_ID + '-registration.json')).read_text())['registered'] is True
  fixture.assert_preserved()


def test_corrected_registration_reuses_retained_compiler_after_source_update(existing):
  first = existing.prepare()
  existing.fixture.compiler.write_text('# later compiler version; corrected package retains its exact source\n')
  second = existing.prepare()
  assert first['manifest_path'] == second['manifest_path']


def test_file_only_preparation_keeps_previous_selection_and_enable_request(candidate):
  fixture, *_ = candidate
  fixture.params.values['NavigationModelDrivingEnabled'] = True
  before = copy.deepcopy(fixture.params.values)
  receipt = fixture.prepare(publish_selection=False)
  assert receipt['registered'] and receipt['model_id'] == corrected.MODEL_ID
  assert fixture.params.values == before
  assert fixture.params.writes == []
  repeated = fixture.prepare(publish_selection=False)
  assert repeated == receipt
  assert fixture.params.values == before


def test_failed_preparation_keeps_complete_corrected_receipt_and_selection(candidate):
  fixture, _, report_path, *_ = candidate
  receipt = prepare(candidate)
  receipt_path = fixture.package_root / (corrected.MODEL_ID + '-registration.json')
  retained = receipt_path.read_bytes()
  fixture.params.values['NavigationModelDrivingEnabled'] = True
  before = copy.deepcopy(fixture.params.values)
  report_path.write_bytes(b'{"changed":true}')
  with pytest.raises(ExperimentalCatalogError):
    fixture.prepare(publish_selection=False)
  assert receipt_path.read_bytes() == retained
  assert fixture.params.values == before
  assert fixture.params.get('NavigationModelManifest') == receipt['manifest_path']
