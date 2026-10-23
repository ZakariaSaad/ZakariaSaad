"""Candidate-specific registration/selection; inert files, no model execution."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import activation, selection
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub

ORIGINAL = 'big-nav-v2'
CORRECTED = 'big-nav-v2-map-fp32'


@pytest.fixture
def two_candidates(tmp_path):
  registry = json.loads(activation.REGISTRY.read_text())
  original = next(entry for entry in registry['candidates'] if entry['id'] == ORIGINAL)
  corrected = copy.deepcopy(original)
  corrected.update(id=CORRECTED, display_name='Bigv2 map FP32 corrected', activation_approved=False,
                   compiler_sha256='c' * 64, compiler_report_sha256='d' * 64)
  registry['candidates'] = [original, corrected]
  packages = {}
  for entry in registry['candidates']:
    root = tmp_path / entry['id']
    root.mkdir()
    artifact, warp, metadata, manifest = (root / name for name in ('artifact.sptg', 'warp.sptg', 'metadata.json', 'manifest.json'))
    artifact.write_bytes(entry['id'].encode())
    warp.write_bytes(b'same reviewed camera warp')
    metadata.write_text('{}')
    records = {}
    for field, key, path in (('compiled_artifact', 'artifact_sha256', artifact), ('warp_artifact', 'warp_sha256', warp),
                              ('compiler_metadata', 'metadata_sha256', metadata)):
      digest = hashlib.sha256(path.read_bytes()).hexdigest()
      entry[key] = digest
      records[field] = {'path': str(path), 'sha256': digest}
    manifest.write_text(json.dumps({'entries': [{'id': entry['id'], **records}]}))
    receipt = {'registered': True, 'model_id': entry['id'], 'manifest_path': str(manifest),
               **{key: entry[key] for key in ('artifact_sha256', 'warp_sha256', 'metadata_sha256')}}
    receipt_path = tmp_path / ('latest-registration.json' if entry['id'] == ORIGINAL else entry['id'] + '-registration.json')
    receipt_path.write_text(json.dumps(receipt))
    packages[entry['id']] = SimpleNamespace(artifact=artifact, warp=warp, metadata=metadata, manifest=manifest, receipt=receipt_path)
  registry_path = tmp_path / 'registry.json'
  registry_path.write_text(json.dumps(registry))
  fallback = tmp_path / 'ordinary.sptg'
  fallback.write_bytes(b'SPTGPLN1-inert-ordinary')
  params = ParamsStub(NavigationModelCandidate=ORIGINAL, NavigationModelManifest=str(packages[ORIGINAL].manifest),
                      NavigationModelDrivingEnabled=False, IsOffroad=True, IsEngaged=False,
                      ModelManager_ActiveBundle={'minimumSelectorVersion': 21, 'runner': 'tinygrad',
                                                'models': [{'artifact': {'fileName': fallback.name}}]},
                      ModelManager_ActiveBundleChestnut={'keep': 'ordinary big model'}, ModelRunnerTypeCache=1)

  def load(manifest, *, model_id, **kwargs):
    entry = next(row for row in json.loads(registry_path.read_text())['candidates'] if row['id'] == model_id)
    package = packages[model_id]
    assert Path(manifest) == package.manifest
    return SimpleNamespace(model_id=model_id, artifact_path=package.artifact, warp_path=package.warp,
                           input_specs={}, warp_frame_size=1622016, compiler_verification_validated=True,
                           metadata={'compiler_verification': {'compiler': {'sha256': entry['compiler_sha256']},
                                                              'report': {'sha256': entry['compiler_report_sha256']}}},
                           **{key: entry[key] for key in ('artifact_sha256', 'warp_sha256', 'metadata_sha256')})

  with ExitStack() as stack:
    for name, value in (('REGISTRY', registry_path), ('REGISTRATION_ROOT', tmp_path)):
      stack.enter_context(patch.object(activation, name, value))
    stack.enter_context(patch.object(activation, 'ordinary_fallback_reason', return_value=''))
    stack.enter_context(patch.object(activation, '_model_root', return_value=tmp_path))
    loader = stack.enter_context(patch.object(activation, 'load_experimental_model', side_effect=load))
    guard = stack.enter_context(patch.object(selection, 'require_offroad'))
    yield SimpleNamespace(params=params, packages=packages, registry=registry_path, loader=loader, guard=guard)


def test_registered_corrected_candidate_does_not_inherit_original_approval(two_candidates):
  fixture = two_candidates
  before = copy.deepcopy(fixture.params.values)
  status = activation.candidate_status(fixture.params, candidate_id=CORRECTED)
  assert status['registered'] and status['installed'] and not status['approved'] and not status['eligible']
  assert status['manifest_path'] == str(fixture.packages[CORRECTED].manifest)
  assert status['blockers'] == ['Explicit user authorization for Navigation Model Driving is still required']
  with pytest.raises(NavigationModelDisabled, match='Explicit user authorization'):
    selection.set_driving_enabled(fixture.params, True, offroad=True, candidate_id=CORRECTED)
  assert fixture.params.values == before
  fixture.loader.assert_not_called()


def test_select_and_rollback_bind_matching_manifests_without_changing_ordinary_slots(two_candidates):
  fixture = two_candidates
  ordinary = {key: copy.deepcopy(value) for key, value in fixture.params.values.items() if not key.startswith('NavigationModel')}
  original_receipt = fixture.packages[ORIGINAL].receipt.read_bytes()
  for candidate in (CORRECTED, ORIGINAL):
    fixture.params.put_bool('NavigationModelDrivingEnabled', True)
    selection.select_candidate(fixture.params, candidate, offroad=True)
    assert fixture.params.get('NavigationModelCandidate') == candidate
    assert fixture.params.get('NavigationModelManifest') == str(fixture.packages[candidate].manifest)
    assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
    assert ordinary == {key: value for key, value in fixture.params.values.items() if not key.startswith('NavigationModel')}
  assert fixture.packages[ORIGINAL].receipt.read_bytes() == original_receipt
  assert fixture.guard.call_count == 2
  fixture.loader.assert_not_called()


def test_corrected_receipt_cannot_redirect_selection_to_original_model(two_candidates):
  fixture = two_candidates
  path = fixture.packages[CORRECTED].receipt
  receipt = json.loads(path.read_text())
  receipt['manifest_path'] = str(fixture.packages[ORIGINAL].manifest)
  path.write_text(json.dumps(receipt))
  before = copy.deepcopy(fixture.params.values)
  assert not activation.candidate_status(fixture.params, candidate_id=CORRECTED)['registered']
  with pytest.raises(NavigationModelDisabled, match='does not declare the reviewed'):
    selection.select_candidate(fixture.params, CORRECTED, offroad=True)
  assert fixture.params.values == before


def test_genuine_offroad_refusal_prevents_selection_changes(two_candidates):
  fixture = two_candidates
  before = copy.deepcopy(fixture.params.values)
  fixture.guard.side_effect = NavigationModelDisabled('deviceState.started is true')
  with pytest.raises(NavigationModelDisabled, match='deviceState.started'):
    selection.select_candidate(fixture.params, CORRECTED, offroad=True)
  assert fixture.params.values == before


def test_future_reviewed_approval_admits_only_the_corrected_artifact(two_candidates):
  fixture = two_candidates
  # Test-only explicit branch review; production corrected approval stays false.
  registry = json.loads(fixture.registry.read_text())
  registry['candidates'][1]['activation_approved'] = True
  fixture.registry.write_text(json.dumps(registry))
  before = copy.deepcopy(fixture.params.values)
  permit = activation.admit_candidate(fixture.params, candidate_id=CORRECTED)
  assert fixture.params.values == before
  assert permit.artifact_path == fixture.packages[CORRECTED].artifact
  assert permit.manifest_path == fixture.packages[CORRECTED].manifest
  activation.validate_admission(permit, fixture.params, allow_unselected_manifest=True)
  with pytest.raises(NavigationModelDisabled, match='manifest changed'):
    activation.validate_runtime_permit(permit, params=fixture.params)
  selection.set_driving_enabled(fixture.params, True, offroad=True, candidate_id=CORRECTED)
  assert fixture.params.get('NavigationModelManifest') == str(fixture.packages[CORRECTED].manifest)
  assert fixture.params.get('NavigationModelCandidate') == CORRECTED
  activation.validate_runtime_permit(permit, params=fixture.params)
  fixture.params.put('NavigationModelManifest', str(fixture.packages[ORIGINAL].manifest))
  with pytest.raises(NavigationModelDisabled, match='manifest changed'):
    activation.validate_runtime_permit(permit, params=fixture.params)


def test_failed_enable_restores_original_manifest_and_selection(two_candidates):
  fixture = two_candidates
  registry = json.loads(fixture.registry.read_text())
  registry['candidates'][1]['activation_approved'] = True
  fixture.registry.write_text(json.dumps(registry))
  before = copy.deepcopy(fixture.params.values)
  put_bool = fixture.params.put_bool

  def fail_enable(key, value, **kwargs):
    if key != 'NavigationModelDrivingEnabled' or not value:
      put_bool(key, value, **kwargs)

  with patch.object(fixture.params, 'put_bool', side_effect=fail_enable):
    with pytest.raises(NavigationModelDisabled, match='Could not persist'):
      selection.set_driving_enabled(fixture.params, True, offroad=True, candidate_id=CORRECTED)
  assert fixture.params.values == before


def test_branch_registry_pins_corrected_candidate_without_changing_original_approval():
  from openpilot.selfdrive.navigation import corrected_bigv2 as corrected
  registry = json.loads(activation.REGISTRY.read_text())
  candidates = {entry['id']: entry for entry in registry['candidates']}
  original, candidate = candidates[ORIGINAL], candidates[CORRECTED]
  assert original['activation_approved'] is True
  assert original['artifact_sha256'] == 'feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4'
  assert candidate['activation_approved'] is False
  assert candidate['artifact_sha256'] == corrected.ARTIFACT_SHA256
  assert candidate['metadata_sha256'] == corrected.METADATA_SHA256
  assert candidate['compiler_sha256'] == corrected.COMPILER_SHA256
  assert candidate['compiler_report_sha256'] == corrected.REPORT_SHA256
  assert candidate['map_precision_implementation_sha256'] == corrected.IMPLEMENTATION_SHA256
  assert candidate['compiler_revision'] == corrected.COMPILER_REVISION
  assert candidate['display_name'] == corrected.DISPLAY_NAME
  assert candidate['warp_sha256'] == original['warp_sha256']


def test_manifest_switch_during_enable_admission_cancels_request(two_candidates):
  fixture = two_candidates
  registry = json.loads(fixture.registry.read_text())
  registry['candidates'][1]['activation_approved'] = True
  fixture.registry.write_text(json.dumps(registry))
  native_admit = selection.admit_candidate

  def change_manifest(*args, **kwargs):
    permit = native_admit(*args, **kwargs)
    fixture.params.put('NavigationModelManifest', str(fixture.packages[CORRECTED].manifest))
    return permit

  with patch.object(selection, 'admit_candidate', side_effect=change_manifest):
    with pytest.raises(NavigationModelDisabled, match='activation cancelled'):
      selection.set_driving_enabled(fixture.params, True, offroad=True, candidate_id=CORRECTED)
  assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
  assert fixture.params.get('NavigationModelCandidate') == ORIGINAL


@pytest.mark.parametrize('operation', ['select', 'enable'])
@pytest.mark.parametrize('restore_behavior', ['raise_before_write', 'ignored_write', 'raise_after_write'])
def test_partial_pointer_write_attempts_both_restores_and_reports_failure(two_candidates, operation, restore_behavior):
  fixture = two_candidates
  registry = json.loads(fixture.registry.read_text())
  registry['candidates'][1]['activation_approved'] = True
  fixture.registry.write_text(json.dumps(registry))
  put = fixture.params.put

  def failing_candidate_write(key, value, **kwargs):
    if key == 'NavigationModelCandidate':
      if value == CORRECTED:
        put(key, value, **kwargs)
        raise OSError('candidate write failed after persistence')
      if value == ORIGINAL:
        if restore_behavior == 'raise_before_write':
          raise OSError('candidate restoration failed')
        if restore_behavior == 'ignored_write':
          return
        put(key, value, **kwargs)
        raise OSError('candidate restoration raised after persistence')
    put(key, value, **kwargs)

  with patch.object(fixture.params, 'put', side_effect=failing_candidate_write):
    with pytest.raises(NavigationModelDisabled, match='candidate write failed after persistence.*rollback failed.*NavigationModelCandidate'):
      if operation == 'select':
        selection.select_candidate(fixture.params, CORRECTED, offroad=True)
      else:
        selection.set_driving_enabled(fixture.params, True, offroad=True, candidate_id=CORRECTED)
  assert fixture.params.get('NavigationModelManifest') == str(fixture.packages[ORIGINAL].manifest)
  assert fixture.params.get('NavigationModelCandidate') == (ORIGINAL if restore_behavior == 'raise_after_write' else CORRECTED)
  assert not fixture.params.get_bool('NavigationModelDrivingEnabled')


def test_enable_flag_write_then_raise_still_restores_both_pointers(two_candidates):
  fixture = two_candidates
  registry = json.loads(fixture.registry.read_text())
  registry['candidates'][1]['activation_approved'] = True
  fixture.registry.write_text(json.dumps(registry))
  before = copy.deepcopy(fixture.params.values)
  put_bool = fixture.params.put_bool

  def write_flag_then_raise(key, value, **kwargs):
    put_bool(key, value, **kwargs)
    raise OSError('flag write raised after persistence')

  with patch.object(fixture.params, 'put_bool', side_effect=write_flag_then_raise):
    with pytest.raises(NavigationModelDisabled, match='rollback failed.*NavigationModelDrivingEnabled'):
      selection.set_driving_enabled(fixture.params, True, offroad=True, candidate_id=CORRECTED)
  assert fixture.params.values == before
