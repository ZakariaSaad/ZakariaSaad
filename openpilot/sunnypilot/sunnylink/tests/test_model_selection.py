"""Existing SunnyLink wire operations reach the real off-road selection queue."""
import base64
import copy
import gzip
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub
from openpilot.common.params import Params
from openpilot.sunnypilot.models import selection
from openpilot.sunnypilot.sunnylink import model_selection as bridge


def catalog_bundle(ref, *, big=True):
  return {'index': 1, 'short_name': ref, 'display_name': ref, 'ref': ref,
          'generation': 10, 'environment': 'AMD' if big else 'QCOM', 'runner': 'tinygrad',
          'minimum_selector_version': 21, 'is_20hz': False, 'models': []}


@pytest.fixture
def params():
  return ParamsStub(IsOffroad=True, IsEngaged=False, NavigationEnabled=True,
                    NavigationModelCandidate=bridge.MODEL_ID, NavigationModelDrivingEnabled=False,
                    NavigationModelManifest='/data/navigation-models/corrected/manifest.json',
                    ModelManager_ActiveJson={'qcom': 'https://example.test/small.json', 'chestnut': 'https://example.test/v28.json'},
                    ModelManager_ModelsCache={'bundles': [catalog_bundle('ordinary-small', big=False)]},
                    ModelManager_ModelsCache_Chestnut={'bundles': [catalog_bundle('ordinary-big')]},
                    ModelManager_ActiveBundle={'ref': 'ordinary-small'},
                    ModelManager_ActiveBundleChestnut={'ref': 'ordinary-big'}, ModelRunnerTypeCache=1)


@pytest.fixture
def offroad():
  with patch.object(selection, 'require_offroad') as guard:
    yield guard


def test_device_catalog_projection_preserves_upstream_bytes_and_has_no_fake_runtime(params):
  before = copy.deepcopy(params.values)
  raw = json.dumps(params.get('ModelManager_ModelsCache_Chestnut')).encode()
  with patch.object(bridge, 'ARTIFACT') as artifact:
    artifact.is_file.return_value = True
    projected = json.loads(bridge.project_get_param(params, 'ModelManager_ModelsCache_Chestnut', raw))
  assert projected['bundles'][0] == before['ModelManager_ModelsCache_Chestnut']['bundles'][0]
  nav = projected['bundles'][-1]
  assert nav['ref'] == bridge.NAV_REF and nav['artifact_sha256'] == bridge.ARTIFACT_SHA256
  assert nav['models'] == [] and 'generation' not in nav and nav['download_available']
  assert nav['release_assets'] == [
    {'url': bridge.MODEL_URL, 'sha256': bridge.ARTIFACT_SHA256, 'size': 810439301},
    {'url': bridge.SIDECAR_URL, 'sha256': 'ebe109ce6352e8d144b60b623c047320b923d6493921f860e48f0efd32a291de', 'size': 3279412},
  ]
  assert nav['runtime'] == 'native-navigation' and 'installed local' in nav['display_name']
  projected_urls = json.loads(bridge.project_get_param(params, 'ModelManager_ActiveJson', json.dumps(before['ModelManager_ActiveJson']).encode()))
  assert projected_urls == {'qcom': before['ModelManager_ActiveJson']['qcom'], 'chestnut': ''}
  assert params.values == before


def test_missing_binary_is_advertised_as_downloadable_without_ordinary_model_parts():
  with patch.object(bridge, 'ARTIFACT') as artifact:
    artifact.is_file.return_value = False
    entry = bridge.local_bundle()
  assert 'downloadable' in entry['display_name']
  assert entry['download_available'] is True and entry['models'] == []
  assert entry['overrides']['folder'] == 'Navigation models'


def test_retired_and_duplicate_navigation_entries_removed_only_from_response(params):
  old = catalog_bundle('navigation:big-nav-v2')
  raw = json.dumps({'bundles': [old, bridge.local_bundle(), catalog_bundle('ordinary-big')]}).encode()
  response = json.loads(bridge.project_get_param(params, 'ModelManager_ModelsCache_Chestnut', raw))
  assert [row['ref'] for row in response['bundles']] == ['ordinary-big', bridge.NAV_REF]
  assert json.loads(raw)['bundles'][0] == old


def test_active_projection_tracks_request_without_overwriting_ordinary_big_slot(params):
  ordinary = json.dumps(params.get('ModelManager_ActiveBundleChestnut')).encode()
  assert bridge.project_get_param(params, 'ModelManager_ActiveBundleChestnut', ordinary) == ordinary
  params.put_bool('NavigationModelDrivingEnabled', True)
  assert json.loads(bridge.project_get_param(params, 'ModelManager_ActiveBundleChestnut', ordinary))['ref'] == bridge.NAV_REF
  assert params.get('ModelManager_ActiveBundleChestnut') == {'ref': 'ordinary-big'}
  params.put_bool('NavigationModelDrivingEnabled', False)
  assert bridge.project_get_param(params, 'ModelManager_ActiveBundleChestnut', ordinary) == ordinary


def test_current_frontend_nav_ref_enters_shared_pending_queue(params, offroad):
  before = copy.deepcopy(params.values)
  remaining = bridge.apply_remote_model_settings(params, {'ModelManager_DownloadRef': bridge.NAV_REF, 'UnrelatedSetting': 3})
  assert remaining == {'UnrelatedSetting': 3}
  request = params.get(selection.REQUEST_KEY)
  assert request['ref'] == bridge.NAV_REF and request['origin'] == 'sunnylink' and request['enable'] is True
  assert selection.selection_status(params)['state'] == 'pending'
  assert not params.get_bool('NavigationModelDrivingEnabled')
  for key in ('NavigationModelManifest', 'ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut'):
    assert params.get(key) == before[key]
  assert offroad.call_count == 2
  assert bridge.project_get_param(params, bridge.STATUS_KEY, None).startswith(b'Pending:')


def test_frontend_default_selection_disables_navigation_only_when_ready(params, offroad):
  params.put_bool('NavigationModelDrivingEnabled', True)
  fallback = copy.deepcopy(params.get('ModelManager_ActiveBundle'))
  bridge.apply_remote_model_settings(params, {'ModelManager_ActiveBundleChestnut': '{}'})
  assert params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('ModelManager_ActiveBundleChestnut') == {'ref': 'ordinary-big'}
  with patch.object(selection, '_default_ready'):
    selection.process_pending_selection(params)
  assert not params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('ModelManager_ActiveBundleChestnut') is None
  assert params.get('ModelManager_ActiveBundle') == fallback
  assert params.get_bool('NavigationEnabled')


def test_ordinary_ref_completion_uses_shared_switch_and_preserves_map_and_small(params, offroad):
  params.put_bool('NavigationModelDrivingEnabled', True)
  bridge.apply_remote_model_settings(params, {'ModelManager_DownloadRef': 'ordinary-big'})
  assert params.get('ModelManager_DownloadRef') == 'ordinary-big'
  assert params.get_bool('NavigationModelDrivingEnabled')
  # Model manager invokes completion only after its existing artifact SHA check.
  assert selection.complete_ordinary_selection(params, 'chestnut', {'ref': 'ordinary-big'}, 'ordinary-big')
  assert not params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('ModelManager_ActiveBundle') == {'ref': 'ordinary-small'}
  assert params.get_bool('NavigationEnabled')


def test_cancel_and_explicit_off_are_real_shared_operations(params, offroad):
  bridge.apply_remote_model_settings(params, {'ModelManager_DownloadRef': bridge.NAV_REF})
  bridge.apply_remote_model_settings(params, {'ModelManager_DownloadRef': ''})
  assert params.get(selection.REQUEST_KEY) is None
  assert selection.selection_status(params)['state'] == 'cancelled'
  params.put_bool('NavigationModelDrivingEnabled', True)
  bridge.apply_remote_model_settings(params, {'NavigationModelDrivingEnabled': False})
  assert not params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('ModelManager_ActiveBundleChestnut') == {'ref': 'ordinary-big'}


def test_revision_guard_refuses_stale_remote_choice_without_mutation(params, offroad):
  revision = bridge.project_get_param(params, bridge.REVISION_KEY, None).decode()
  bridge.apply_remote_model_settings(params, {bridge.REVISION_KEY: revision, 'ModelManager_DownloadRef': bridge.NAV_REF})
  before = copy.deepcopy(params.values)
  with pytest.raises(NavigationModelDisabled, match='selection changed'):
    bridge.apply_remote_model_settings(params, {bridge.REVISION_KEY: revision, 'ModelManager_ActiveBundleChestnut': {}})
  assert params.values == before


@pytest.mark.parametrize('updates', [
  {'ModelManager_DownloadRef': 'navigation:big-nav-v2'},
  {'ModelManager_DownloadRef': 'not-in-current-catalog'},
  {'ModelManager_ActiveBundleChestnut': {'ref': 'navigation:big-nav-v2'}},
  {'NavigationModelManifest': '/tmp/unverified.json'},
  {'NavigationModelCandidate': 'big-nav-v2'},
  {'NavigationModelDrivingEnabled': 'yes'},
  {'ModelManager_DownloadRef': bridge.NAV_REF, 'NavigationModelDrivingEnabled': False},
])
def test_unsafe_or_conflicting_wire_values_do_not_mutate_selection(params, updates):
  before = copy.deepcopy(params.values)
  with pytest.raises(NavigationModelDisabled):
    bridge.apply_remote_model_settings(params, updates)
  assert params.values == before


def test_frontend_forceoffroad_cannot_bypass_device_guard(params):
  before = copy.deepcopy(params.values)
  with patch.object(selection, 'require_offroad', side_effect=NavigationModelDisabled('deviceState.started is true')):
    with pytest.raises(NavigationModelDisabled, match='deviceState.started'):
      bridge.apply_remote_model_settings(params, {'ModelManager_DownloadRef': bridge.NAV_REF})
  assert params.values == before


def test_backup_never_restores_proof_status_or_pending_downloads(params):
  before = copy.deepcopy(params.values)
  unsafe = {key: base64.b64encode(b'ignored').decode() for key in bridge.PROTECTED_KEYS |
            {bridge.REVISION_KEY, 'ModelManager_DownloadRef', 'NavigationModelDrivingEnabled'}}
  unsafe['UnrelatedSetting'] = base64.b64encode(b'keep').decode()
  assert bridge.model_restore_settings(params, unsafe) == {'UnrelatedSetting': unsafe['UnrelatedSetting']}
  assert params.values == before


@pytest.fixture
def rpc_device(tmp_path, monkeypatch, params):
  """Actual RPC methods, typed persistent Params, and the production queue."""
  root = str(tmp_path / 'params')
  monkeypatch.setenv('PARAMS_ROOT', root)
  native = Params(root)
  for key, value in params.values.items():
    native.put(key, value, block=True)
  from openpilot.sunnypilot.sunnylink.athena import sunnylinkd
  from openpilot.selfdrive.navigation import download
  monkeypatch.setattr(sunnylinkd, 'params', native)
  # These RPC tests exercise queue/admission publication; streamed installer
  # bytes and registration are covered separately by the download/preparation tests.
  monkeypatch.setattr(download, 'prepare_downloaded_model', lambda *args, **kwargs: None)
  return SimpleNamespace(params=native, rpc=sunnylinkd, root=tmp_path)


def wire(value):
  return base64.b64encode((value if isinstance(value, str) else json.dumps(value)).encode()).decode()


def rpc_value(response, key):
  return base64.b64decode(response[key]).decode()


def test_actual_rpc_catalog_active_and_schema_status_are_consistent(rpc_device):
  p, rpc = rpc_device.params, rpc_device.rpc
  before = {key: p.get(key) for key in ('ModelManager_ActiveJson', 'ModelManager_ModelsCache_Chestnut', 'ModelManager_ActiveBundleChestnut')}
  response = rpc.getParams(list(before))
  assert json.loads(rpc_value(response, 'ModelManager_ActiveJson'))['chestnut'] == ''
  catalog = json.loads(rpc_value(response, 'ModelManager_ModelsCache_Chestnut'))
  assert [item['ref'] for item in catalog['bundles']] == ['ordinary-big', bridge.NAV_REF]
  assert json.loads(rpc_value(response, 'ModelManager_ActiveBundleChestnut'))['ref'] == 'ordinary-big'
  assert bridge.STATUS_KEY in response and bridge.REVISION_KEY in response
  schema = json.loads(gzip.decompress(base64.b64decode(rpc.getParamsMetadata())))
  panel = next(panel for panel in schema['panels'] if panel['id'] == 'models')
  section = next(section for section in panel['sections'] if section['id'] == 'navigation_model')
  assert {item['key'] for item in section['items']} == {'NavigationModelDrivingEnabled', bridge.STATUS_KEY}
  assert before == {key: p.get(key) for key in before}


def test_actual_rpc_nav_selection_is_pending_until_shared_admission_completes(rpc_device, offroad):
  p, rpc = rpc_device.params, rpc_device.rpc
  manifest = rpc_device.root / 'registered-manifest.json'
  permit = SimpleNamespace(manifest_path=manifest, model_id=bridge.MODEL_ID)
  before = {key: p.get(key) for key in ('ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut')}
  rpc.saveParams({'ModelManager_DownloadRef': wire(bridge.NAV_REF)})
  assert selection.selection_status(p)['state'] == 'pending'
  # The existing hosted frontend polls this ref and sends an empty ref to
  # cancel. It must remain visible until admission commits the new selection.
  assert rpc_value(rpc.getParams(['ModelManager_DownloadRef']), 'ModelManager_DownloadRef') == bridge.NAV_REF
  assert not p.get_bool('NavigationModelDrivingEnabled')
  with patch.object(selection, 'candidate_status', return_value={'registered': True}), \
       patch.object(selection, 'admit_candidate', return_value=permit) as admit, \
       patch.object(selection, 'validate_admission') as validate:
    selection.process_pending_selection(p)
  admit.assert_called_once_with(p, candidate_id=bridge.MODEL_ID)
  validate.assert_called_once_with(permit, p, allow_unselected_manifest=True)
  assert p.get_bool('NavigationModelDrivingEnabled')
  assert p.get('NavigationModelManifest') == str(manifest)
  assert p.get('ModelManager_DownloadRef') is None
  assert before == {key: p.get(key) for key in before}
  response = rpc.getParams(['ModelManager_ActiveBundleChestnut'])
  assert json.loads(rpc_value(response, 'ModelManager_ActiveBundleChestnut'))['ref'] == bridge.NAV_REF
  assert 'requested' in rpc_value(response, bridge.STATUS_KEY).lower()


def test_current_hosted_cancel_clears_native_download_request_and_keeps_selected_model(rpc_device, offroad):
  p, rpc = rpc_device.params, rpc_device.rpc
  before = {key: p.get(key) for key in selection.SELECTION_KEYS}
  rpc.saveParams({'ModelManager_DownloadRef': wire(bridge.NAV_REF)})
  assert p.get('ModelManager_DownloadRef') == bridge.NAV_REF
  rpc.saveParams({'ModelManager_DownloadRef': wire('')})
  response = rpc.getParams(['ModelManager_DownloadRef', 'ModelManager_ActiveBundleChestnut'])
  assert rpc_value(response, 'ModelManager_DownloadRef') == ''
  assert p.get(selection.REQUEST_KEY) is None
  assert rpc_value(response, bridge.STATUS_KEY).startswith('Cancelled:')
  assert before == {key: p.get(key) for key in before}


def test_actual_rpc_admission_failure_preserves_previous_selection(rpc_device, offroad):
  p, rpc = rpc_device.params, rpc_device.rpc
  before = {key: p.get(key) for key in selection.SELECTION_KEYS}
  rpc.saveParams({'ModelManager_DownloadRef': wire(bridge.NAV_REF)})
  with patch.object(selection, 'candidate_status', return_value={'registered': True}), \
       patch.object(selection, 'admit_candidate', side_effect=NavigationModelDisabled('Artifact integrity mismatch')):
    selection.process_pending_selection(p)
  assert selection.selection_status(p)['state'] == 'failed'
  assert 'Artifact integrity mismatch' in rpc_value(rpc.getParams(['ModelManager_DownloadRef']), bridge.STATUS_KEY)
  assert before == {key: p.get(key) for key in before}


def test_actual_rpc_default_commits_off_after_ready_without_touching_small_or_maps(rpc_device, offroad):
  p, rpc = rpc_device.params, rpc_device.rpc
  p.put_bool('NavigationModelDrivingEnabled', True, block=True)
  fallback = p.get('ModelManager_ActiveBundle')
  rpc.saveParams({'ModelManager_ActiveBundleChestnut': wire({})})
  assert p.get_bool('NavigationModelDrivingEnabled')
  with patch.object(selection, '_default_ready') as ready:
    selection.process_pending_selection(p)
  ready.assert_called_once_with('chestnut')
  assert not p.get_bool('NavigationModelDrivingEnabled')
  assert p.get('ModelManager_ActiveBundleChestnut') is None
  assert p.get('ModelManager_ActiveBundle') == fallback and p.get_bool('NavigationEnabled')
  assert selection.selection_status(p)['selected_ref'] == selection.DEFAULT_REF


@pytest.mark.parametrize('update', [{'ModelManager_DownloadRef': ''}, {'ModelManager_ActiveBundleChestnut': {}}])
def test_actual_rpc_stale_client_cannot_cancel_or_replace_new_request(rpc_device, offroad, update):
  p, rpc = rpc_device.params, rpc_device.rpc
  revision = rpc_value(rpc.getParams([bridge.REVISION_KEY]), bridge.REVISION_KEY)
  rpc.saveParams({'ModelManager_DownloadRef': wire(bridge.NAV_REF), bridge.REVISION_KEY: wire(revision)})
  pending = p.get(selection.REQUEST_KEY)
  with pytest.raises(NavigationModelDisabled, match='selection changed'):
    rpc.saveParams({**{key: wire(value) for key, value in update.items()}, bridge.REVISION_KEY: wire(revision)})
  assert p.get(selection.REQUEST_KEY) == pending
  assert not p.get_bool('NavigationModelDrivingEnabled')


def test_actual_rpc_onroad_refusal_does_not_queue_or_change_model(rpc_device):
  p, rpc = rpc_device.params, rpc_device.rpc
  before = {key: p.get(key) for key in (*selection.SELECTION_KEYS, selection.REQUEST_KEY, selection.STATE_KEY)}
  with patch.object(selection, 'require_offroad', side_effect=NavigationModelDisabled('deviceState.started is true')):
    with pytest.raises(NavigationModelDisabled, match='deviceState.started'):
      rpc.saveParams({'ModelManager_DownloadRef': wire(bridge.NAV_REF)})
  assert before == {key: p.get(key) for key in before}


@pytest.mark.parametrize('key,value', [
  ('ModelManager_ActiveBundleUSBGPU', {'ref': 'stale-legacy-big'}),
  ('ModelManager_BundleSlotsMigrated', False),
  ('ModelRunnerTypeCache', 0),
])
def test_actual_rpc_cannot_write_legacy_or_derived_model_state(rpc_device, key, value):
  p, rpc = rpc_device.params, rpc_device.rpc
  before = {name: p.get(name) for name in (*selection.SELECTION_KEYS, key)}
  rpc.saveParams({key: wire(value)})
  assert before == {name: p.get(name) for name in before}
  with pytest.raises(NavigationModelDisabled, match='not writable settings'):
    bridge.apply_remote_model_settings(p, {key: value})
  assert before == {name: p.get(name) for name in before}


def test_native_camelcase_retired_hash_is_not_a_restorable_model(params):
  updates = {'ModelManager_ActiveBundleChestnut': {'ref': 'ordinary-big', 'models': [
    {'artifact': {'downloadUri': {'sha256': bridge.RETIRED_SHA256}}}]}}
  before = copy.deepcopy(params.values)
  with pytest.raises(NavigationModelDisabled, match='Original Bigv2 is retired'):
    bridge.apply_remote_model_settings(params, updates)
  assert params.values == before


def test_shared_restore_also_discards_legacy_and_derived_model_fields(params):
  before = copy.deepcopy(params.values)
  updates = {'ModelManager_ActiveBundleUSBGPU': {'ref': 'stale-big'}, 'ModelManager_BundleSlotsMigrated': False,
             'ModelRunnerTypeCache': 0, 'UnrelatedSetting': 'preserved'}
  assert selection.model_restore_settings(params, updates) == {'UnrelatedSetting': 'preserved'}
  assert params.values == before
