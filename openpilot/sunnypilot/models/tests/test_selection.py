"""Real persistent Params and shared model transactions; no GPU or live messages."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.navigation import preparation, download
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.sunnypilot.models import selection


class PowerLoss(BaseException):
  pass


def snapshot(params):
  return {key: copy.deepcopy(params.get(key)) for key in (*selection.SELECTION_KEYS, 'NavigationEnabled', 'NavDestination')}


@pytest.fixture
def device(tmp_path, monkeypatch):
  params = Params(str(tmp_path / 'params'))
  params.put_bool('IsOffroad', True, block=True)
  params.put_bool('IsEngaged', False, block=True)
  params.put_bool('NavigationEnabled', True, block=True)
  params.put('NavDestination', {'latitude': 1., 'longitude': 2.}, block=True)
  params.put_bool('ModelManager_NativeSelectionMigrated', True, block=True)
  params.put('ModelManager_ActiveBundle', {'ref': 'small-existing', 'runner': 'tinygrad'}, block=True)
  params.put('ModelManager_ActiveBundleChestnut', {'ref': 'big-existing', 'runner': 'tinygrad'}, block=True)
  params.put('ModelRunnerTypeCache', 1, block=True)
  manifest = tmp_path / 'corrected' / 'manifest.json'
  manifest.parent.mkdir()
  manifest.write_text('{}')
  permit = SimpleNamespace(model_id=selection.MODEL_ID, manifest_path=manifest)
  guard = Mock()
  inspect = Mock(return_value={'registered': True})
  admit = Mock(return_value=permit)
  validate = Mock()
  prepare = Mock(return_value={'registered': True, 'manifest_path': str(manifest)})
  defaults = Mock()
  monkeypatch.setattr(selection, 'require_offroad', guard)
  monkeypatch.setattr(selection, 'candidate_status', inspect)
  monkeypatch.setattr(selection, 'admit_candidate', admit)
  monkeypatch.setattr(selection, 'validate_admission', validate)
  monkeypatch.setattr(preparation, 'prepare_existing_model', prepare)
  def install(params, *, current, progress, registration):
    assert current()
    if not registration.get('registered'):
      prepare(params, is_offroad=True, model_id=selection.MODEL_ID, publish_selection=False)
  installer = Mock(side_effect=install)
  monkeypatch.setattr(download, 'prepare_downloaded_model', installer)
  monkeypatch.setattr(selection, '_default_ready', defaults)
  monkeypatch.setattr(selection, '_ordinary_bundle', lambda _params, _source, ref: {'ref': ref, 'runner': 'tinygrad'})
  return SimpleNamespace(params=params, root=tmp_path, permit=permit, guard=guard, inspect=inspect,
                         admit=admit, validate=validate, prepare=prepare, defaults=defaults, installer=installer)


def request(device, ref=selection.NAV_REF, source='chestnut', **kwargs):
  return selection.request_model_selection(device.params, source=source, ref=ref, offroad=True, **kwargs)


def enable_existing(device):
  request(device)
  selection.process_pending_selection(device.params)
  assert selection.is_navigation_selected(device.params)


def test_explicit_corrected_selection_prepares_before_publishing_and_requests_on(device):
  params = device.params
  before = snapshot(params)
  device.inspect.return_value = {'registered': False}
  order = []
  def prepare(*args, **kwargs):
    assert snapshot(params) == before
    assert kwargs == {'is_offroad': True, 'model_id': selection.MODEL_ID, 'publish_selection': False}
    order.append('prepare')
  def admit(*args, **kwargs):
    assert snapshot(params) == before
    order.append('admit')
    return device.permit
  device.prepare.side_effect = prepare
  device.admit.side_effect = admit
  pending = request(device)
  assert pending['state'] == 'pending' and pending['selected_ref'] == 'big-existing'
  assert snapshot(params) == before
  assert selection.process_pending_selection(params)
  assert order == ['prepare', 'admit']
  assert params.get('NavigationModelManifest') == str(device.permit.manifest_path)
  assert selection.is_navigation_selected(params)
  assert params.get('ModelManager_ActiveBundleChestnut') == selection.navigation_bundle()
  for key in ('ModelManager_ActiveBundle', 'NavigationEnabled', 'NavDestination'):
    assert params.get(key) == before[key]
  assert params.get(selection.REQUEST_KEY) is None
  assert selection.selection_status(params)['selected_ref'] == selection.NAV_REF
  device.validate.assert_called_once_with(device.permit, params, allow_unselected_manifest=True)


def test_existing_corrected_registration_is_reused_without_repreparation(device):
  enable_existing(device)
  device.prepare.assert_not_called()


def test_navigation_ref_stays_visible_to_existing_frontend_until_install_and_commit(device):
  request(device)
  assert device.params.get('ModelManager_DownloadRef') == selection.NAV_REF
  def installing(params, *, current, progress, registration):
    assert current() and params.get('ModelManager_DownloadRef') == selection.NAV_REF
    progress('downloading', 'Downloading corrected Bigv2: 40%')
    assert selection.selection_status(params)['message'].endswith('40%')
  device.installer.side_effect = installing
  selection.process_pending_selection(device.params)
  assert device.params.get('ModelManager_DownloadRef') is None
  assert selection.is_navigation_selected(device.params)


def test_explicit_off_cancels_download_ref_and_prevents_late_activation(device):
  request(device)
  def installing(params, **kwargs):
    selection.cancel_model_selection(params)
    assert not kwargs['current']()
    kwargs['progress']('downloading', 'Stale progress must not replace cancellation')
  device.installer.side_effect = installing
  selection.process_pending_selection(device.params)
  assert device.params.get('ModelManager_DownloadRef') is None
  assert not selection.is_navigation_selected(device.params)
  assert 'cancelled' in selection.selection_status(device.params)['message']


def test_retired_legacy_choice_never_automatically_selects_corrected(device):
  device.params.put_bool('ModelManager_NativeSelectionMigrated', False, block=True)
  device.params.put('NavigationModelCandidate', selection.RETIRED_MODEL_ID, block=True)
  device.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
  before = snapshot(device.params)
  selection.process_pending_selection(device.params)
  device.installer.assert_not_called()
  device.admit.assert_not_called()
  assert snapshot(device.params) == before
  assert not selection.is_navigation_selected(device.params)
  assert device.params.get('NavigationModelDrivingEnabled') is None



@pytest.mark.parametrize('failure_stage', ['prepare', 'admit', 'validate'])
def test_failed_corrected_preparation_or_admission_preserves_previous_configuration(device, failure_stage):
  before = snapshot(device.params)
  device.inspect.return_value = {'registered': failure_stage != 'prepare'}
  getattr(device, failure_stage).side_effect = NavigationModelDisabled('Synthetic artifact mismatch')
  request(device)
  selection.process_pending_selection(device.params)
  assert snapshot(device.params) == before
  state = selection.selection_status(device.params)
  assert state['state'] == 'failed' and 'Synthetic artifact mismatch' in state['message']
  assert not state['request_id']


def test_ordinary_big_switch_waits_for_download_then_withdraws_navigation_only(device):
  enable_existing(device)
  params = device.params
  small, destination = params.get('ModelManager_ActiveBundle'), params.get('NavDestination')
  ordinary = {'ref': 'ordinary-ready', 'runner': 'tinygrad'}
  request(device, ordinary['ref'])
  assert selection.is_navigation_selected(params)
  assert params.get('ModelManager_ActiveBundleChestnut')['ref'] == selection.NAV_REF
  assert selection.process_pending_selection(params) is False
  assert selection.complete_ordinary_selection(params, 'chestnut', ordinary, ordinary['ref'])
  assert not selection.is_navigation_selected(params)
  assert params.get('NavigationModelCandidate') is None
  assert params.get('ModelManager_ActiveBundleChestnut') == ordinary
  assert params.get('ModelManager_ActiveBundle') == small
  assert params.get_bool('NavigationEnabled') and params.get('NavDestination') == destination
  assert params.get('ModelManager_DownloadRef') is None


def test_default_mountain_dew_switch_clears_big_slot_only(device):
  enable_existing(device)
  small = device.params.get('ModelManager_ActiveBundle')
  request(device, selection.DEFAULT_REF)
  selection.process_pending_selection(device.params)
  device.defaults.assert_called_once_with('chestnut')
  assert device.params.get('ModelManager_ActiveBundleChestnut') is None
  assert not selection.is_navigation_selected(device.params)
  assert device.params.get('ModelManager_ActiveBundle') == small
  assert device.params.get_bool('NavigationEnabled')


@pytest.mark.parametrize('target,previous_runner,expected_runner', [('Default', 1, 2), ('ordinary-ready', 2, 1)])
def test_completed_switch_resolves_new_runner_before_another_manager_tick(device, monkeypatch, target, previous_runner, expected_runner):
  from openpilot.sunnypilot.models import helpers
  params = device.params
  monkeypatch.setattr(helpers, 'chestnut_present', lambda: True)
  params.put('ModelRunnerTypeCache', previous_runner, block=True)
  request(device, target)
  if target == 'Default':
    selection.process_pending_selection(params)
  else:
    bundle = {'ref': target, 'minimumSelectorVersion': 21, 'runner': 'tinygrad'}
    assert selection.complete_ordinary_selection(params, 'chestnut', bundle, target)
  assert params.get('ModelRunnerTypeCache') is None
  # Exercise the actual startup runner resolver immediately, without a manager
  # validation tick, model execution, or alteration of the ordinary small slot.
  assert helpers.get_active_model_runner(params) == expected_runner


def test_missing_default_keeps_previous_valid_model_and_explains_failure(device):
  enable_existing(device)
  before = snapshot(device.params)
  device.defaults.side_effect = NavigationModelDisabled('The installed default model is incomplete')
  request(device, selection.DEFAULT_REF)
  selection.process_pending_selection(device.params)
  assert snapshot(device.params) == before
  assert 'incomplete' in selection.selection_status(device.params)['message']


@pytest.mark.parametrize('state', ['failed', 'cancelled'])
def test_download_failure_or_cancellation_keeps_current_model(device, state):
  enable_existing(device)
  before = snapshot(device.params)
  request(device, 'ordinary-download')
  selection.report_download_state(device.params, 'ordinary-download', state, 'Synthetic transfer failure')
  assert snapshot(device.params) == before
  assert device.params.get(selection.REQUEST_KEY) is None
  assert device.params.get('ModelManager_DownloadRef') is None
  assert selection.selection_status(device.params)['state'] == state


def test_ordinary_selection_survives_native_reopen_and_manager_clears(device):
  enable_existing(device)
  request(device, selection.DEFAULT_REF)
  selection.process_pending_selection(device.params)
  before = snapshot(device.params)
  # These are intentionally transient in the existing branch; selection
  # persistence must not change their normal lifecycle semantics.
  before['ModelRunnerTypeCache'] = None
  before['NavDestination'] = None
  params = Params(str(device.root / 'params'))
  for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
    params.clear_all(flag)
  params.put_bool('IsOffroad', True, block=True)
  params.put_bool('IsEngaged', False, block=True)
  device.admit.reset_mock()
  for _ in range(3):
    assert selection.process_pending_selection(params) is False
  assert snapshot(params) == before
  assert not selection.is_navigation_selected(params)
  device.admit.assert_not_called()


def test_onroad_request_and_onroad_download_completion_do_not_change_model(device):
  before = snapshot(device.params)
  device.guard.side_effect = NavigationModelDisabled('deviceState.started is true')
  with pytest.raises(NavigationModelDisabled, match='deviceState.started'):
    request(device)
  assert snapshot(device.params) == before
  assert device.params.get(selection.REQUEST_KEY) is None
  device.guard.side_effect = None
  request(device, 'ordinary-ready')
  device.params.put_bool('IsOffroad', False, block=True)
  assert not selection.complete_ordinary_selection(device.params, 'chestnut', {'ref': 'ordinary-ready'}, 'ordinary-ready')
  assert snapshot(device.params) == before
  assert selection.selection_status(device.params)['state'] == 'pending'


def test_stale_revision_and_cancelled_completion_cannot_commit(device):
  first = request(device, 'ordinary-ready')
  with pytest.raises(NavigationModelDisabled, match='Stale model selection'):
    request(device, expected_revision='stale-revision')
  assert selection.selection_status(device.params)['request_id'] == first['request_id']
  before = snapshot(device.params)
  selection.cancel_model_selection(device.params)
  assert not selection.complete_ordinary_selection(device.params, 'chestnut', {'ref': 'ordinary-ready'}, 'ordinary-ready')
  assert snapshot(device.params) == before


def test_cancel_refuses_to_report_success_when_request_removal_fails(device, monkeypatch):
  request(device)
  remove = device.params.remove
  monkeypatch.setattr(device.params, 'remove', lambda key: None if key == selection.REQUEST_KEY else remove(key))
  with pytest.raises(NavigationModelDisabled, match='Could not persist'):
    selection.cancel_model_selection(device.params)
  assert selection.selection_status(device.params)['state'] != 'cancelled'


def test_new_request_cannot_overwrite_an_interrupted_commit_journal(device):
  request(device)
  journal = {**device.params.get(selection.REQUEST_KEY), 'phase': 'committing', 'previous': snapshot(device.params)}
  device.params.put(selection.REQUEST_KEY, journal, block=True)
  with pytest.raises(NavigationModelDisabled, match='committ|recover|interrupted'):
    request(device, 'ordinary-new')
  assert device.params.get(selection.REQUEST_KEY) == journal


@pytest.mark.parametrize('compatible', [False, True])
def test_ordinary_small_switch_requires_navigation_fallback_contract_and_preserves_request(device, monkeypatch, compatible):
  from openpilot.selfdrive.navigation import software_compatibility
  enable_existing(device)
  before = snapshot(device.params)
  bundle = {'ref': 'small-new', 'runner': 'tinygrad'}
  verify = Mock(side_effect=None if compatible else NavigationModelDisabled('Configured fallback is not a supported native QCOM model'))
  monkeypatch.setattr(software_compatibility, 'validate_ordinary_fallback_bundle', verify)
  request(device, bundle['ref'], 'qcom')
  if compatible:
    assert selection.complete_ordinary_selection(device.params, 'qcom', bundle, bundle['ref'])
    assert device.params.get('ModelManager_ActiveBundle') == bundle
    for key in ('NavigationEnabled', 'ModelManager_ActiveBundleChestnut'):
      assert device.params.get(key) == before[key]
  else:
    with pytest.raises(NavigationModelDisabled, match='supported native QCOM'):
      selection.complete_ordinary_selection(device.params, 'qcom', bundle, bundle['ref'])
    assert snapshot(device.params) == before
  verify.assert_called_once()
  assert verify.call_args.args[0] == bundle


def test_stock_small_default_cannot_remove_required_navigation_fallback(device):
  enable_existing(device)
  before = snapshot(device.params)
  request(device, selection.DEFAULT_REF, 'qcom')
  selection.process_pending_selection(device.params)
  assert snapshot(device.params) == before
  assert selection.selection_status(device.params)['state'] == 'failed'
  assert 'small-model fallback' in selection.selection_status(device.params)['message']


def test_off_during_corrected_preparation_cancels_without_late_enable(device):
  device.inspect.return_value = {'registered': False}
  device.prepare.side_effect = lambda *args, **kwargs: selection.cancel_model_selection(device.params)
  request(device)
  selection.process_pending_selection(device.params)
  assert not selection.is_navigation_selected(device.params)
  assert device.params.get(selection.REQUEST_KEY) is None
  assert device.params.get('ModelManager_ActiveBundleChestnut')['ref'] == 'big-existing'


@pytest.mark.parametrize('flag,completed,expected', [(True, True, True), (False, True, False), (True, False, False)])
def test_legacy_migration_requires_completed_explicit_corrected_selection(device, flag, completed, expected):
  params = device.params
  params.put_bool('ModelManager_NativeSelectionMigrated', False, block=True)
  params.put('NavigationModelCandidate', selection.MODEL_ID, block=True)
  params.put_bool('NavigationModelDrivingEnabled', flag, block=True)
  params.put('NavigationModelManifest', str(device.permit.manifest_path), block=True)
  params.put(selection.STATE_KEY, {'id': 'old-completed', 'source': 'chestnut', 'ref': selection.NAV_REF,
                                   'state': 'selected' if completed else 'failed'}, block=True)
  selection.migrate_native_selection(params)
  assert selection.is_navigation_selected(params) is expected
  assert params.get('NavigationModelDrivingEnabled') is None
  assert params.get('NavigationModelCandidate') is None
  assert params.get_bool('ModelManager_NativeSelectionMigrated')
  device.installer.assert_not_called()
  device.admit.assert_not_called()
  # A stale legacy write after migration cannot reactivate Bigv2.
  params.put('ModelManager_ActiveBundleChestnut', {'ref': 'flip-flop'}, block=True)
  params.put_bool('NavigationModelDrivingEnabled', True, block=True)
  params.put('NavigationModelCandidate', selection.MODEL_ID, block=True)
  selection.migrate_native_selection(params)
  assert params.get('ModelManager_ActiveBundleChestnut')['ref'] == 'flip-flop'


@pytest.mark.parametrize('pending', ['retired_bigv2_migration', 'device_ui'])
def test_legacy_pending_requests_do_not_resurrect_original_or_override_newer_choice(device, pending):
  params = device.params
  params.put_bool('ModelManager_NativeSelectionMigrated', False, block=True)
  params.put('NavigationModelCandidate', selection.RETIRED_MODEL_ID, block=True)
  params.put_bool('NavigationModelDrivingEnabled', True, block=True)
  ref = selection.NAV_REF if pending == 'retired_bigv2_migration' else 'flip-flop'
  params.put(selection.REQUEST_KEY, {'schema': 1, 'id': 'interrupted', 'source': 'chestnut', 'ref': ref,
                                    'origin': pending, 'phase': 'pending', 'enable': True}, block=True)
  selection._recover_or_migrate(params)
  assert not selection.is_navigation_selected(params)
  assert params.get('ModelManager_ActiveBundleChestnut')['ref'] == 'big-existing'
  assert params.get(selection.REQUEST_KEY) is None if pending == 'retired_bigv2_migration' else params.get(selection.REQUEST_KEY)['ref'] == ref


@pytest.mark.parametrize('after_completion', [False, True])
def test_native_commit_journal_recovers_partial_or_completed_power_loss(device, monkeypatch, after_completion):
  params = device.params
  before = snapshot(params)
  original_put, original_remove = params.put, params.remove
  def put(key, value, **kwargs):
    original_put(key, value, **kwargs)
    if not after_completion and key == 'NavigationModelManifest':
      raise PowerLoss()
  def remove(key):
    if after_completion and key == selection.REQUEST_KEY:
      raise PowerLoss()
    original_remove(key)
  request(device)
  with monkeypatch.context() as patcher:
    patcher.setattr(params, 'put', put)
    patcher.setattr(params, 'remove', remove)
    with pytest.raises(PowerLoss):
      selection.process_pending_selection(params)
  assert params.get(selection.REQUEST_KEY)['phase'] == 'committing'
  reopened = Params(str(device.root / 'params'))
  selection.process_pending_selection(reopened)
  assert reopened.get(selection.REQUEST_KEY) is None
  if after_completion:
    assert selection.is_navigation_selected(reopened)
    assert reopened.get('ModelManager_ActiveBundleChestnut')['ref'] == selection.NAV_REF
  else:
    assert snapshot(reopened) == before
    assert selection.selection_status(reopened)['state'] == 'failed'


@pytest.mark.parametrize('after_completion', [False, True])
def test_ordinary_commit_recovery_drops_only_its_stale_download_request(device, monkeypatch, after_completion):
  enable_existing(device)
  params = device.params
  before = snapshot(params)
  bundle = {'ref': 'ordinary-ready', 'runner': 'tinygrad'}
  request(device, bundle['ref'])
  original_put, original_remove = params.put, params.remove
  def put(key, value, **kwargs):
    original_put(key, value, **kwargs)
    if not after_completion and key == 'ModelManager_ActiveBundleChestnut':
      raise PowerLoss()
  def remove(key):
    if after_completion and key == selection.REQUEST_KEY:
      raise PowerLoss()
    original_remove(key)
  with monkeypatch.context() as patcher:
    patcher.setattr(params, 'put', put)
    patcher.setattr(params, 'remove', remove)
    with pytest.raises(PowerLoss):
      selection.complete_ordinary_selection(params, 'chestnut', bundle, bundle['ref'])
  assert params.get('ModelManager_DownloadRef') == bundle['ref']
  reopened = Params(str(device.root / 'params'))
  selection.process_pending_selection(reopened)
  assert reopened.get(selection.REQUEST_KEY) is None
  assert reopened.get('ModelManager_DownloadRef') is None
  if after_completion:
    assert reopened.get('ModelManager_ActiveBundleChestnut') == bundle
    assert not selection.is_navigation_selected(reopened)
  else:
    assert snapshot(reopened) == before


def test_silently_failed_completion_state_write_rolls_back_model_switch(device, monkeypatch):
  before = snapshot(device.params)
  put = device.params.put
  def ignore_selected_state(key, value, **kwargs):
    if key != selection.STATE_KEY or value.get('state') != 'selected':
      put(key, value, **kwargs)
  monkeypatch.setattr(device.params, 'put', ignore_selected_state)
  request(device)
  selection.process_pending_selection(device.params)
  assert snapshot(device.params) == before
  assert selection.selection_status(device.params)['state'] == 'failed'
  assert 'Could not persist' in selection.selection_status(device.params)['message']


def test_selection_status_distinguishes_requested_loading_active_and_failed(device):
  enable_existing(device)
  assert selection.selection_status(device.params)['state'] == 'selected'
  device.params.put_bool('IsOffroad', False, block=True)
  device.params.put_bool('ChestnutLoading', True, block=True)
  assert selection.selection_status(device.params)['state'] == 'loading'
  device.params.put('NavigationModelRuntimeState', {'candidate_id': selection.MODEL_ID, 'role': 'candidate'}, block=True)
  assert selection.selection_status(device.params)['state'] == 'reported'
  assert selection.selection_status(device.params)['last_reported_ref'] == selection.NAV_REF
  device.params.put('NavigationModelRuntimeState', {'candidate_id': selection.MODEL_ID, 'role': 'ordinary_small'}, block=True)
  assert selection.selection_status(device.params)['state'] == 'reported'
  assert selection.selection_status(device.params)['last_reported_ref'] == 'small-existing'


def test_retired_choice_never_enters_queue(device):
  for ref in (selection.RETIRED_MODEL_ID, 'navigation:' + selection.RETIRED_MODEL_ID):
    with pytest.raises(NavigationModelDisabled, match='retired'):
      request(device, ref)
  assert device.params.get(selection.REQUEST_KEY) is None


@pytest.mark.parametrize('stage', ['slot', 'marker'])
def test_legacy_migration_power_loss_cannot_resurrect_stale_selection(device, monkeypatch, stage):
  params = device.params
  params.put_bool('ModelManager_NativeSelectionMigrated', False, block=True)
  params.put('NavigationModelCandidate', selection.MODEL_ID, block=True)
  params.put_bool('NavigationModelDrivingEnabled', True, block=True)
  params.put('NavigationModelManifest', str(device.permit.manifest_path), block=True)
  params.put(selection.STATE_KEY, {'id': 'old-selected', 'source': 'chestnut', 'ref': selection.NAV_REF, 'state': 'selected'}, block=True)
  original = params.put
  target = 'ModelManager_ActiveBundleChestnut' if stage == 'slot' else 'ModelManager_NativeSelectionMigrated'
  def interrupted(key, value, **kwargs):
    original(key, value, **kwargs)
    if key == target:
      raise PowerLoss()
  with monkeypatch.context() as patcher:
    patcher.setattr(params, 'put', interrupted)
    with pytest.raises(PowerLoss):
      selection.migrate_native_selection(params)
  reopened = Params(str(device.root / 'params'))
  selection.migrate_native_selection(reopened)
  assert selection.is_navigation_selected(reopened)
  assert reopened.get_bool('ModelManager_NativeSelectionMigrated')
  assert reopened.get('NavigationModelDrivingEnabled') is None


def test_native_selection_survives_first_restart_and_repeated_ignition_cycles(device):
  enable_existing(device)
  for _ in range(4):
    params = Params(str(device.root / 'params'))
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
      params.clear_all(flag)
    params.put_bool('IsOffroad', True, block=True)
    params.put_bool('IsEngaged', False, block=True)
    selection._recover_or_migrate(params)
    assert selection.is_navigation_selected(params)
    assert params.get('NavigationModelManifest') == str(device.permit.manifest_path)
    assert params.get('NavigationModelDrivingEnabled') is None


@pytest.mark.parametrize('ordinary', ['flip-flop', 'mountain-dew'])
def test_bigv2_ordinary_bigv2_roundtrip_uses_one_slot(device, ordinary):
  enable_existing(device)
  request(device, ordinary)
  assert selection.complete_ordinary_selection(device.params, 'chestnut', {'ref': ordinary}, ordinary)
  assert not selection.is_navigation_selected(device.params)
  assert device.params.get('ModelManager_ActiveBundleChestnut')['ref'] == ordinary
  enable_existing(device)
  assert device.params.get('ModelManager_ActiveBundleChestnut')['ref'] == selection.NAV_REF


def test_bare_background_download_does_not_replace_explicit_big_selection(device):
  enable_existing(device)
  device.params.put('ModelManager_DownloadRef', 'flip-flop', block=True)
  assert not selection.complete_ordinary_selection(device.params, 'chestnut', {'ref': 'flip-flop'}, 'flip-flop')
  assert selection.is_navigation_selected(device.params)
  assert device.params.get('ModelManager_DownloadRef') is None


@pytest.mark.parametrize('role', ['loading', 'ready', 'small_loading', 'failed'])
def test_runtime_status_never_claims_unpublished_fallback_is_running(device, role):
  enable_existing(device)
  device.params.put_bool('IsOffroad', False, block=True)
  device.params.put('NavigationModelRuntimeState', {'candidate_id': selection.MODEL_ID, 'role': role, 'reason': 'test'}, block=True)
  state = selection.selection_status(device.params)
  assert state['last_reported_ref'] is None
  assert 'publishing predictions' not in state['message']
  assert state['state'] == ('failed' if role == 'failed' else 'loading')


def test_native_bundle_resolves_shared_runner_and_survives_missing_files_or_upstream_catalog(device, monkeypatch):
  from openpilot.cereal import custom
  from openpilot.sunnypilot.models import helpers
  enable_existing(device)
  monkeypatch.setattr(helpers, 'chestnut_present', lambda: True)
  monkeypatch.setattr(helpers, '_bundle_is_valid_locally', lambda bundle: False)
  helpers.validate_active_bundles(device.params, {'chestnut': [custom.ModelManagerSP.ModelBundle(ref='flip-flop')]})
  assert selection.is_navigation_selected(device.params)
  bundle = helpers.get_selected_bundle(device.params, 'chestnut')
  assert bundle.ref == selection.NAV_REF
  assert bundle.generation == 0
  assert bundle.models[0].type == custom.ModelManagerSP.Model.Type.navigation
  assert helpers.get_active_model_runner(device.params) == int(custom.ModelManagerSP.Runner.tinygrad)


def test_backup_native_slot_or_legacy_flags_do_not_silently_reselect(device):
  before = snapshot(device.params)
  remaining = selection.model_restore_settings(device.params, {
    'ModelManager_ActiveBundleChestnut': selection.navigation_bundle(),
    'NavigationModelDrivingEnabled': True, 'NavigationModelCandidate': selection.MODEL_ID,
    'ModelManager_NativeSelectionMigrated': False, 'NavigationEnabled': True})
  assert remaining == {'NavigationEnabled': True}
  assert snapshot(device.params) == before
  assert device.params.get(selection.REQUEST_KEY) is None
