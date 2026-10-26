"""Real persistent Params and shared model transactions; no GPU or live messages."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.navigation import preparation
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
  params.put_bool('NavigationModelDrivingEnabled', False, block=True)
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
  monkeypatch.setattr(selection, '_default_ready', defaults)
  monkeypatch.setattr(selection, '_ordinary_bundle', lambda _params, _source, ref: {'ref': ref, 'runner': 'tinygrad'})
  return SimpleNamespace(params=params, root=tmp_path, permit=permit, guard=guard, inspect=inspect,
                         admit=admit, validate=validate, prepare=prepare, defaults=defaults)


def request(device, ref=selection.NAV_REF, source='chestnut', **kwargs):
  return selection.request_model_selection(device.params, source=source, ref=ref, offroad=True, **kwargs)


def enable_existing(device):
  request(device)
  selection.process_pending_selection(device.params)
  assert device.params.get_bool('NavigationModelDrivingEnabled')


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
  assert params.get('NavigationModelCandidate') == selection.MODEL_ID
  assert params.get_bool('NavigationModelDrivingEnabled')
  for key in ('ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut', 'NavigationEnabled', 'NavDestination'):
    assert params.get(key) == before[key]
  assert params.get(selection.REQUEST_KEY) is None
  assert selection.selection_status(params)['selected_ref'] == selection.NAV_REF
  device.validate.assert_called_once_with(device.permit, params, allow_unselected_manifest=True)


def test_existing_corrected_registration_is_reused_without_repreparation(device):
  enable_existing(device)
  device.prepare.assert_not_called()


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
  assert params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('ModelManager_ActiveBundleChestnut')['ref'] == 'big-existing'
  assert selection.process_pending_selection(params) is False
  assert selection.complete_ordinary_selection(params, 'chestnut', ordinary, ordinary['ref'])
  assert not params.get_bool('NavigationModelDrivingEnabled')
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
  assert not device.params.get_bool('NavigationModelDrivingEnabled')
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


@pytest.mark.parametrize('choice', ['off', 'ordinary'])
def test_explicit_off_or_ordinary_selection_survives_native_reopen_and_manager_clears(device, choice):
  enable_existing(device)
  if choice == 'off':
    selection.request_navigation_disabled(device.params)
  else:
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
  assert not params.get_bool('NavigationModelDrivingEnabled')
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
    for key in ('NavigationModelCandidate', 'NavigationModelDrivingEnabled', 'NavigationEnabled', 'ModelManager_ActiveBundleChestnut'):
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
  device.prepare.side_effect = lambda *args, **kwargs: selection.request_navigation_disabled(device.params)
  request(device)
  selection.process_pending_selection(device.params)
  assert not device.params.get_bool('NavigationModelDrivingEnabled')
  assert device.params.get(selection.REQUEST_KEY) is None
  assert device.params.get('ModelManager_ActiveBundleChestnut')['ref'] == 'big-existing'


@pytest.mark.parametrize('enabled', [False, True])
def test_retired_request_migrates_only_after_corrected_admission(device, enabled):
  params = device.params
  params.put('NavigationModelCandidate', selection.RETIRED_MODEL_ID, block=True)
  params.put_bool('NavigationModelDrivingEnabled', enabled, block=True)
  def admit(*args, **kwargs):
    assert not params.get_bool('NavigationModelDrivingEnabled')
    assert kwargs['candidate_id'] == selection.MODEL_ID
    return device.permit
  device.admit.side_effect = admit
  selection.process_pending_selection(params)
  assert params.get('NavigationModelCandidate') == selection.MODEL_ID
  assert params.get_bool('NavigationModelDrivingEnabled') is enabled
  device.admit.reset_mock()
  selection.process_pending_selection(params)
  device.admit.assert_not_called()


@pytest.mark.parametrize('interrupted', [False, True])
def test_failed_or_interrupted_legacy_migration_keeps_navigation_off(device, interrupted):
  params = device.params
  params.put('NavigationModelCandidate', selection.RETIRED_MODEL_ID, block=True)
  params.put_bool('NavigationModelDrivingEnabled', True, block=True)
  if interrupted:
    params.put(selection.REQUEST_KEY, {'schema': 1, 'id': 'interrupted', 'source': 'chestnut', 'ref': selection.NAV_REF,
                                      'origin': 'retired_bigv2_migration', 'phase': 'pending', 'enable': True}, block=True)
  device.admit.side_effect = NavigationModelDisabled('Corrected artifact missing')
  selection.process_pending_selection(params)
  assert not params.get_bool('NavigationModelDrivingEnabled')
  assert selection.selection_status(params)['state'] == 'failed'
  assert 'Corrected artifact missing' in selection.selection_status(params)['message']
  assert params.get('ModelManager_ActiveBundleChestnut')['ref'] == 'big-existing'


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
    assert reopened.get_bool('NavigationModelDrivingEnabled')
    assert reopened.get('NavigationModelCandidate') == selection.MODEL_ID
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
    assert not reopened.get_bool('NavigationModelDrivingEnabled')
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
  assert selection.selection_status(device.params)['state'] == 'active'
  device.params.put('NavigationModelRuntimeState', {'candidate_id': selection.MODEL_ID, 'role': 'ordinary_small'}, block=True)
  assert selection.selection_status(device.params)['state'] == 'failed'


def test_retired_choice_never_enters_queue(device):
  for ref in (selection.RETIRED_MODEL_ID, 'navigation:' + selection.RETIRED_MODEL_ID):
    with pytest.raises(NavigationModelDisabled, match='retired'):
      request(device, ref)
  assert device.params.get(selection.REQUEST_KEY) is None
