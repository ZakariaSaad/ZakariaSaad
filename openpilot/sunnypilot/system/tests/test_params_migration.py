"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import base64
from pathlib import Path
import tempfile
from unittest import mock

from openpilot.common.params import Params, ParamKeyFlag, UnknownKeyName
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.candidate_identity import RETIRED_ARTIFACT_SHA256
from openpilot.sunnypilot.system.params_migration import _migrate_model_bundle_slots, run_migration


class TestRetiredLongitudinalSettings(OpenpilotTestCase):
  LEGACY_KEYS = ('ExperimentalMode', 'ExperimentalModeConfirmed', 'DynamicExperimentalControl')

  def _seed_previous_install(self, params):
    # An update keeps the old files even though the new native registry no
    # longer permits creating these keys through Params.put().
    for key in self.LEGACY_KEYS:
      Path(params.get_param_path(key)).write_bytes(b'1')
    persistent = {
      'HondaEpsFirmwareEnabled': True,
      'HondaEpsFirmwareVerifiedSha256': 'existing-owner-confirmation',
      'HondaEpsFirmwareGeometry': 'mvl',
      'HondaEpsFirmwareResponse': 'route32',
      'NavigationEnabled': True,
      'NavigationModelDrivingEnabled': False,
      'NavigationModelCandidate': 'big-nav-v2-map-fp32',
      'NavigationModelManifest': '/data/navigation-models/existing/manifest.json',
      'ModelManager_ActiveBundle': {'ref': 'existing-small'},
      'ModelManager_ActiveBundleChestnut': {'ref': 'existing-ordinary-big'},
      'LongitudinalPersonality': 1,
      'InitialCruiseSpeedKph': 50,
    }
    for key, value in persistent.items():
      params.put(key, value, block=True)
    return persistent

  def test_manager_cleanup_removes_old_modes_without_changing_current_choices(self):
    params = Params()
    persistent = self._seed_previous_install(params)
    for _ in range(2):
      # manager_init performs this native unknown-key cleanup before migrations.
      # It owns the lock/fsync; no custom filesystem migration is necessary.
      params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
      run_migration(params)
      for key in self.LEGACY_KEYS:
        assert not Path(params.get_param_path(key)).exists()
        with self.assertRaises(UnknownKeyName):
          params.get_bool(key)
      for key, value in persistent.items():
        assert params.get(key) == value

  def test_old_backups_cannot_restore_retired_modes_or_disrupt_model_choices(self):
    from openpilot.sunnypilot.sunnylink.backups.manager import BackupManagerSP

    params = Params()
    persistent = self._seed_previous_install(params)
    params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
    manager = object.__new__(BackupManagerSP)
    manager.params = params
    old_backup = dict.fromkeys(self.LEGACY_KEYS, 'malformed obsolete value must never be decoded')
    old_backup['experimentalmode'] = base64.b64encode(b'1').decode()
    old_backup['IsMetric'] = base64.b64encode(b'1').decode()
    manager._apply_config(old_backup)
    assert params.get_bool('IsMetric')
    for key in self.LEGACY_KEYS:
      assert not Path(params.get_param_path(key)).exists()
    for key, value in persistent.items():
      assert params.get(key) == value

  def test_current_backups_exclude_stale_longitudinal_files(self):
    from openpilot.sunnypilot.sunnylink.backups.manager import BackupManagerSP

    params = Params()
    self._seed_previous_install(params)
    manager = object.__new__(BackupManagerSP)
    manager.params = params
    backup = manager._collect_config_data()
    assert set(self.LEGACY_KEYS).isdisjoint(backup)
    assert base64.b64decode(backup['HondaEpsFirmwareEnabled']) == b'True'
    assert 'HondaEpsFirmwareVerifiedSha256' not in backup
    assert 'NavigationModelManifest' not in backup


class TestModelBundleSlotMigration(OpenpilotTestCase):
  """Pre-split, a chestnut user's big-model selection lived in the single ActiveBundle.
  The migration seeds both slots; per-source validation later drops whichever does not
  match its own manifest."""

  def test_seeds_chestnut_slot_from_active_bundle(self):
    params = Params()
    bundle = {"ref": "big", "minimumSelectorVersion": 18}
    params.put("ModelManager_ActiveBundle", bundle, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") == bundle
    assert params.get("ModelManager_ActiveBundle") == bundle

  def test_noop_when_chestnut_slot_already_set(self):
    params = Params()
    params.put("ModelManager_ActiveBundle", {"ref": "small"}, block=True)
    params.put("ModelManager_ActiveBundleChestnut", {"ref": "big"}, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") == {"ref": "big"}

  def test_noop_when_no_selection(self):
    params = Params()
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") is None
    assert params.get_bool('ModelManager_BundleSlotsMigrated')

  def test_legacy_usb_slot_seeds_once_and_default_survives_later_boots(self):
    params = Params()
    legacy = {'ref': 'legacy-big', 'minimumSelectorVersion': 18}
    params.put('ModelManager_ActiveBundleUSBGPU', legacy, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get('ModelManager_ActiveBundleChestnut') == legacy
    assert params.get_bool('ModelManager_BundleSlotsMigrated')
    params.remove('ModelManager_ActiveBundleChestnut')  # Explicit default selection.
    _migrate_model_bundle_slots(params)
    assert params.get('ModelManager_ActiveBundleChestnut') is None
    assert params.get('ModelManager_ActiveBundleUSBGPU') == legacy

  def test_existing_modern_settings_are_never_seeded_from_stale_legacy_bytes(self):
    indicators = {
      'ModelManager_SelectionState': {'source': 'chestnut', 'ref': 'Default', 'state': 'selected'},
      'ModelManager_SelectionRequest': {'source': 'chestnut', 'ref': 'Default', 'phase': 'pending'},
      'ModelManager_ModelsCache_Chestnut': {'bundles': []},
      'ModelManager_ActiveJson': {'chestnut': 'existing catalog'},
      'NavigationModelCandidate': 'big-nav-v2-map-fp32',
    }
    for key, value in indicators.items():
      with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
        params = Params(directory)
        legacy = {'ref': 'legacy-big'}
        small = {'ref': 'preserved-small'}
        params.put(key, value, block=True)
        params.put('ModelManager_ActiveBundleUSBGPU', legacy, block=True)
        params.put('ModelManager_ActiveBundle', small, block=True)
        _migrate_model_bundle_slots(params)
        assert params.get('ModelManager_ActiveBundleChestnut') is None
        assert params.get('ModelManager_ActiveBundleUSBGPU') == legacy
        assert params.get('ModelManager_ActiveBundle') == small
        assert params.get(key) == value
        assert params.get_bool('ModelManager_BundleSlotsMigrated')

  def test_retired_native_or_catalog_identity_is_never_seeded(self):
    bundles = [
      {'ref': 'big-nav-v2'}, {'ref': 'navigation:big-nav-v2'},
      {'ref': 'alias', 'models': [{'artifact': {'downloadUri': {'sha256': RETIRED_ARTIFACT_SHA256}}}]},
      {'ref': 'alias', 'models': [{'artifact': {'download_uri': {'sha256': RETIRED_ARTIFACT_SHA256}}}]},
    ]
    for bundle in bundles:
      with self.subTest(bundle=bundle), tempfile.TemporaryDirectory() as directory:
        params = Params(directory)
        params.put('ModelManager_ActiveBundleUSBGPU', bundle, block=True)
        _migrate_model_bundle_slots(params)
        assert params.get('ModelManager_ActiveBundleChestnut') is None
        assert params.get('ModelManager_ActiveBundleUSBGPU') == bundle
        assert params.get_bool('ModelManager_BundleSlotsMigrated')

  def test_marker_write_interruption_is_repeatable_without_overwriting_slot(self):
    params = Params()
    legacy = {'ref': 'legacy-big'}
    params.put('ModelManager_ActiveBundleUSBGPU', legacy, block=True)
    with mock.patch.object(params, 'put_bool', side_effect=OSError('interrupted marker write')):
      _migrate_model_bundle_slots(params)
    assert params.get('ModelManager_ActiveBundleChestnut') == legacy
    assert not params.get_bool('ModelManager_BundleSlotsMigrated')
    _migrate_model_bundle_slots(params)
    assert params.get('ModelManager_ActiveBundleChestnut') == legacy
    assert params.get_bool('ModelManager_BundleSlotsMigrated')
