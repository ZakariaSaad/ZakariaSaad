"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import tempfile
from unittest import mock

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.candidate_identity import RETIRED_ARTIFACT_SHA256
from openpilot.sunnypilot.system.params_migration import _migrate_model_bundle_slots


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
