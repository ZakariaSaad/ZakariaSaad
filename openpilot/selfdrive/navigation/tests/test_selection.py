"""Real Params selection boundaries preserve ordinary models and display settings."""
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.activation import candidate_status
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled
from openpilot.selfdrive.navigation.selection import select_candidate, set_driving_enabled
from openpilot.sunnypilot.models.helpers import get_active_model_runner


class TestNavigationSelection(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.params = Params()
    self.params.put('ModelRunnerTypeCache', 0, block=True)
    self.params.put('ModelManager_ActiveBundle', {'untouched': 'small'}, block=True)
    self.params.put('ModelManager_ActiveBundleChestnut', {'untouched': 'big'}, block=True)
    self.params.put_bool('NavigationEnabled', True, block=True)
    self.before = {key: self.params.get(key) for key in ('ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut',
                                                       'ModelRunnerTypeCache', 'NavigationEnabled')}

  def assert_normal_settings_preserved(self):
    self.assertEqual(self.before, {key: self.params.get(key) for key in self.before})

  def test_default_and_unreviewed_candidate_never_activate(self):
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertFalse(candidate_status(self.params)['eligible'])
    select_candidate(self.params, 'big-nav-v2', offroad=True)
    with self.assertRaises(NavigationModelDisabled):
      set_driving_enabled(self.params, True, offroad=True)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(get_active_model_runner(self.params), 0)
    self.assert_normal_settings_preserved()

  def test_explicit_reviewed_opt_in_and_disable_preserve_slots(self):
    with patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value={'eligible': True}):
      with self.assertRaises(NavigationModelDisabled):
        set_driving_enabled(self.params, True, offroad=False)
      set_driving_enabled(self.params, True, offroad=True)
    with patch('openpilot.selfdrive.navigation.activation.candidate_status', return_value={'eligible': True}):
      self.assertNotEqual(get_active_model_runner(self.params), 0)
    set_driving_enabled(self.params, False, offroad=True)
    self.assertEqual(get_active_model_runner(self.params), 0)
    self.assert_normal_settings_preserved()

  def test_invalid_selection_and_preparation_cannot_change_normal_models(self):
    for offroad, identifier in ((False, 'big-nav-v2'), (True, 'unrecognized')):
      with self.subTest(offroad=offroad), self.assertRaises(NavigationModelDisabled):
        select_candidate(self.params, identifier, offroad=offroad)
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    # A tampered checkbox is insufficient to select the native runner.
    self.assertEqual(get_active_model_runner(self.params), 0)
    select_candidate(self.params, 'big-nav-v2', offroad=True)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assert_normal_settings_preserved()
