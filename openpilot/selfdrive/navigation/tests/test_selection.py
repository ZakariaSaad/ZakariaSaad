"""Real Params selection boundaries preserve ordinary models and display settings."""
from unittest.mock import patch
from types import SimpleNamespace

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.activation import candidate_status
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled
from openpilot.selfdrive.navigation.selection import select_candidate, set_driving_enabled
from openpilot.sunnypilot.models.helpers import get_active_model_runner


class TestNavigationSelection(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(patch('openpilot.selfdrive.navigation.selection.require_offroad'))
    self.params = Params()
    self.params.put('ModelRunnerTypeCache', 0, block=True)
    self.params.put('ModelManager_ActiveBundle', {'untouched': 'small'}, block=True)
    self.params.put('ModelManager_ActiveBundleChestnut', {'untouched': 'big'}, block=True)
    self.params.put_bool('NavigationEnabled', True, block=True)
    self.params.put_bool('IsOffroad', True, block=True)
    self.params.put_bool('IsEngaged', False, block=True)
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
    with patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value={'eligible': True}), \
         patch('openpilot.selfdrive.navigation.selection.require_offroad') as offroad, \
         patch('openpilot.selfdrive.navigation.selection.admit_candidate', return_value=SimpleNamespace(model_id='big-nav-v2')) as admit, \
         patch('openpilot.selfdrive.navigation.selection.validate_admission') as validate:
      with self.assertRaises(NavigationModelDisabled):
        set_driving_enabled(self.params, True, offroad=False)
      self.assertFalse(admit.called)
      set_driving_enabled(self.params, True, offroad=True, candidate_id='big-nav-v2')
      self.assertEqual(offroad.call_count, 2)
      admit.assert_called_once_with(self.params, candidate_id='big-nav-v2')
      validate.assert_called_once()
    self.assertEqual(self.params.get('NavigationModelCandidate'), 'big-nav-v2')
    with patch('openpilot.selfdrive.navigation.activation.candidate_status', return_value={'eligible': True}):
      self.assertNotEqual(get_active_model_runner(self.params), 0)
    # Withdrawal is always permitted, even if the device transitions on road.
    set_driving_enabled(self.params, False, offroad=False)
    self.assertEqual(get_active_model_runner(self.params), 0)
    self.assert_normal_settings_preserved()

  def test_failed_enable_preserves_selection_and_flag_at_every_admission_stage(self):
    self.params.put('NavigationModelCandidate', 'previous-selection', block=True)
    for stage in ('first_offroad', 'file_admission', 'second_offroad', 'admission_changed'):
      with self.subTest(stage=stage), \
           patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value={'eligible': True}), \
           patch('openpilot.selfdrive.navigation.selection.require_offroad') as guard, \
           patch('openpilot.selfdrive.navigation.selection.admit_candidate', return_value=SimpleNamespace(model_id='big-nav-v2')) as admit, \
           patch('openpilot.selfdrive.navigation.selection.validate_admission') as validate:
        error = NavigationModelDisabled(stage)
        if stage == 'first_offroad':
          guard.side_effect = error
        elif stage == 'file_admission':
          admit.side_effect = error
        elif stage == 'second_offroad':
          guard.side_effect = [None, error]
        else:
          validate.side_effect = error
        with self.assertRaisesRegex(NavigationModelDisabled, stage):
          set_driving_enabled(self.params, True, offroad=True, candidate_id='big-nav-v2')
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assertEqual(self.params.get('NavigationModelCandidate'), 'previous-selection')
        self.assert_normal_settings_preserved()

  def test_ssh_disable_during_ui_admission_cancels_the_pending_enable(self):
    self.params.put('NavigationModelCandidate', 'previous-selection', block=True)
    def disable_during_read(*_args, **_kwargs):
      # Even writing OFF over an existing OFF must cancel the earlier request.
      set_driving_enabled(self.params, False, offroad=True)
      return SimpleNamespace(model_id='big-nav-v2')
    with patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value={'eligible': True}), \
         patch('openpilot.selfdrive.navigation.selection.require_offroad'), \
         patch('openpilot.selfdrive.navigation.selection.admit_candidate', side_effect=disable_during_read), \
         patch('openpilot.selfdrive.navigation.selection.validate_admission'):
      with self.assertRaisesRegex(NavigationModelDisabled, 'activation cancelled'):
        set_driving_enabled(self.params, True, offroad=True, candidate_id='big-nav-v2')
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(self.params.get('NavigationModelCandidate'), 'previous-selection')
    self.assert_normal_settings_preserved()

  def test_silent_native_write_failure_does_not_report_success_or_leave_selection_changed(self):
    self.params.put('NavigationModelCandidate', 'previous-selection', block=True)
    put_bool = self.params.put_bool
    def fail_enable(key, value, **kwargs):
      if key != 'NavigationModelDrivingEnabled' or not value:
        put_bool(key, value, **kwargs)
    with patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value={'eligible': True}), \
         patch('openpilot.selfdrive.navigation.selection.require_offroad'), \
         patch('openpilot.selfdrive.navigation.selection.admit_candidate', return_value=SimpleNamespace(model_id='big-nav-v2')), \
         patch('openpilot.selfdrive.navigation.selection.validate_admission'), \
         patch.object(self.params, 'put_bool', side_effect=fail_enable):
      with self.assertRaisesRegex(NavigationModelDisabled, 'Could not persist'):
        set_driving_enabled(self.params, True, offroad=True, candidate_id='big-nav-v2')
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(self.params.get('NavigationModelCandidate'), 'previous-selection')
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
