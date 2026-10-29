"""UI identity follows canonical selection and actual published runtime roles."""
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.corrected_bigv2 import MODEL_ID, DISPLAY_NAME
from openpilot.sunnypilot.models.navigation_bundle import navigation_bundle


class ModelMessages(dict):
  def __init__(self, *, big=True, alive=True):
    super().__init__(modelV2=SimpleNamespace(big=big))
    self.alive = {'modelV2': alive}
    # Calibration can make a published model invalid without stopping inference.
    self.valid = {'modelV2': False}


class TestNavigationModelIdentity(OpenpilotTestCase):
  def test_only_published_candidate_and_small_roles_are_reported_as_running(self):
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.sunnypilot import model_info
    params = Params()
    params.put('ModelManager_ActiveBundleChestnut', navigation_bundle(), block=True)
    selected = params.get('ModelManager_ActiveBundleChestnut')
    messages = ModelMessages()
    with patch.object(ui_state, 'params', params), patch.object(ui_state, 'is_offroad', return_value=False), patch.object(ui_state, 'sm', messages):
      for legacy_flag in (False, True):
        params.put_bool('NavigationModelDrivingEnabled', legacy_flag, block=True)
        params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': 'candidate', 'active_source': 'chestnut'}, block=True)
        self.assertEqual(model_info.carrying_model(), ('chestnut', MODEL_ID, DISPLAY_NAME))
        self.assertEqual(model_info.model_info()[1], DISPLAY_NAME)
      for role in ('loading', 'ready', 'small_loading', 'failed'):
        params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': role, 'active_source': None}, block=True)
        self.assertEqual(model_info.carrying_model(), (None, None, None))
      messages['modelV2'].big = False
      params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': 'ordinary_small', 'active_source': 'qcom'}, block=True)
      self.assertEqual(model_info.carrying_model()[0], 'qcom')
      self.assertEqual(model_info.model_info()[1], model_info.default_model_name('qcom'))
      self.assertEqual(params.get('ModelManager_ActiveBundleChestnut'), selected)
      params.clear_all(ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)
      self.assertIsNone(params.get('NavigationModelRuntimeState'))
      self.assertEqual(model_info.carrying_model(), (None, None, None))
      self.assertEqual(params.get('ModelManager_ActiveBundleChestnut'), selected)

  def test_stale_navigation_runtime_does_not_override_an_ordinary_selection(self):
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.sunnypilot import model_info
    params = Params()
    params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    params.put('NavigationModelCandidate', MODEL_ID, block=True)
    params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': 'candidate', 'active_source': 'chestnut'}, block=True)
    with patch.object(ui_state, 'params', params), patch.object(ui_state, 'is_offroad', return_value=False):
      self.assertIsNone(model_info._navigation_model_info())

  def test_stale_published_role_needs_live_matching_predictions(self):
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.sunnypilot import model_info
    params = Params()
    params.put('ModelManager_ActiveBundleChestnut', navigation_bundle(), block=True)
    messages = ModelMessages()
    with patch.object(ui_state, 'params', params), patch.object(ui_state, 'is_offroad', return_value=False), patch.object(ui_state, 'sm', messages):
      for role, source, big in (('candidate', 'chestnut', True), ('ordinary_small', 'qcom', False)):
        with self.subTest(role=role):
          params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': role, 'active_source': source}, block=True)
          messages['modelV2'].big = big
          messages.alive['modelV2'] = False
          self.assertEqual(model_info.carrying_model(), (None, None, None))
          self.assertEqual(model_info.model_info()[1], 'No model predictions')
          messages.alive['modelV2'] = True
          messages['modelV2'].big = not big
          self.assertEqual(model_info.carrying_model(), (None, None, None))
          messages['modelV2'].big = big
          self.assertEqual(model_info.carrying_model()[0], source)
