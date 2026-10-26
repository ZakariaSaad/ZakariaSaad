"""Actual UI identity helpers follow runtime role without overwriting user slots."""
from unittest.mock import patch

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.corrected_bigv2 import MODEL_ID, DISPLAY_NAME


class TestNavigationModelIdentity(OpenpilotTestCase):
  def test_candidate_and_small_runtime_roles_preserve_ordinary_selection_and_clear_offroad(self):
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.sunnypilot import model_info
    params = Params()
    params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    params.put('NavigationModelCandidate', MODEL_ID, block=True)
    ordinary = params.get('ModelManager_ActiveBundleChestnut')
    with patch.object(ui_state, 'params', params):
      params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': 'candidate', 'active_source': 'chestnut'}, block=True)
      self.assertEqual(model_info.carrying_model(), ('chestnut', MODEL_ID, DISPLAY_NAME))
      self.assertEqual(model_info.model_info()[1], DISPLAY_NAME)
      params.put('NavigationModelRuntimeState', {'candidate_id': MODEL_ID, 'role': 'ordinary_small', 'active_source': 'qcom'}, block=True)
      self.assertEqual(model_info.carrying_model()[0], 'qcom')
      self.assertEqual(model_info.model_info()[1], model_info.default_model_name('qcom'))
      self.assertEqual(params.get('ModelManager_ActiveBundleChestnut'), ordinary)
      params.put_bool('NavigationModelDrivingEnabled', False, block=True)
      self.assertIsNone(model_info._navigation_model_info())
      params.clear_all(ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)
      self.assertIsNone(params.get('NavigationModelRuntimeState'))
      self.assertEqual(params.get('ModelManager_ActiveBundleChestnut'), ordinary)
