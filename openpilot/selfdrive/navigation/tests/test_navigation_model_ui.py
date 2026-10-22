"""Real rendered selection widgets with isolated Params; no model/device inference.

Only the positive eligibility case uses a synthetic reviewed status. All default,
unreviewed, unknown-candidate and on-road gates use the actual admission helpers.
"""
import os
import threading
import unittest
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.selection import selection_description
from openpilot.sunnypilot.models.helpers import get_active_model_runner


@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires real raylib rendering under Xvfb')
class TestNavigationModelUI(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    gui_app.init_window('Navigation model selection tests')

  @classmethod
  def tearDownClass(cls):
    from openpilot.system.ui.lib.application import gui_app
    gui_app.close()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.system.ui.lib.application import gui_app
    self.params = Params()
    self.params.put('ModelManager_ActiveBundle', {'preserve': 'ordinary-small'}, block=True)
    self.params.put('ModelManager_ActiveBundleChestnut', {'preserve': 'ordinary-big'}, block=True)
    self.params.put('ModelRunnerTypeCache', 0, block=True)
    self.params.put_bool('NavigationEnabled', True, block=True)
    self.params.put_bool('IsOffroad', True, block=True)
    self.params.put_bool('IsEngaged', False, block=True)
    self.preserved = {key: self.params.get(key) for key in ('ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut',
                                                          'ModelRunnerTypeCache', 'NavigationEnabled')}
    self.enterContext(patch.object(ui_state, 'params', self.params))
    self.offroad = self.enterContext(patch.object(ui_state, 'is_offroad', return_value=True))
    self.enterContext(patch.object(gui_app, '_nav_stack', []))

  def assert_ordinary_settings_preserved(self):
    self.assertEqual(self.preserved, {key: self.params.get(key) for key in self.preserved})

  def draw(self, widget, width=536, height=240):
    import pyray as rl
    rl.begin_drawing()
    try:
      widget.render(rl.Rectangle(0, 0, width, height))
    finally:
      rl.end_drawing()

  def open_mici(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets.scroller import NavScroller
    layout = ModelsLayoutMici()
    gui_app.push_widget(layout)
    self.draw(layout)
    # Execute the actual parent button callback, rather than constructing an
    # isolated substitute toggle without the installed Models layout.
    layout.navigation_models_btn._handle_mouse_release(rl.Vector2(0, 0))
    submenu = gui_app.get_active_widget()
    self.assertIsInstance(submenu, NavScroller)
    self.draw(submenu)
    details, toggle, *candidates = submenu._scroller.items
    self.assertTrue(candidates)
    return layout, details, toggle, candidates

  def test_mici_default_gate_visible_and_stale_enable_callback_rejected(self):
    import pyray as rl
    from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
    from openpilot.system.ui.lib.application import gui_app
    _layout, details, toggle, _candidates = self.open_mici()
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertFalse(toggle._checked)
    self.assertFalse(toggle.enabled)
    details._handle_mouse_release(rl.Vector2(0, 0))
    dialog = gui_app.get_active_widget()
    self.assertIsInstance(dialog, BigDialog)
    self.assertIn('Nav Model Integration Bigv2', dialog._card.value)
    self.assertIn('Explicit user authorization', dialog._card.value)
    self.draw(dialog)
    gui_app.pop_widget()
    # A stale/injected callback cannot bypass the disabled visible widget.
    toggle._toggle_callback(True)
    self.assertIsInstance(gui_app.get_active_widget(), BigDialog)
    self.assertIn('Explicit user authorization', gui_app.get_active_widget()._card.value)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertFalse(toggle._checked)
    self.assert_ordinary_settings_preserved()

  def test_mici_prepare_candidate_persists_only_separate_selection(self):
    import pyray as rl
    layout, _details, _toggle, candidates = self.open_mici()
    self.assertTrue(candidates[0].enabled)
    candidates[0]._handle_mouse_release(rl.Vector2(0, 0))
    self.assertEqual(Params().get('NavigationModelCandidate'), 'big-nav-v2')
    self.assertFalse(Params().get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(get_active_model_runner(self.params), self.preserved['ModelRunnerTypeCache'])
    self.assert_ordinary_settings_preserved()
    self.draw(layout)

  def test_mici_offroad_and_unknown_candidate_gates_cannot_be_overridden(self):
    import pyray as rl
    layout, _details, toggle, candidates = self.open_mici()
    self.offroad.return_value = False
    self.assertFalse(candidates[0].enabled)
    candidates[0]._handle_mouse_release(rl.Vector2(0, 0))
    self.assertIsNone(self.params.get('NavigationModelCandidate'))
    self.offroad.return_value = True
    self.params.put('NavigationModelCandidate', 'unknown-candidate', block=True)
    self.assertFalse(toggle.enabled)
    toggle._toggle_callback(True)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    # Merely writing the checkbox cannot select an unreviewed native model.
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    self.assertEqual(get_active_model_runner(self.params), self.preserved['ModelRunnerTypeCache'])
    self.assertIn('Blocked', selection_description(self.params))
    layout._update_state()
    self.assertEqual(layout.navigation_models_btn.value, 'requested')
    toggle._toggle_callback(False)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assert_ordinary_settings_preserved()

  def test_mici_synthetic_reviewed_status_enable_disable(self):
    import pyray as rl
    status = {'id': 'big-nav-v2', 'display_name': 'Synthetic reviewed fixture', 'eligible': True, 'reason': 'fixture only'}
    with patch('openpilot.selfdrive.ui.sunnypilot.mici.layouts.models.candidate_status', return_value=status), \
         patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value=status):
      _layout, _details, toggle, _candidates = self.open_mici()
      self.assertTrue(toggle.enabled)
      toggle._handle_mouse_release(rl.Vector2(0, 0))
      self.assertTrue(toggle._checked)
      self.assertTrue(Params().get_bool('NavigationModelDrivingEnabled'))
      self.assertIn('Requested on', selection_description(self.params))
      toggle._handle_mouse_release(rl.Vector2(0, 0))
      self.assertFalse(toggle._checked)
      self.assertFalse(Params().get_bool('NavigationModelDrivingEnabled'))
    self.assert_ordinary_settings_preserved()

  def test_tici_real_layout_renders_status_and_rejects_unreviewed_enable(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
    layout = ModelsLayout()
    gui_app.push_widget(layout)
    self.draw(layout, 2160, 1080)
    self.assertFalse(layout.navigation_driving.action_item.enabled)
    self.assertFalse(layout.navigation_driving.action_item.get_state())
    assert layout.navigation_status.callback is not None
    layout.navigation_status.callback()
    dialog = gui_app.get_active_widget()
    self.assertIsInstance(dialog, ConfirmDialog)
    self.assertIn('Explicit user authorization', dialog._label._text)
    self.draw(dialog, 2160, 1080)
    gui_app.pop_widget()
    layout._set_navigation_driving(True)
    self.assertFalse(layout.navigation_driving.action_item.get_state())
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assert_ordinary_settings_preserved()

  def test_mici_register_existing_model_uses_background_worker_and_keeps_driving_off(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ExistingBigv2RegistrationDialog
    from openpilot.system.ui.lib.application import gui_app
    entered, release = threading.Event(), threading.Event()
    ui_thread = threading.get_ident()
    def prepare(params, *, is_offroad):
      self.assertNotEqual(threading.get_ident(), ui_thread)
      self.assertIs(params, self.params)
      self.assertTrue(is_offroad)
      entered.set()
      if not release.wait(3):
        raise RuntimeError('Test did not release registration worker')
      params.put('NavigationModelManifest', '/fixture/validated/manifest.json', block=True)
      return {'prepared': True, 'activation_allowed': False}
    self.addCleanup(release.set)
    with patch('openpilot.selfdrive.navigation.preparation.prepare_existing_model', side_effect=prepare) as service:
      layout, _details, _toggle, _candidates = self.open_mici()
      button = layout.navigation_register_btn
      self.assertEqual(button.text, 'Register existing Bigv2')
      button._handle_mouse_release(rl.Vector2(0, 0))
      dialog = gui_app.get_active_widget()
      self.assertIsInstance(dialog, ExistingBigv2RegistrationDialog)
      self.assertTrue(entered.wait(1))
      self.draw(dialog)
      self.assertIn('Checking existing', dialog._card.value)
      self.assertFalse(button.enabled)
      layout._register_existing_navigation()  # Stale callbacks cannot launch a duplicate job.
      self.assertEqual(service.call_count, 1)
      release.set()
      dialog.worker.join(3)
      self.assertFalse(dialog.worker.is_alive())
      self.draw(dialog)
      self.assertEqual(dialog._card.text, 'Bigv2 registered')
      self.assertTrue(button.enabled)
      self.assertEqual(self.params.get('NavigationModelManifest'), '/fixture/validated/manifest.json')
      self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
      self.assert_ordinary_settings_preserved()

  def test_mici_registration_onroad_and_failure_keep_selection_and_report_reason(self):
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    self.params.put('NavigationModelManifest', '/fixture/previous.json', block=True)
    with patch('openpilot.selfdrive.navigation.preparation.prepare_existing_model', side_effect=ValueError('Camera warp hash mismatch')) as service:
      layout, _details, _toggle, _candidates = self.open_mici()
      self.offroad.return_value = False
      self.assertFalse(layout.navigation_register_btn.enabled)
      layout._register_existing_navigation()
      service.assert_not_called()
      self.assertIn('off road', gui_app.get_active_widget()._card.value)
      gui_app.pop_widget()
      self.offroad.return_value = True
      layout.navigation_register_btn._handle_mouse_release(rl.Vector2(0, 0))
      dialog = gui_app.get_active_widget()
      dialog.worker.join(3)
      self.draw(dialog)
      self.assertEqual(dialog._card.text, 'Registration failed')
      self.assertEqual(dialog._card.value, 'Camera warp hash mismatch')
      self.assertEqual(self.params.get('NavigationModelManifest'), '/fixture/previous.json')
      self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
      self.assert_ordinary_settings_preserved()

  def test_tici_register_existing_model_success_and_failure_are_explicit_and_never_enable(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout, ExistingBigv2RegistrationDialog
    from openpilot.system.ui.lib.application import gui_app
    layout = ModelsLayout()
    gui_app.push_widget(layout)
    self.draw(layout, 2160, 1080)
    for error in (None, ValueError('Source report does not match installed model')):
      with self.subTest(error=error), patch('openpilot.selfdrive.navigation.preparation.prepare_existing_model',
                                           return_value={'prepared': True, 'activation_allowed': False}, side_effect=error) as service:
        self.assertTrue(layout.navigation_register.action_item.enabled)
        assert layout.navigation_register.callback is not None
        layout.navigation_register.callback()
        dialog = gui_app.get_active_widget()
        self.assertIsInstance(dialog, ExistingBigv2RegistrationDialog)
        dialog.worker.join(3)
        self.draw(dialog, 2160, 1080)
        service.assert_called_once_with(self.params, is_offroad=True)
        if error:
          self.assertIn(str(error), dialog._label._text)
        else:
          self.assertIn('registered', dialog._label._text)
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assert_ordinary_settings_preserved()
        gui_app.pop_widget()
    self.offroad.return_value = False
    with patch('openpilot.selfdrive.navigation.preparation.prepare_existing_model') as service:
      self.assertFalse(layout.navigation_register.action_item.enabled)
      layout._register_existing_navigation()
      service.assert_not_called()
      self.assertIn('off road', gui_app.get_active_widget()._label._text)
