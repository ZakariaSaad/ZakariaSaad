"""Real rendered selection widgets with isolated Params; no model/device inference.

Negative approval cases use an isolated unapproved registry. Only the positive
eligibility case uses synthetic reviewed status; admission helpers remain real.
"""
import os
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation import activation
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
    registry = json.loads(activation.REGISTRY.read_text())
    registry['candidates'][0]['activation_approved'] = False
    registry_path = Path(self.enterContext(tempfile.TemporaryDirectory())) / 'unapproved-registry.json'
    registry_path.write_text(json.dumps(registry))
    self.enterContext(patch.object(activation, 'REGISTRY', registry_path))
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

  def finish_activation(self, layout):
    dialog = layout._navigation_activation
    assert dialog is not None
    dialog.worker.join(3)
    self.assertFalse(dialog.worker.is_alive())
    self.draw(dialog)
    return dialog

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
    self.finish_activation(_layout)
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
    self.finish_activation(layout)
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

  def test_mici_toggle_tracks_permission_changed_by_shared_ssh_path(self):
    _layout, _details, toggle, _candidates = self.open_mici()
    self.assertFalse(toggle._checked)
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    self.draw(toggle)
    self.assertTrue(toggle._checked)
    self.params.put_bool('NavigationModelDrivingEnabled', False, block=True)
    self.draw(toggle)
    self.assertFalse(toggle._checked)
    self.assert_ordinary_settings_preserved()

  def test_mici_synthetic_reviewed_status_enable_disable(self):
    import pyray as rl
    status = {'id': 'big-nav-v2', 'display_name': 'Synthetic reviewed fixture', 'eligible': True, 'reason': 'fixture only'}
    with patch('openpilot.selfdrive.ui.sunnypilot.mici.layouts.models.candidate_status', return_value=status), \
         patch('openpilot.selfdrive.navigation.selection.candidate_status', return_value=status), \
         patch('openpilot.selfdrive.navigation.selection.require_offroad'), \
         patch('openpilot.selfdrive.navigation.selection.admit_candidate', return_value=SimpleNamespace(model_id='big-nav-v2')), \
         patch('openpilot.selfdrive.navigation.selection.validate_admission', create=True):
      _layout, _details, toggle, _candidates = self.open_mici()
      self.assertTrue(toggle.enabled)
      toggle._handle_mouse_release(rl.Vector2(0, 0))
      self.finish_activation(_layout)
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
    self.finish_activation(layout)
    self.assertFalse(layout.navigation_driving.action_item.get_state())
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assert_ordinary_settings_preserved()

  def test_mici_big_models_contains_installed_local_navigation_without_catalog_or_download(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    from openpilot.system.ui.lib.application import gui_app
    status = {'id': 'big-nav-v2', 'display_name': 'Nav Model Integration Bigv2', 'installed': True, 'eligible': False,
              'selected': False, 'enabled': False, 'reason': 'Explicit user authorization is still required'}
    self.params.put('NavigationModelManifest', '/fixture/existing/manifest.json', block=True)
    self.params.put('ModelManager_DownloadRef', 'preserve-existing-queue', block=True)
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    with patch('openpilot.selfdrive.ui.sunnypilot.mici.layouts.models.candidate_status', return_value=status), \
         patch('openpilot.selfdrive.ui.sunnypilot.mici.layouts.models.bundles_for_source', return_value=[]):
      layout = ModelsLayoutMici()
      gui_app.push_widget(layout)
      layout._select_hardware('chestnut')
      submenu = gui_app.get_active_widget()
      self.draw(submenu)
      local = submenu._scroller.items[1]
      self.assertEqual(local.text, 'Nav Model Integration Bigv2')
      self.assertIn('installed local', local.value)
      self.assertIn('activation blocked', local.value)
      self.assertTrue(local.enabled)
      local._handle_mouse_release(rl.Vector2(0, 0))
    self.assertEqual(self.params.get('NavigationModelCandidate'), 'big-nav-v2')
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(self.params.get('NavigationModelManifest'), '/fixture/existing/manifest.json')
    self.assertEqual(self.params.get('ModelManager_DownloadRef'), 'preserve-existing-queue')
    self.assertIn('Driving stays off', gui_app.get_active_widget()._card.value)
    self.assert_ordinary_settings_preserved()

  def test_mici_uninstalled_big_models_row_and_stale_callback_cannot_select(self):
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    from openpilot.system.ui.lib.application import gui_app
    status = {'id': 'big-nav-v2', 'display_name': 'Nav Model Integration Bigv2', 'installed': False, 'eligible': False}
    with patch('openpilot.selfdrive.ui.sunnypilot.mici.layouts.models.candidate_status', return_value=status), \
         patch('openpilot.selfdrive.ui.sunnypilot.mici.layouts.models.bundles_for_source', return_value=[]):
      layout = ModelsLayoutMici()
      gui_app.push_widget(layout)
      layout._select_hardware('chestnut')
      self.assertFalse(gui_app.get_active_widget()._scroller.items[1].enabled)
      layout._select_navigation_candidate('big-nav-v2', require_installed=True)
    self.assertIsNone(self.params.get('NavigationModelCandidate'))
    self.assertIn('not installed', gui_app.get_active_widget()._card.value)
    self.assert_ordinary_settings_preserved()

  def test_tici_big_models_local_node_uses_dedicated_navigation_selection(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout, NAVIGATION_REF
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets import DialogResult
    status = {'id': 'big-nav-v2', 'display_name': 'Nav Model Integration Bigv2', 'installed': True, 'eligible': False,
              'selected': False, 'enabled': False, 'reason': 'Explicit user authorization is still required'}
    self.params.put('NavigationModelManifest', '/fixture/existing/manifest.json', block=True)
    self.params.put('ModelManager_DownloadRef', 'preserve-existing-queue', block=True)
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    with patch('openpilot.selfdrive.ui.sunnypilot.layouts.settings.models.candidate_status', return_value=status), \
         patch('openpilot.selfdrive.ui.sunnypilot.layouts.settings.models.bundles_for_source', return_value=[]):
      layout = ModelsLayout()
      gui_app.push_widget(layout)
      folders = layout._source_folders(set(), 'chestnut')
      local = folders[0].nodes[1]
      self.assertEqual(local.ref, NAVIGATION_REF)
      self.assertIn('installed local, navigation aware', local.data['display_name'])
      self.assertIn('activation blocked', local.data['display_name'])
      self.assertEqual(layout._source_folders(set(), 'qcom'), [])
      layout._open_source_dialog('chestnut')
      assert layout.model_dialog is not None
      layout.model_dialog._select_node(local)
      layout._on_model_selected(DialogResult.CONFIRM)
    self.assertEqual(self.params.get('NavigationModelCandidate'), 'big-nav-v2')
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertEqual(self.params.get('NavigationModelManifest'), '/fixture/existing/manifest.json')
    self.assertEqual(self.params.get('ModelManager_DownloadRef'), 'preserve-existing-queue')
    self.assertIn('Driving stays off', gui_app.get_active_widget()._label._text)
    self.assert_ordinary_settings_preserved()

  def test_tici_uninstalled_navigation_node_cannot_select(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout, NAVIGATION_REF
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets import DialogResult
    status = {'id': 'big-nav-v2', 'display_name': 'Nav Model Integration Bigv2', 'installed': False, 'eligible': False}
    with patch('openpilot.selfdrive.ui.sunnypilot.layouts.settings.models.candidate_status', return_value=status):
      layout = ModelsLayout()
      gui_app.push_widget(layout)
      layout._open_source_dialog('chestnut')
      assert layout.model_dialog is not None
      layout.model_dialog.selection_ref = NAVIGATION_REF
      layout._on_model_selected(DialogResult.CONFIRM)
    self.assertIsNone(self.params.get('NavigationModelCandidate'))
    self.assertIn('not installed', gui_app.get_active_widget()._label._text)
    self.assert_ordinary_settings_preserved()

  def test_selecting_ordinary_big_model_withdraws_navigation_without_altering_small_slot(self):
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets import DialogResult
    small = self.params.get('ModelManager_ActiveBundle')
    layout = ModelsLayoutMici()
    gui_app.push_widget(layout)
    layout._selection_source = 'chestnut'
    self.params.put('NavigationModelCandidate', 'big-nav-v2', block=True)
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    layout._select_model(SimpleNamespace(ref='ordinary-big-ref'))
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertIsNone(self.params.get('NavigationModelCandidate'))
    self.assertEqual(self.params.get('ModelManager_DownloadRef', block=True), 'ordinary-big-ref')
    layout = ModelsLayout()
    layout._open_source_dialog('chestnut')
    assert layout.model_dialog is not None
    layout.model_dialog.selection_ref = 'Default'
    self.params.put('NavigationModelCandidate', 'big-nav-v2', block=True)
    self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    layout._on_model_selected(DialogResult.CONFIRM)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    self.assertIsNone(self.params.get('NavigationModelCandidate'))
    self.assertIsNone(self.params.get('ModelManager_ActiveBundleChestnut'))
    self.assertEqual(self.params.get('ModelManager_ActiveBundle'), small)

  def test_activation_admission_runs_off_render_thread_and_ignores_duplicate_request(self):
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
    from openpilot.system.ui.lib.application import gui_app
    ui_thread = threading.get_ident()
    for path, layout_type in (('mici.layouts', ModelsLayoutMici), ('layouts.settings', ModelsLayout)):
      with self.subTest(path=path):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def admission(params, enabled, *, offroad, entered=entered, release=release):
          self.assertNotEqual(threading.get_ident(), ui_thread)
          self.assertTrue(enabled)
          self.assertTrue(offroad)
          entered.set()
          if not release.wait(3):
            raise RuntimeError('Test did not release activation worker')
          params.put_bool('NavigationModelDrivingEnabled', True, block=True)
        with patch(f'openpilot.selfdrive.ui.sunnypilot.{path}.models.set_driving_enabled', side_effect=admission) as service:
          self.params.put_bool('NavigationModelDrivingEnabled', False, block=True)
          layout = layout_type()
          gui_app.push_widget(layout)
          if isinstance(layout, ModelsLayoutMici):
            layout._show_navigation_models()
            toggle = gui_app.get_active_widget()._scroller.items[1]
            def trigger(toggle=toggle):
              toggle._toggle_callback(True)
          else:
            def trigger(layout=layout):
              layout._set_navigation_driving(True)
          trigger()
          self.assertTrue(entered.wait(1))
          dialog = gui_app.get_active_widget()
          self.draw(dialog)
          self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
          trigger()
          self.assertEqual(service.call_count, 1)
          release.set()
          self.finish_activation(layout)
          self.assertTrue(self.params.get_bool('NavigationModelDrivingEnabled'))
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
