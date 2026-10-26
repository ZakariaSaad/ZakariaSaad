"""Rendered Models widgets using the real shared request queue and native Params.

There is no vehicle messaging, GPU, model preparation or download. Fresh-state
subscriptions are substituted; the queue/selection interfaces stay real.
"""
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.corrected_bigv2 import MODEL_ID, DISPLAY_NAME
from openpilot.sunnypilot.models import selection


@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires real raylib rendering under Xvfb')
class TestNavigationModelUI(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    gui_app.init_window('Unified model selection tests')

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
    self.params.put_bool('NavigationModelDrivingEnabled', False, block=True)
    self.params.put_bool('IsOffroad', True, block=True)
    self.params.put_bool('IsEngaged', False, block=True)
    self.preserved = {key: self.params.get(key) for key in ('ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut',
                                                          'ModelRunnerTypeCache', 'NavigationEnabled')}
    self.enterContext(patch.object(ui_state, 'params', self.params))
    self.offroad = self.enterContext(patch.object(ui_state, 'is_offroad', return_value=True))
    self.enterContext(patch.object(selection, 'require_offroad'))
    self.enterContext(patch.object(gui_app, '_nav_stack', []))

  def layouts(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    return [('layouts.settings', ModelsLayout), ('mici.layouts', ModelsLayoutMici)]

  def draw(self, widget, width=536, height=240):
    import pyray as rl
    rl.begin_drawing()
    try:
      widget.render(rl.Rectangle(0, 0, width, height))
    finally:
      rl.end_drawing()

  def mount(self, layout_type):
    from openpilot.system.ui.lib.application import gui_app
    layout = layout_type()
    gui_app.push_widget(layout)
    self.draw(layout)
    return layout

  def finish_request(self, layout):
    dialog = layout._model_selection
    self.assertIsNotNone(dialog)
    dialog.worker.join(4)
    self.assertFalse(dialog.worker.is_alive())
    self.draw(dialog)
    return dialog

  def assert_preserved(self):
    self.assertEqual(self.preserved, {key: self.params.get(key) for key in self.preserved})

  def choose_corrected(self, module, layout):
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets import DialogResult
    if module == 'mici.layouts':
      layout._select_hardware('chestnut')
      submenu = gui_app.get_active_widget()
      self.draw(submenu)
      corrected = submenu._scroller.items[1]
      self.assertEqual(corrected.text, DISPLAY_NAME)
      self.assertTrue(corrected.enabled)
      corrected._handle_mouse_release(rl.Vector2(0, 0))
    else:
      layout._open_source_dialog('chestnut')
      nodes = layout._source_folders(set(), 'chestnut')[0].nodes
      self.assertEqual(nodes[1].ref, selection.NAV_REF)
      layout.model_dialog._select_node(nodes[1])
      layout._on_model_selected(DialogResult.CONFIRM)
    return self.finish_request(layout)

  def test_both_big_models_lists_retire_original_and_allow_automatic_preparation(self):
    from openpilot.system.ui.lib.application import gui_app
    for module, layout_type in self.layouts():
      with self.subTest(module=module), patch(f'openpilot.selfdrive.ui.sunnypilot.{module}.models.bundles_for_source', return_value=[]):
        layout = self.mount(layout_type)
        self.assertFalse(hasattr(layout, 'navigation_register'))
        self.assertFalse(hasattr(layout, 'navigation_register_btn'))
        if module == 'mici.layouts':
          layout._select_hardware('chestnut')
          rows = gui_app.get_active_widget()._scroller.items
          self.assertEqual(len(rows), 2)
          self.assertEqual(rows[1].text, DISPLAY_NAME)
          self.assertIn('local files checked on selection', rows[1].value)
          self.assertTrue(rows[1].enabled)
        else:
          rows = layout._source_folders(set(), 'chestnut')[0].nodes
          self.assertEqual([row.ref for row in rows], ['Default', selection.NAV_REF])
          self.assertIn('local files checked on selection', rows[1].data['display_name'])
        self.draw(gui_app.get_active_widget())
    self.assert_preserved()

  def test_corrected_buttons_use_actual_queue_and_do_not_claim_active_before_preparation(self):
    for module, layout_type in self.layouts():
      with self.subTest(module=module), patch(f'openpilot.selfdrive.ui.sunnypilot.{module}.models.bundles_for_source', return_value=[]):
        layout = self.mount(layout_type)
        dialog = self.choose_corrected(module, layout)
        self.assertIsNone(dialog.request.error)
        status = selection.selection_status(self.params)
        self.assertEqual(status['ref'], selection.NAV_REF)
        self.assertEqual(status['source'], 'chestnut')
        self.assertEqual(status['state'], 'pending')
        self.assertIn('Pending', dialog.request.description())
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assert_preserved()
        selection.cancel_model_selection(self.params, offroad=True)

  def test_ordinary_big_selection_queues_without_prematurely_disabling_working_navigation(self):
    from openpilot.system.ui.widgets import DialogResult
    for module, layout_type in self.layouts():
      with self.subTest(module=module):
        self.params.put('NavigationModelCandidate', MODEL_ID, block=True)
        self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
        layout = self.mount(layout_type)
        if module == 'mici.layouts':
          layout._select_default('chestnut')
        else:
          layout._selection_source = 'chestnut'
          layout.model_dialog = SimpleNamespace(selection_ref='Default')
          layout._on_model_selected(DialogResult.CONFIRM)
        self.assertIsNone(self.finish_request(layout).request.error)
        status = selection.selection_status(self.params)
        self.assertEqual((status['source'], status['ref'], status['state']), ('chestnut', 'Default', 'pending'))
        self.assertTrue(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assert_preserved()
        selection.cancel_model_selection(self.params, offroad=True)

  def test_explicit_off_uses_shared_path_and_survives_widget_recreation(self):
    from openpilot.system.ui.lib.application import gui_app
    for module, layout_type in self.layouts():
      with self.subTest(module=module):
        self.params.put('NavigationModelCandidate', MODEL_ID, block=True)
        self.params.put_bool('NavigationModelDrivingEnabled', True, block=True)
        layout = self.mount(layout_type)
        if module == 'mici.layouts':
          layout._show_navigation_models()
          toggle = gui_app.get_active_widget()._scroller.items[1]
          toggle._toggle_callback(False)
        else:
          layout._set_navigation_driving(False)
        self.assertIsNone(self.finish_request(layout).request.error)
        self.assertFalse(Params().get_bool('NavigationModelDrivingEnabled'))
        self.assert_preserved()
        self.draw(self.mount(layout_type))
        self.assertFalse(Params().get_bool('NavigationModelDrivingEnabled'))

  def test_retired_id_stale_callback_is_rejected_by_shared_service(self):
    from openpilot.system.ui.widgets import DialogResult
    for module, layout_type in self.layouts():
      with self.subTest(module=module):
        layout = self.mount(layout_type)
        if module == 'mici.layouts':
          layout._select_navigation_candidate('big-nav-v2')
        else:
          layout._selection_source = 'chestnut'
          layout.model_dialog = SimpleNamespace(selection_ref='navigation:big-nav-v2')
          layout._on_model_selected(DialogResult.CONFIRM)
        dialog = self.finish_request(layout)
        self.assertIsNotNone(dialog.request.error)
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assert_preserved()

  def test_onroad_stale_callback_cannot_queue_or_mutate_current_selection(self):
    for module, layout_type in self.layouts():
      with self.subTest(module=module):
        layout = self.mount(layout_type)
        self.offroad.return_value = False
        self.params.put_bool('IsOffroad', False, block=True)
        layout._request_selection('chestnut', selection.NAV_REF)
        dialog = self.finish_request(layout)
        self.assertIsNotNone(dialog.request.error)
        self.assertIn('off', dialog.request.error.lower())
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assert_preserved()
        self.offroad.return_value = True
        self.params.put_bool('IsOffroad', True, block=True)

  def test_shared_guard_runs_off_render_thread_and_duplicate_requests_are_ignored(self):
    ui_thread = threading.get_ident()
    for module, layout_type in self.layouts():
      with self.subTest(module=module):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def guard(_params, entered=entered, release=release):
          self.assertNotEqual(threading.get_ident(), ui_thread)
          entered.set()
          if not release.wait(3):
            raise RuntimeError('Test did not release request guard')
        with patch.object(selection, 'require_offroad', side_effect=guard):
          layout = self.mount(layout_type)
          layout._request_selection('chestnut', selection.NAV_REF)
          self.assertTrue(entered.wait(1))
          first = layout._model_selection
          self.draw(first)
          layout._request_selection('chestnut', 'Default')
          self.assertIs(layout._model_selection, first)
          release.set()
          self.assertIsNone(self.finish_request(layout).request.error)
        selection.cancel_model_selection(self.params, offroad=True)

  def test_cancel_buttons_cancel_the_actual_pending_request_without_changing_models(self):
    for module, layout_type in self.layouts():
      with self.subTest(module=module):
        selection.request_model_selection(self.params, source='chestnut', ref=selection.NAV_REF,
                                           offroad=True, origin='device_ui')
        layout = self.mount(layout_type)
        layout._cancel_selection()
        dialog = self.finish_request(layout)
        self.assertIsNone(dialog.request.error)
        self.assertEqual(selection.selection_status(self.params)['state'], 'cancelled')
        self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
        self.assert_preserved()

  def test_preparation_and_failure_status_are_visible_without_claiming_active_model(self):
    states = ('selected', 'pending', 'preparing', 'downloading', 'loading', 'active', 'failed', 'cancelled')
    for module, layout_type in self.layouts():
      path = f'openpilot.selfdrive.ui.sunnypilot.{module}.models'
      for state in states:
        with self.subTest(module=module, state=state), patch(path + '.selection_status', return_value={
          'state': state, 'source': 'chestnut', 'ref': selection.NAV_REF, 'selected_ref': selection.NAV_REF,
          'message': 'Exact registration or download status'}):
          layout = self.mount(layout_type)
          self.draw(layout)
          if module == 'mici.layouts':
            self.assertEqual(layout.current_model_info.info_header._text, state)
            self.assertEqual(layout.current_model_info.info_text._text, 'Exact registration or download status')
          else:
            self.assertEqual(layout._status_note(), state.capitalize() + ': Exact registration or download status')
          self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
          self.assert_preserved()
