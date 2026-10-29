"""Rendered Blendful controls and visualization with isolated Params/messages.

No vehicle processes, model execution, model downloads or device access are used.
"""
import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.cereal import custom, messaging
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase


@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires raylib rendering under Xvfb')
class TestBlendfulUI(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    gui_app.init_window('Blendful UI regression')

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
    self.enterContext(patch.object(ui_state, 'params', self.params))
    self.enterContext(patch.object(ui_state, 'is_offroad', return_value=True))
    self.enterContext(patch.object(gui_app, '_nav_stack', []))
    self.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.tinygrad, block=True)
    self.refresh_runner()

  @staticmethod
  def layouts():
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.models import ModelsLayout
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts.models import ModelsLayoutMici
    return [('tici', ModelsLayout), ('mici', ModelsLayoutMici)]

  @staticmethod
  def refresh_runner():
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP
    UIStateSP.update_params(ui_state)

  @staticmethod
  def draw(widget):
    import pyray as rl
    rl.begin_drawing()
    try:
      widget.render(rl.Rectangle(0, 0, 536, 1080))
    finally:
      rl.end_drawing()

  def mount(self, cls):
    from openpilot.system.ui.lib.application import gui_app
    widget = cls()
    gui_app.push_widget(widget)
    self.draw(widget)
    return widget

  @staticmethod
  def control(kind, layout):
    return layout.blendful_toggle.action_item if kind == 'tici' else layout.blendful_toggle

  @staticmethod
  def checked(kind, control):
    return control.get_state() if kind == 'tici' else control._checked

  def test_both_widgets_default_off_write_same_parameter_and_survive_recreation(self):
    import pyray as rl
    for kind, cls in self.layouts():
      with self.subTest(layout=kind):
        self.params.remove('Blendful')
        layout = self.mount(cls)
        toggle = self.control(kind, layout)
        self.assertTrue(toggle.enabled)
        self.assertFalse(self.checked(kind, toggle))
        self.assertFalse(self.params.get_bool('Blendful'))
        if kind == 'tici':
          toggle.toggle.set_enabled(True)
          toggle.toggle._handle_mouse_release(rl.Vector2(0, 0))
          # ToggleSP uses the normal async Params writer.
          for _ in range(200):
            if self.params.get_bool('Blendful'):
              break
            time.sleep(.005)
        else:
          toggle._handle_mouse_release(rl.Vector2(0, 0))
        self.assertTrue(Params().get_bool('Blendful'))
        recreated = self.mount(cls)
        self.assertTrue(self.checked(kind, self.control(kind, recreated)))

  def test_launched_runner_gates_both_widgets_without_erasing_saved_setting(self):
    from openpilot.selfdrive.ui.ui_state import ui_state
    for kind, cls in self.layouts():
      with self.subTest(layout=kind):
        self.params.put_bool('Blendful', True, block=True)
        # A highlighted/selected tinygrad model cannot override a stock runner.
        self.params.put('ModelManager_ActiveBundle', {'runner': 'tinygrad'}, block=True)
        self.params.put('ModelManager_ActiveBundleChestnut', {'runner': 'tinygrad'}, block=True)
        self.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.stock, block=True)
        self.refresh_runner()
        self.assertFalse(ui_state.active_runner_tinygrad)
        layout = self.mount(cls)
        toggle = self.control(kind, layout)
        self.assertFalse(toggle.enabled)
        self.assertFalse(self.checked(kind, toggle))
        self.assertTrue(Params().get_bool('Blendful'))
        self.assertIn('non-Default' if kind == 'tici' else 'custom model only',
                      layout.blendful_toggle.description if kind == 'tici' else toggle.value)
        # Conversely, the actual tinygrad launch wins over a Default-looking slot.
        self.params.remove('ModelManager_ActiveBundle')
        self.params.remove('ModelManager_ActiveBundleChestnut')
        self.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.tinygrad, block=True)
        self.refresh_runner()
        self.draw(layout)
        self.assertTrue(toggle.enabled)
        self.assertTrue(self.checked(kind, toggle))
        if kind == 'mici':
          self.assertEqual(toggle.value, '')
        self.params.put_bool('Blendful', False, block=True)
        self.draw(layout)
        self.assertFalse(self.checked(kind, toggle))

  def test_visualization_ticks_on_new_data_fades_and_preserves_lateral_gate(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.onroad import blendful as visual
    from openpilot.selfdrive.ui.ui_state import UIStatus

    report = messaging.new_message('modelDataV2SP').modelDataV2SP
    report.blendfulTrust = .8
    report.blendfulCurvature = .0003
    report.blendfulLeftProb = report.blendfulRightProb = .9
    report.blendfulLaneWidth = 3.6
    model = messaging.new_message('modelV2').modelV2
    model.laneLineStds = [.1] * 4
    selfdrive = messaging.new_message('selfdriveStateSP').selfdriveStateSP
    selfdrive.mads.available = False

    class Messages(dict):
      updated = {'modelDataV2SP': True}
      valid = {'selfdriveStateSP': True}

    messages = Messages(modelDataV2SP=report, modelV2=model, carState=SimpleNamespace(vEgo=20.), selfdriveStateSP=selfdrive)
    ui = SimpleNamespace(sm=messages, status=UIStatus.ENGAGED)
    x = np.linspace(0., 100., 33)
    lines = [SimpleNamespace(raw_points=np.column_stack((x, np.full(33, offset), np.zeros(33))))
             for offset in (-5.4, -1.8, 1.8, 5.4)]
    projection = Mock(side_effect=lambda points, *_: np.array([[100., 100.], [110., 100.], [110., 400.], [100., 400.]], np.float32))
    renderer = SimpleNamespace(_lane_lines=lines, _map_line_to_polygon=projection)
    with patch.object(visual, 'ui_state', ui):
      for relative, readout in ((False, False), (True, True)):
        with self.subTest(rect_relative=relative, readout=readout):
          overlay = visual.BlendfulRenderer(readout=readout, rect_relative=relative)
          for _ in range(30):
            overlay.update(renderer, 30)
          self.assertTrue(overlay._center_points.size)
          tick = overlay._alpha.x
          messages.updated['modelDataV2SP'] = False
          overlay.update(renderer, 30)
          self.assertEqual(overlay._alpha.x, tick)
          with patch.object(visual, 'draw_polygon', wraps=visual.draw_polygon) as polygon:
            rl.begin_drawing()
            try:
              overlay.draw(rl.Rectangle(10, 20, 536, 240))
              polygon.assert_called_once()
              np.testing.assert_array_equal(polygon.call_args.args[1], overlay._center_points + ((10, 20) if relative else (0, 0)))
              ui.status = UIStatus.DISENGAGED
              overlay.draw(rl.Rectangle(10, 20, 536, 240))
              self.assertEqual(polygon.call_count, 1)
            finally:
              rl.end_drawing()
          ui.status = UIStatus.ENGAGED
          messages.updated['modelDataV2SP'] = True
          report.blendfulTrust = 0.
          for _ in range(100):
            overlay.update(renderer, 30)
          self.assertFalse(overlay._center_points.size)
          report.blendfulTrust = .8

  def test_shared_lateral_gate_preserves_existing_mads_and_status_cases(self):
    from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP, MADSState
    from openpilot.selfdrive.ui.ui_state import UIStatus
    for available in (False, True):
      for enabled in (False, True):
        for paused in (False, True):
          for status in UIStatus:
            with self.subTest(available=available, enabled=enabled, paused=paused, status=status):
              message = SimpleNamespace(mads=SimpleNamespace(available=available, enabled=enabled,
                                        state=MADSState.paused if paused else MADSState.disabled))
              class Messages(dict):
                valid = {'selfdriveStateSP': True}
              ui = SimpleNamespace(sm=Messages(selfdriveStateSP=message), status=status)
              expected = enabled and not paused if available else status in (UIStatus.ENGAGED, UIStatus.LAT_ONLY)
              self.assertEqual(UIStateSP.lateral_active(ui), expected)
