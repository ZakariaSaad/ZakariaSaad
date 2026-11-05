"""Render the upstream Cruise toggle with isolated Params and vehicle metadata."""
import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase


@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires raylib rendering under Xvfb')
class TestPredictiveBoostUI(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    gui_app.init_window('Predictive boost UI regression')

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
    # Normal manager startup seeds registered defaults before UI is launched.
    self.params.put('AccelBoostSP', self.params.get_default_value('AccelBoostSP'), block=True)
    self.enterContext(patch.object(ui_state, 'params', self.params))
    self.enterContext(patch.object(ui_state, 'is_offroad', return_value=True))
    self.enterContext(patch.object(gui_app, '_nav_stack', []))
    self.enterContext(patch.object(ui_state, 'CP', SimpleNamespace(pcmCruise=False, alphaLongitudinalAvailable=False)))
    self.enterContext(patch.object(ui_state, 'CP_SP', SimpleNamespace(intelligentCruiseButtonManagementAvailable=False)))
    self.enterContext(patch.object(ui_state, 'has_longitudinal_control', True))
    self.enterContext(patch.object(ui_state, 'has_icbm', False))

  @staticmethod
  def draw(layout):
    import pyray as rl
    rl.begin_drawing()
    try:
      layout.render(rl.Rectangle(0, 0, 1920, 1080))
    finally:
      rl.end_drawing()

  def mount(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise import CruiseLayout
    from openpilot.system.ui.lib.application import gui_app
    layout = CruiseLayout()
    gui_app.push_widget(layout)
    self.draw(layout)
    return layout

  def test_default_on_saved_off_and_recreation(self):
    import pyray as rl
    layout = self.mount()
    control = layout.accel_boost_toggle.action_item
    self.assertTrue(control.enabled)
    self.assertTrue(control.get_state())
    control.toggle.set_enabled(True)
    control.toggle._handle_mouse_release(rl.Vector2(0, 0))
    for _ in range(200):
      if not self.params.get_bool('AccelBoostSP'):
        break
      time.sleep(.005)
    self.assertFalse(self.params.get_bool('AccelBoostSP'))
    self.assertFalse(self.mount().accel_boost_toggle.action_item.get_state())

  def test_stock_or_missing_vehicle_disables_control_without_erasing_preference(self):
    from openpilot.selfdrive.ui.ui_state import ui_state
    layout = self.mount()
    with patch.object(ui_state, 'has_longitudinal_control', False):
      self.draw(layout)
      self.assertFalse(layout.accel_boost_toggle.action_item.enabled)
      self.assertTrue(self.params.get_bool('AccelBoostSP'))
    with patch.object(ui_state, 'CP', None), patch.object(ui_state, 'CP_SP', None):
      self.draw(layout)
      self.assertFalse(layout.accel_boost_toggle.action_item.enabled)
      self.assertTrue(self.params.get_bool('AccelBoostSP'))
    self.draw(layout)
    self.assertTrue(layout.accel_boost_toggle.action_item.enabled)
