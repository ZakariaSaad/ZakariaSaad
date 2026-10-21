"""Real raylib/VisionIPC rendering under Xvfb; no vehicle or network needed."""
import os
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from openpilot.common.test import OpenpilotTestCase


@unittest.skipUnless(os.environ.get('DISPLAY'), 'requires a display (run under Xvfb on a PC)')
class TestNavigationUI(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    import pyray as rl
    from openpilot.system.ui.lib.application import gui_app
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    gui_app.init_window('Navigation Alpha UI tests')

  @classmethod
  def tearDownClass(cls):
    from openpilot.system.ui.lib.application import gui_app
    gui_app.close()
    super().tearDownClass()

  def test_setting_persistence_without_changing_custom_settings(self):
    import pyray as rl
    from openpilot.common.params import Params
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
    params = Params()
    params.put_bool('LaneCentering', True, block=True)
    params.put_bool('HondaEpsFirmwareEnabled', False, block=True)
    with patch.object(ui_state, 'params', params):
      settings = DeveloperLayoutMici()
      self.assertFalse(settings._navigation_toggle._checked)
      settings._navigation_toggle._handle_mouse_release(rl.Vector2(0, 0))
      self.assertTrue(params.get_bool('NavigationEnabled'))
      self.assertTrue(DeveloperLayoutMici()._navigation_toggle._checked)
      settings._navigation_toggle._handle_mouse_release(rl.Vector2(0, 0))
      self.assertFalse(params.get_bool('NavigationEnabled'))
      self.assertTrue(params.get_bool('LaneCentering'))
      self.assertFalse(params.get_bool('HondaEpsFirmwareEnabled'))

  def test_frames_instructions_and_stale_frame_handling(self):
    import pyray as rl
    from msgq.visionipc import VisionIpcServer
    from openpilot.cereal import messaging
    from openpilot.cereal.visionipc import VisionStreamType
    from openpilot.selfdrive.navigation.renderer import SIZE, ROUTE_INTENSITY
    from openpilot.selfdrive.ui.mici.onroad.navigation import NavigationPanel, format_distance
    from openpilot.selfdrive.ui.ui_state import ui_state
    pm = messaging.PubMaster(['navInstruction'])
    sm = messaging.SubMaster(['navInstruction', 'gpsLocation', 'gpsLocationExternal'])
    server = VisionIpcServer('mapsd')
    server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
    server.start_listener()
    with patch.object(ui_state, 'sm', sm), patch.object(ui_state, 'is_metric', True):
      panel = NavigationPanel()
      self.addCleanup(panel.close)
      panel._last_connect = -1
      self.assertFalse(panel.map_ready)
      self.assertFalse(panel.routing)
      self.assertEqual(format_distance(250), '250 m')
      with patch.object(ui_state, 'is_metric', False):
        self.assertEqual(format_distance(1609.344), '1.0 mi')
      msg = messaging.new_message('navInstruction', valid=True)
      msg.navInstruction.maneuverDistance = 240
      msg.navInstruction.maneuverModifier = 'right'
      msg.navInstruction.timeRemaining = 120
      msg.navInstruction.distanceRemaining = 1000
      pm.send('navInstruction', msg)
      sm.update(1000)
      self.assertTrue(panel.routing)
      deadline = time.monotonic() + 5
      while time.monotonic() < deadline and not panel._client.is_connected():
        panel._last_connect = -1
        panel._update_state()
        time.sleep(0.01)
      self.assertTrue(panel._client.is_connected())
      image = np.zeros((SIZE, SIZE), np.uint8)
      image[:, 128] = ROUTE_INTENSITY
      server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), 1, time.monotonic_ns(), time.monotonic_ns())
      deadline = time.monotonic() + 5
      while time.monotonic() < deadline and not panel.map_ready:
        panel._update_state()
        time.sleep(0.01)
      self.assertTrue(panel.map_ready)
      rl.begin_drawing()
      panel.render(rl.Rectangle(0, 0, 261, 240))
      rl.end_drawing()
      self.assertEqual(panel._eta_label._text, '2 min / 1.0 km')
      panel._frame_time = rl.get_time() - 4
      self.assertFalse(panel.map_ready)
      server.send(VisionStreamType.VISION_STREAM_MAP, np.zeros_like(image).tobytes(), 2, 0, time.monotonic_ns())
      deadline = time.monotonic() + 5
      while time.monotonic() < deadline and panel._client.frame_id != 2:
        panel._update_state()
        time.sleep(0.01)
      self.assertEqual(panel._client.frame_id, 2)
      self.assertFalse(panel.map_ready)

  def test_alert_priority_and_disabled_camera_layout(self):
    import pyray as rl
    from openpilot.selfdrive.ui.mici.onroad.augmented_road_view import AugmentedRoadView
    from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
    from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
    from openpilot.selfdrive.ui.mici.onroad.navigation import PANEL_WIDTH
    from openpilot.selfdrive.ui.ui_state import ui_state
    view = AugmentedRoadView()
    self.addCleanup(view.close)
    view.set_rect(rl.Rectangle(0, 0, 536, 240))
    with patch.object(ui_state, 'started', True), patch.object(ui_state, 'navigation_enabled', True), \
         patch.object(ui_state, 'nav_destination_set', True), patch.object(view, '_switch_stream_if_needed'), \
         patch.object(view, '_update_calibration'), patch.object(CameraView, '_render'), \
         patch.object(view._model_renderer, 'render'), patch.object(view._hud_renderer, 'render'), \
         patch.object(view._driver_state_renderer, 'render'), patch.object(view._alert_renderer, 'render'), \
         patch.object(view._confidence_ball, 'render') as confidence, patch.object(view._bookmark_icon, 'render'), \
         patch.object(view._navigation, 'render') as nav, patch.object(view._alert_renderer, 'will_render') as alert:
      alert.return_value = (None, False)
      view._render(view.rect)
      nav.assert_called_once()
      self.assertEqual(view._content_rect.width, 536 - PANEL_WIDTH)
      confidence.assert_not_called()
      nav.reset_mock()
      alert.return_value = (MagicMock(), True)
      view._render(view.rect)
      nav.assert_not_called()
      self.assertEqual(view._content_rect.width, 536 - SIDE_PANEL_WIDTH)
      alert.return_value = (None, False)
      with patch.object(ui_state, 'navigation_enabled', False):
        view._render(view.rect)
      nav.assert_not_called()
      self.assertEqual(view._content_rect.width, 536 - SIDE_PANEL_WIDTH)
