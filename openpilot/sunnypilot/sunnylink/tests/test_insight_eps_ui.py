"""Rendered owner-confirmation controls; backend admission is tested separately."""
import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='requires real raylib rendering under Xvfb')


@pytest.fixture
def gui(tmp_path, monkeypatch):
  import pyray as rl
  from openpilot.common.params import Params
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.lib.application import gui_app
  params = Params(str(tmp_path / 'params'))
  params.put_bool('IsOffroad', True, block=True)
  monkeypatch.setattr(ui_state, 'params', params)
  monkeypatch.setattr(ui_state, 'is_offroad', lambda: True)
  monkeypatch.setattr(type(ui_state), 'engaged', property(lambda self: False))
  monkeypatch.setattr(gui_app, '_nav_stack', [])
  rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
  gui_app.init_window('Insight EPS settings tests')
  yield gui_app, params
  gui_app.close()


def draw(widget):
  import pyray as rl
  rl.begin_drawing()
  try:
    widget.render(rl.Rectangle(0, 0, 536, 240))
  finally:
    rl.end_drawing()


def test_mici_recovery_and_toggle_call_shared_guard_only_after_explicit_action(gui):
  from openpilot.sunnypilot.sunnylink import insight_eps
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.insight_eps import InsightEpsLayout, FirmwareOwnerConfirmation
  app, params = gui
  with patch.object(insight_eps, 'confirm_installed_firmware') as confirm, patch.object(insight_eps, 'apply_enabled') as enable:
    page = InsightEpsLayout()
    app.push_widget(page)
    draw(page)
    recovery = FirmwareOwnerConfirmation(page._confirm)
    app.push_widget(recovery)
    draw(recovery)
    confirm.assert_not_called()
    enable.assert_not_called()
    page._confirm()
    confirm.assert_called_once_with(params, True)
    assert params.get('HondaEpsFirmwareVerifiedSha256') is None
    page._set_enabled(True)
    enable.assert_called_once_with(params, True)


def test_mici_rejection_restores_requested_toggle_and_shows_error(gui):
  from openpilot.sunnypilot.sunnylink import insight_eps
  from openpilot.selfdrive.ui.sunnypilot.mici.layouts.insight_eps import InsightEpsLayout
  from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
  app, params = gui
  with patch.object(insight_eps, 'apply_enabled', side_effect=ValueError('fresh off-road state required')):
    page = InsightEpsLayout()
    app.push_widget(page)
    page._set_enabled(True)
    assert not params.get_bool(insight_eps.ENABLED)
    assert isinstance(app.get_active_widget(), BigDialog)
    draw(app.get_active_widget())


def test_mici_toggles_use_upstream_longitudinal_controls_without_legacy_modes(gui):
  from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
  app, _ = gui
  page = TogglesLayoutMici()
  app.push_widget(page)
  assert not hasattr(page, "_experimental_btn")
  assert all(key not in {"ExperimentalMode", "DynamicExperimentalControl"} for key, _ in page._refresh_toggles)
  assert page._initial_cruise_speed in page._scroller.items
  draw(page)


def test_tici_steering_uses_same_guarded_owner_settings(gui):
  from openpilot.sunnypilot.sunnylink import insight_eps
  from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering import SteeringLayout
  app, params = gui
  with patch.object(insight_eps, 'apply_enabled') as enable:
    page = SteeringLayout()
    app.push_widget(page)
    draw(page)
    enable.assert_not_called()
    page._on_insight_eps(True)
    enable.assert_called_once_with(params, True)


def test_tici_toggles_use_upstream_longitudinal_controls_without_legacy_modes(gui):
  from openpilot.selfdrive.ui.layouts.settings.toggles import TogglesLayout
  app, _ = gui
  page = TogglesLayout()
  app.push_widget(page)
  assert 'ExperimentalMode' not in page._toggles
  assert 'DynamicExperimentalControl' not in page._toggles
  assert 'InitialCruiseSpeedKph' in page._toggles
  draw(page)


@pytest.mark.parametrize('layout', ['onroad', 'mici.onroad'])
def test_camera_choice_uses_longitudinal_capability_without_deprecated_mode_state(gui, layout):
  from opendbc.car import structs
  module = importlib.import_module(f'openpilot.selfdrive.ui.{layout}.augmented_road_view')
  view = SimpleNamespace(available_streams=[module.NARROW_ROAD_CAM, module.WIDE_CAM],
                         stream_type=module.NARROW_ROAD_CAM, switch_stream=Mock())
  # The current wire contract has no active experimentalMode setting. Maps and
  # navigation driving are independent of this ordinary camera selection path.
  cp = structs.CarParams.new_message(openpilotLongitudinalControl=True)
  messages = {'carParams': cp, 'carState': structs.CarState.new_message(vEgo=0.)}
  module.AugmentedRoadView._switch_stream_if_needed(view, messages)
  view.switch_stream.assert_called_once_with(module.WIDE_CAM)
  view.switch_stream.reset_mock()
  view.stream_type = module.WIDE_CAM
  cp.openpilotLongitudinalControl = False
  module.AugmentedRoadView._switch_stream_if_needed(view, messages)
  view.switch_stream.assert_called_once_with(module.NARROW_ROAD_CAM)


def test_mici_developer_renders_new_descriptions_and_guarded_eps_entry(gui):
  from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
  from openpilot.sunnypilot.sunnylink import insight_eps
  app, _ = gui
  with patch.object(insight_eps, 'confirm_installed_firmware') as confirm:
    page = DeveloperLayoutMici()
    app.push_widget(page)
    assert any(getattr(item, 'text', '') == 'Insight EPS controller' for item in page._scroller.items)
    draw(page)
    confirm.assert_not_called()
