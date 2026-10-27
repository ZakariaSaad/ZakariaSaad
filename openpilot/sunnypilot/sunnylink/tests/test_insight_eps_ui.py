"""Rendered owner-confirmation controls; backend admission is tested separately."""
import os
from unittest.mock import patch

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


def test_mici_toggles_keep_explicit_e2e_and_upstream_initial_max_speed(gui):
  from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
  app, _ = gui
  page = TogglesLayoutMici()
  app.push_widget(page)
  assert page._experimental_btn in page._scroller.items
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


def test_tici_toggles_keep_experimental_mode_and_upstream_initial_speed(gui):
  from openpilot.selfdrive.ui.layouts.settings.toggles import TogglesLayout
  app, _ = gui
  page = TogglesLayout()
  app.push_widget(page)
  assert 'ExperimentalMode' in page._toggles
  assert 'InitialCruiseSpeedKph' in page._toggles
  draw(page)


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
