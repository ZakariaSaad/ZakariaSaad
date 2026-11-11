"""Exercise real action layout state without a GL window or renderer."""
from types import SimpleNamespace

import pyray as rl
import pytest

from opendbc.car.honda.values import CAR, HondaFlags
from openpilot.selfdrive.ui.layouts.settings import toggles
from openpilot.system.ui.sunnypilot.widgets.list_view import MultipleButtonActionSP
from openpilot.system.ui.widgets.list_view import MultipleButtonAction, RIGHT_ITEM_PADDING


@pytest.mark.parametrize("action_class", [MultipleButtonAction, MultipleButtonActionSP])
def test_vehicle_mode_transition_updates_width_selection_and_animation(monkeypatch, action_class):
  # Constructors load fonts/textures; the state and sizing methods tested here
  # are the production methods and require only their ordinary stored fields.
  action = object.__new__(action_class)
  action._rect = rl.Rectangle(0, 0, 1, 80)
  action.buttons = ["Aggressive", "Standard", "Relaxed", "Econ"]
  action.button_width = 220
  action.selected_button = 3
  if action_class is MultipleButtonActionSP:
    action._anim_x = 660.0
  page = SimpleNamespace(_personality_buttons=list(action.buttons),
                         _long_personality_setting=SimpleNamespace(action_item=action))
  cp = SimpleNamespace(brand="honda", carFingerprint=CAR.HONDA_INSIGHT,
                       openpilotLongitudinalControl=True, flags=int(HondaFlags.MVL_LONGITUDINAL))
  monkeypatch.setattr(toggles, "ui_state", SimpleNamespace(CP=cp))

  assert not toggles.TogglesLayout._update_personality_choices(page)
  assert action.buttons == ["Aggressive", "Standard", "Relaxed"]
  assert action.get_selected_button() == 2
  assert action.get_width_hint() == 3 * 300 + 2 * RIGHT_ITEM_PADDING
  if action_class is MultipleButtonActionSP:
    assert action._anim_x is None
    action._anim_x = 600.0

  cp.flags = 0
  assert toggles.TogglesLayout._update_personality_choices(page)
  assert action.buttons[-1] == "Econ"
  assert action.get_selected_button() == 2
  assert action.get_width_hint() == 4 * 220 + 3 * RIGHT_ITEM_PADDING
  if action_class is MultipleButtonActionSP:
    assert action._anim_x is None
