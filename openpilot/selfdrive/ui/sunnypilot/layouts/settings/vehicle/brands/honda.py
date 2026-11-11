"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp
from openpilot.nrdr.ui.honda_longitudinal import (
  MVL_DESCRIPTION, RADAR_DESCRIPTION, live_gas_available, save_live_gas_choice, save_radar_choice,
)
from openpilot.nrdr.ui.settings.live_gas_reset import large_reset_button

DESCRIPTIONS = {
  'enforce_stock_longitudinal': tr_noop(
    'sunnypilot will not take over control of gas and brakes. Factory Honda Nidec longitudinal control will be used.'
  ),
}


class HondaSettings(BrandSettings):
  def __init__(self):
    super().__init__()

    self.enforce_stock_longitudinal = toggle_item_sp(
      lambda: tr("Nidec: Enforce Factory Longitudinal Control"),
      description=lambda: tr(DESCRIPTIONS["enforce_stock_longitudinal"]),
      initial_state=ui_state.params.get_bool("HondaEnforceStockLongitudinal"),
      callback=self._on_enable_enforce_stock_longitudinal,
      enabled=lambda: not ui_state.engaged,
    )

    self.live_learning_gas = toggle_item_sp(
      lambda: tr("Live Learning Gas - Full MVL Longitudinal"),
      description=lambda: tr(MVL_DESCRIPTION),
      initial_state=ui_state.params.get_bool("HondaLiveLearningGas"),
      callback=self._set_live_gas,
    )
    self.bosch_a_radar = toggle_item_sp(
      lambda: tr("Honda Bosch-A Radar (Default: ON)"),
      description=lambda: tr(RADAR_DESCRIPTION),
      initial_state=bool(ui_state.params.get("HondaBoschARadar", return_default=True)),
      callback=self._set_radar,
    )
    self.items = [self.enforce_stock_longitudinal, self.live_learning_gas, large_reset_button(), self.bosch_a_radar]
    self.update_settings()

  def _set_live_gas(self, value):
    save_live_gas_choice(ui_state, value)
    self.update_settings()

  def _set_radar(self, value):
    save_radar_choice(ui_state, value)
    self.update_settings()

  def _request_onroad_cycle(self):
    # Bosch (alphaLongitudinalAvailable): the CarController applies the change through an onroad cycle
    # itself, after handing longitudinal control back to the stock radar; cycling from here first would
    # stop card before the radar is transmitting again and latch a brake-module fault for the drive.
    # Nidec has no radar to hand back and no controller-side handling, so the cycle is requested here.
    if ui_state.CP is not None and ui_state.CP.alphaLongitudinalAvailable:
      return
    ui_state.params.put_bool("OnroadCycleRequested", True)

  def _on_enable_enforce_stock_longitudinal(self, state: bool):
    if state:
      def confirm_callback(result: int):
        if result == DialogResult.CONFIRM:
          ui_state.params.put_bool("HondaEnforceStockLongitudinal", True)
          self._request_onroad_cycle()
        else:
          self.enforce_stock_longitudinal.action_item.set_state(False)

      content = (f"<h1>{self.enforce_stock_longitudinal.title}</h1><br>" +
                 f"<p>{self.enforce_stock_longitudinal.description}</p>")

      dlg = ConfirmDialog(content, tr("Enable"), rich=True)
      gui_app.set_modal_overlay(dlg, callback=confirm_callback)

    else:
      ui_state.params.put_bool("HondaEnforceStockLongitudinal", False)
      self._request_onroad_cycle()

  def update_settings(self):
    self.live_learning_gas.action_item.set_enabled(
      ui_state.is_offroad() and live_gas_available(ui_state.CP, ui_state.CP_SP, ui_state.params))
    self.live_learning_gas.action_item.set_state(ui_state.params.get_bool("HondaLiveLearningGas"))
    self.bosch_a_radar.action_item.set_enabled(getattr(ui_state.CP, "brand", None) == "honda")
    self.bosch_a_radar.action_item.set_state(bool(ui_state.params.get("HondaBoschARadar", return_default=True)))
