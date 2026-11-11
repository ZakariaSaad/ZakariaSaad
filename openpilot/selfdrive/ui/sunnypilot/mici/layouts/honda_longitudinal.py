"""Comma 4 presentations of the donor's saved Honda longitudinal choices."""
from openpilot.nrdr.ui.honda_longitudinal import (
  MVL_DESCRIPTION, RADAR_DESCRIPTION, live_gas_available, live_gas_status, save_live_gas_choice, save_radar_choice, save_stock_choice,
)
from openpilot.nrdr.ui.settings.live_gas_reset import mici_reset_button
from openpilot.selfdrive.ui.mici.widgets.button import BigToggle
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.scroller import NavScroller


class LiveGasControl(BigToggle):
  def __init__(self):
    super().__init__(tr("Live Learning Gas"), initial_state=ui_state.params.get_bool("HondaLiveLearningGas"),
                     toggle_callback=self._select, font_size=30, description=tr(MVL_DESCRIPTION))
    self.set_enabled(lambda: ui_state.is_offroad() and live_gas_available(ui_state.CP, ui_state.CP_SP, ui_state.params))
    self._refresh()

  def _title_width_hint(self):
    return max(1, int(self._rect.width - self.LABEL_HORIZONTAL_PADDING - self._txt_enabled_toggle.width - 12))

  def _refresh(self):
    self.set_checked(ui_state.params.get_bool("HondaLiveLearningGas"))
    self.set_value(tr(live_gas_status(ui_state.CP, ui_state.params, is_offroad=ui_state.is_offroad())))

  def _select(self, value):
    save_live_gas_choice(ui_state, value)
    self._refresh()

  def _update_state(self):
    super()._update_state()
    self._refresh()


class BoschRadarControl(BigToggle):
  def __init__(self):
    super().__init__(tr("Honda Bosch-A Radar"),
                     initial_state=bool(ui_state.params.get("HondaBoschARadar", return_default=True)),
                     toggle_callback=self._select, font_size=30, description=tr(RADAR_DESCRIPTION))
    self.set_enabled(lambda: getattr(ui_state.CP, "brand", None) == "honda")

  def _select(self, value):
    save_radar_choice(ui_state, value)
    self.set_checked(bool(ui_state.params.get("HondaBoschARadar", return_default=True)))

  def _update_state(self):
    super()._update_state()
    self.set_checked(bool(ui_state.params.get("HondaBoschARadar", return_default=True)))


class HondaLongitudinalLayout(NavScroller):
  def __init__(self):
    super().__init__()
    self._stock = BigToggle(tr("Enforce Factory Longitudinal Control"),
                            initial_state=ui_state.params.get_bool("HondaEnforceStockLongitudinal"),
                            toggle_callback=self._set_stock, font_size=30,
                            description=tr("Use factory Honda ACC instead of openpilot gas and brake control. Change while parked."))
    self._stock.set_enabled(lambda: ui_state.is_offroad() and live_gas_available(ui_state.CP, ui_state.CP_SP, ui_state.params))
    self._scroller.add_widgets([self._stock, LiveGasControl(), mici_reset_button(), BoschRadarControl()])

  def _set_stock(self, value):
    save_stock_choice(ui_state, value)
    self._stock.set_checked(ui_state.params.get_bool("HondaEnforceStockLongitudinal"))

  def _update_state(self):
    super()._update_state()
    self._stock.set_checked(ui_state.params.get_bool("HondaEnforceStockLongitudinal"))
