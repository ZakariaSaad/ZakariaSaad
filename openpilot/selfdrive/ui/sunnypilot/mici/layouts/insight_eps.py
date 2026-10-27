"""On-device access to the same guarded Insight owner settings as SunnyLink."""
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigToggle, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationCircleButton, BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import FIRMWARE_BUILD
from openpilot.sunnypilot.sunnylink import insight_eps
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


class FirmwareOwnerConfirmation(NavScroller):
  def __init__(self, on_confirm):
    super().__init__()
    self._scroller.add_widgets([
      GreyBigButton("owner installation statement", FIRMWARE_BUILD),
      GreyBigButton("", "Confirm only if this exact corrected build is already installed in this Insight."),
      GreyBigButton("", "The old incorrect image has the same part number. A part-number match is not firmware verification."),
      GreyBigButton("", "This records your statement. It does not read back or flash firmware, enable the controller, or engage controls."),
      BigConfirmationCircleButton("I confirm this build is installed",
                                  gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 64, 64),
                                  lambda: self.dismiss(on_confirm)),
    ])


class InsightEpsLayout(NavScroller):
  def __init__(self):
    super().__init__()
    self._eps_toggle = BigToggle("Insight EPS controller", toggle_callback=self._set_enabled,
                              description="Use the custom Insight controller on the next drive. OFF uses the ordinary sp-steer controller.")
    self._status = BigButton("requested and effective status", "tap for details")
    self._status_details = ""
    self._status.set_click_callback(lambda: gui_app.push_widget(BigDialog("Insight EPS status", self._status_details)))
    confirm = BigButton("recover installed-firmware confirmation", FIRMWARE_BUILD,
                        description="Explicit owner installation attestation for this Insight, not an ECU firmware readback.")
    confirm.set_click_callback(lambda: gui_app.push_widget(FirmwareOwnerConfirmation(self._confirm)))
    confirm.set_enabled(lambda: ui_state.is_offroad() and not ui_state.engaged)
    self._eps_toggle.set_enabled(lambda: ui_state.is_offroad() and not ui_state.engaged)
    self._scroller.add_widgets([self._eps_toggle, self._status, confirm])

  def _perform(self, operation):
    try:
      operation()
    except Exception as error:
      gui_app.push_widget(BigDialog("Insight EPS request refused", str(error)))
    self._refresh()

  def _set_enabled(self, enabled):
    self._perform(lambda: insight_eps.apply_enabled(ui_state.params, enabled))

  def _confirm(self):
    self._perform(lambda: insight_eps.confirm_installed_firmware(ui_state.params, True))

  def _refresh(self):
    state = insight_eps.status(ui_state.params)
    self._eps_toggle.set_checked(state['requested'])
    self._status_details = state['description']
    requested = 'ON' if state['requested'] else 'OFF'
    effective = 'active' if state['effective'] else 'off road / next drive' if ui_state.is_offroad() else 'ordinary controller'
    self._status.set_value(f'Requested {requested}; {effective}. Tap for details.')

  def show_event(self):
    super().show_event()
    self._refresh()
