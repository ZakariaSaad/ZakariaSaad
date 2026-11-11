from openpilot.cereal import log

from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigToggle, BigParamControl, BigMultiParamToggle
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.mici.layouts.settings.initial_cruise_speed import InitialCruiseSpeedButton
from openpilot.nrdr.ui.honda_longitudinal import (
  EXPERIMENTAL_DESCRIPTION, donor_bosch_a_mode, personality_value, save_experimental_choice,
)

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants


class TogglesLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._personality_options = ["aggressive", "standard", "relaxed", "econ"]
    self._personality_toggle = BigMultiParamToggle("following distance", "LongitudinalPersonality", self._personality_options,
                                                   description="Standard is recommended.\n" +
                                                               "Aggressive follows closer, with firmer gas and braking.\n" +
                                                               "Relaxed leaves more space.\n" +
                                                               "Use the steering wheel distance button on supported cars.")
    self._experimental_mode = BigToggle("experimental mode", initial_state=ui_state.params.get_bool("ExperimentalMode"),
                                        toggle_callback=self._set_experimental_mode, description=EXPERIMENTAL_DESCRIPTION)
    self._experimental_mode.set_visible(lambda: donor_bosch_a_mode(ui_state.CP))
    self._experimental_mode.set_enabled(lambda: donor_bosch_a_mode(ui_state.CP))
    self._initial_cruise_speed = InitialCruiseSpeedButton()
    self._initial_cruise_speed.set_visible(lambda: ui_state.CP is not None and ui_state.has_longitudinal_control and not ui_state.CP.pcmCruise)
    is_metric_toggle = BigToggle("use metric units", initial_state=ui_state.params.get_bool("IsMetric"), toggle_callback=self._on_metric_toggle)
    ldw_toggle = BigParamControl("lane departure warnings", "IsLdwEnabled",
                                 description="Warn when you drift across a detected lane line.\n" +
                                             "Only above 31 mph (50 km/h), with no turn signal.")
    always_on_dm_toggle = BigParamControl("always-on driver monitor", "AlwaysOnDM", description="Monitor the driver even when sunnypilot is not engaged.")
    record_front = BigParamControl("record & upload cabin camera", "RecordFront",
                                   description_icon=gui_app.texture("icons_mici/settings/device/cameras.png", 64, 64),
                                   toggle_callback=restart_needed_callback, description="Upload cabin camera data to help improve driver monitoring.")
    record_mic = BigParamControl("record & upload mic audio", "RecordAudio", description_icon=gui_app.texture("icons_mici/microphone.png", 64, 64),
                                 toggle_callback=restart_needed_callback,
                                 description="Record microphone audio while driving.\n" +
                                             "Audio is included in dashcam videos in sunnylink.")
    enable_openpilot = BigParamControl("enable sunnypilot", "OpenpilotEnabledToggle", toggle_callback=restart_needed_callback,
                                       description="Enable to use sunnypilot driver assistance.\n" +
                                                   "Disable to use your car's stock driver assistance.")


    self._scroller.add_widgets([
      self._personality_toggle,
      self._experimental_mode,
      self._initial_cruise_speed,
      is_metric_toggle,
      ldw_toggle,
      always_on_dm_toggle,
      record_front,
      record_mic,
      enable_openpilot,
    ])

    # Toggle lists
    self._refresh_toggles = (
      ("ExperimentalMode", self._experimental_mode),
      ("IsMetric", is_metric_toggle),
      ("IsLdwEnabled", ldw_toggle),
      ("AlwaysOnDM", always_on_dm_toggle),
      ("RecordFront", record_front),
      ("RecordAudio", record_mic),
      ("OpenpilotEnabledToggle", enable_openpilot),
    )

    enable_openpilot.set_enabled(lambda: not ui_state.engaged)
    record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))
    record_mic.set_enabled(lambda: not ui_state.engaged)

    if ui_state.params.get_bool("ShowDebugInfo"):
      gui_app.set_show_touches(True)
      gui_app.set_show_fps(True)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    super()._update_state()
    self._personality_toggle._options = self._personality_options[:4 if donor_bosch_a_mode(ui_state.CP) else 3]
    if self._personality_toggle.value not in self._personality_toggle._options:
      self._personality_toggle.set_value(self._personality_toggle._options[-1])

    if ui_state.sm.updated["selfdriveState"]:
      personality = personality_value(ui_state.CP, PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality])
      if personality != ui_state.personality and ui_state.started:
        self._personality_toggle.set_value(self._personality_toggle._options[personality])
      ui_state.personality = personality

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _on_metric_toggle(self, state: bool):
    ui_state.params.put_bool("IsMetric", state, block=True)
    self._update_toggles()

  def _set_experimental_mode(self, value):
    save_experimental_choice(ui_state, value)
    self._experimental_mode.set_checked(ui_state.params.get_bool("ExperimentalMode"))

  def _update_toggles(self):
    ui_state.update_params()
    self._initial_cruise_speed.refresh()
    self._personality_toggle._options = self._personality_options[:4 if donor_bosch_a_mode(ui_state.CP) else 3]
    selected = personality_value(ui_state.CP, ui_state.params.get("LongitudinalPersonality", return_default=True))
    self._personality_toggle.set_value(self._personality_options[selected])

    if ui_state.CP is not None:
      self._personality_toggle.set_visible(ui_state.has_longitudinal_control)

    # Refresh toggles from params to mirror external changes
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))
