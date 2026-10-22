"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import threading
import time

import pyray as rl

from openpilot.cereal import custom
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, BigDialog
from openpilot.sunnypilot.models.helpers import ACTIVE_BUNDLE_KEYS, get_selected_bundle
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigToggle
from openpilot.selfdrive.navigation.activation import candidate_status, list_candidates
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled
from openpilot.selfdrive.navigation.selection import select_candidate, selection_description, set_driving_enabled
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.selfdrive.ui.sunnypilot.model_info import (active_source, big_model_state, bundles_for_source, carrying_model,
                                                           default_model_name, model_cache_size_mb, model_info, queued_name,
                                                           refresh_in_progress, refresh_model_list)
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller


class ExistingBigv2RegistrationDialog(BigDialog):
  """Hash existing files off the render thread; result widgets stay on the UI thread."""
  def __init__(self, params, *, is_offroad: bool):
    super().__init__(tr("Register existing Bigv2"), tr("Checking existing model and camera warp. Navigation driving stays off."))
    self._finished = threading.Event()
    self._error: str | None = None
    self._shown_result = False
    def run():
      try:
        from openpilot.selfdrive.navigation.preparation import prepare_existing_model
        receipt = prepare_existing_model(params, is_offroad=is_offroad)
        if receipt.get('prepared') is not True or receipt.get('activation_allowed') is not False:
          raise ValueError('Existing model registration was not completed')
      except Exception as error:
        self._error = str(error)
      finally:
        self._finished.set()
    self.worker = threading.Thread(target=run, daemon=True, name='navigation-model-registration')
    self.worker.start()

  def _update_state(self):
    super()._update_state()
    if self._finished.is_set() and not self._shown_result:
      self._shown_result = True
      self._card.set_text(tr("Registration failed") if self._error else tr("Bigv2 registered"))
      self._card.set_value(self._error or tr("Existing model and warp registered. Navigation driving stays off."))


class NavigationActivationDialog(BigDialog):
  """Run the shared file admission off the render thread; no model execution."""
  def __init__(self, params, *, is_offroad: bool, on_finished):
    super().__init__(tr("Checking navigation activation"), tr("Checking approval and existing registered files. No model is downloaded or executed."))
    self._finished = threading.Event()
    self._error: str | None = None
    self._shown_result = False
    self._on_finished = on_finished
    def run():
      try:
        set_driving_enabled(params, True, offroad=is_offroad)
      except Exception as error:
        self._error = str(error)
      finally:
        self._finished.set()
    self.worker = threading.Thread(target=run, daemon=True, name='navigation-model-activation')
    self.worker.start()

  def _back_enabled(self):
    return not self.worker.is_alive()

  def _update_state(self):
    super()._update_state()
    if self._finished.is_set() and not self._shown_result:
      self._shown_result = True
      self._card.set_text(tr("Navigation activation blocked") if self._error else tr("Navigation driving requested"))
      self._card.set_value(self._error or tr("Admission passed. The selected navigation model is requested for the next drive."))
      self._on_finished()


class NavigationDrivingToggle(BigToggle):
  """Reflect permission withdrawn or requested through the shared SSH command."""
  def _update_state(self):
    super()._update_state()
    self.set_checked(ui_state.params.get_bool('NavigationModelDrivingEnabled'))


def _model_info() -> tuple[str, str, str]:
  """(active model, info header, info text) for the panel. Runner-matched: the
  active line names what actually drives, and a notable big-model state takes
  the info pair."""
  source, active_name, other_name = model_info()
  state = big_model_state()
  _, _, carry_display = carrying_model()
  if carry_display is None:
    big = get_selected_bundle(ui_state.params, "chestnut")
    carry_display = big.displayName if big else default_model_name("chestnut")
  active_text = (carry_display or active_name).lower()
  if state == 'failed':
    return active_text, tr("big model"), tr("unavailable")
  if state == 'loading':
    return active_text, tr("big model"), tr("getting ready")
  header = tr("small model") if source == "chestnut" else tr("big model")
  return active_text, header, other_name.lower()


class CurrentModelInfo(Widget):
  def __init__(self):
    super().__init__()

    self.set_rect(rl.Rectangle(0, 0, 360, 180))

    header_color = rl.Color(255, 255, 255, int(255 * 0.9))
    subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)
    active_text, info_header, info_text = _model_info()
    self.current_model_header = UnifiedLabel(tr("active model"), 48, max_width=max_width, text_color=header_color, font_weight=FontWeight.DISPLAY)
    self.current_model_text = UnifiedLabel(active_text, 32, max_width=max_width, text_color=subheader_color, font_weight=FontWeight.ROMAN, scroll=True)

    self.info_header = UnifiedLabel(info_header, 48, max_width=max_width, text_color=header_color, font_weight=FontWeight.DISPLAY)
    self.info_text = UnifiedLabel(info_text, 32, max_width=max_width, text_color=subheader_color, font_weight=FontWeight.ROMAN, scroll=True)

  def _render(self, _):
    self.current_model_header.set_position(self._rect.x + 20, self._rect.y - 10)
    self.current_model_header.render()

    self.current_model_text.set_position(self._rect.x + 20, self._rect.y + 68 - 25)
    self.current_model_text.render()

    self.info_header.set_position(self._rect.x + 20, self._rect.y + 114 - 30)
    self.info_header.render()

    self.info_text.set_position(self._rect.x + 20, self._rect.y + 161 - 25)
    self.info_text.render()

class ModelsLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self.focused_widget = None

    self.current_model_info = CurrentModelInfo()
    self._download_progress = "."
    self._download_frame = 0
    self._was_downloading = False
    self._selection_source: str | None = None
    self._navigation_registration: ExistingBigv2RegistrationDialog | None = None
    self._navigation_activation: NavigationActivationDialog | None = None

    self.select_model_btn = BigButton(tr("select model"))
    self.select_model_btn.set_click_callback(self._show_folders)

    self.refresh_btn = BigButton(tr("refresh models"))
    self.refresh_btn.set_click_callback(self._refresh_models)
    self._refresh_start: float | None = None

    self.cancel_download_btn = BigButton(tr("cancel download"))
    self.cancel_download_btn.set_click_callback(lambda: ui_state.params.remove("ModelManager_DownloadRef"))

    self.clear_cache_btn = BigButton(tr("clear cache"), value=f"{model_cache_size_mb():.1f} MB")
    self.clear_cache_btn.set_click_callback(self._confirm_clear_cache)
    self._cache_size_time = 0.0

    self.navigation_models_btn = BigButton(tr("navigation models"), value=tr("off by default"))
    self.navigation_models_btn.set_click_callback(self._show_navigation_models)

    self.main_items = [self.current_model_info, self.select_model_btn, self.navigation_models_btn,
                       self.cancel_download_btn, self.refresh_btn, self.clear_cache_btn]
    self._scroller.add_widgets(self.main_items)

  def _show_navigation_models(self):
    status = candidate_status(ui_state.params)
    details = BigButton(tr("navigation model status"), value=status['display_name'], font_size=36)
    details.set_click_callback(lambda: gui_app.push_widget(BigDialog(
      tr("Navigation models"), selection_description(ui_state.params))))
    enable = NavigationDrivingToggle(tr("navigation driving"), initial_state=ui_state.params.get_bool('NavigationModelDrivingEnabled'))
    enable.set_enabled(lambda: not self._navigation_activation_running() and
                       ((ui_state.is_offroad() and candidate_status(ui_state.params)['eligible']) or
                        ui_state.params.get_bool('NavigationModelDrivingEnabled')))

    def on_enable(checked):
      if checked:
        if self._navigation_activation_running():
          return
        self._navigation_activation = NavigationActivationDialog(
          ui_state.params, is_offroad=ui_state.is_offroad(),
          on_finished=lambda: enable.set_checked(ui_state.params.get_bool('NavigationModelDrivingEnabled')))
        gui_app.push_widget(self._navigation_activation)
        enable.set_checked(ui_state.params.get_bool('NavigationModelDrivingEnabled'))
        return
      try:
        set_driving_enabled(ui_state.params, False, offroad=ui_state.is_offroad())
      except NavigationModelDisabled as error:
        gui_app.push_widget(BigDialog(tr("Navigation model unavailable"), str(error)))
      enable.set_checked(ui_state.params.get_bool('NavigationModelDrivingEnabled'))
    enable._toggle_callback = on_enable
    items = [details, enable]
    for candidate in list_candidates():
      button = BigButton(candidate['display_name'], value=tr("select navigation candidate"), font_size=36)
      button.set_enabled(ui_state.is_offroad)
      button.set_click_callback(lambda candidate_id=candidate['id']: self._select_navigation_candidate(candidate_id))
      items.append(button)
    self.navigation_register_btn = BigButton(tr("Register existing Bigv2"), value=tr("use installed files"), font_size=36)
    self.navigation_register_btn.set_enabled(lambda: ui_state.is_offroad() and not self._navigation_registration_running())
    self.navigation_register_btn.set_click_callback(self._register_existing_navigation)
    items.append(self.navigation_register_btn)
    self._push_selection_view(items)

  def _navigation_activation_running(self):
    return self._navigation_activation is not None and self._navigation_activation.worker.is_alive()

  def _navigation_registration_running(self):
    return self._navigation_registration is not None and self._navigation_registration.worker.is_alive()

  def _register_existing_navigation(self):
    offroad = ui_state.is_offroad()
    if not offroad:
      gui_app.push_widget(BigDialog(tr("Registration unavailable"), tr("Register navigation models only while off road.")))
      return
    if self._navigation_registration_running():
      return
    self._navigation_registration = ExistingBigv2RegistrationDialog(ui_state.params, is_offroad=offroad)
    gui_app.push_widget(self._navigation_registration)

  @property
  def model_manager(self):
    return ui_state.sm["modelManagerSP"]

  def _get_grouped_bundles(self, bundles, favorites = None):
    folders = {}
    for bundle in bundles:
      folder = next((override.value for override in bundle.overrides if override.key == "folder"), "")
      folders.setdefault(folder, []).append(bundle)

    if favorites:
      for fav_bundle in [bundle for bundle in bundles if bundle.ref in favorites]:
        folders.setdefault("favorites", []).append(fav_bundle)

    return folders

  def _push_selection_view(self, items):
    scroller = NavScroller()
    scroller._scroller.add_widgets(items)
    gui_app.push_widget(scroller)

  def _show_folders(self):
    self.focused_widget = self.select_model_btn

    hardware_btns = []
    active = active_source()
    for source, label in (("qcom", tr("small models")), ("chestnut", tr("big models"))):
      bundle = get_selected_bundle(ui_state.params, source)
      value = (bundle.internalName if bundle else default_model_name(source)).lower()
      navigation = candidate_status(ui_state.params) if source == 'chestnut' else None
      if navigation and navigation.get('selected'):
        value = navigation['display_name'] + (tr(" (navigation requested)") if navigation['enabled'] else tr(" (local; driving off)"))
      if source == active and (not navigation or not navigation.get('selected') or carrying_model()[1] == navigation['id']):
        value += f" ({tr('active')})"
      btn = BigButton(label.lower(), value=value)
      btn.set_click_callback(lambda s=source: self._select_hardware(s))
      hardware_btns.append(btn)
    self._push_selection_view(hardware_btns)

  def _select_hardware(self, source):
    self._selection_source = source

    favs = ui_state.params.get("ModelManager_Favs")
    favorites = set(favs.split(';')) if favs else set()

    bundles = bundles_for_source(source)
    navigation = candidate_status(ui_state.params, candidate_id='big-nav-v2') if source == 'chestnut' else None
    if not bundles and navigation is None:
      gui_app.push_widget(BigDialog(title=tr("No models available"),
                                    description=tr("No models are available for this hardware yet. Connect to the internet and refresh the model list.")))
      return
    folders = self._get_grouped_bundles(bundles, favorites)

    folder_buttons = []
    default_btn = BigButton(default_model_name(source).lower())
    default_btn.set_click_callback(lambda s=source: self._select_default(s))
    folder_buttons.append(default_btn)

    if navigation is not None:
      value = tr("installed local · navigation aware") if navigation.get('installed') else tr("local model not installed")
      value += tr(" · eligible") if navigation['eligible'] else tr(" · activation blocked")
      button = BigButton(navigation['display_name'], value=value, font_size=36)
      button.set_enabled(lambda: ui_state.is_offroad() and
                         candidate_status(ui_state.params, candidate_id='big-nav-v2').get('installed', False))
      button.set_click_callback(lambda: self._select_navigation_candidate('big-nav-v2', require_installed=True))
      folder_buttons.append(button)

    for folder in sorted(folders.keys(), key=lambda f: max((bundle.index for bundle in folders[f]), default=-1), reverse=True):
      btn = BigButton(folder.lower())
      btn.set_click_callback(lambda f=folder: self._select_folder(f))
      if folder.lower() == "favorites":
        folder_buttons.insert(0, btn)
      else:
        folder_buttons.append(btn)
    self._push_selection_view(folder_buttons)

  def _pop_to_main(self):
    gui_app.pop_widgets_to(self)
    self._scroller.scroll_panel.set_offset(0.0)

  def _select_model(self, bundle):
    if self._selection_source == 'chestnut':
      set_driving_enabled(ui_state.params, False, offroad=ui_state.is_offroad())
      ui_state.params.remove('NavigationModelCandidate')
    ui_state.params.put("ModelManager_DownloadRef", bundle.ref)
    self._pop_to_main()

  def _select_default(self, source):
    if source == 'chestnut':
      set_driving_enabled(ui_state.params, False, offroad=ui_state.is_offroad())
      ui_state.params.remove('NavigationModelCandidate')
    ui_state.params.remove(ACTIVE_BUNDLE_KEYS[source])
    self._pop_to_main()

  def _select_navigation_candidate(self, candidate_id, *, require_installed=False):
    try:
      if require_installed and not candidate_status(ui_state.params, candidate_id=candidate_id).get('installed'):
        raise NavigationModelDisabled('The registered local Bigv2 model is not installed')
      select_candidate(ui_state.params, candidate_id, offroad=ui_state.is_offroad())
      self._pop_to_main()
      gui_app.push_widget(BigDialog(tr("Navigation model selected"), selection_description(ui_state.params) + "\n" +
        tr("Driving stays off. Use Models → navigation models → navigation driving to request activation after approval.")))
    except NavigationModelDisabled as error:
      gui_app.push_widget(BigDialog(tr("Navigation model unavailable"), str(error)))

  def _confirm_clear_cache(self):
    icon = gui_app.texture("icons_mici/settings/network/new/trash.png", 54, 64)
    gui_app.push_widget(BigConfirmationDialog(f"{tr('slide to')}\n{tr('clear cache')}", icon,
                                              lambda: ui_state.params.put_bool("ModelManager_ClearCache", True), red=True))

  def _refresh_models(self):
    refresh_model_list()
    self._refresh_start = time.monotonic()

  def _select_folder(self, folder_name):
    source = self._selection_source
    if source is None:  # folders are only reachable after picking a hardware
      return
    favs = ui_state.params.get("ModelManager_Favs")
    favorites = set(favs.split(';')) if favs else set()

    folders = self._get_grouped_bundles(bundles_for_source(source), favorites)
    bundles = sorted(folders.get(folder_name, []), key=lambda b: b.index, reverse=True)

    btns = []
    for bundle in bundles:
      btn = BigButton(bundle.displayName.lower())
      btn.set_click_callback(lambda b=bundle: self._select_model(b))
      btns.append(btn)
    self._push_selection_view(btns)

  def hide_event(self):
    super().hide_event()
    if self._was_downloading:
      device.set_override_interactive_timeout(None)
      self._was_downloading = False

  def _update_state(self):
    super()._update_state()

    self.navigation_models_btn.set_value(tr('requested') if ui_state.params.get_bool('NavigationModelDrivingEnabled') else tr('off'))
    self.select_model_btn.set_enabled(ui_state.is_offroad())
    self.cancel_download_btn.set_visible(False)
    self.current_model_info.current_model_header._shimmer = False
    self.current_model_info.info_header._shimmer = False

    manager = self.model_manager
    self._download_frame += 1
    should_update = self._download_frame % (gui_app.target_fps / 2) == 0
    if should_update:
      self._download_progress = self._download_progress + "." if len(self._download_progress) < 3 else ""

    is_downloading = (manager.selectedBundle
                      and manager.selectedBundle.status == custom.ModelManagerSP.DownloadStatus.downloading)
    if self._was_downloading and not is_downloading:
      device.set_override_interactive_timeout(None)
    self._was_downloading = is_downloading

    # manager is offroad-only, so an onroad clear would never be serviced
    clearing = ui_state.params.get_bool("ModelManager_ClearCache")
    self.clear_cache_btn.set_enabled(ui_state.is_offroad() and not is_downloading and not clearing)
    if clearing:
      self.clear_cache_btn.set_value(tr("clearing..."))
      self._cache_size_time = 0.0  # refresh the size as soon as clearing finishes
    elif (now := time.monotonic()) - self._cache_size_time > 0.5:
      self._cache_size_time = now
      self.clear_cache_btn.set_value(f"{model_cache_size_mb():.1f} MB")

    # manager is offroad-only, so a refresh queued onroad would never be serviced
    refreshing = refresh_in_progress(self._refresh_start)
    self.refresh_btn.set_enabled(ui_state.is_offroad() and not is_downloading and not refreshing)
    self.refresh_btn.set_value(tr("fetching...") if refreshing else "")

    self.current_model_info.current_model_header.set_text(tr("active model"))
    active_text, info_header, info_text = _model_info()
    self.current_model_info.current_model_text.set_text(active_text)
    self.current_model_info.info_header.set_text(info_header)
    self.current_model_info.info_text.set_text(info_text)

    if manager.selectedBundle and manager.selectedBundle.status == custom.ModelManagerSP.DownloadStatus.failed:
      self.current_model_info.info_header.set_text(tr("error") + self._download_progress)
      self.current_model_info.info_text.set_text(tr("download failed"))

    elif manager.selectedBundle and manager.selectedBundle.status == custom.ModelManagerSP.DownloadStatus.downloading:
      self.cancel_download_btn.set_visible(True)
      device.set_override_interactive_timeout(5)
      progress = 0.0
      count = 0
      verifying = False
      for model in manager.selectedBundle.models:
        count += 1
        p = model.artifact.downloadProgress
        if p.status in (custom.ModelManagerSP.DownloadStatus.downloading,
                        custom.ModelManagerSP.DownloadStatus.verifying):
          progress += p.progress
          verifying = verifying or p.status == custom.ModelManagerSP.DownloadStatus.verifying
        elif p.status in (custom.ModelManagerSP.DownloadStatus.downloaded,
                          custom.ModelManagerSP.DownloadStatus.cached):
          progress += 100.0

      self.current_model_info.current_model_header.set_text(tr("verifying") if verifying else tr("downloading"))
      self.cancel_download_btn.set_text(tr("cancel verification") if verifying else tr("cancel download"))
      self.current_model_info.current_model_header._shimmer = True
      name_text = manager.selectedBundle.internalName.lower()
      if queued := queued_name(manager.selectedBundle.ref):
        name_text += f"  |  {queued.lower()} {tr('queued')}"
      self.current_model_info.current_model_text.set_text(name_text)
      self.current_model_info.info_header.set_text(tr("progress") + self._download_progress)
      self.current_model_info.info_header._shimmer = True
      self.current_model_info.info_text.set_text(f"{progress/count:.2f}%")

    elif manager.selectedBundle and manager.selectedBundle.status == custom.ModelManagerSP.DownloadStatus.downloaded:
      self.current_model_info.info_header.set_text(tr("downloaded"))
      self.current_model_info.info_text.set_text(tr("downloaded"))
