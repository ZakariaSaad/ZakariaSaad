"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

import pyray as rl

from openpilot.cereal import custom
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, BigDialog
from openpilot.sunnypilot.models.helpers import get_selected_bundle
from openpilot.sunnypilot.models.selection import NAV_REF, selection_status
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigToggle
from openpilot.selfdrive.navigation.activation import candidate_status, list_candidates
from openpilot.selfdrive.navigation.selection import selection_description
from openpilot.selfdrive.ui.sunnypilot.model_selection import ModelSelectionRequest
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.selfdrive.ui.sunnypilot.model_info import (active_source, big_model_state, bundles_for_source, carrying_model,
                                                           default_model_name, model_cache_size_mb, model_info, queued_name,
                                                           refresh_in_progress, refresh_model_list)
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller


class ModelSelectionDialog(BigDialog):
  """Keep the shared off-road request guard out of the render loop."""
  def __init__(self, params, *, source: str, ref: str | None, is_offroad: bool, on_finished, cancel: bool = False):
    super().__init__(tr("Model selection"), tr("Checking off-road status. The current model stays selected until the replacement is ready."))
    self.request = ModelSelectionRequest(params, source=source, ref=ref, offroad=is_offroad, cancel=cancel)
    self.worker = self.request.worker
    self._on_finished, self._shown_result = on_finished, False

  def _back_enabled(self):
    return not self.worker.is_alive()

  def _update_state(self):
    super()._update_state()
    if self.request.finished.is_set():
      self._card.set_text(tr("Model selection blocked") if self.request.error else tr("Model selection"))
      self._card.set_value(self.request.description())
      if not self._shown_result:
        self._shown_result = True
        self._on_finished()


class NavigationDrivingToggle(BigToggle):
  """Reflect permission withdrawn or requested through the shared selection service."""
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
  navigation = candidate_status(ui_state.params)
  if ui_state.is_offroad() and navigation.get('selected') and navigation['enabled']:
    active_text = navigation['display_name'].lower()
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
    self._model_selection: ModelSelectionDialog | None = None

    self.select_model_btn = BigButton(tr("select model"))
    self.select_model_btn.set_click_callback(self._show_folders)

    self.refresh_btn = BigButton(tr("refresh models"))
    self.refresh_btn.set_click_callback(self._refresh_models)
    self._refresh_start: float | None = None

    self.cancel_download_btn = BigButton(tr("cancel download"))
    self.cancel_download_btn.set_click_callback(self._cancel_selection)

    self.clear_cache_btn = BigButton(tr("clear cache"), value=f"{model_cache_size_mb():.1f} MB")
    self.clear_cache_btn.set_click_callback(self._confirm_clear_cache)
    self._cache_size_time = 0.0

    self.navigation_models_btn = BigButton(tr("navigation driving"), value=tr("off"))
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
    enable.set_enabled(lambda: not self._model_selection_running() and ui_state.is_offroad())

    def on_enable(checked):
      self._request_selection('chestnut', NAV_REF if checked else None,
                              on_finished=lambda: enable.set_checked(ui_state.params.get_bool('NavigationModelDrivingEnabled')))
      enable.set_checked(ui_state.params.get_bool('NavigationModelDrivingEnabled'))
    enable._toggle_callback = on_enable
    self._push_selection_view([details, enable])

  def _model_selection_running(self):
    return self._model_selection is not None and self._model_selection.worker.is_alive()

  def _request_selection(self, source: str, ref: str | None, *, on_finished=lambda: None, cancel: bool = False):
    if self._model_selection_running():
      return
    self._model_selection = ModelSelectionDialog(
      ui_state.params, source=source, ref=ref, is_offroad=ui_state.is_offroad(), on_finished=on_finished, cancel=cancel)
    gui_app.push_widget(self._model_selection)

  def _cancel_selection(self):
    request = selection_status(ui_state.params)
    self._request_selection(request.get('source', 'chestnut'), request.get('ref'), cancel=True)

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
        value += f" ({tr('selected') if ui_state.is_offroad() else tr('active')})"
      btn = BigButton(label.lower(), value=value)
      btn.set_click_callback(lambda s=source: self._select_hardware(s))
      hardware_btns.append(btn)
    self._push_selection_view(hardware_btns)

  def _select_hardware(self, source):
    self._selection_source = source

    favs = ui_state.params.get("ModelManager_Favs")
    favorites = set(favs.split(';')) if favs else set()

    bundles = bundles_for_source(source)
    navigation = [candidate_status(ui_state.params, candidate_id=candidate['id']) for candidate in list_candidates()] if source == 'chestnut' else []
    if not bundles and not navigation:
      gui_app.push_widget(BigDialog(title=tr("No models available"),
                                    description=tr("No models are available for this hardware yet. Connect to the internet and refresh the model list.")))
      return
    folders = self._get_grouped_bundles(bundles, favorites)

    folder_buttons = []
    default_btn = BigButton(default_model_name(source).lower())
    default_btn.set_click_callback(lambda s=source: self._select_default(s))
    folder_buttons.append(default_btn)

    for candidate in navigation:
      value = tr("installed local · navigation aware") if candidate.get('installed') else tr("local files checked on selection")
      value += tr(" · enables navigation driving after verification")
      button = BigButton(candidate['display_name'], value=value, font_size=36)
      button.set_enabled(lambda: ui_state.is_offroad() and not self._model_selection_running())
      button.set_click_callback(lambda candidate_id=candidate['id']: self._select_navigation_candidate(candidate_id))
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
    self._pop_to_main()
    self._request_selection(self._selection_source, bundle.ref)

  def _select_default(self, source):
    self._pop_to_main()
    self._request_selection(source, 'Default')

  def _select_navigation_candidate(self, candidate_id):
    self._pop_to_main()
    self._request_selection('chestnut', 'navigation:' + candidate_id)

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
    self.select_model_btn.set_enabled(ui_state.is_offroad() and not self._model_selection_running())
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

    self.current_model_info.current_model_header.set_text(tr("selected model") if ui_state.is_offroad() else tr("active model"))
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

    requested = selection_status(ui_state.params)
    if requested.get('state') in ('pending', 'preparing', 'downloading'):
      self.cancel_download_btn.set_visible(True)
      self.cancel_download_btn.set_text(tr("cancel model selection"))
    self.cancel_download_btn.set_enabled(ui_state.is_offroad() and not self._model_selection_running())
    if requested.get('state') in ('pending', 'preparing', 'downloading', 'failed', 'cancelled') or requested.get('selected_ref') == NAV_REF:
      self.current_model_info.info_header.set_text(requested['state'])
      self.current_model_info.info_text.set_text(requested.get('message', ''))
