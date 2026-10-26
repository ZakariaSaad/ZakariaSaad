"""Background requests for the same model-selection service used by SunnyLink.

The manager performs file preparation and downloads. This worker only keeps the
fresh off-road check out of the render loop; it never writes model parameters.
"""
import threading

from openpilot.sunnypilot.models.selection import cancel_model_selection, request_model_selection, request_navigation_disabled, selection_status


class ModelSelectionRequest:
  def __init__(self, params, *, source: str, ref: str | None, offroad: bool, cancel: bool = False):
    self.params, self.source, self.ref = params, source, ref
    self.error: str | None = None
    self.result: dict = {}
    self.finished = threading.Event()

    def run():
      try:
        if cancel:
          self.result = cancel_model_selection(params, offroad=offroad, origin='device_ui')
        elif ref is None:
          self.result = request_navigation_disabled(params, offroad=offroad)
        else:
          self.result = request_model_selection(params, source=source, ref=ref, offroad=offroad, origin='device_ui')
      except Exception as error:
        self.error = str(error)
      finally:
        self.finished.set()

    self.worker = threading.Thread(target=run, daemon=True, name='model-selection-request')
    self.worker.start()

  def description(self) -> str:
    if self.error:
      return self.error
    current = selection_status(self.params)
    status = current if current.get('source') == self.source and current.get('ref') == self.ref else self.result
    return selection_message(status)


def selection_message(status: dict) -> str:
  state, message = status.get('state', ''), status.get('message', '')
  return f'{state.capitalize()}: {message}' if state and message else message or state.capitalize()
