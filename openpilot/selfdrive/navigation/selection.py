"""Navigation driving selection, separate from ordinary Selector 21 bundle slots."""
from openpilot.selfdrive.navigation.activation import candidate_status, list_candidates
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled


def select_candidate(params, candidate_id: str, *, offroad: bool) -> None:
  if not offroad:
    raise NavigationModelDisabled('Navigation model selection is available only while off road')
  if candidate_id not in {candidate['id'] for candidate in list_candidates()}:
    raise NavigationModelDisabled('Unknown navigation model candidate')
  # Selecting a candidate for preparation does not select it for inference.
  params.put_bool('NavigationModelDrivingEnabled', False, block=True)
  params.put('NavigationModelCandidate', candidate_id, block=True)


def set_driving_enabled(params, enabled: bool, *, offroad: bool) -> None:
  if not offroad:
    raise NavigationModelDisabled('Change navigation driving selection only while off road')
  if enabled:
    status = candidate_status(params)
    if not status['eligible']:
      raise NavigationModelDisabled(status['reason'])
  params.put_bool('NavigationModelDrivingEnabled', bool(enabled), block=True)


def selection_description(params) -> str:
  status = candidate_status(params)
  requested = params.get_bool('NavigationModelDrivingEnabled')
  state = 'Requested on' if requested and status['eligible'] else 'Blocked' if requested else 'Off'
  return f"{status['display_name']} — {state}. {status['reason']}"
