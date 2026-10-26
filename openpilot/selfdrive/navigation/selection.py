"""Compatibility entrypoints using the shared off-road model-selection service."""
from openpilot.selfdrive.navigation.activation import candidate_status
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled


def select_candidate(params, candidate_id: str, *, offroad: bool) -> None:
  """An explicit corrected-model selection prepares it and requests driving ON."""
  set_driving_enabled(params, True, offroad=offroad, candidate_id=candidate_id)


def set_driving_enabled(params, enabled: bool, *, offroad: bool, candidate_id: str | None = None) -> None:
  from openpilot.sunnypilot.models.selection import select_navigation_now
  identifier = candidate_id if candidate_id is not None else MODEL_ID
  if identifier != MODEL_ID:
    raise NavigationModelDisabled('Original Bigv2 is retired or the navigation candidate is unknown')
  select_navigation_now(params, enabled=enabled, offroad=offroad)


def selection_description(params) -> str:
  status = candidate_status(params)
  requested = params.get_bool('NavigationModelDrivingEnabled')
  state = 'Requested on' if requested and status['eligible'] else 'Blocked' if requested else 'Off'
  registration = 'Registered' if status.get('registered') else 'Not registered'
  software = 'Software integration reviewed' if status.get('software_compatible') else 'Software review incomplete'
  artifact = 'Compiler/artifact evidence reviewed' if status.get('artifact_evidence_reviewed') else 'Artifact evidence incomplete'
  runtime = status.get('runtime_compatibility', 'unverified')
  return f"{status['display_name']} — {state}. {registration}. {software}. {artifact}. Runtime compatibility: {runtime}. {status['reason']}"
