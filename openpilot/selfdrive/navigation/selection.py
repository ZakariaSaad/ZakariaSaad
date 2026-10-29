"""Optional CLI access to the shared off-road model-selection service."""
from openpilot.selfdrive.navigation.activation import candidate_status
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled


def select_candidate(params, candidate_id: str, *, offroad: bool) -> None:
  from openpilot.sunnypilot.models.navigation_bundle import NAV_REF
  from openpilot.sunnypilot.models.selection import select_model_now
  if candidate_id != MODEL_ID:
    raise NavigationModelDisabled('Original Bigv2 is retired or the navigation candidate is unknown')
  select_model_now(params, source='chestnut', ref=NAV_REF, offroad=offroad)


def selection_description(params) -> str:
  status = candidate_status(params)
  state = 'Selected' if status.get('selected') else 'Not selected'
  registration = 'Registered' if status.get('registered') else 'Not registered'
  runtime = status.get('runtime_compatibility', 'unverified')
  return f"{status['display_name']} — {state}. {registration}. Runtime compatibility: {runtime}. {status['reason']}"
