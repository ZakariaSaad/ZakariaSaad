"""Navigation driving selection, separate from ordinary Selector 21 bundle slots."""
from pathlib import Path

from openpilot.selfdrive.navigation.activation import admit_candidate, candidate_status, list_candidates, validate_admission
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import require_offroad


def _selection_identity(params):
  """Notice an SSH/UI selection or withdrawal during slow file admission."""
  values = (params.get('NavigationModelCandidate'), params.get_bool('NavigationModelDrivingEnabled'))
  if not hasattr(params, 'get_param_path'):
    return values
  files = []
  for key in ('NavigationModelCandidate', 'NavigationModelDrivingEnabled'):
    try:
      stat = Path(params.get_param_path(key)).stat()
      files.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
    except FileNotFoundError:
      files.append(None)
  return values, tuple(files)


def select_candidate(params, candidate_id: str, *, offroad: bool) -> None:
  if not offroad:
    raise NavigationModelDisabled('Navigation model selection is available only while off road')
  if candidate_id not in {candidate['id'] for candidate in list_candidates()}:
    raise NavigationModelDisabled('Unknown navigation model candidate')
  # Selecting a candidate for preparation does not select it for inference.
  params.put_bool('NavigationModelDrivingEnabled', False, block=True)
  params.put('NavigationModelCandidate', candidate_id, block=True)


def set_driving_enabled(params, enabled: bool, *, offroad: bool, candidate_id: str | None = None) -> None:
  """Shared Settings/SSH transaction. A failed admission never changes selection.

  Disabling always withdraws permission, including during a transition on road.
  Enabling hashes the existing registered files without executing model code.
  The driving flag is written last, after a second fresh off-road check.
  """
  if not enabled:
    params.put_bool('NavigationModelDrivingEnabled', False, block=True)
    if params.get_bool('NavigationModelDrivingEnabled'):
      raise NavigationModelDisabled('Could not persist Navigation Model Driving OFF')
    return
  if not offroad:
    raise NavigationModelDisabled('Enable navigation driving only while off road')
  status = candidate_status(params, candidate_id=candidate_id)
  if not status['eligible']:
    raise NavigationModelDisabled(status['reason'])
  try:
    selection_identity = _selection_identity(params)
    require_offroad(params)
    permit = admit_candidate(params, candidate_id=candidate_id)
    require_offroad(params)
    validate_admission(permit, params)
    if _selection_identity(params) != selection_identity:
      raise NavigationModelDisabled('Navigation selection or driving permission changed during admission; activation cancelled')
  except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
    raise NavigationModelDisabled(str(error)) from error
  previous = params.get('NavigationModelCandidate')
  try:
    params.put('NavigationModelCandidate', permit.model_id, block=True)
    if params.get('NavigationModelCandidate') != permit.model_id:
      raise OSError('Could not persist the admitted navigation model selection')
    params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    if not params.get_bool('NavigationModelDrivingEnabled'):
      raise OSError('Could not persist Navigation Model Driving ON')
  except (OSError, RuntimeError, ValueError, TypeError) as error:
    # Never leave a partially written activation requested after an I/O failure.
    params.put_bool('NavigationModelDrivingEnabled', False, block=True)
    if previous is None:
      params.remove('NavigationModelCandidate')
    else:
      params.put('NavigationModelCandidate', previous, block=True)
    raise NavigationModelDisabled(str(error)) from error


def selection_description(params) -> str:
  status = candidate_status(params)
  requested = params.get_bool('NavigationModelDrivingEnabled')
  state = 'Requested on' if requested and status['eligible'] else 'Blocked' if requested else 'Off'
  registration = 'Registered' if status.get('registered') else 'Not registered'
  software = 'Software integration reviewed' if status.get('software_compatible') else 'Software review incomplete'
  artifact = 'Compiler/artifact evidence reviewed' if status.get('artifact_evidence_reviewed') else 'Artifact evidence incomplete'
  runtime = status.get('runtime_compatibility', 'unverified')
  return f"{status['display_name']} — {state}. {registration}. {software}. {artifact}. Runtime compatibility: {runtime}. {status['reason']}"
