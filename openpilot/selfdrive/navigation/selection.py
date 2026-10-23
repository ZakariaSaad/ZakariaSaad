"""Navigation driving selection, separate from ordinary Selector 21 bundle slots."""
from pathlib import Path

from openpilot.selfdrive.navigation.activation import admit_candidate, candidate_manifest, candidate_status, list_candidates, validate_admission
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import require_offroad


def _selection_identity(params):
  """Notice an SSH/UI selection or withdrawal during slow file admission."""
  values = (params.get('NavigationModelCandidate'), params.get('NavigationModelManifest'), params.get_bool('NavigationModelDrivingEnabled'))
  if not hasattr(params, 'get_param_path'):
    return values
  files = []
  for key in ('NavigationModelCandidate', 'NavigationModelManifest', 'NavigationModelDrivingEnabled'):
    try:
      stat = Path(params.get_param_path(key)).stat()
      files.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
    except FileNotFoundError:
      files.append(None)
  return values, tuple(files)


def _restore_selection(params, previous):
  """Withdraw permission and attempt every pointer restore despite I/O failures."""
  failures = []
  for key, value in (('NavigationModelDrivingEnabled', False), *previous.items()):
    try:
      if key == 'NavigationModelDrivingEnabled':
        params.put_bool(key, False, block=True)
      elif params.get(key) != value:
        if value is None:
          params.remove(key)
        else:
          params.put(key, value, block=True)
    except Exception as error:
      failures.append(f'{key}: {error}')
    # A native write may silently fail or write successfully before raising.
    # Always inspect its result, and still attempt the other pointer afterward.
    try:
      actual = params.get_bool(key) if key == 'NavigationModelDrivingEnabled' else params.get(key)
      if actual != value:
        failures.append(f'Could not restore {key}')
    except Exception as error:
      failures.append(f'Could not verify restored {key}: {error}')
  return failures


def _write_selection(params, candidate_id, manifest):
  if manifest is not None:
    params.put('NavigationModelManifest', str(manifest), block=True)
    if params.get('NavigationModelManifest') != str(manifest):
      raise OSError('Could not persist the registered navigation model manifest')
  params.put('NavigationModelCandidate', candidate_id, block=True)
  if params.get('NavigationModelCandidate') != candidate_id:
    raise OSError('Could not persist the admitted navigation model selection')


def select_candidate(params, candidate_id: str, *, offroad: bool) -> None:
  if not offroad:
    raise NavigationModelDisabled('Navigation model selection is available only while off road')
  if candidate_id not in {candidate['id'] for candidate in list_candidates()}:
    raise NavigationModelDisabled('Unknown navigation model candidate')
  require_offroad(params)
  manifest = candidate_manifest(params, candidate_id)
  if manifest is not None:
    status = candidate_status(params, candidate_id=candidate_id)
    if not status['registered'] or not status['installed']:
      raise NavigationModelDisabled(status['reason'])
  previous = {key: params.get(key) for key in ('NavigationModelCandidate', 'NavigationModelManifest')}
  # Selecting a candidate for preparation does not select it for inference.
  params.put_bool('NavigationModelDrivingEnabled', False, block=True)
  if params.get_bool('NavigationModelDrivingEnabled'):
    raise NavigationModelDisabled('Could not persist Navigation Model Driving OFF')
  try:
    _write_selection(params, candidate_id, manifest)
  except (OSError, RuntimeError, ValueError, TypeError) as error:
    failures = _restore_selection(params, previous)
    detail = f'; navigation selection rollback failed: {"; ".join(failures)}' if failures else ''
    raise NavigationModelDisabled(f'{error}{detail}') from error


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
    validate_admission(permit, params, allow_unselected_manifest=True)
    if _selection_identity(params) != selection_identity:
      raise NavigationModelDisabled('Navigation selection or driving permission changed during admission; activation cancelled')
  except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
    raise NavigationModelDisabled(str(error)) from error
  previous = {key: params.get(key) for key in ('NavigationModelCandidate', 'NavigationModelManifest')}
  try:
    _write_selection(params, permit.model_id, getattr(permit, 'manifest_path', None))
    params.put_bool('NavigationModelDrivingEnabled', True, block=True)
    if not params.get_bool('NavigationModelDrivingEnabled'):
      raise OSError('Could not persist Navigation Model Driving ON')
  except (OSError, RuntimeError, ValueError, TypeError) as error:
    # Never leave a partially written activation requested after an I/O failure.
    failures = _restore_selection(params, previous)
    detail = f'; navigation selection rollback failed: {"; ".join(failures)}' if failures else ''
    raise NavigationModelDisabled(f'{error}{detail}') from error


def selection_description(params) -> str:
  status = candidate_status(params)
  requested = params.get_bool('NavigationModelDrivingEnabled')
  state = 'Requested on' if requested and status['eligible'] else 'Blocked' if requested else 'Off'
  registration = 'Registered' if status.get('registered') else 'Not registered'
  software = 'Software integration reviewed' if status.get('software_compatible') else 'Software review incomplete'
  artifact = 'Compiler/artifact evidence reviewed' if status.get('artifact_evidence_reviewed') else 'Artifact evidence incomplete'
  runtime = status.get('runtime_compatibility', 'unverified')
  return f"{status['display_name']} — {state}. {registration}. {software}. {artifact}. Runtime compatibility: {runtime}. {status['reason']}"
