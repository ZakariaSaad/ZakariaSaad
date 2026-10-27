"""One off-road selection queue for SunnyLink, Models, downloads and the CLI.

The queue records explicit user intent, never infers activation from a catalog
refresh or a reboot. Artifact preparation happens before publishing any selection.
"""
from contextlib import contextmanager
import copy
import fcntl
from pathlib import Path
import threading
import uuid

from openpilot.selfdrive.navigation.activation import admit_candidate, candidate_status, validate_admission
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID, RETIRED_MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import require_offroad

NAV_REF = 'navigation:' + MODEL_ID
DEFAULT_REF = 'Default'
REQUEST_KEY = 'ModelManager_SelectionRequest'
STATE_KEY = 'ModelManager_SelectionState'
SLOTS = {'qcom': 'ModelManager_ActiveBundle', 'chestnut': 'ModelManager_ActiveBundleChestnut'}
SELECTION_KEYS = ('NavigationModelCandidate', 'NavigationModelManifest', 'NavigationModelDrivingEnabled',
                  'ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut', 'ModelRunnerTypeCache')
_THREAD_LOCK = threading.RLock()


@contextmanager
def _lock(params):
  with _THREAD_LOCK:
    if not hasattr(params, 'get_param_path'):
      yield
      return
    path = Path(params.get_param_path(STATE_KEY)).parent / '.model-selection.lock'
    with path.open('a') as stream:
      fcntl.flock(stream, fcntl.LOCK_EX)
      try:
        yield
      finally:
        fcntl.flock(stream, fcntl.LOCK_UN)


def _record(params, key):
  value = params.get(key)
  return value if isinstance(value, dict) else {}


def _put(params, key, value):
  if value is None:
    params.remove(key)
  elif key == 'NavigationModelDrivingEnabled':
    params.put_bool(key, value, block=True)
  else:
    params.put(key, value, block=True)
  actual = params.get_bool(key) if key == 'NavigationModelDrivingEnabled' else params.get(key)
  if actual != value:
    raise NavigationModelDisabled('Could not persist ' + key)


def _current(params, request):
  return bool(request.get('id')) and _record(params, REQUEST_KEY).get('id') == request['id']


def _state(params, request, state, message):
  _put(params, STATE_KEY, {'id': request.get('id', ''), 'ref': request.get('ref', ''),
                          'source': request.get('source', 'chestnut'), 'state': state, 'message': message})


def _selected_ref(params, source='chestnut'):
  if source == 'chestnut' and params.get('NavigationModelCandidate') == MODEL_ID and params.get_bool('NavigationModelDrivingEnabled'):
    return NAV_REF
  bundle = params.get(SLOTS[source])
  return bundle.get('ref', DEFAULT_REF) if isinstance(bundle, dict) and bundle else DEFAULT_REF


def selection_status(params):
  record = _record(params, STATE_KEY)
  request = _record(params, REQUEST_KEY)
  source = request.get('source', record.get('source', 'chestnut'))
  source = source if source in SLOTS else 'chestnut'
  selected = _selected_ref(params, source)
  state, message = record.get('state', 'selected'), record.get('message', 'Selected model applies on the next drive.')
  if request and request.get('id') != record.get('id'):
    state, message = 'pending', 'Model selection is pending; current selections remain in use.'
  runtime = _record(params, 'NavigationModelRuntimeState')
  if not request and record.get('state') not in ('failed', 'cancelled'):
    if selected == NAV_REF:
      if not params.get_bool('IsOffroad') and runtime.get('candidate_id') == MODEL_ID and runtime.get('role') in ('ordinary_small', 'failed'):
        state, message = 'failed', 'Corrected Bigv2 is unavailable; the ordinary small model is carrying this drive.'
      elif not params.get_bool('IsOffroad') and runtime.get('candidate_id') == MODEL_ID and runtime.get('role') == 'candidate':
        state, message = 'active', 'Corrected Bigv2 is active. Driver supervision remains required.'
      elif not params.get_bool('IsOffroad') and params.get_bool('ChestnutLoading'):
        state, message = 'loading', 'Loading corrected Bigv2; this does not engage controls.'
      else:
        state, message = 'selected', 'Corrected Bigv2 selected; navigation driving requested. Applies on the next drive.'
    elif params.get('NavigationModelCandidate') == MODEL_ID:
      state, message = 'selected', 'Navigation driving is OFF. The ordinary model selection is in use.'
  return {**record, 'state': state, 'message': message, 'source': source,
          'ref': request.get('ref', record.get('ref', selected)), 'selected_ref': selected,
          'navigation_enabled': params.get_bool('NavigationModelDrivingEnabled'),
          'request_id': request.get('id', ''), 'revision': record.get('id', '')}


def _guard(params, offroad):
  if not offroad:
    raise NavigationModelDisabled('Model changes are available only while genuinely off road and disengaged')
  require_offroad(params)


def _ordinary_bundle(params, source, ref):
  from openpilot.sunnypilot.models.fetcher import get_cached_bundles
  for bundle in get_cached_bundles(params, source):
    if bundle.ref == ref:
      return bundle.to_dict()
  raise NavigationModelDisabled('The requested ordinary model is not in the current device catalog; refresh the model list')


def request_model_selection(params, *, source, ref, offroad, origin='device_ui', expected_revision=None):
  if source not in SLOTS or not isinstance(ref, str) or not ref:
    raise NavigationModelDisabled('Invalid model selection')
  if ref in (RETIRED_MODEL_ID, 'navigation:' + RETIRED_MODEL_ID):
    raise NavigationModelDisabled('Original Bigv2 is retired; select corrected Bigv2')
  if ref.startswith('navigation:') and (source != 'chestnut' or ref != NAV_REF):
    raise NavigationModelDisabled('Unknown navigation model selection')
  if ref not in (NAV_REF, DEFAULT_REF):
    _ordinary_bundle(params, source, ref)
  _guard(params, offroad)
  with _lock(params):
    require_offroad(params)
    if expected_revision is not None and expected_revision != _record(params, STATE_KEY).get('id', ''):
      raise NavigationModelDisabled('Stale model selection; refresh SunnyLink before selecting again')
    # Repeated in-flight delivery is idempotent. Completed explicit selections
    # may intentionally re-enable navigation after a deliberate OFF.
    current = _record(params, REQUEST_KEY)
    if current.get('phase') == 'committing':
      raise NavigationModelDisabled('Model selection is being committed; wait for recovery/completion')
    if current.get('ref') == ref and current.get('source') == source and current.get('phase') != 'committing':
      return selection_status(params)
    request = {'schema': 1, 'id': uuid.uuid4().hex, 'source': source, 'ref': ref, 'origin': origin,
               'phase': 'pending', 'enable': ref == NAV_REF}
    previous = {key: copy.deepcopy(params.get(key)) for key in (REQUEST_KEY, STATE_KEY, 'ModelManager_DownloadRef')}
    try:
      _put(params, REQUEST_KEY, request)
      _put(params, 'ModelManager_DownloadRef', None if ref == DEFAULT_REF else ref)
      _state(params, request, 'pending', 'Selection pending; the previous model remains selected until preparation succeeds.')
    except Exception as error:
      failures = []
      for key, value in previous.items():
        try:
          if params.get(key) != value:
            _put(params, key, value)
        except Exception as restore_error:
          failures.append(str(restore_error))
      detail = '; request rollback failed: ' + '; '.join(failures) if failures else ''
      raise NavigationModelDisabled(str(error) + detail) from error
    return selection_status(params)


def cancel_model_selection(params, *, offroad=True, origin='device_ui', expected_revision=None):
  _guard(params, offroad)
  with _lock(params):
    require_offroad(params)
    if expected_revision is not None and expected_revision != _record(params, STATE_KEY).get('id', ''):
      raise NavigationModelDisabled('Stale model selection; refresh SunnyLink before selecting again')
    request = _record(params, REQUEST_KEY)
    if request.get('phase') == 'committing':
      raise NavigationModelDisabled('Model selection is being committed; wait for completion')
    _put(params, REQUEST_KEY, None)
    _put(params, 'ModelManager_DownloadRef', None)
    _state(params, {'id': uuid.uuid4().hex, 'source': request.get('source', 'chestnut'), 'ref': _selected_ref(params)},
           'cancelled', 'Selection cancelled; previous model settings preserved.')
    return selection_status(params)


def request_navigation_disabled(params, *, offroad=True, origin='device_ui', expected_revision=None):
  _guard(params, offroad)
  with _lock(params):
    require_offroad(params)
    if expected_revision is not None and expected_revision != _record(params, STATE_KEY).get('id', ''):
      raise NavigationModelDisabled('Stale model selection; refresh SunnyLink before selecting again')
    request = _record(params, REQUEST_KEY)
    if request.get('phase') == 'committing':
      raise NavigationModelDisabled('Model selection is being committed; wait for completion')
    # Explicit OFF cancels an unfinished corrected selection, not an ordinary
    # download requested separately. Never alter NavigationEnabled (map display).
    if request.get('ref') == NAV_REF:
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == NAV_REF:
        _put(params, 'ModelManager_DownloadRef', None)
    _put(params, 'NavigationModelDrivingEnabled', False)
    _state(params, {'id': uuid.uuid4().hex, 'ref': _selected_ref(params), 'source': 'chestnut'},
           'selected', 'Navigation driving is OFF; ordinary model selections are preserved.')
    return selection_status(params)


def report_download_state(params, ref, state, message=''):
  with _lock(params):
    request = _record(params, REQUEST_KEY)
    if request.get('ref') != ref:
      return
    _state(params, request, state, message or ('Preparing ordinary model: ' + state))
    if state in ('failed', 'cancelled'):
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request.get('ref'):
        _put(params, 'ModelManager_DownloadRef', None)


def _restore(params, previous):
  failures = []
  try:
    _put(params, 'NavigationModelDrivingEnabled', False)
  except Exception as error:
    failures.append(str(error))
  for key, value in previous.items():
    if key == 'NavigationModelDrivingEnabled':
      continue
    try:
      if params.get(key) != value:
        _put(params, key, value)
    except Exception as error:
      failures.append(str(error))
  if not failures and previous.get('NavigationModelCandidate') != RETIRED_MODEL_ID:
    try:
      _put(params, 'NavigationModelDrivingEnabled', bool(previous.get('NavigationModelDrivingEnabled')))
    except Exception as error:
      failures.append(str(error))
  if failures:
    raise NavigationModelDisabled('Selection rollback needs attention: ' + '; '.join(failures))


def _commit(params, request, changes, permit=None):
  require_offroad(params)
  with _lock(params):
    require_offroad(params)
    if not _current(params, request):
      return False
    if permit is not None:
      validate_admission(permit, params, allow_unselected_manifest=True)
    previous = {key: copy.deepcopy(params.get(key)) for key in SELECTION_KEYS}
    previous['NavigationModelDrivingEnabled'] = params.get_bool('NavigationModelDrivingEnabled')
    journal = {**request, 'phase': 'committing', 'previous': previous}
    _put(params, REQUEST_KEY, journal)
    try:
      _put(params, 'NavigationModelDrivingEnabled', False)
      for key, value in changes.items():
        if key != 'NavigationModelDrivingEnabled':
          _put(params, key, value)
      _put(params, 'NavigationModelDrivingEnabled', bool(changes.get('NavigationModelDrivingEnabled', False)))
      # Completed state is written before clearing the journal. Recovery can
      # distinguish an interrupted successful commit from a partial write.
      _state(params, request, 'selected', 'Model selection completed; applies on the next drive. Controls are not engaged.')
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request['ref']:
        _put(params, 'ModelManager_DownloadRef', None)
      return True
    except Exception as error:
      try:
        _restore(params, previous)
      except Exception as rollback_error:
        _state(params, request, 'failed', f'{error}; {rollback_error}')
        raise
      _state(params, request, 'failed', str(error))
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request['ref']:
        _put(params, 'ModelManager_DownloadRef', None)
      raise


def _default_ready(source):
  from openpilot.common.basedir import BASEDIR
  directory = Path(BASEDIR) / 'openpilot/selfdrive/modeld/models'
  names = ['big_driving_tinygrad.pkl'] if source == 'chestnut' else ['driving_tinygrad.pkl']
  for name in names:
    path = directory / name
    with path.open('rb') as stream:
      header = stream.read(100)
    if not header or header.startswith(b'version https://git-lfs.github.com/'):
      raise NavigationModelDisabled('The installed default model is incomplete: ' + name)


def _recover_or_migrate(params):
  with _lock(params):
    request = _record(params, REQUEST_KEY)
    if request.get('origin') == 'retired_bigv2_migration' and request.get('phase') == 'pending':
      _put(params, 'NavigationModelDrivingEnabled', False)
      if params.get('NavigationModelCandidate') == RETIRED_MODEL_ID:
        _put(params, 'NavigationModelCandidate', None)
    if request.get('phase') == 'committing':
      state = _record(params, STATE_KEY)
      if state.get('id') != request.get('id') or state.get('state') != 'selected':
        _restore(params, request.get('previous', {}))
        _state(params, request, 'failed', 'An interrupted model switch was rolled back. Select the intended model again.')
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request.get('ref'):
        _put(params, 'ModelManager_DownloadRef', None)
      return
    if request or params.get('NavigationModelCandidate') != RETIRED_MODEL_ID:
      return
    enabled = params.get_bool('NavigationModelDrivingEnabled')
    # Record old intent before withdrawing it, so power loss cannot accidentally
    # re-enable the retired model or repeatedly reinterpret the saved request.
    request = {'schema': 1, 'id': uuid.uuid4().hex, 'source': 'chestnut', 'ref': NAV_REF,
               'origin': 'retired_bigv2_migration', 'phase': 'pending', 'enable': enabled}
    _put(params, REQUEST_KEY, request)
    _put(params, 'NavigationModelDrivingEnabled', False)
    params.remove('NavigationModelCandidate')
    _state(params, request, 'pending', 'Original Bigv2 retired. Checking the corrected local model; navigation driving is OFF.')


def process_pending_selection(params):
  """Manager tick. True means this request is handled/deferred, not a download."""
  if not params.get_bool('IsOffroad') or params.get_bool('IsEngaged'):
    return True
  try:
    require_offroad(params)
  except NavigationModelDisabled:
    return True
  _recover_or_migrate(params)
  request = _record(params, REQUEST_KEY)
  if not request:
    return False
  if request.get('ref') not in (NAV_REF, DEFAULT_REF):
    # DownloadRef clears on reboot/road transition; only a persisted explicit
    # request may restore it. No request means no automatic model reselection.
    if params.get('ModelManager_DownloadRef') != request.get('ref'):
      _put(params, 'ModelManager_DownloadRef', request['ref'])
    return False
  with _lock(params):
    if not _current(params, request):
      return True
    _state(params, request, 'preparing', 'Checking existing model files; previous selection remains in use.')
  try:
    if request['ref'] == NAV_REF:
      registered = candidate_status(params, candidate_id=MODEL_ID)
      if registered.get('approved') is False:
        raise NavigationModelDisabled('Explicit user authorization for Navigation Model Driving is still required')
      if request.get('origin') == 'retired_bigv2_migration':
        # Updating an old saved ID is not consent to a large download. Retain
        # the previous local-only migration and leave navigation OFF on failure.
        if not registered.get('registered'):
          from openpilot.selfdrive.navigation.preparation import prepare_existing_model
          prepare_existing_model(params, is_offroad=True, model_id=MODEL_ID, publish_selection=False)
      else:
        from openpilot.selfdrive.navigation.download import prepare_downloaded_model
        def progress(state, message):
          with _lock(params):
            if _current(params, request):
              _state(params, request, state, message)
        with _lock(params):
          if not _current(params, request):
            return True
          _put(params, 'ModelManager_DownloadRef', NAV_REF)
        prepare_downloaded_model(params, current=lambda: _current(params, request), progress=progress, registration=registered)
      permit = admit_candidate(params, candidate_id=MODEL_ID)
      changes = {'NavigationModelManifest': str(permit.manifest_path), 'NavigationModelCandidate': MODEL_ID,
                 'NavigationModelDrivingEnabled': bool(request.get('enable', True))}
      _commit(params, request, changes, permit)
    else:
      if request['source'] == 'qcom' and params.get_bool('NavigationModelDrivingEnabled'):
        raise NavigationModelDisabled('Corrected Bigv2 requires the configured native small-model fallback. ' +
                                      'Turn navigation driving OFF before selecting the stock small default.')
      _default_ready(request['source'])
      changes = {SLOTS[request['source']]: None, 'ModelRunnerTypeCache': None}
      if request['source'] == 'chestnut':
        changes['NavigationModelCandidate'] = None
      else:
        changes['NavigationModelDrivingEnabled'] = params.get_bool('NavigationModelDrivingEnabled')
      _commit(params, request, changes)
  except Exception as error:
    with _lock(params):
      if _current(params, request):
        if _record(params, REQUEST_KEY).get('phase') != 'committing':
          _put(params, REQUEST_KEY, None)
          if params.get('ModelManager_DownloadRef') == request['ref']:
            _put(params, 'ModelManager_DownloadRef', None)
        _state(params, request, 'failed', str(error))
    return True
  return True


def complete_ordinary_selection(params, source, bundle_dict, expected_ref):
  """Called only after manager SHA checks; refuses stale or on-road completion."""
  if source not in SLOTS or not isinstance(bundle_dict, dict) or bundle_dict.get('ref') != expected_ref:
    raise NavigationModelDisabled('Ordinary model completion identity mismatch')
  if params.get('ModelManager_DownloadRef') != expected_ref:
    return False
  if not params.get_bool('IsOffroad') or params.get_bool('IsEngaged'):
    report_download_state(params, expected_ref, 'pending', 'Downloaded; waiting for a genuine off-road transition.')
    return False
  try:
    require_offroad(params)
  except NavigationModelDisabled as error:
    report_download_state(params, expected_ref, 'pending', str(error))
    return False
  with _lock(params):
    request = _record(params, REQUEST_KEY)
    if request and (request.get('ref') != expected_ref or request.get('source') != source):
      return False
    if not request:
      # Existing manager/auto-fallback requests also get the same guarded commit.
      request = {'schema': 1, 'id': uuid.uuid4().hex, 'source': source, 'ref': expected_ref, 'phase': 'pending', 'origin': 'manager'}
      _put(params, REQUEST_KEY, request)
    # The next drive can start before the manager's next validation tick. Let
    # startup derive the runner from the new slot, never from the previous model.
    changes = {SLOTS[source]: bundle_dict, 'ModelRunnerTypeCache': None}
    if source == 'chestnut':
      changes['NavigationModelCandidate'] = None
    else:
      changes['NavigationModelDrivingEnabled'] = params.get_bool('NavigationModelDrivingEnabled')
  if source == 'qcom' and params.get_bool('NavigationModelDrivingEnabled'):
    from openpilot.common.hardware.hw import Paths
    from openpilot.selfdrive.navigation.software_compatibility import validate_ordinary_fallback_bundle
    validate_ordinary_fallback_bundle(bundle_dict, Path(Paths.model_root()))
  return _commit(params, request, changes)


def select_navigation_now(params, *, enabled, offroad):
  """Synchronous CLI compatibility path through the same explicit request queue."""
  if not enabled:
    return request_navigation_disabled(params, offroad=offroad, origin='cli')
  queued = request_model_selection(params, source='chestnut', ref=NAV_REF, offroad=offroad, origin='cli')
  process_pending_selection(params)
  with _lock(params):
    state = selection_status(params)
    if state.get('revision') == queued['request_id'] and state['state'] in ('selected', 'active', 'loading') and state['navigation_enabled']:
      return state
    request = _record(params, REQUEST_KEY)
    if request.get('id') == queued['request_id'] and request.get('phase') != 'committing':
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request['ref']:
        _put(params, 'ModelManager_DownloadRef', None)
      _state(params, request, 'cancelled', 'Selection did not complete; previous model settings preserved. Select again while off road.')
    raise NavigationModelDisabled(state['message'])


def model_restore_settings(params, updates):
  """Never restore approval/manifest pointers or a stale pending command."""
  protected = {'NavigationModelCandidate', 'NavigationModelManifest', 'NavigationModelDrivingEnabled',
               'NavigationModelRuntimeState', REQUEST_KEY, STATE_KEY, 'ModelManager_DownloadRef',
               'ModelManager_SelectionStatus', 'ModelManager_SelectionRevision',
               'ModelManager_ActiveBundleUSBGPU', 'ModelManager_BundleSlotsMigrated', 'ModelRunnerTypeCache'}
  remaining = {key: value for key, value in updates.items() if key not in protected}
  if sum(key in remaining for key in SLOTS.values()) > 1:
    raise NavigationModelDisabled('Restore one model selection at a time; current selections are preserved')
  for source, key in SLOTS.items():
    if key in remaining:
      value = remaining.pop(key)
      if not isinstance(value, dict):
        raise NavigationModelDisabled('Invalid backed-up ordinary model selection')
      request_model_selection(params, source=source, ref=value.get('ref', DEFAULT_REF), offroad=True, origin='restore')
  return remaining
