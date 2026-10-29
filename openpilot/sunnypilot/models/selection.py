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
from openpilot.sunnypilot.models.navigation_bundle import NAV_REF, navigation_bundle, is_navigation_selected

DEFAULT_REF = 'Default'
REQUEST_KEY = 'ModelManager_SelectionRequest'
STATE_KEY = 'ModelManager_SelectionState'
SLOTS = {'qcom': 'ModelManager_ActiveBundle', 'chestnut': 'ModelManager_ActiveBundleChestnut'}
SELECTION_KEYS = ('NavigationModelManifest',
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
  else:
    params.put(key, value, block=True)
  actual = params.get(key)
  if actual != value:
    raise NavigationModelDisabled('Could not persist ' + key)


def _current(params, request):
  return bool(request.get('id')) and _record(params, REQUEST_KEY).get('id') == request['id']


def _state(params, request, state, message):
  _put(params, STATE_KEY, {'id': request.get('id', ''), 'ref': request.get('ref', ''),
                          'source': request.get('source', 'chestnut'), 'state': state, 'message': message})


def _selected_ref(params, source='chestnut'):
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
  last_reported_ref = None
  if not params.get_bool('IsOffroad') and runtime.get('candidate_id') == MODEL_ID:
    role = runtime.get('role')
    if role == 'candidate':
      last_reported_ref = NAV_REF
    elif role == 'ordinary_small':
      last_reported_ref = _selected_ref(params, 'qcom')
    if not request and selected == NAV_REF:
      if role == 'candidate':
        state, message = 'reported', 'Last model-process report: corrected Bigv2 published predictions. Check current live health on the device.'
      elif role == 'ordinary_small':
        state, message = 'reported', 'Last model-process report: ordinary small fallback published predictions. Check current live health on the device.'
      elif role == 'failed':
        state, message = 'failed', 'Model prediction failed: ' + str(runtime.get('reason', 'unknown error'))
      elif role in ('loading', 'ready', 'small_loading'):
        state, message = 'loading', 'Preparing model predictions: ' + str(runtime.get('reason', role))
  if not request and record.get('state') not in ('failed', 'cancelled') and selected == NAV_REF and params.get_bool('IsOffroad'):
    state, message = 'selected', 'Corrected Bigv2 selected. Applies on the next drive; no separate activation is needed.'
  elif not request and selected == NAV_REF and not runtime and params.get_bool('ChestnutLoading'):
    state, message = 'loading', 'Loading corrected Bigv2; predictions are not yet confirmed.'
  return {**record, 'state': state, 'message': message, 'source': source,
          'ref': request.get('ref', record.get('ref', selected)), 'selected_ref': selected,
          'last_reported_ref': last_reported_ref, 'navigation_enabled': is_navigation_selected(params),
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
    # Repeated in-flight delivery is idempotent. Only an explicit selection
    # can replace the canonical slot after artifact preparation.
    current = _record(params, REQUEST_KEY)
    if current.get('phase') == 'committing':
      raise NavigationModelDisabled('Model selection is being committed; wait for recovery/completion')
    if current.get('ref') == ref and current.get('source') == source and current.get('phase') != 'committing':
      return selection_status(params)
    request = {'schema': 1, 'id': uuid.uuid4().hex, 'source': source, 'ref': ref, 'origin': origin,
               'phase': 'pending'}
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
  for key in SELECTION_KEYS:
    if key not in previous:
      continue
    try:
      if params.get(key) != previous[key]:
        _put(params, key, previous[key])
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
    journal = {**request, 'phase': 'committing', 'previous': previous}
    _put(params, REQUEST_KEY, journal)
    try:
      for key, value in changes.items():
        _put(params, key, value)
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
    if request.get('phase') == 'committing':
      state = _record(params, STATE_KEY)
      if state.get('id') != request.get('id') or state.get('state') != 'selected':
        _restore(params, request.get('previous', {}))
        _state(params, request, 'failed', 'An interrupted model switch was rolled back. Select the intended model again.')
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request.get('ref'):
        _put(params, 'ModelManager_DownloadRef', None)
  migrate_native_selection(params)


def migrate_native_selection(params):
  """Consume a completed legacy choice once; never infer selection from files.

  Called during manager startup before any model process and before off-road
  selection processing. The slot write is atomic, then the migration marker is
  persisted. A restart between those writes recognizes the canonical slot.
  """
  with _lock(params):
    if params.get_bool('ModelManager_NativeSelectionMigrated'):
      params.remove('NavigationModelDrivingEnabled')
      params.remove('NavigationModelCandidate')
      return
    request = _record(params, REQUEST_KEY)
    state = _record(params, STATE_KEY)
    # Never continue the former policy that converted retired model intent into
    # corrected-model activation. It was not a new explicit corrected choice.
    retired_request = request.get('origin') == 'retired_bigv2_migration'
    if retired_request:
      _put(params, REQUEST_KEY, None)
      if params.get('ModelManager_DownloadRef') == request.get('ref'):
        _put(params, 'ModelManager_DownloadRef', None)
      _state(params, request, 'cancelled', 'Original Bigv2 is retired. Select the corrected model explicitly.')
    completed = (not retired_request and not request and state.get('state') == 'selected' and
                 state.get('source') == 'chestnut' and state.get('ref') == NAV_REF and
                 params.get('NavigationModelCandidate') == MODEL_ID and
                 params.get_bool('NavigationModelDrivingEnabled') and bool(params.get('NavigationModelManifest')))
    if not is_navigation_selected(params) and completed:
      # Integrity is rechecked by the normal loader; migration preserves identity
      # even if removable storage or Chestnut is temporarily unavailable.
      _put(params, SLOTS['chestnut'], navigation_bundle())
      _put(params, 'ModelRunnerTypeCache', None)
    _put(params, 'ModelManager_NativeSelectionMigrated', True)
    params.remove('NavigationModelDrivingEnabled')
    params.remove('NavigationModelCandidate')


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
      changes = {'NavigationModelManifest': str(permit.manifest_path),
                 SLOTS['chestnut']: navigation_bundle(), 'ModelRunnerTypeCache': None}
      _commit(params, request, changes, permit)
    else:
      if request['source'] == 'qcom' and is_navigation_selected(params):
        raise NavigationModelDisabled('Corrected Bigv2 requires the configured native small-model fallback. ' +
                                      'Select another big model before selecting the stock small default.')
      _default_ready(request['source'])
      changes = {SLOTS[request['source']]: None, 'ModelRunnerTypeCache': None}
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
      # Upstream may prepare the ordinary small fallback when its slot is empty.
      # A bare/background big download is not an explicit selection request.
      if source != 'qcom' or params.get(SLOTS['qcom']):
        _put(params, 'ModelManager_DownloadRef', None)
        return False
      request = {'schema': 1, 'id': uuid.uuid4().hex, 'source': source, 'ref': expected_ref, 'phase': 'pending', 'origin': 'manager'}
      _put(params, REQUEST_KEY, request)
    # The next drive can start before the manager's next validation tick. Let
    # startup derive the runner from the new slot, never from the previous model.
    changes = {SLOTS[source]: bundle_dict, 'ModelRunnerTypeCache': None}
  if source == 'qcom' and is_navigation_selected(params):
    from openpilot.common.hardware.hw import Paths
    from openpilot.selfdrive.navigation.software_compatibility import validate_ordinary_fallback_bundle
    validate_ordinary_fallback_bundle(bundle_dict, Path(Paths.model_root()))
  return _commit(params, request, changes)


def select_model_now(params, *, source, ref, offroad):
  """Optional synchronous caller of the same explicit model selection queue."""
  queued = request_model_selection(params, source=source, ref=ref, offroad=offroad, origin='cli')
  process_pending_selection(params)
  with _lock(params):
    state = selection_status(params)
    if state.get('revision') == queued['request_id'] and state['state'] in ('selected', 'active', 'loading') and state['selected_ref'] == ref:
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
               'ModelManager_ActiveBundleUSBGPU', 'ModelManager_BundleSlotsMigrated', 'ModelManager_NativeSelectionMigrated', 'ModelRunnerTypeCache'}
  remaining = {key: value for key, value in updates.items() if key not in protected}
  if sum(key in remaining for key in SLOTS.values()) > 1:
    raise NavigationModelDisabled('Restore one model selection at a time; current selections are preserved')
  for source, key in SLOTS.items():
    if key in remaining:
      value = remaining.pop(key)
      if not isinstance(value, dict):
        raise NavigationModelDisabled('Invalid backed-up ordinary model selection')
      if value.get('ref') == NAV_REF:
        # A backup is not a fresh explicit native-model choice. Keep the current
        # model; selecting Bigv2 in Models performs registration/admission.
        continue
      request_model_selection(params, source=source, ref=value.get('ref', DEFAULT_REF), offroad=True, origin='restore')
  return remaining
