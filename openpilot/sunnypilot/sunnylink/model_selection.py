"""SunnyLink's existing model-picker protocol, backed by shared safe selection.

The hosted catalog overlay presents the native Bigv2 adapter. Accepted model
choices use the same canonical Chestnut slot as ordinary big models.
"""
import base64
import copy
import json
from pathlib import Path

from openpilot.selfdrive.navigation.corrected_bigv2 import ARTIFACT_SHA256, DISPLAY_NAME, MODEL_ID
from openpilot.selfdrive.navigation.candidate_identity import is_retired_bundle, RETIRED_ARTIFACT_SHA256
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.download import MODEL_URL, MODEL_SIZE, SIDECAR_URL, SIDECAR_SHA256, SIDECAR_SIZE

NAV_REF = 'navigation:' + MODEL_ID
RETIRED_ID = 'big-nav-v2'
RETIRED_SHA256 = RETIRED_ARTIFACT_SHA256
ARTIFACT = Path('/data/nav-bigv2/bigv2-amd-map-fp32.sptg')
STATUS_KEY = 'ModelManager_SelectionStatus'
REVISION_KEY = 'ModelManager_SelectionRevision'
WRITE_KEYS = frozenset(('ModelManager_DownloadRef', 'ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut',
                        REVISION_KEY))
PROTECTED_KEYS = frozenset(('NavigationModelDrivingEnabled', 'NavigationModelCandidate', 'NavigationModelManifest', 'NavigationModelRuntimeState',
                            'NavigationModelInputsEnabled', 'ModelManager_SelectionRequest', 'ModelManager_SelectionState', STATUS_KEY,
                            'ModelManager_ActiveBundleUSBGPU', 'ModelManager_BundleSlotsMigrated',
                            'ModelManager_NativeSelectionMigrated', 'ModelRunnerTypeCache'))


def _json(value, default=None):
  if isinstance(value, (bytes, str)):
    try:
      return json.loads(value)
    except (ValueError, UnicodeError):
      return default
  return value if value is not None else default


def _encode(value):
  return json.dumps(value, separators=(',', ':')).encode()


def _selection():
  # Keep the RPC import cheap, and avoid a model-manager import cycle.
  from openpilot.sunnypilot.models import selection
  return selection


def local_bundle() -> dict:
  """Hosted catalog metadata for the native adapter; no fabricated generation."""
  availability = 'installed local' if ARTIFACT.is_file() else 'downloadable'
  return {'short_name': 'NAV-BIG-V2-MAP-FP32', 'display_name': f'{DISPLAY_NAME} ({availability})',
          'ref': NAV_REF, 'is_big': True, 'is_20hz': True, 'environment': 'AMD LLVM gfx1200 USB',
          'runner': 'tinygrad', 'minimum_selector_version': '21', 'models': [],
          'overrides': {'folder': 'Navigation models'},
          'navigation_model_id': MODEL_ID, 'artifact_sha256': ARTIFACT_SHA256,
          'runtime': 'native-navigation', 'download_available': True,
          'release_assets': [{'url': MODEL_URL, 'sha256': ARTIFACT_SHA256, 'size': MODEL_SIZE},
                             {'url': SIDECAR_URL, 'sha256': SIDECAR_SHA256, 'size': SIDECAR_SIZE}]}


def _retired(bundle):
  return is_retired_bundle(bundle)


def _status_text(params):
  status = _selection().selection_status(params)
  state = str(status.get('state', 'idle'))
  message = str(status.get('message', ''))
  return state.capitalize() + (': ' + message if message else '')


def project_get_param(params, key: str, raw: bytes | None) -> bytes | None:
  """Project native navigation selection for the existing hosted Models page."""
  if key == STATUS_KEY:
    return _status_text(params).encode()
  if key == REVISION_KEY:
    return str(_selection().selection_status(params).get('revision', '')).encode()
  if key == 'ModelManager_ActiveJson':
    urls = _json(raw, {})
    urls = dict(urls) if isinstance(urls, dict) else {}
    # The current frontend otherwise fetches the upstream URL and never asks
    # for the device overlay. Empty means use its existing cache fallback.
    urls['chestnut'] = ''
    urls.pop('usbgpu', None)
    return _encode(urls)
  if key == 'ModelManager_ModelsCache_Chestnut':
    catalog = _json(raw, {})
    catalog = copy.deepcopy(catalog) if isinstance(catalog, dict) else {}
    bundles = catalog.get('bundles', [])
    bundles = bundles if isinstance(bundles, list) else []
    catalog['bundles'] = [bundle for bundle in bundles if not _retired(bundle) and
                          not (isinstance(bundle, dict) and bundle.get('ref') == NAV_REF)] + [local_bundle()]
    return _encode(catalog)
  return raw


def _ordinary_source(params, ref):
  found = []
  for source, key in (('qcom', 'ModelManager_ModelsCache'), ('chestnut', 'ModelManager_ModelsCache_Chestnut')):
    catalog = _json(params.get(key), {})
    if isinstance(catalog, dict):
      bundles = catalog.get('bundles', [])
      for bundle in bundles if isinstance(bundles, list) else []:
        if isinstance(bundle, dict) and bundle.get('ref') == ref and not _retired(bundle):
          found.append(source)
  if len(found) != 1:
    raise NavigationModelDisabled('Unknown or ambiguous model selection; refresh the Models list and select again')
  return found[0]


def apply_remote_model_settings(params, updates: dict, *, origin='sunnylink') -> dict:
  """Consume model writes as explicit guarded requests; return unrelated settings.

  Legacy SunnyLink sends bare refs/default {}; the optional revision handshake
  additionally rejects stale-client requests when the frontend supplies it.
  """
  protected = PROTECTED_KEYS.intersection(updates)
  if protected:
    raise NavigationModelDisabled('Model identity/status is managed by selection, not writable settings: ' + ', '.join(sorted(protected)))
  selection = _selection()
  expected = updates.get(REVISION_KEY)
  if expected is not None:
    if not isinstance(expected, str) or expected != str(selection.selection_status(params).get('revision', '')):
      raise NavigationModelDisabled('Model selection changed; refresh SunnyLink before applying this request')
  requests = []
  if 'ModelManager_DownloadRef' in updates:
    ref = updates['ModelManager_DownloadRef']
    if not isinstance(ref, str):
      raise NavigationModelDisabled('Model selection ref must be a string')
    if ref in (RETIRED_ID, 'navigation:' + RETIRED_ID):
      raise NavigationModelDisabled('Original Bigv2 is retired; select the corrected FP32 model')
    if ref:
      requests.append(('chestnut' if ref == NAV_REF else _ordinary_source(params, ref), ref))
    else:
      requests.append(('cancel', ''))
  for source, key in (('qcom', 'ModelManager_ActiveBundle'), ('chestnut', 'ModelManager_ActiveBundleChestnut')):
    if key in updates:
      bundle = _json(updates[key])
      if not isinstance(bundle, dict):
        raise NavigationModelDisabled('Restore/select a catalog model rather than writing an invalid model bundle')
      if _retired(bundle):
        raise NavigationModelDisabled('Original Bigv2 is retired; select the corrected FP32 model')
      ref = bundle.get('ref') if bundle else 'Default'
      valid_navigation = source == 'chestnut' and ref == NAV_REF
      if ref != 'Default' and not valid_navigation and (not isinstance(ref, str) or _ordinary_source(params, ref) != source):
        raise NavigationModelDisabled('Restored bundle must identify a model in the matching current catalog')
      requests.append((source, ref))
  if len(requests) > 1:
    raise NavigationModelDisabled('Apply one model selection at a time')
  if requests:
    source, ref = requests[0]
    if source == 'cancel':
      selection.cancel_model_selection(params, offroad=True, origin=origin, expected_revision=expected)
    else:
      selection.request_model_selection(params, source=source, ref=ref, offroad=True, origin=origin, expected_revision=expected)
  return {key: value for key, value in updates.items() if key not in WRITE_KEYS}


def model_restore_settings(params, updates: dict[str, str]) -> dict[str, str]:
  """Route restored model choices through the same admission as explicit choices.

  Never restore a manifest/proof/status or execute saved download requests.
  Ordinary bundle restoration identifies the current catalog entry by ref.
  """
  canonical = {key.lower(): key for key in WRITE_KEYS | PROTECTED_KEYS}
  relevant = {canonical[key.lower()]: value for key, value in updates.items() if key.lower() in canonical}
  # Saved transient/proof fields and an old activation bit are not new explicit
  # selections. Drop them while preserving the current approved registration.
  ignored = PROTECTED_KEYS | {REVISION_KEY, 'ModelManager_DownloadRef', 'NavigationModelDrivingEnabled'}
  decoded = {}
  for key, value in relevant.items():
    if key not in ignored:
      raw = base64.b64decode(value, validate=True).decode()
      decoded[key] = json.loads(raw)
  if decoded:
    _selection().model_restore_settings(params, decoded)
  return {key: value for key, value in updates.items() if key.lower() not in canonical}
