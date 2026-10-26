"""Branch-reviewed navigation model admission, separate from map display.

A local registration or a UI checkbox cannot approve a driving artifact.
Only a branch-reviewed candidate with pinned compiled hashes can issue a permit.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import json

from openpilot.selfdrive.navigation.experimental_catalog import SOURCE_ONNX_SHA256, _json, _json_file, load_experimental_model
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID, RETIRED_REASON, is_retired

REGISTRY = Path(__file__).with_name('driving_models.json')
REGISTRATION_ROOT = Path('/data/navigation-models')
_PERMIT_SEAL = object()


def _text(value, default=''):
  return value.decode() if isinstance(value, bytes) else value if isinstance(value, str) else default


def _candidate(params, candidate_id=None):
  identifier = candidate_id if candidate_id is not None else _text(params.get('NavigationModelCandidate', return_default=True), MODEL_ID) or MODEL_ID
  if is_retired(identifier):
    raise NavigationModelDisabled(RETIRED_REASON)
  registry = _json_file(REGISTRY)
  if registry.get('schema_version') != 1:
    raise NavigationModelDisabled('Unsupported branch navigation model registry')
  matches = [entry for entry in registry.get('candidates', []) if entry.get('id') == identifier]
  if len(matches) != 1 or identifier != MODEL_ID:
    raise NavigationModelDisabled('Unknown branch navigation model candidate')
  if is_retired(artifact_sha256=matches[0].get('artifact_sha256')):
    raise NavigationModelDisabled(RETIRED_REASON)
  return matches[0]


def list_candidates() -> list[dict]:
  return [{'id': entry['id'], 'display_name': entry['display_name']} for entry in _json_file(REGISTRY).get('candidates', [])
          if entry.get('id') == MODEL_ID and not is_retired(artifact_sha256=entry.get('artifact_sha256'))]


def candidate_manifest(params, candidate_id: str) -> Path | None:
  """Find only this candidate's registered package; receipts never grant approval.

  Keep a matching configured manifest authoritative, including malformed or
  mismatched identity declarations, so a damaged selection is reported rather
  than silently replaced. A different candidate's package is never substituted.
  """
  entry = _candidate(params, candidate_id)
  configured = _text(params.get('NavigationModelManifest'))
  if configured and Path(configured).is_file():
    records = _json_file(Path(configured)).get('entries', [])
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
      raise NavigationModelDisabled('The configured navigation manifest has invalid entries')
    if any(record.get('id') == candidate_id for record in records):
      return Path(configured)
  filename = candidate_id + '-registration.json'
  if Path(filename).name != filename:
    raise NavigationModelDisabled('Invalid registered navigation candidate identifier')
  receipt_path = REGISTRATION_ROOT / filename
  if not receipt_path.is_file():
    return None
  receipt = _json_file(receipt_path)
  if receipt.get('registered') is not True or receipt.get('model_id') != candidate_id or any(
    receipt.get(key) != entry.get(key) for key in ('artifact_sha256', 'warp_sha256', 'metadata_sha256')
  ):
    raise NavigationModelDisabled('Registration receipt does not match the selected candidate identity')
  manifest = receipt.get('manifest_path')
  if not isinstance(manifest, str) or not manifest or '://' in manifest:
    raise NavigationModelDisabled('Registration receipt has no local manifest path')
  return Path(manifest)


def Params():
  """Load the installed native parameter interface only when a caller needs it."""
  from openpilot.common.params import Params as NativeParams
  return NativeParams()


def _ordinary_bundle(params):
  bundle = params.get('ModelManager_ActiveBundle')
  if isinstance(bundle, str):
    bundle = _json(bundle.encode())
  elif isinstance(bundle, bytes):
    bundle = _json(bundle)
  if not isinstance(bundle, dict):
    raise NavigationModelDisabled('A configured ordinary tinygrad small-model fallback is required')
  return bundle


def _model_root() -> Path:
  # Match Paths.model_root without importing hardware/cereal into the CLI.
  if Path('/AGNOS').is_file():
    return Path('/data/media/0/models')
  return Path.home() / ('.comma' + os.environ.get('OPENPILOT_PREFIX', '')) / 'media/0/models'


def ordinary_fallback_reason(params, *, verify=False) -> str:
  """Inspect the ordinary small slot; full admission reuses its native verifier."""
  try:
    bundle = _ordinary_bundle(params)
    if verify:
      from openpilot.selfdrive.navigation.software_compatibility import validate_ordinary_fallback_bundle
      validate_ordinary_fallback_bundle(bundle, _model_root())
      return ''
    if bundle.get('minimumSelectorVersion') != 21 or bundle.get('runner') != 'tinygrad':
      return 'Ordinary small fallback requires Selector21 tinygrad'
    models = bundle.get('models')
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
      return 'A configured ordinary tinygrad small-model fallback is required'
    artifact = models[0].get('artifact', {})
    name = artifact.get('fileName') if isinstance(artifact, dict) else None
    if not isinstance(name, str) or not name or Path(name).name != name:
      return 'Invalid ordinary fallback artifact name'
    with (_model_root() / name).open('rb') as stream:
      if stream.read(8) == b'SPTGPLN1':
        return ''
    return 'The configured ordinary fallback compiled artifact is not installed in native SPTG format'
  except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
    return str(error)


def candidate_status(params=None, *, candidate_id=None) -> dict:
  """Lightweight review/registration status; no weights, kernels or device I/O.

  Registration checks declared identities only. Full byte/provenance validation
  occurs in the read-only software validator and again on runtime admission.
  Software review and completed physical compiler evidence do not assert live
  inference performance, camera execution, or candidate/fallback handover.
  """
  params = params if params is not None else Params()
  try:
    entry = _candidate(params, candidate_id)
    reasons = []
    if entry.get('software_contract_reviewed') is not True:
      reasons.append('Model action and temporal contract has not been reviewed')
    if entry.get('software_integration_reviewed') is not True:
      reasons.append('Navigation model software integration regressions have not been reviewed')
    if entry.get('artifact_evidence_reviewed') is not True:
      reasons.append('Artifact, compiler verification and camera-warp evidence have not been reviewed')
    approved = entry.get('activation_approved') is True
    if not approved:
      reasons.append('Explicit user authorization for Navigation Model Driving is still required')
    if any(not isinstance(entry.get(key), str) or re.fullmatch('[0-9a-f]{64}', entry[key]) is None
           for key in ('artifact_sha256', 'warp_sha256', 'metadata_sha256', 'compiler_sha256', 'compiler_report_sha256')):
      reasons.append('Reviewed model, warp, metadata, compiler and verification-report hashes are not pinned')
    manifest_path = candidate_manifest(params, entry['id'])
    manifest = str(manifest_path) if manifest_path is not None else ''
    registered = installed = False
    artifact_path = ''
    if manifest and Path(manifest).is_file():
      records = _json_file(Path(manifest)).get('entries', [])
      if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError('The configured navigation manifest has invalid entries')
      matches = [record for record in records if record.get('id') == entry['id']]
      registered = len(matches) == 1 and all(isinstance(matches[0].get(field), dict) and
        matches[0][field].get('sha256') == entry.get(key)
        for field, key in (('compiled_artifact', 'artifact_sha256'), ('warp_artifact', 'warp_sha256'),
                           ('compiler_metadata', 'metadata_sha256')))
      if registered:
        name = matches[0]['compiled_artifact'].get('path')
        if isinstance(name, str) and name and '://' not in name:
          artifact = Path(name)
          artifact = artifact if artifact.is_absolute() else Path(manifest).parent / artifact
          artifact_path = str(artifact)
          installed = artifact.is_file()
        if not installed:
          reasons.append('The registered local compiled artifact is not installed')
      if not registered:
        reasons.append('The configured manifest does not declare the reviewed model, warp and metadata identities')
    else:
      reasons.append('A matching local verified manifest has not been configured')
    fallback_reason = ordinary_fallback_reason(params)
    if fallback_reason:
      reasons.append(fallback_reason)
    software_reviewed = entry.get('software_contract_reviewed') is True and entry.get('software_integration_reviewed') is True
    return {'id': entry['id'], 'display_name': entry['display_name'], 'eligible': not reasons,
            'reason': '; '.join(reasons), 'enabled': params.get_bool('NavigationModelDrivingEnabled'),
            'registered': registered, 'installed': installed, 'artifact_path': artifact_path, 'manifest_path': manifest, 'approved': approved,
            'selected': _text(params.get('NavigationModelCandidate')) == entry['id'],
            'blockers': reasons, 'software_compatible': software_reviewed,
            'artifact_evidence_reviewed': entry.get('artifact_evidence_reviewed') is True,
            'runtime_compatibility': entry.get('runtime_compatibility', 'unverified'),
            'local_file_validation': 'Required at runtime admission; registration status checks declared identities only'}
  except (ValueError, OSError, TypeError, KeyError) as error:
    return {'id': '', 'display_name': 'Navigation model', 'eligible': False, 'reason': str(error), 'enabled': False,
            'registered': False, 'installed': False, 'artifact_path': '', 'manifest_path': '', 'approved': False, 'selected': False,
            'blockers': [str(error)], 'software_compatible': False, 'artifact_evidence_reviewed': False,
            'runtime_compatibility': 'unverified'}


def _fallback_identity(params):
  bundle = _ordinary_bundle(params)
  models = bundle.get('models')
  if not isinstance(models, list) or not models or not isinstance(models[0], dict):
    raise NavigationModelDisabled('Configured ordinary fallback is empty or malformed')
  artifact = models[0].get('artifact')
  name = artifact.get('fileName') if isinstance(artifact, dict) else None
  if not isinstance(name, str) or not name or Path(name).name != name:
    raise NavigationModelDisabled('Invalid ordinary fallback artifact name')
  path = (_model_root() / name).resolve(strict=True)
  return json.dumps(bundle, sort_keys=True, separators=(',', ':'), allow_nan=False), path, _fingerprint(path)


def _fingerprint(path):
  stat = Path(path).stat()
  return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class DrivingPermit:
  model_id: str
  artifact_path: Path
  warp_path: Path
  manifest_path: Path
  metadata_path: Path
  input_specs: dict
  metadata: dict
  action_contract: str
  warp_frame_size: int
  registry_fingerprint: tuple
  file_fingerprints: tuple
  fallback_identity: tuple
  seal: object


def admit_candidate(params=None, *, candidate_id=None) -> DrivingPermit:
  """Read-only enable/startup admission, shared by Settings, SSH and modeld.

  A selected or enabled flag is not required here. No Params are written and no
  model is loaded or executed. The returned permit still requires the actual
  driving flag and selection before runtime use.
  """
  params = params if params is not None else Params()
  registry_fingerprint = _fingerprint(REGISTRY)
  status = candidate_status(params, candidate_id=candidate_id)
  if not status['eligible']:
    raise NavigationModelDisabled(status['reason'])
  approved = _candidate(params, candidate_id)
  if approved.get('source_onnx_sha256') != SOURCE_ONNX_SHA256 or approved.get('action_contract') != \
     'comma-nav-v2-20hz-lateral-acceleration-longitudinal-acceleration':
    raise NavigationModelDisabled('Unapproved navigation checkpoint or action contract')
  fallback_identity = _fallback_identity(params)
  fallback_reason = ordinary_fallback_reason(params, verify=True)
  if fallback_reason:
    raise NavigationModelDisabled(fallback_reason)
  chosen_manifest = candidate_manifest(params, approved['id'])
  if chosen_manifest is None:
    raise NavigationModelDisabled('The selected candidate has no registered manifest')
  manifest = chosen_manifest.resolve(strict=True)
  manifest_fingerprint = _fingerprint(manifest)
  manifest_entry = next(item for item in _json_file(manifest)['entries'] if item['id'] == approved['id'])
  artifact_paths = []
  for key in ('compiled_artifact', 'warp_artifact', 'compiler_metadata'):
    path = Path(manifest_entry[key]['path'])
    artifact_paths.append((path if path.is_absolute() else manifest.parent / path).resolve(strict=True))
  fingerprints = tuple(_fingerprint(path) for path in artifact_paths)
  entry = load_experimental_model(manifest, experimental=True, backend='AMD', model_id=approved['id'])
  metadata_path = Path(manifest_entry['compiler_metadata']['path'])
  metadata_path = (metadata_path if metadata_path.is_absolute() else manifest.parent / metadata_path).resolve(strict=True)
  if not entry.compiler_verification_validated or any(
    getattr(entry, key) != approved[key] for key in ('artifact_sha256', 'warp_sha256', 'metadata_sha256')
  ):
    raise NavigationModelDisabled('Local artifacts do not match the branch-reviewed compiler and registration evidence')
  proof = entry.metadata.get('compiler_verification', {})
  if proof.get('compiler', {}).get('sha256') != approved['compiler_sha256'] or \
     proof.get('report', {}).get('sha256') != approved['compiler_report_sha256']:
    raise NavigationModelDisabled('Local compiler source or verification report differs from the reviewed evidence')
  if _fingerprint(REGISTRY) != registry_fingerprint or _fingerprint(manifest) != manifest_fingerprint or \
     tuple(_fingerprint(path) for path in artifact_paths) != fingerprints or _fallback_identity(params) != fallback_identity:
    raise NavigationModelDisabled('Navigation artifacts or branch approval changed during admission')
  permit = DrivingPermit(entry.model_id, entry.artifact_path, entry.warp_path, manifest, metadata_path, entry.input_specs, entry.metadata,
                         approved['action_contract'], entry.warp_frame_size, registry_fingerprint,
                         tuple(_fingerprint(path) for path in (entry.artifact_path, entry.warp_path, manifest, metadata_path)), fallback_identity, _PERMIT_SEAL)
  return permit


def authorize_runtime(params=None) -> DrivingPermit | None:
  """Hash and verify once at startup. OFF mode never opens candidate files."""
  params = params if params is not None else Params()
  if not params.get_bool('NavigationModelDrivingEnabled'):
    return None
  permit = admit_candidate(params)
  validate_runtime_permit(permit, params=params)
  return permit


def validate_admission(permit, params=None, *, allow_unselected_manifest=False) -> None:
  """Recheck admitted file/selection identities without requiring activation."""
  params = params if params is not None else Params()
  if not isinstance(permit, DrivingPermit) or permit.seal is not _PERMIT_SEAL:
    raise NavigationModelDisabled('A branch-reviewed runtime permit is required')
  if permit.model_id != MODEL_ID:
    raise NavigationModelDisabled(RETIRED_REASON if is_retired(permit.model_id) else 'Unknown navigation runtime candidate')
  current = candidate_manifest(params, permit.model_id) if allow_unselected_manifest else Path(_text(params.get('NavigationModelManifest')))
  if current is None or current.resolve() != permit.manifest_path:
    raise NavigationModelDisabled('Navigation model manifest changed')
  if _fingerprint(REGISTRY) != permit.registry_fingerprint or \
     tuple(_fingerprint(path) for path in (permit.artifact_path, permit.warp_path, permit.manifest_path, permit.metadata_path)) != permit.file_fingerprints:
    raise NavigationModelDisabled('Navigation artifacts or branch eligibility changed after admission')
  try:
    fallback_changed = _fallback_identity(params) != permit.fallback_identity
  except (OSError, ValueError, TypeError, KeyError) as error:
    raise NavigationModelDisabled('Configured ordinary small-model fallback changed after admission: ' + str(error)) from error
  if fallback_changed:
    raise NavigationModelDisabled('Configured ordinary small-model fallback changed after admission')


def validate_runtime_permit(permit, input_specs=None, params=None) -> None:
  """Cheap per-frame revocation and native contract check; never rehash weights."""
  params = params if params is not None else Params()
  validate_admission(permit, params)
  if not params.get_bool('NavigationModelDrivingEnabled') or _candidate(params)['id'] != permit.model_id:
    raise NavigationModelDisabled('Navigation model driving permission was withdrawn')
  if input_specs is not None:
    import numpy as np

    actual = {key: (tuple(shape), np.dtype(dtype), device) for key, (shape, dtype, device) in input_specs.items()}
    expected = {key: (tuple(shape), np.dtype(dtype), device) for key, (shape, dtype, device) in permit.input_specs.items()}
    if actual != expected:
      raise NavigationModelDisabled('Loaded native navigation input/state contract differs from the approved artifact')


def validate_runtime_metadata(permit, extras, params=None) -> None:
  import numpy as np

  from openpilot.sunnypilot.modeld_v2.model_adapters import diagnostic_output_slices

  validate_runtime_permit(permit, extras.get('input_specs', {}), params)
  expected_outputs = permit.metadata['output_specs']
  actual_outputs = extras.get('output_specs', {})
  normalized = {name: {'shape': list(shape), 'dtype': np.dtype(dtype).name, 'device': device}
                for name, (shape, dtype, device) in actual_outputs.items()}
  if normalized != expected_outputs:
    raise NavigationModelDisabled('Loaded native navigation output/state contract differs from the approved artifact')
  raw = extras['metadata']['metadata']['output_slices']
  slices = diagnostic_output_slices(raw, actual_outputs['outputs'][0][1])
  size = actual_outputs['outputs'][0][1]
  bounds = {name: value.indices(size) for name, value in slices.items()}
  approved = {name: slice(value['start'], value['stop'], value.get('step')).indices(size)
              for name, value in permit.metadata['output_slices'].items()}
  if bounds != approved:
    raise NavigationModelDisabled('Loaded native navigation action slices differ from the approved artifact')
