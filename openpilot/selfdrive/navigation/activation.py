"""Branch-reviewed navigation model admission, separate from map display.

A local registration or a UI checkbox cannot approve a driving artifact.
Only a branch-reviewed candidate with pinned compiled hashes can issue a permit.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.experimental_catalog import SOURCE_ONNX_SHA256, _json_file, load_experimental_model
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled

REGISTRY = Path(__file__).with_name('driving_models.json')
_PERMIT_SEAL = object()


def _text(value, default=''):
  return value.decode() if isinstance(value, bytes) else value if isinstance(value, str) else default


def _candidate(params, candidate_id=None):
  identifier = candidate_id if candidate_id is not None else _text(params.get('NavigationModelCandidate', return_default=True), 'big-nav-v2')
  registry = _json_file(REGISTRY)
  if registry.get('schema_version') != 1:
    raise NavigationModelDisabled('Unsupported branch navigation model registry')
  matches = [entry for entry in registry.get('candidates', []) if entry.get('id') == identifier]
  if len(matches) != 1:
    raise NavigationModelDisabled('Unknown branch navigation model candidate')
  return matches[0]


def list_candidates() -> list[dict]:
  return [{'id': entry['id'], 'display_name': entry['display_name']} for entry in _json_file(REGISTRY).get('candidates', [])]


def ordinary_fallback_reason(params) -> str:
  from openpilot.cereal import custom
  from openpilot.common.hardware.hw import Paths
  from openpilot.sunnypilot.models.helpers import get_active_bundle

  bundle = get_active_bundle(params, chestnut=False)
  if bundle is None or bundle.runner.raw != custom.ModelManagerSP.Runner.tinygrad or not bundle.models:
    return 'A configured ordinary tinygrad small-model fallback is required'
  path = Path(Paths.model_root()) / bundle.models[0].artifact.fileName
  try:
    with path.open('rb') as stream:
      if stream.read(8) == b'SPTGPLN1':
        return ''
  except OSError:
    pass
  return 'The configured ordinary fallback compiled artifact is not installed'


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
    if entry.get('activation_approved') is not True:
      reasons.append('Explicit user authorization for Navigation Model Driving is still required')
    if any(not isinstance(entry.get(key), str) or re.fullmatch('[0-9a-f]{64}', entry[key]) is None
           for key in ('artifact_sha256', 'warp_sha256', 'metadata_sha256', 'compiler_sha256', 'compiler_report_sha256')):
      reasons.append('Reviewed model, warp, metadata, compiler and verification-report hashes are not pinned')
    manifest = _text(params.get('NavigationModelManifest'))
    registered = False
    if manifest and Path(manifest).is_file():
      records = _json_file(Path(manifest)).get('entries', [])
      if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError('The configured navigation manifest has invalid entries')
      matches = [record for record in records if record.get('id') == entry['id']]
      registered = len(matches) == 1 and all(isinstance(matches[0].get(field), dict) and
        matches[0][field].get('sha256') == entry.get(key)
        for field, key in (('compiled_artifact', 'artifact_sha256'), ('warp_artifact', 'warp_sha256'),
                           ('compiler_metadata', 'metadata_sha256')))
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
            'registered': registered, 'software_compatible': software_reviewed,
            'artifact_evidence_reviewed': entry.get('artifact_evidence_reviewed') is True,
            'runtime_compatibility': entry.get('runtime_compatibility', 'unverified'),
            'local_file_validation': 'Required at runtime admission; registration status checks declared identities only'}
  except (ValueError, OSError, TypeError, KeyError) as error:
    return {'id': '', 'display_name': 'Navigation model', 'eligible': False, 'reason': str(error), 'enabled': False,
            'registered': False, 'software_compatible': False, 'artifact_evidence_reviewed': False,
            'runtime_compatibility': 'unverified'}


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
  seal: object


def authorize_runtime(params=None) -> DrivingPermit | None:
  """Hash and verify once at startup. OFF mode never opens any candidate files."""
  params = params if params is not None else Params()
  if not params.get_bool('NavigationModelDrivingEnabled'):
    return None
  status = candidate_status(params)
  if not status['eligible']:
    raise NavigationModelDisabled(status['reason'])
  approved = _candidate(params)
  if approved.get('source_onnx_sha256') != SOURCE_ONNX_SHA256 or approved.get('action_contract') != \
     'comma-nav-v2-20hz-lateral-acceleration-longitudinal-acceleration':
    raise NavigationModelDisabled('Unapproved navigation checkpoint or action contract')
  manifest = Path(_text(params.get('NavigationModelManifest'))).resolve(strict=True)
  entry = load_experimental_model(manifest, experimental=True, backend='AMD', model_id=approved['id'])
  manifest_entry = next(item for item in _json_file(manifest)['entries'] if item['id'] == approved['id'])
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
  permit = DrivingPermit(entry.model_id, entry.artifact_path, entry.warp_path, manifest, metadata_path, entry.input_specs, entry.metadata,
                         approved['action_contract'], entry.warp_frame_size, _fingerprint(REGISTRY),
                         tuple(_fingerprint(path) for path in (entry.artifact_path, entry.warp_path, manifest, metadata_path)), _PERMIT_SEAL)
  validate_runtime_permit(permit, params=params)
  return permit


def validate_runtime_permit(permit, input_specs=None, params=None) -> None:
  """Cheap per-frame revocation and native contract check; never rehash weights."""
  params = params if params is not None else Params()
  if not isinstance(permit, DrivingPermit) or permit.seal is not _PERMIT_SEAL:
    raise NavigationModelDisabled('A branch-reviewed runtime permit is required')
  if not params.get_bool('NavigationModelDrivingEnabled') or _candidate(params)['id'] != permit.model_id:
    raise NavigationModelDisabled('Navigation model driving permission was withdrawn')
  if Path(_text(params.get('NavigationModelManifest'))).resolve() != permit.manifest_path:
    raise NavigationModelDisabled('Navigation model manifest changed')
  if _fingerprint(REGISTRY) != permit.registry_fingerprint or \
     tuple(_fingerprint(path) for path in (permit.artifact_path, permit.warp_path, permit.manifest_path, permit.metadata_path)) != permit.file_fingerprints:
    raise NavigationModelDisabled('Navigation artifacts or branch eligibility changed after admission')
  if input_specs is not None:
    actual = {key: (tuple(shape), np.dtype(dtype), device) for key, (shape, dtype, device) in input_specs.items()}
    expected = {key: (tuple(shape), np.dtype(dtype), device) for key, (shape, dtype, device) in permit.input_specs.items()}
    if actual != expected:
      raise NavigationModelDisabled('Loaded native navigation input/state contract differs from the approved artifact')


def validate_runtime_metadata(permit, extras, params=None) -> None:
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
