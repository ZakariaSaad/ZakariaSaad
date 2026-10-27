"""Register existing Bigv2 files without GPU access, downloads or compilation.

Preparation is separate from branch approval and never enables driving. Only the
small warp and provenance are copied; existing model/source weights stay in place.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.experimental_catalog import (
  SOURCE_ONNX_SHA256, ExperimentalCatalogError, _json_file, _local_file, _require, load_experimental_model,
)
from openpilot.selfdrive.navigation.export_experimental_model import _digest, export_experimental_model
from openpilot.selfdrive.navigation import corrected_bigv2 as corrected
from openpilot.selfdrive.navigation.candidate_identity import RETIRED_REASON, is_retired
from openpilot.selfdrive.navigation.offroad import require_offroad
from openpilot.selfdrive.navigation.warp_compatibility import installed_warp_path, validate_existing_warp

CHECKOUT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path('/data/nav-bigv2')
PACKAGE_ROOT = Path('/data/navigation-models')
COMPILER = CHECKOUT / 'openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py'
DEFAULT_WARP = installed_warp_path(1344, 760)
PRECISION_IMPLEMENTATION = CHECKOUT / 'openpilot/sunnypilot/modeld_v2/nav_map_precision.py'
RELEASE_NOTICES = CHECKOUT / 'docs/honda-eps/navigation-model-release-notices.md'


def _offroad(params, is_offroad):
  _require(is_offroad is True and params.get_bool('IsOffroad') and not params.get_bool('IsEngaged'),
           'Register the existing model only while software is genuinely off-road and disengaged')


def _write_json(path, value):
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.registration-', delete=False) as stream:
      temporary = Path(stream.name)
      json.dump(value, stream, indent=2)
      stream.write('\n')
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


@contextmanager
def _preparation_lock(root):
  with (root / '.preparation.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise ExperimentalCatalogError('Another existing-model registration is already running') from error
    try:
      yield
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


def _report_file(directory, explicit=None, *, basename='bigv2-amd-map-fp32'):
  if explicit is not None:
    return Path(explicit).resolve(strict=True)
  reports = [path for path in (directory / (basename + '.verification.json'), directory / (basename + '.sptg.verification.json')) if path.is_file()]
  _require(bool(reports), f'Existing Bigv2 compiler report is missing; expected {basename}.verification.json or {basename}.sptg.verification.json')
  _require(len({_digest(path) for path in reports}) == 1, 'Conflicting Bigv2 reports exist; select the intended report with --verification')
  return reports[0]


def _package_receipt(package, identity, profile, model_id=corrected.MODEL_ID):
  entry = load_experimental_model(package / 'manifest.json', experimental=True, backend='AMD', model_id=model_id)
  _require(entry.compiler_verification_validated, 'The existing model does not have validated compiler provenance')
  _require(entry.artifact_sha256 == corrected.ARTIFACT_SHA256, 'Registered model differs from the uploaded compiler report')
  _require(entry.metadata_sha256 == corrected.METADATA_SHA256, 'Corrected registration metadata differs from the reviewed portable contract')
  proof = entry.metadata['compiler_verification']
  _require(entry.warp_sha256 == identity['warp_sha256'] and entry.metadata.get('camera_profile') == profile
           and proof['report']['sha256'] == identity['report_sha256'] and proof['compiler']['sha256'] == identity['compiler_sha256'],
           'Existing registration package differs from the reviewed warp/compiler/report identity; it was not overwritten')
  return {'prepared': True, 'activation_allowed': False, 'model_id': entry.model_id, 'manifest_path': str(package / 'manifest.json'),
          'artifact_sha256': entry.artifact_sha256, 'warp_sha256': entry.warp_sha256, 'metadata_sha256': entry.metadata_sha256,
          'source_onnx_sha256': SOURCE_ONNX_SHA256, 'source_onnx_bytes_rechecked': entry.metadata.get('source_onnx_bytes_rechecked'),
          'camera_profile': entry.metadata.get('camera_profile'), 'gpu_execution_performed': False,
          'model_recompiled': False, 'models_downloaded': False, 'normal_service_modified': False}


def _compiler_source(root, expected):
  """Reuse the corrected package's exact retained source after compiler updates."""
  sources = [COMPILER, *sorted(root.glob(corrected.MODEL_ID + '-*/compiler-source.py'))]
  for source in sources:
    if source.is_file() and _digest(source) == expected:
      return source
  raise ExperimentalCatalogError('The retained report requires different compiler source; preserve the artifact and send the report for source review')


def _release_sources(directory):
  """Accept only pinned provenance primitives, never foreign path instructions.

  The downloader separately authenticates the complete archive and bounds its
  extraction. These checks also protect direct callers and a changed staging
  file. Uploaded manifest/registration paths are not used for installation.
  """
  directory = Path(directory).resolve(strict=True)
  pins = {'compiler-report.json': corrected.REPORT_SHA256,
          'compiler-source.py': corrected.COMPILER_SHA256,
          'map-precision-source.py': corrected.IMPLEMENTATION_SHA256,
          'manifest.json.metadata.json': corrected.METADATA_SHA256}
  sources = {}
  for name in (*pins, 'camera-warp.sptg'):
    path = directory / name
    _require(path.is_file() and not path.is_symlink() and path.resolve().parent == directory,
             'Released provenance must be a regular file: ' + name)
    if name in pins:
      _require(_digest(path) == pins[name], 'Released provenance differs from reviewed bytes: ' + name)
    sources[name] = path
  # validate_existing_warp below binds the camera warp to its approved digest,
  # native bounds, camera geometry and saved NumPy reference.
  return sources


def _publish_selection(params, model_id, manifest):
  """Publish both pointers with readback; retain the old pair on failed writes."""
  previous = {key: params.get(key) for key in ('NavigationModelCandidate', 'NavigationModelManifest')}
  try:
    for key, value in (('NavigationModelCandidate', model_id), ('NavigationModelManifest', manifest)):
      params.put(key, value, block=True)
      _require(params.get(key) == value, f'Could not persist {key}; registration cancelled')
    _require(not params.get_bool('NavigationModelDrivingEnabled'), 'Driving permission changed while publishing registration')
  except Exception as error:
    failures = []
    try:
      params.put_bool('NavigationModelDrivingEnabled', False, block=True)
      _require(not params.get_bool('NavigationModelDrivingEnabled'), 'Could not persist Navigation Model Driving OFF')
    except Exception as restore_error:
      failures.append(str(restore_error))
    for key, value in previous.items():
      try:
        if params.get(key) != value:
          if value is None:
            params.remove(key)
          else:
            params.put(key, value, block=True)
        _require(params.get(key) == value, f'Could not restore {key}')
      except Exception as restore_error:
        failures.append(str(restore_error))
    if failures:
      raise ExperimentalCatalogError(f'{error}; registration rollback failed: {"; ".join(failures)}') from error
    raise


def prepare_existing_model(params=None, *, is_offroad: bool, model_dir=MODEL_DIR, package_root=PACKAGE_ROOT,
                           verification=None, model_id: str = corrected.MODEL_ID, publish_selection: bool = True,
                           sidecars_dir=None) -> dict:
  """Verify existing corrected bytes and retain their immutable local package.

  Shared model selection uses publish_selection=False: preparation never changes
  the previous selected model or enable request while replacement files are being
  checked. Direct registration publishes a disabled selection after validation.
  Neither mode executes kernels, deletes legacy evidence, or copies model weights.
  """
  params = params if params is not None else Params()
  _require(not is_retired(model_id), RETIRED_REASON)
  _require(model_id == corrected.MODEL_ID, 'Unknown existing navigation model identity')
  _require(os.geteuid() == CHECKOUT.stat().st_uid,
           'Run existing-model registration as the checkout owner (comma), without sudo; no elevated hardware access is needed')
  _offroad(params, is_offroad)
  require_offroad(params)
  if publish_selection:
    params.put_bool('NavigationModelDrivingEnabled', False, block=True)
    _require(not params.get_bool('NavigationModelDrivingEnabled'), 'Could not persist Navigation Model Driving OFF')
  directory, root = Path(model_dir).resolve(strict=True), Path(package_root).resolve()
  _require(root != CHECKOUT and not root.is_relative_to(CHECKOUT), 'Registration packages must remain outside the installed source checkout')
  root.mkdir(parents=True, exist_ok=True)
  receipt_path = root / (model_id + '-registration.json')
  basename = 'bigv2-amd-map-fp32'
  with _preparation_lock(root):
    artifact = directory / (basename + '.sptg')
    _require(artifact.is_file(), f'Existing {basename}.sptg is missing; this workflow never downloads or recompiles it')
    _, artifact_sha = _local_file(artifact, {'path': str(artifact), 'sha256': corrected.ARTIFACT_SHA256}, 'Existing Bigv2 artifact')
    released = _release_sources(sidecars_dir) if sidecars_dir is not None else None
    _require(released is None or verification is None, 'Choose released provenance or an explicit compiler report, not both')
    report = released['compiler-report.json'] if released else _report_file(directory, verification, basename=basename)
    report_data = _json_file(report)
    _require(report_data.get('artifact_sha256') == artifact_sha, 'Compiler report refers to different Bigv2 artifact bytes')
    compiler = released['compiler-source.py'] if released else _compiler_source(root, corrected.COMPILER_SHA256)
    compiler_sha = _digest(compiler)
    _require(report_data.get('compiler_sha256') == compiler_sha,
             'The retained report requires different compiler source; preserve the artifact and send the report for source review')
    _require(_digest(report) == corrected.REPORT_SHA256, 'Corrected compiler report differs from the reviewed evidence')
    precision_sources = ([released['map-precision-source.py']] if released else
                         [PRECISION_IMPLEMENTATION, *sorted(root.glob(model_id + '-*/map-precision-source.py'))])
    precision = next((path for path in precision_sources if path.is_file() and _digest(path) == corrected.IMPLEMENTATION_SHA256), None)
    _require(precision is not None, 'Corrected compiler map precision implementation source changed')
    warp = released['camera-warp.sptg'] if released else DEFAULT_WARP
    profile = validate_existing_warp(warp)
    profile.pop('path', None)  # The copied warp's byte identity is the durable binding.
    identity = {'schema': 1, 'artifact_sha256': artifact_sha, 'artifact_path': str(artifact),
                'report_sha256': _digest(report), 'compiler_sha256': compiler_sha,
                'warp_sha256': profile['artifact_sha256'], 'model_id': model_id,
                'implementation_sha256': corrected.IMPLEMENTATION_SHA256}
    package_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    package = root / (model_id + '-' + package_id)
    if not package.exists():
      sources = [(compiler, 'compiler-source.py'), (report, 'compiler-report.json'),
                 (warp, 'camera-warp.sptg'), (precision, 'map-precision-source.py')]
      if released:
        _require(RELEASE_NOTICES.is_file(), 'Released model redistribution notices are missing from this checkout')
        sources.append((RELEASE_NOTICES, 'REDISTRIBUTION-NOTICES.md'))
      needed = sum(path.stat().st_size for path, _ in sources) + (2 << 20)
      _require(shutil.disk_usage(root).free >= needed,
               f'Existing-model registration needs {needed} free bytes for warp/provenance; model weights are not copied')
      stage = Path(tempfile.mkdtemp(dir=root, prefix='.prepare-existing-'))
      try:
        for original, name in sources:
          shutil.copyfile(original, stage / name)
        _require(_digest(stage / 'compiler-source.py') == compiler_sha and _digest(stage / 'compiler-report.json') == identity['report_sha256']
                 and _digest(stage / 'camera-warp.sptg') == profile['artifact_sha256'], 'Source files changed during preparation')
        export_experimental_model(artifact, stage / 'camera-warp.sptg', stage / 'compiler-report.json', None,
                                  stage / 'compiler-source.py', CHECKOUT / 'tinygrad_repo', stage / 'manifest.json',
                                  experimental=True, portable=True, camera_profile=profile, model_id=model_id,
                                  precision_implementation_path=stage / 'map-precision-source.py')
        load_experimental_model(stage / 'manifest.json', experimental=True, backend='AMD', model_id=model_id)
        _require(not package.exists(), 'A package appeared during preparation; no existing package was overwritten')
        os.rename(stage, package)
      finally:
        shutil.rmtree(stage, ignore_errors=True)
    receipt = _package_receipt(package, identity, profile, model_id=model_id)
    _offroad(params, is_offroad)
    require_offroad(params)
    if publish_selection:
      _require(not params.get_bool('NavigationModelDrivingEnabled'), 'Driving permission changed during preparation; registration refused')
    receipt.update(status='registered_disabled', registered=True)
    # Keep previous complete receipts intact until all file checks succeed. Each
    # receipt publication is atomic, and interrupted preparations are reusable.
    _write_json(receipt_path, receipt)
    if not (package / 'registration.json').exists():
      _write_json(package / 'registration.json', receipt)
    if publish_selection:
      _publish_selection(params, model_id, receipt['manifest_path'])
    return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', choices=(corrected.MODEL_ID,), default=corrected.MODEL_ID)
  parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
  parser.add_argument('--verification', type=Path, help='Optional explicit existing compiler report when filenames are ambiguous')
  args = parser.parse_args()
  try:
    params = Params()
    receipt = prepare_existing_model(params, is_offroad=params.get_bool('IsOffroad'), model_dir=args.model_dir,
                                     verification=args.verification, model_id=args.model)
  except (ValueError, OSError) as error:
    parser.exit(2, f'Existing-model registration failed: {error}\nNo Bigv2 compilation or GPU diagnostic was run.\n')
  print(json.dumps(receipt, indent=2))
  print('Receipt: ' + str(PACKAGE_ROOT / (args.model + '-registration.json')))


if __name__ == '__main__':
  main()
