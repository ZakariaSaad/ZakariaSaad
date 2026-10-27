"""Pinned corrected Bigv2 release installation; no model execution or selection."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import zipfile

from openpilot.selfdrive.navigation.corrected_bigv2 import ARTIFACT_SHA256
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import require_offroad

RELEASE_URL = 'https://github.com/ZakariaSaad/ZakariaSaad/releases/download/nav-bigv2-map-fp32-sidecars/'
MODEL_URL = RELEASE_URL + 'bigv2-amd-map-fp32.sptg'
SIDECAR_URL = RELEASE_URL + 'bigv2-sidecars-from-comma.zip'
MODEL_SIZE = 810439301
SIDECAR_SIZE = 3279412
SIDECAR_SHA256 = 'ebe109ce6352e8d144b60b623c047320b923d6493921f860e48f0efd32a291de'
MODEL_DIR = Path('/data/nav-bigv2')
PACKAGE_ROOT = Path('/data/navigation-models')
SIDECAR_FILES = frozenset(('camera-warp.sptg', 'compiler-report.json', 'compiler-source.py',
                           'map-precision-source.py', 'manifest.json.metadata.json', 'manifest.json', 'registration.json'))
MAX_EXTRACTED_BYTES = 4 << 20


class DownloadCancelled(NavigationModelDisabled):
  pass


def _check(params, current):
  if not current():
    raise DownloadCancelled('Model selection cancelled or superseded; previous selection preserved')
  if not params.get_bool('IsOffroad') or params.get_bool('IsEngaged'):
    raise DownloadCancelled('Download stopped because the device left off-road/disengaged state; select again off road')


def _matches(path, expected, params, current):
  if not path.is_file():
    return False
  digest = hashlib.sha256()
  with path.open('rb') as source:
    while chunk := source.read(4 << 20):
      _check(params, current)
      digest.update(chunk)
  return digest.hexdigest() == expected


@contextmanager
def _lock(directory):
  with (directory / '.release-download.lock').open('a') as stream:
    try:
      fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise NavigationModelDisabled('Another corrected Bigv2 installation is in progress') from error
    try:
      yield
    finally:
      fcntl.flock(stream, fcntl.LOCK_UN)


def _download(url, destination, expected_hash, expected_size, params, current, progress, label):
  import requests
  _check(params, current)
  digest, count, reported = hashlib.sha256(), 0, 0.0
  with requests.get(url, stream=True, timeout=(30, 30)) as response:
    response.raise_for_status()
    if not response.url.startswith('https://') or response.status_code != 200:
      raise NavigationModelDisabled('Release asset requires a complete HTTPS response')
    content_length = response.headers.get('content-length')
    if content_length is not None and int(content_length) != expected_size:
      raise NavigationModelDisabled(label + ' download size differs from the verified release')
    with destination.open('wb') as output:
      for chunk in response.iter_content(chunk_size=128 << 10):
        _check(params, current)
        count += len(chunk)
        if count > expected_size:
          raise NavigationModelDisabled(label + ' exceeds the verified release size')
        output.write(chunk)
        digest.update(chunk)
        now = time.monotonic()
        if now - reported >= 0.5:
          progress('downloading', f'Downloading {label}: {count * 100 // expected_size}% ({count}/{expected_size} bytes)')
          reported = now
      output.flush()
      os.fsync(output.fileno())
  _check(params, current)
  progress('preparing', 'Verifying ' + label + ' SHA256')
  if count != expected_size or digest.hexdigest() != expected_hash:
    raise NavigationModelDisabled(label + ' SHA256 or size verification failed; installed files were preserved')


def _extract_sidecars(archive, destination):
  """Read only the pinned flat package; copied manifests are evidence, not paths."""
  with zipfile.ZipFile(archive) as package:
    entries = package.infolist()
    if len(entries) != len(SIDECAR_FILES) or {entry.filename for entry in entries} != SIDECAR_FILES:
      raise NavigationModelDisabled('Unexpected or duplicate files in corrected Bigv2 sidecars')
    if sum(entry.file_size for entry in entries) > MAX_EXTRACTED_BYTES:
      raise NavigationModelDisabled('Corrected Bigv2 sidecars exceed the extraction bound')
    for entry in entries:
      mode = stat.S_IFMT(entry.external_attr >> 16)
      if entry.flag_bits & 1 or mode not in (0, stat.S_IFREG) or entry.is_dir():
        raise NavigationModelDisabled('Sidecars must contain unencrypted regular files')
      # No extractall: neither foreign absolute paths nor archive attributes
      # control the destination, ownership, permissions or runtime registration.
      data = package.read(entry)
      if len(data) != entry.file_size:
        raise NavigationModelDisabled('Truncated corrected Bigv2 sidecar')
      (destination / entry.filename).write_bytes(data)


def prepare_downloaded_model(params, *, current, progress, registration=None, model_dir=MODEL_DIR, package_root=PACKAGE_ROOT):
  """Reuse verified registered bytes, or install and prepare an exact release.

  Returns without changing selected model parameters. The shared selection
  transaction still performs ordinary fallback admission and the final commit.
  """
  _check(params, current)
  require_offroad(params)
  registration = registration or {}
  if registration.get('registered') and registration.get('artifact_path'):
    progress('preparing', 'Verifying the registered corrected Bigv2; reuse needs no download')
    if _matches(Path(registration['artifact_path']), ARTIFACT_SHA256, params, current):
      return
  directory = Path(model_dir)
  directory.mkdir(parents=True, exist_ok=True)
  with _lock(directory):
    artifact = directory / 'bigv2-amd-map-fp32.sptg'
    progress('preparing', 'Checking the existing corrected Bigv2 file')
    reuse = _matches(artifact, ARTIFACT_SHA256, params, current)
    needed = (0 if reuse else MODEL_SIZE) + SIDECAR_SIZE + MAX_EXTRACTED_BYTES + (16 << 20)
    if shutil.disk_usage(directory).free < needed:
      raise NavigationModelDisabled(f'Corrected Bigv2 installation needs {needed} free bytes')
    with tempfile.TemporaryDirectory(dir=directory, prefix='.release-') as temporary:
      stage = Path(temporary)
      sidecar_zip = stage / 'sidecars.zip'
      _download(SIDECAR_URL, sidecar_zip, SIDECAR_SHA256, SIDECAR_SIZE, params, current, progress, 'registration files')
      sidecars = stage / 'sidecars'
      sidecars.mkdir()
      _extract_sidecars(sidecar_zip, sidecars)
      if not reuse:
        staged_model = stage / 'model.part'
        _download(MODEL_URL, staged_model, ARTIFACT_SHA256, MODEL_SIZE, params, current, progress, 'corrected Bigv2')
        require_offroad(params)
        _check(params, current)
        # Another complete installation is safe to reuse. A verified existing
        # model is never replaced; a failed/partial download never reaches here.
        if not _matches(artifact, ARTIFACT_SHA256, params, current):
          os.replace(staged_model, artifact)
          descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
          try:
            os.fsync(descriptor)
          finally:
            os.close(descriptor)
      require_offroad(params)
      _check(params, current)
      progress('preparing', 'Preparing verified model registration and camera warp')
      from openpilot.selfdrive.navigation.preparation import prepare_existing_model
      prepare_existing_model(params, is_offroad=True, model_dir=directory, package_root=package_root,
                             sidecars_dir=sidecars, publish_selection=False)
      _check(params, current)
