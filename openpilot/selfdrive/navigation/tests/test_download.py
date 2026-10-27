"""Pinned HTTP installation with bounded synthetic payloads; no model execution."""
import hashlib
import io
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest
import requests

from openpilot.common.params import Params
from openpilot.selfdrive.navigation import download, preparation
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled


def archive_bytes(names=None):
  stream = io.BytesIO()
  with zipfile.ZipFile(stream, 'w') as archive:
    for name in names or sorted(download.SIDECAR_FILES):
      archive.writestr(name, ('synthetic ' + name).encode())
  return stream.getvalue()


@pytest.fixture
def install(tmp_path, monkeypatch):
  params = Params(str(tmp_path / 'params'))
  params.put_bool('IsOffroad', True, block=True)
  params.put_bool('IsEngaged', False, block=True)
  params.put_bool('NavigationModelDrivingEnabled', False, block=True)
  params.put('ModelManager_ActiveBundleChestnut', {'ref': 'ordinary-preserved'}, block=True)
  directory = tmp_path / 'models'
  directory.mkdir()
  model = b'synthetic corrected native artifact, downloaded without execution' * 30
  sidecars = archive_bytes()
  monkeypatch.setattr(download, 'MODEL_SIZE', len(model))
  monkeypatch.setattr(download, 'ARTIFACT_SHA256', hashlib.sha256(model).hexdigest())
  monkeypatch.setattr(download, 'SIDECAR_SIZE', len(sidecars))
  monkeypatch.setattr(download, 'SIDECAR_SHA256', hashlib.sha256(sidecars).hexdigest())
  monkeypatch.setattr(download, 'require_offroad', Mock())
  prepared = Mock()
  monkeypatch.setattr(preparation, 'prepare_existing_model', prepared)
  context = SimpleNamespace(params=params, directory=directory, artifact=directory / 'bigv2-amd-map-fp32.sptg',
                            root=tmp_path / 'packages', model=model, sidecars=sidecars, prepared=prepared,
                            calls=[], updates=[], current=True, chunk_hook=lambda url, index: None)

  class Response:
    def __init__(self, url):
      self.url, self.status_code = url, 200
      self.body = context.model if url == download.MODEL_URL else context.sidecars
      self.headers = {'content-length': str(len(self.body))}

    def __enter__(self):
      return self

    def __exit__(self, *args):
      pass

    def raise_for_status(self):
      pass

    def iter_content(self, chunk_size):
      for index in range(0, len(self.body), 29):
        context.chunk_hook(self.url, index)
        yield self.body[index:index + 29]

  def get(url, **kwargs):
    assert kwargs == {'stream': True, 'timeout': (30, 30)}
    context.calls.append(url)
    return Response(url)
  monkeypatch.setattr(requests, 'get', get)
  context.run = lambda **kwargs: download.prepare_downloaded_model(
    params, current=lambda: context.current, progress=lambda state, message: context.updates.append((state, message)),
    model_dir=directory, package_root=context.root, **kwargs)
  return context


def test_registered_verified_model_is_reused_without_any_download_or_preparation(install):
  install.artifact.write_bytes(install.model)
  before = install.artifact.stat().st_mtime_ns
  install.run(registration={'registered': True, 'artifact_path': str(install.artifact)})
  assert install.calls == []
  install.prepared.assert_not_called()
  assert install.artifact.stat().st_mtime_ns == before


def test_existing_unregistered_model_reuses_bytes_and_prepares_release_sidecars(install):
  install.artifact.write_bytes(install.model)
  before = install.artifact.stat().st_mtime_ns
  seen = []
  def prepare(params, **kwargs):
    assert params is install.params and kwargs['publish_selection'] is False and kwargs['is_offroad'] is True
    assert kwargs['model_dir'] == install.directory and kwargs['package_root'] == install.root
    sidecars = kwargs['sidecars_dir']
    seen.extend(p.name for p in sidecars.iterdir())
    assert (sidecars / 'registration.json').read_bytes() == b'synthetic registration.json'
  install.prepared.side_effect = prepare
  install.run()
  assert set(seen) == download.SIDECAR_FILES
  assert install.calls == [download.SIDECAR_URL]
  assert install.artifact.stat().st_mtime_ns == before
  assert not list(install.directory.glob('.release-*/*'))


def test_complete_download_atomically_replaces_only_invalid_bytes_then_prepares(install):
  previous = b'invalid old artifact'
  install.artifact.write_bytes(previous)
  def inspect_during_transfer(url, index):
    assert install.artifact.read_bytes() == previous
  install.chunk_hook = inspect_during_transfer
  install.run()
  assert install.calls == [download.SIDECAR_URL, download.MODEL_URL]
  assert install.artifact.read_bytes() == install.model
  assert install.prepared.call_count == 1
  assert install.params.get('ModelManager_ActiveBundleChestnut') == {'ref': 'ordinary-preserved'}
  assert not install.params.get_bool('NavigationModelDrivingEnabled')
  assert any(state == 'downloading' and '%' in message for state, message in install.updates)


@pytest.mark.parametrize('failure', ['hash', 'size', 'cancel', 'onroad'])
def test_failed_or_interrupted_model_download_preserves_old_file_and_cleans_stage(install, failure):
  previous = b'old file retained'
  install.artifact.write_bytes(previous)
  if failure == 'hash':
    install.model = bytes([install.model[0] ^ 1]) + install.model[1:]
  elif failure == 'size':
    install.model += b'extra bytes'
  else:
    def interrupt(url, index):
      if url == download.MODEL_URL and index > 0:
        if failure == 'cancel':
          install.current = False
        else:
          install.params.put_bool('IsOffroad', False, block=True)
    install.chunk_hook = interrupt
  with pytest.raises(NavigationModelDisabled):
    install.run()
  assert install.artifact.read_bytes() == previous
  install.prepared.assert_not_called()
  assert not list(install.directory.glob('.release-*/*'))


def test_cancelled_request_does_not_fetch_or_publish(install):
  install.current = False
  with pytest.raises(download.DownloadCancelled, match='superseded'):
    install.run()
  assert install.calls == [] and not install.artifact.exists()


def test_sidecar_hash_failure_stops_before_model_download(install):
  install.sidecars = install.sidecars[:-1] + bytes([install.sidecars[-1] ^ 1])
  with pytest.raises(NavigationModelDisabled, match='verification failed'):
    install.run()
  assert install.calls == [download.SIDECAR_URL]
  assert not install.artifact.exists()


@pytest.mark.parametrize('failure', ['http', 'timeout'])
def test_http_error_or_midstream_timeout_preserves_installed_file(install, monkeypatch, failure):
  install.artifact.write_bytes(b'existing bytes')
  if failure == 'http':
    get = requests.get
    def rejected(url, **kwargs):
      response = get(url, **kwargs)
      if url == download.MODEL_URL:
        response.raise_for_status = Mock(side_effect=requests.HTTPError('503 unavailable'))
      return response
    monkeypatch.setattr(requests, 'get', rejected)
  else:
    def interrupted(url, index):
      if url == download.MODEL_URL and index > 0:
        raise requests.ReadTimeout('Interrupted release transfer')
    install.chunk_hook = interrupted
  with pytest.raises(requests.RequestException):
    install.run()
  assert install.artifact.read_bytes() == b'existing bytes'
  install.prepared.assert_not_called()
  assert not list(install.directory.glob('.release-*/*'))


def test_cancellation_after_transfer_before_atomic_publication_preserves_old_file(install, monkeypatch):
  install.artifact.write_bytes(b'existing bytes')
  calls = []
  def guard(params):
    calls.append(params)
    if len(calls) == 2:
      install.current = False
  monkeypatch.setattr(download, 'require_offroad', guard)
  with pytest.raises(download.DownloadCancelled):
    install.run()
  assert install.artifact.read_bytes() == b'existing bytes'
  install.prepared.assert_not_called()
  assert not list(install.directory.glob('.release-*/*'))


@pytest.mark.parametrize('names', [
  ['../escape'] + sorted(download.SIDECAR_FILES)[1:],
  ['camera-warp.sptg'] * 7,
  list(download.SIDECAR_FILES) + ['unexpected'],
])
def test_unsafe_or_ambiguous_archive_is_rejected_without_extracting_paths(tmp_path, names):
  archive = tmp_path / 'sidecars.zip'
  with pytest.warns(UserWarning) if len(set(names)) != len(names) else nullcontext():
    archive.write_bytes(archive_bytes(names))
  destination = tmp_path / 'extract'
  destination.mkdir()
  with pytest.raises(NavigationModelDisabled, match='Unexpected or duplicate'):
    download._extract_sidecars(archive, destination)
  assert list(destination.iterdir()) == [] and not (tmp_path / 'escape').exists()


def test_zip_symlink_and_extraction_bounds_are_rejected(tmp_path, monkeypatch):
  archive = tmp_path / 'sidecars.zip'
  with zipfile.ZipFile(archive, 'w') as output:
    for name in download.SIDECAR_FILES:
      entry = zipfile.ZipInfo(name)
      entry.create_system = 3
      entry.external_attr = 0o120777 << 16
      output.writestr(entry, '/outside')
  destination = tmp_path / 'extract'
  destination.mkdir()
  with pytest.raises(NavigationModelDisabled, match='regular files'):
    download._extract_sidecars(archive, destination)
  archive.write_bytes(archive_bytes())
  monkeypatch.setattr(download, 'MAX_EXTRACTED_BYTES', 5)
  with pytest.raises(NavigationModelDisabled, match='extraction bound'):
    download._extract_sidecars(archive, destination)


def test_low_storage_fails_before_network_or_file_replacement(install, monkeypatch):
  monkeypatch.setattr(download.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
  with pytest.raises(NavigationModelDisabled, match='free bytes'):
    install.run()
  assert install.calls == [] and not install.artifact.exists()


def test_registration_failure_preserves_selection_and_keeps_only_verified_cached_model(install):
  install.prepared.side_effect = NavigationModelDisabled('Synthetic fallback/provenance refusal')
  with pytest.raises(NavigationModelDisabled, match='refusal'):
    install.run()
  assert install.artifact.read_bytes() == install.model
  assert not install.params.get_bool('NavigationModelDrivingEnabled')
  assert install.params.get('ModelManager_ActiveBundleChestnut') == {'ref': 'ordinary-preserved'}
  assert not list(install.directory.glob('.release-*/*'))
