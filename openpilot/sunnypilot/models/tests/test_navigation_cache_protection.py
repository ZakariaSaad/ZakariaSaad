"""Real file/Params cache cleanup preserves registered navigation and fallback."""
import json

import pytest

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.sunnypilot.models import manager as manager_module
from openpilot.sunnypilot.models.manager import ModelManagerSP


@pytest.fixture
def cache(tmp_path, monkeypatch):
  model_root = tmp_path / 'models'
  model_root.mkdir()
  params = Params(str(tmp_path / 'params'))
  manager = ModelManagerSP.__new__(ModelManagerSP)
  manager.params = params
  monkeypatch.setattr(manager_module.Paths, 'model_root', lambda: str(model_root))
  return model_root, params, manager


def selected_bundle(filename):
  bundle = custom.ModelManagerSP.ModelBundle.new_message()
  bundle.minimumSelectorVersion = 21
  bundle.ref = filename
  bundle.runner = 'tinygrad'
  bundle.init('models', 1)
  bundle.models[0].artifact.fileName = filename
  return bundle.to_dict()


def test_cleanup_preserves_native_model_warp_proof_and_ordinary_fallback(cache):
  root, params, manager = cache
  names = ('corrected.sptg', 'warp.sptg', 'compiler.py', 'lowering.py', 'report.json', 'fallback.sptg', 'ordinary-big.sptg')
  for name in names:
    (root / name).write_bytes(name.encode())
  params.put('ModelManager_ActiveBundle', selected_bundle('fallback.sptg'), block=True)
  params.put('ModelManager_ActiveBundleChestnut', selected_bundle('ordinary-big.sptg'), block=True)
  metadata = root / 'metadata.json'
  metadata.write_text(json.dumps({'compiler_verification': {
    'compiler': {'path': 'compiler.py'}, 'map_precision_implementation': {'path': 'lowering.py'},
    'report': {'path': 'report.json'}}}))
  manifest = root / 'manifest.json'
  manifest.write_text(json.dumps({'entries': [{
    'compiled_artifact': {'path': 'corrected.sptg'}, 'warp_artifact': {'path': 'warp.sptg'},
    'compiler_metadata': {'path': 'metadata.json'}}]}))
  params.put('NavigationModelManifest', str(manifest), block=True)
  # Preserve registration even when navigation driving is explicitly OFF.
  params.put_bool('NavigationModelDrivingEnabled', False, block=True)
  for name in ('unused.sptg', '.interrupted.part'):
    (root / name).write_bytes(b'unreferenced')
  before = {path.name: path.read_bytes() for path in root.iterdir() if not path.name.startswith(('unused', '.'))}
  manager.clear_model_cache()
  assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_unreadable_registration_prevents_destructive_cleanup(cache):
  root, params, manager = cache
  (root / 'retained.sptg').write_bytes(b'preserve if registration cannot be inspected')
  params.put('NavigationModelManifest', str(root / 'missing-manifest.json'), block=True)
  manager.clear_model_cache()
  assert (root / 'retained.sptg').read_bytes() == b'preserve if registration cannot be inspected'


def test_external_corrected_artifact_and_registered_cache_alias_survive(cache, tmp_path):
  root, params, manager = cache
  package = tmp_path / 'registered'
  package.mkdir()
  artifact = package / 'corrected.sptg'
  artifact.write_bytes(b'existing corrected artifact, never loaded')
  warp = package / 'warp.sptg'
  warp.write_bytes(b'existing warp')
  (package / 'metadata.json').write_text('{}')
  manifest = package / 'manifest.json'
  manifest.write_text(json.dumps({'entries': [{
    'compiled_artifact': {'path': 'corrected.sptg'}, 'warp_artifact': {'path': 'warp.sptg'},
    'compiler_metadata': {'path': 'metadata.json'}}]}))
  params.put('NavigationModelManifest', str(manifest), block=True)
  alias = root / 'corrected-alias.sptg'
  alias.symlink_to(artifact)
  manager.clear_model_cache()
  assert alias.is_symlink() and artifact.read_bytes() == b'existing corrected artifact, never loaded'
  assert warp.read_bytes() == b'existing warp'
