"""Published provenance is checked and rebound locally without executing code."""
import copy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import preparation
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError, load_experimental_model
from openpilot.selfdrive.navigation.tests.test_corrected_preparation import existing  # noqa: F401


@pytest.fixture
def released(existing):  # noqa: F811 - pytest imports this shared fixture by name
  receipt = existing.prepare(publish_selection=False)
  package = Path(receipt['manifest_path']).parent
  directory = existing.root / 'released-sidecars'
  shutil.copytree(package, directory)
  # Downloaded paths and requested permission must have no authority locally.
  manifest = json.loads((directory / 'manifest.json').read_text())
  manifest['entries'][0]['compiled_artifact']['path'] = '/foreign/device/untrusted.sptg'
  (directory / 'manifest.json').write_text(json.dumps(manifest))
  (directory / 'registration.json').write_text(json.dumps({'manifest_path': '/foreign/device/manifest.json', 'activation_allowed': True}))
  shutil.rmtree(package)
  (existing.package_root / 'big-nav-v2-map-fp32-registration.json').unlink()
  notices = existing.root / 'release-notices.md'
  notices.write_text('Synthetic fixture redistribution notices.\n')
  with patch.object(preparation, 'RELEASE_NOTICES', notices):
    yield existing, directory, notices


def test_released_provenance_rebuilds_paths_receipt_and_preserves_selected_model(released):
  fixture, directory, notices = released
  before = copy.deepcopy(fixture.params.values)
  artifact = fixture.artifact.read_bytes()
  missing = fixture.root / 'missing-installed-provenance'
  with patch.object(preparation, 'COMPILER', missing), patch.object(preparation, 'DEFAULT_WARP', missing), \
       patch.object(preparation, 'PRECISION_IMPLEMENTATION', missing):
    receipt = fixture.prepare(sidecars_dir=directory, publish_selection=False)
  package = Path(receipt['manifest_path']).parent
  assert package.parent == fixture.package_root
  assert receipt['activation_allowed'] is False
  assert fixture.params.values == before and fixture.params.writes == []
  assert fixture.artifact.read_bytes() == artifact
  assert (package / 'REDISTRIBUTION-NOTICES.md').read_bytes() == notices.read_bytes()
  shutil.rmtree(directory)
  entry = load_experimental_model(package / 'manifest.json', experimental=True, backend='AMD')
  assert entry.artifact_path == fixture.artifact
  assert entry.compiler_verification_validated
  proof = entry.metadata['compiler_verification']
  assert proof['report']['path'] == 'compiler-report.json'
  assert proof['compiler']['path'] == 'compiler-source.py'
  assert proof['map_precision_implementation']['path'] == 'map-precision-source.py'
  assert '/foreign/' not in (package / 'manifest.json').read_text()
  assert '/foreign/' not in (package / 'registration.json').read_text()


@pytest.mark.parametrize('name', ['compiler-report.json', 'compiler-source.py', 'map-precision-source.py', 'manifest.json.metadata.json'])
def test_changed_provenance_rejected_without_selection_or_receipt(released, name):
  fixture, directory, _ = released
  before = copy.deepcopy(fixture.params.values)
  path = directory / name
  path.write_bytes(path.read_bytes() + b'\n')
  with pytest.raises(ExperimentalCatalogError, match='Released provenance differs'):
    fixture.prepare(sidecars_dir=directory, publish_selection=False)
  assert fixture.params.values == before and fixture.params.writes == []
  assert not (fixture.package_root / 'big-nav-v2-map-fp32-registration.json').exists()


def test_symlink_provenance_is_rejected_even_with_matching_bytes(released):
  fixture, directory, _ = released
  path = directory / 'compiler-source.py'
  path.unlink()
  path.symlink_to(fixture.fixture.compiler)
  with pytest.raises(ExperimentalCatalogError, match='regular file'):
    fixture.prepare(sidecars_dir=directory, publish_selection=False)
  assert fixture.params.writes == []


def test_missing_distribution_notices_prevent_new_package_without_touching_model(released):
  fixture, directory, notices = released
  notices.unlink()
  before = fixture.artifact.read_bytes()
  with pytest.raises(ExperimentalCatalogError, match='redistribution notices'):
    fixture.prepare(sidecars_dir=directory, publish_selection=False)
  assert fixture.artifact.read_bytes() == before and fixture.params.writes == []
