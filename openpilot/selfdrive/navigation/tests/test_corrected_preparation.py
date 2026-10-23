"""Corrected candidate registration with inert native bytes and real file hashes.

Only the reviewed identities/profile/runtime are replaced with synthetic fixture
identities. No device, compiler, pickle reader, download or service is used.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import corrected_bigv2 as corrected, export_experimental_model as exporter, preparation
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError, load_experimental_model
from openpilot.selfdrive.navigation.tests import test_preparation as legacy


@pytest.fixture
def existing():
  fixture = legacy.TestExistingModelPreparation('runTest')
  fixture.setUp()
  yield fixture
  fixture.doCleanups()


@pytest.fixture
def candidate(existing):
  fixture = existing
  artifact = fixture.model_dir / 'bigv2-amd-map-fp32.sptg'
  raw = fixture.artifact.read_bytes()
  size = struct.unpack('<Q', raw[8:16])[0]
  container = json.loads(raw[16:16 + size])
  header = container['parts'][0]['header']
  target = header['calls'][0]['target']
  header['calls'] = [{'kind': 'program', 'target': target, 'name': name, 'lib': [0, 1], 'ins': [], 'outs': [0],
                     'args': [{'buffer': index, 'offset': 0, 'nbytes': 131072, 'dtype': 'dtypes.f32'}]}
                    for index, name in enumerate(('synthetic_dot', 'synthetic_scale'))]
  header['buffers'] = [{'device': 'AMD', 'dtype': 'dtypes.f16', 'nbytes': 1, 'init': [0, 1]}]
  encoded = json.dumps(container).encode()
  artifact.write_bytes(b'SPTGPLN1' + struct.pack('<Q', len(encoded)) + encoded + bytes(-(16 + len(encoded)) % 256)
                       + raw[16 + size + -(16 + size) % 256:])
  implementation = fixture.model_dir / 'precision.py'
  implementation.write_text('# inert reviewed lowering fixture; never executed\n')
  report = copy.deepcopy(fixture.report)
  report.update(artifact_sha256=exporter._digest(artifact), compiler_revision=corrected.COMPILER_REVISION,
                candidate_activation_approved=False)
  kernel_sha = hashlib.sha256(b'\0').hexdigest()
  correction = {'id': corrected.CORRECTION_ID, 'implementation_sha256': exporter._digest(implementation),
                'source_onnx_sha256': corrected.SOURCE_ONNX_SHA256, 'compiler_sha256': report['compiler_sha256'],
                'compiler_revision': corrected.COMPILER_REVISION, 'tinygrad_revision': corrected.TINYGRAD_REF,
                'operand_dtype': 'float16', 'dot_bias_gamma_dtype': 'float32', 'residual_input_dtype': 'float16',
                'materialized_fp32_shape': [1, 512, 8, 8], 'original_matmul': 'node_MatMul_599',
                'requires_separate_registration_and_approval': True,
                'initializers': {'synthetic_initializer': {'nbytes': 1, 'sha256': kernel_sha}},
                'compiled_precision': {'status': 'passed_compiled_ir_and_native_buffer_checks',
                                       'residual_boundary': {'materialized_dtype': 'float16', 'nbytes': 65536}}}
  for index, key in enumerate(('dot', 'bias_gamma')):
    correction['compiled_precision'][key] = {'call': index, 'name': header['calls'][index]['name'], 'target': target,
                                             'kernel_sha256': kernel_sha, 'early_half_casts': 0,
                                             'llvm_early_half_narrowing': False, 'floating_arithmetic': ['float32']}
  report['map_precision_correction'] = correction
  report['map_fixture_verification'] = {'cases': {
    name: {'synthetic_uint8_map_sha256': digest, 'other_inputs_seed': 42, 'independent_buffers': True,
           'outputs': copy.deepcopy(report['verification']['runs'][0]['outputs'])} for name, digest in corrected.FIXTURES.items()}}
  report_path = fixture.model_dir / 'bigv2-amd-map-fp32.verification.json'
  report_path.write_text(json.dumps(report))
  fixture.enterContext(patch.object(corrected, 'ARTIFACT_SHA256', report['artifact_sha256']))
  fixture.enterContext(patch.object(corrected, 'COMPILER_SHA256', report['compiler_sha256']))
  fixture.enterContext(patch.object(corrected, 'IMPLEMENTATION_SHA256', exporter._digest(implementation)))
  fixture.enterContext(patch.object(corrected, 'REPORT_SHA256', exporter._digest(report_path)))
  fixture.enterContext(patch.object(preparation, 'PRECISION_IMPLEMENTATION', implementation))
  fixture.enterContext(patch.object(preparation, 'require_offroad'))
  # The original helper asserts the original ID only; this shared Params write
  # spy instead validates the currently selected ID, exactly as native admission.
  original_put = fixture.params.put

  def put(key, value, **kwargs):
    if key != 'NavigationModelManifest':
      return original_put(key, value, **kwargs)
    assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
    assert load_experimental_model(value, experimental=True, backend='AMD',
                                   model_id=fixture.params.get('NavigationModelCandidate')).compiler_verification_validated
    fixture.params.writes.append((key, value))
    fixture.params.values[key] = value
  fixture.params.put = put

  # Expected portable metadata remains byte-pinned for the synthetic profile.
  proof = {'scope': 'source_reload_consistency_only',
           'report': {'path': 'compiler-report.json', 'sha256': corrected.REPORT_SHA256},
           'compiler': {'path': 'compiler-source.py', 'sha256': corrected.COMPILER_SHA256},
           'source_onnx': {'sha256': corrected.SOURCE_ONNX_SHA256, 'bytes_retained': False},
           'tinygrad_root': str(preparation.CHECKOUT / 'tinygrad_repo'),
           'map_precision_implementation': {'path': 'map-precision-source.py', 'sha256': corrected.IMPLEMENTATION_SHA256}}
  import os
  proof['tinygrad_root'] = os.path.relpath(proof['tinygrad_root'], fixture.package_root / 'stage')
  contract = fixture.fixture.fixture.contract
  metadata = exporter.build_metadata(header, contract, fixture.fixture.fixture.entry['target'], proof,
                                      camera_profile=fixture.profile, correction=correction)
  fixture.enterContext(patch.object(corrected, 'METADATA_SHA256', hashlib.sha256((json.dumps(metadata, indent=2) + '\n').encode()).hexdigest()))
  return fixture, artifact, report_path, report, header


def prepare(candidate):
  fixture, *_ = candidate
  return fixture.prepare(model_id=corrected.MODEL_ID)


def test_corrected_registration_preserves_original_and_snapshots_exact_provenance(candidate):
  fixture, artifact, report_path, report, _ = candidate
  original = fixture.prepare()
  original_manifest = Path(original['manifest_path'])
  original_files = {path: path.read_bytes() for path in original_manifest.parent.iterdir() if path.is_file()}
  original_receipt = (fixture.package_root / 'latest-registration.json').read_bytes()
  artifact_before = artifact.read_bytes()
  result = prepare(candidate)
  assert result['model_id'] == corrected.MODEL_ID and result['registered'] and not result['activation_allowed']
  assert (fixture.package_root / 'latest-registration.json').read_bytes() == original_receipt
  assert all(path.read_bytes() == data for path, data in original_files.items())
  assert artifact.read_bytes() == artifact_before
  assert fixture.params.get('NavigationModelCandidate') == corrected.MODEL_ID
  assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
  entry = load_experimental_model(result['manifest_path'], experimental=True, backend='AMD', model_id=corrected.MODEL_ID)
  assert entry.artifact_path == artifact and entry.metadata['map_precision_correction'] == report['map_precision_correction']
  assert entry.metadata['compiler_verification']['map_precision_implementation']['path'] == 'map-precision-source.py'
  package = Path(result['manifest_path']).parent
  assert (package / 'compiler-report.json').read_bytes() == report_path.read_bytes()
  assert json.loads((package / 'registration.json').read_text())['model_id'] == corrected.MODEL_ID
  fixture.assert_preserved()
  with pytest.raises(ExperimentalCatalogError, match='Unknown experimental model'):
    load_experimental_model(result['manifest_path'], experimental=True, backend='AMD', model_id='big-nav-v2')
  with patch.object(preparation, 'export_experimental_model', side_effect=AssertionError('Reuse immutable package')):
    assert prepare(candidate)['manifest_path'] == result['manifest_path']


@pytest.mark.parametrize('mutation', [
  lambda r: r.update(candidate_activation_approved=True),
  lambda r: r.update(compiler_revision='0' * 40),
  lambda r: r['map_precision_correction'].update(implementation_sha256='0' * 64),
  lambda r: r['map_precision_correction']['compiled_precision']['dot'].update(early_half_casts=1),
  lambda r: r['map_precision_correction']['compiled_precision']['bias_gamma'].update(kernel_sha256='0' * 64),
  lambda r: r['map_fixture_verification']['cases']['overflow']['outputs']['next_state_feat_q'].update(source_finite=False),
  lambda r: r['map_fixture_verification']['cases'].pop('working'),
])
def test_corrected_report_or_native_kernel_discrepancy_refuses_registration(candidate, mutation):
  fixture, _, report_path, report, _ = candidate
  changed = copy.deepcopy(report)
  mutation(changed)
  report_path.write_text(json.dumps(changed))
  # Exercise semantic/kernel checks in addition to the reviewed whole-file pin.
  with patch.object(corrected, 'REPORT_SHA256', exporter._digest(report_path)), pytest.raises(ExperimentalCatalogError):
    prepare(candidate)
  assert fixture.params.get('NavigationModelManifest') == 'previous-manifest.json'
  assert fixture.params.get('NavigationModelCandidate') == 'old-candidate'
  fixture.assert_preserved()


def test_corrected_registration_rejects_unreviewed_report_bytes(candidate):
  fixture, _, report_path, *_ = candidate
  report_path.write_text(report_path.read_text() + '\n')
  with pytest.raises(ExperimentalCatalogError, match='report differs'):
    prepare(candidate)
  assert fixture.params.get('NavigationModelManifest') == 'previous-manifest.json'


def test_corrected_registration_requires_fresh_offroad_before_any_selection(candidate):
  fixture, *_ = candidate
  with patch.object(preparation, 'require_offroad', side_effect=ValueError('deviceState.started is true')):
    with pytest.raises(ValueError, match='deviceState.started'):
      prepare(candidate)
  assert fixture.params.get('NavigationModelManifest') == 'previous-manifest.json'
  assert fixture.params.writes == []


@pytest.mark.parametrize('failure', ['raise', 'ignore', 'write_then_raise'])
def test_failed_manifest_publication_restores_original_selection_and_keeps_driving_off(candidate, failure):
  fixture, *_ = candidate
  original = fixture.prepare()
  original_receipt = (fixture.package_root / 'latest-registration.json').read_bytes()
  original_put = fixture.params.put
  failed = False

  def put(key, value, **kwargs):
    nonlocal failed
    if key == 'NavigationModelManifest' and not failed:
      failed = True
      if failure == 'ignore':
        return None
      if failure == 'write_then_raise':
        original_put(key, value, **kwargs)
      raise OSError('synthetic manifest storage error')
    return original_put(key, value, **kwargs)

  with patch.object(fixture.params, 'put', side_effect=put), pytest.raises((OSError, ExperimentalCatalogError)):
    prepare(candidate)
  assert failed
  assert fixture.params.get('NavigationModelCandidate') == 'big-nav-v2'
  assert fixture.params.get('NavigationModelManifest') == original['manifest_path']
  assert not fixture.params.get_bool('NavigationModelDrivingEnabled')
  assert (fixture.package_root / 'latest-registration.json').read_bytes() == original_receipt
  assert json.loads((fixture.package_root / (corrected.MODEL_ID + '-registration.json')).read_text())['status'] == 'registration_failed'
  fixture.assert_preserved()


def test_old_registration_reuses_retained_original_compiler_after_source_update(existing):
  first = existing.prepare()
  existing.fixture.compiler.write_text('# later compiler version; original package retains its exact source\n')
  second = existing.prepare()
  assert first['manifest_path'] == second['manifest_path']
