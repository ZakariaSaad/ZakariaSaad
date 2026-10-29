"""Stdlib-only compatibility tests; inert buffers never execute as model kernels."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation import software_compatibility as compatibility
from openpilot.selfdrive.navigation.experimental_catalog import DEFAULT_CATALOG, ExperimentalCatalogError, SOURCE_ONNX_SHA256, TINYGRAD_REF

ROOT = Path(__file__).resolve().parents[4]
DTYPES = {'UINT8': 'uint8', 'FLOAT': 'float32'}
NATIVE = {'uint8': 'dtypes.u8', 'float32': 'dtypes.f32'}


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def container(path, header):
  blob_size = max([16] + [row[1]['data'][1] for row in header['check']['inputs']] +
                  [row['data'][1] for row in header['check']['outputs']])
  raw = json.dumps({'parts': [{'key': 'run', 'blob': [0, blob_size], 'header': header}], 'extras': [0, 0]}).encode()
  with path.open('wb') as output:
    output.write(b'SPTGPLN1' + struct.pack('<Q', len(raw)) + raw + bytes(-(16 + len(raw)) % 256))
    output.truncate(output.tell() + blob_size)


def captured(spec):
  return {'shape': spec['shape'], 'dtype': '|u1' if spec['dtype'] == 'uint8' else '<f4',
          'data': [0, math.prod(spec['shape']) * (1 if spec['dtype'] == 'uint8' else 4)]}


def profile():
  return {
    'scope': 'branch_pinned_warp_static_and_saved_capture',
    'artifact_sha256': 'f1689356ff8e516f87e3d75f6b53e66953a11ee5a97e859b4808c9a16bfe671b',
    'tinygrad_ref': TINYGRAD_REF, 'target': {'backend': 'AMD', 'arch': 'gfx1200', 'interface': 'USB', 'renderer': 'LLVM'},
    'camera': {'width': 1344, 'height': 760, 'stride': 1408, 'y_height': 768, 'uv_height': 384,
               'copied_frame_bytes': 1622016, 'allocated_frame_bytes': 2428928},
    'width': 1344, 'height': 760, 'stride': 1408, 'y_height': 768, 'uv_height': 384, 'frame_size': 1622016,
    'allocation_size': 2428928, 'output_shape': [2, 6, 128, 256], 'output_dtype': 'uint8', 'captured_check_passed': True,
    'source_bindings': {
      'openpilot/sunnypilot/modeld_v2/compile_warp.py': 'a182425b51918b35de8b7317461c4386e465bc5a4c9064a39e4ff64e29ca0282',
      'openpilot/system/camerad/cameras/nv12_info.py': 'b3f39bbfdd75c43d396b01f746d460068aa8e98862217d46bf3cd426cd034e1a',
      'tinygrad:examples/openpilot/compile_warp.py': '97a6814fde7f16e1ce265c6b6388cd5799c8e2f8969ac78cfc4fd5a929b1921d'},
    'captured_reference': {'passed': True, 'output_bytes': 393216, 'differing_bytes': 0,
                           'output_sha256': '124d4a811290f1c4bc1bbf0d922a80ca870a72e46cfea909f6431c97f46d72b1'},
  }


class TestSoftwareCompatibility(unittest.TestCase):
  metadata: dict[str, Any]
  header: dict[str, Any]
  fallback_header: dict[str, Any]
  report: dict[str, Any]

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.catalog = json.loads(DEFAULT_CATALOG.read_bytes())
    self.entry = self.catalog['entries'][0]
    # Generic inert format fixture, never the retired artifact or a production
    # registry identity. Corrected precision binding is tested independently.
    self.entry['id'] = 'inert-native-compatibility'
    self.entry['enabled'] = True
    self.metadata = {'schema_version': 1, 'source_onnx_sha256': SOURCE_ONNX_SHA256, 'tinygrad_ref': TINYGRAD_REF,
                     'target': self.entry['target'], 'output_slices': self.entry['contract']['output_slices'], 'camera_profile': profile()}
    for kind in ('input', 'output'):
      self.metadata[kind + '_specs'] = {row['name']: {'shape': row['shape'], 'dtype': DTYPES[row['dtype']], 'device': 'AMD'}
                                       for row in self.entry['contract'][kind + 's']}
    inputs, outputs = self.metadata['input_specs'], self.metadata['output_specs']
    names = sorted(inputs)
    slots = [(name, inputs[name]) for name in names] + [(None, value) for value in outputs.values()]
    self.header = {
      'input_names': names, 'buffers': [],
      'inputs': [{'slot': index, 'name': name, 'numel': math.prod(spec['shape']), 'dtype': NATIVE[spec['dtype']], 'device': 'AMD'}
                 for index, (name, spec) in enumerate(slots)],
      'returns': {'kind': 'NoneType', 'tensors': []},
      'calls': [{'kind': 'program', 'name': 'INERT_NOT_EXECUTABLE', 'lib': [0, 1],
                 'target': {'device': 'AMD', 'renderer': 'LLVM', 'arch': 'gfx1200', 'interface': 'USB', 'indices': ''},
                 'global_size': [1, 1, 1], 'local_size': [1, 1, 1], 'globals': list(range(len(slots))),
                 'ins': list(range(len(inputs))), 'outs': list(range(len(inputs), len(slots))),
                 'args': [{'input': index} for index in range(len(slots))],
                 'signature': [[None, index, NATIVE[spec['dtype']], [math.prod(spec['shape'])]] for index, (_, spec) in enumerate(slots)]}],
      'check': {'output_container': 'output_buffers',
                'inputs': [[[name], captured(spec), 'AMD'] for name, spec in inputs.items()] +
                          [[['output_buffers', name], captured(spec), 'AMD'] for name, spec in outputs.items()],
                'outputs': [captured(spec) for spec in outputs.values()]},
    }
    self.model, self.compiler, self.compiler_report = self.root / 'model.sptg', self.root / 'compiler.py', self.root / 'compiler-report.json'
    self.compiler.write_text('# Inert fixture compiler identity. Never executed.\n')
    self.manifest, self.sidecar, self.receipt = self.root / 'manifest.json', self.root / 'metadata.json', self.root / 'receipt.json'
    self.registry = self.root / 'registry.json'
    self.params, self.models = self.root / 'params', self.root / 'models'
    self.params.mkdir()
    self.models.mkdir()
    self.fallback_header = copy.deepcopy(self.header)
    for row in self.fallback_header['inputs']:
      row['device'] = 'QCOM'
    # Generic native layout fixture: actual ordinary-model behavior is covered
    # separately by production adapter/fallback tests, not asserted from bytes.
    self.fallback_header['input_names'] = ['ordinary_' + name for name in self.fallback_header['input_names']]
    for row in self.fallback_header['inputs']:
      if row['name'] is not None:
        row['name'] = 'ordinary_' + row['name']
    self.fallback_header['calls'][0]['target']['device'] = 'QCOM'
    container(self.models / 'small.sptg', self.fallback_header)
    (self.params / 'ModelManager_ActiveBundle').write_text(json.dumps({'minimumSelectorVersion': 21, 'runner': 'tinygrad',
                                                                   'models': [{'artifact': {'fileName': 'small.sptg',
                                                                                           'downloadUri': {'sha256': digest(self.models / 'small.sptg')}}}]}))
    self.warp = ROOT / 'openpilot/sunnypilot/modeld_v2/models/big_driving_warp_1344x760_tinygrad.pkl'
    self.approved = {'id': 'inert-native-compatibility', 'artifact_evidence_reviewed': True, 'software_contract_reviewed': True,
                     'software_integration_reviewed': True, 'runtime_compatibility': 'unverified', 'activation_approved': False,
                     'sunnypilot_generation': None,
                     'action_contract': 'comma-nav-v2-20hz-lateral-acceleration-longitudinal-acceleration'}
    def git(_root, *args, **_kwargs):
      return 'a' * 40 if args == ('rev-parse', 'HEAD') else ''
    self.enterContext(patch.object(compatibility, 'read_only_git', side_effect=git))
    self.enterContext(patch('openpilot.selfdrive.navigation.export_experimental_model._runtime_checkout'))
    self.enterContext(patch.object(os, 'geteuid', return_value=self.root.stat().st_uid if self.root.stat().st_uid else 1000))
    self.save()

  def save(self):
    container(self.model, self.header)
    target = copy.deepcopy(self.header['calls'][0]['target'])
    def result(name, spec, seed):
      value = hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest()
      return {'shape': spec['shape'], 'dtype': spec['dtype'], 'passed': True, 'source_finite': True, 'reloaded_finite': True,
              'max_abs_difference': 0.0, 'source_sha256': value, 'reloaded_sha256': value}
    outputs = [{name: result(name, spec, seed) for name, spec in self.metadata['output_specs'].items()} for seed in (42, 43)]
    self.report = {'status': 'passed_source_reload_consistency', 'artifact_promoted': True,
                   'artifact_sha256': digest(self.model), 'source_onnx_sha256': SOURCE_ONNX_SHA256, 'compiler_sha256': digest(self.compiler),
                   'tinygrad': {'revision': TINYGRAD_REF, 'dirty': False, 'git_status': ''},
                   'backend': {'kind': 'physical_device', 'emulated': False, 'targets': [target]}, 'environment': {'NO_HIPCC': None},
                   'verification': {'seeds': [42, 43], 'rtol': 0.001, 'atol': 0.001, 'equal_nan': False, 'independent_buffers': True,
                                    'primary_output_changes_with_seed': True,
                                    'runs': [{'seed': 42, 'source_repeatability': copy.deepcopy(outputs[0]), 'outputs': outputs[0]},
                                             {'seed': 43, 'outputs': outputs[1]}]}}
    self.compiler_report.write_text(json.dumps(self.report))
    self.metadata['compiler_verification'] = {
      'scope': 'source_reload_consistency_only', 'report': {'path': self.compiler_report.name, 'sha256': digest(self.compiler_report)},
      'compiler': {'path': self.compiler.name, 'sha256': digest(self.compiler)},
      'source_onnx': {'sha256': SOURCE_ONNX_SHA256, 'bytes_retained': False}, 'tinygrad_root': str(ROOT / 'tinygrad_repo')}
    self.save_metadata()

  def save_metadata(self):
    self.sidecar.write_text(json.dumps(self.metadata))
    self.entry.update(compiled_artifact={'path': self.model.name, 'sha256': digest(self.model)},
                      warp_artifact={'path': str(self.warp), 'sha256': digest(self.warp)},
                      compiler_metadata={'path': self.sidecar.name, 'sha256': digest(self.sidecar)})
    self.manifest.write_text(json.dumps(self.catalog))
    self.registration = {'prepared': True, 'registered': True, 'model_id': self.entry['id'], 'source_onnx_sha256': SOURCE_ONNX_SHA256,
                          'manifest_path': str(self.manifest), 'activation_allowed': False, 'status': 'registered_disabled',
                          'artifact_sha256': digest(self.model), 'warp_sha256': digest(self.warp), 'metadata_sha256': digest(self.sidecar),
                          'camera_profile': copy.deepcopy(self.metadata['camera_profile']), 'gpu_execution_performed': False}
    self.receipt.write_text(json.dumps(self.registration))
    self.approved.update(artifact_sha256=digest(self.model), warp_sha256=digest(self.warp), metadata_sha256=digest(self.sidecar),
                         compiler_sha256=digest(self.compiler), compiler_report_sha256=digest(self.compiler_report))
    self.registry.write_text(json.dumps({'schema_version': 1, 'candidates': [self.approved]}))

  def validate(self):
    return compatibility.validate_software_compatibility(self.receipt, registry_path=self.registry, checkout=ROOT,
                                                        params_dir=self.params, model_root=self.models)

  def test_full_real_data_only_header_proof_warp_and_fallback_validation(self):
    before = {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
    with patch('pickle.load', side_effect=AssertionError('No pickle execution')), \
         patch('pickle.loads', side_effect=AssertionError('No pickle execution')), \
         patch('subprocess.run', side_effect=AssertionError('No services, compilers or devices')), \
         patch('subprocess.Popen', side_effect=AssertionError('No process spawning')):
      report = self.validate()
    self.assertTrue(report['passed'], report)
    self.assertEqual(len(report['checks']), 7)
    self.assertEqual(report['status'], 'software_compatible')
    self.assertEqual(report['source_sha'], 'a' * 40)
    self.assertEqual(report['runtime_compatibility'], 'unverified')
    self.assertFalse(report['activation_allowed'])
    for field in ('gpu_execution_performed', 'model_recompiled', 'models_downloaded', 'normal_service_modified', 'params_modified'):
      self.assertFalse(report[field])
    self.assertEqual(before, {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()})

  def test_current_and_legacy_registration_receipts_never_grant_selection(self):
    receipt = json.loads(self.receipt.read_text())
    for status in ('registered', 'registered_disabled'):
      with self.subTest(status=status):
        receipt.update(status=status, activation_allowed=False)
        self.receipt.write_text(json.dumps(receipt))
        report = self.validate()
        self.assertTrue(report['passed'], report)
        receipt['activation_allowed'] = True
        self.receipt.write_text(json.dumps(receipt))
        self.assertFalse(self.validate()['passed'])

  def test_reviewed_integration_has_no_extra_activation_history_requirement(self):
    report = self.validate()
    self.assertTrue(report['passed'], report)
    details = next(check['details'] for check in report['checks'] if check['name'] == 'branch_reviewed_software_integration')
    self.assertEqual(details['execution_policy'], 'single_selected_model')
    self.assertEqual(details['fallback_policy'], 'initialized_for_failure_recovery')
    self.assertNotIn('valid_frame_priming', details)
    self.assertNotIn('minimum_valid_frames', self.approved)
    self.assertFalse(report['activation_allowed'])
    self.assertEqual(details['selection_policy'], 'canonical_chestnut_bundle')

  def test_receipt_candidate_uses_its_own_registry_identity_without_inheriting_approval(self):
    # Inert native fixtures test candidate dispatch and file bindings only;
    # correction-specific compiler evidence has separate registration tests.
    original = copy.deepcopy(self.approved)
    original.update(activation_approved=True, artifact_sha256='0' * 64)
    self.entry['id'] = self.approved['id'] = 'big-nav-v2-map-fp32'
    self.save_metadata()
    self.registry.write_text(json.dumps({'schema_version': 1, 'candidates': [original, self.approved]}))
    before = {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
    report = self.validate()
    # The synthetic legacy report must not become corrected-model evidence
    # merely because candidate/receipt labels agree.
    self.assertFalse(report['passed'], report)
    self.assertEqual(next(check['status'] for check in report['checks'] if check['name'] == 'registered_files_and_hashes'), 'passed')
    proof_check = next(check for check in report['checks'] if check['name'] == 'compiler_provenance_and_source_reload_evidence')
    self.assertEqual(proof_check['status'], 'failed')
    self.assertIn('reviewed evidence', proof_check['reason'])
    self.assertEqual(report['model_id'], 'big-nav-v2-map-fp32')
    self.assertEqual(report['artifact_sha256'], self.approved['artifact_sha256'])
    details = next(check['details'] for check in report['checks'] if check['name'] == 'branch_reviewed_software_integration')
    self.assertEqual(details['selection_policy'], 'canonical_chestnut_bundle')
    self.assertFalse(report['activation_allowed'])
    self.assertEqual(before, {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()})

  def test_unknown_duplicate_or_cross_candidate_receipts_are_rejected(self):
    original = copy.deepcopy(self.registration)
    for identifier in ('unknown-nav-candidate', '', None):
      with self.subTest(identifier=identifier):
        self.receipt.write_text(json.dumps(original | {'model_id': identifier}))
        self.assertFalse(self.validate()['passed'])
    self.receipt.write_text(json.dumps(original))
    self.registry.write_text(json.dumps({'schema_version': 1, 'candidates': [self.approved, self.approved]}))
    self.assertFalse(self.validate()['passed'])
    # Matching hashes cannot substitute a different model's entry/receipt.
    self.entry['id'] = 'big-nav-v2-map-fp32'
    self.manifest.write_text(json.dumps(self.catalog))
    self.registry.write_text(json.dumps({'schema_version': 1, 'candidates': [self.approved]}))
    self.assertFalse(self.validate()['passed'])

  def test_registered_entry_requires_the_same_candidate_as_its_branch_record(self):
    other = self.approved | {'id': 'big-nav-v2-map-fp32'}
    with self.assertRaisesRegex(ExperimentalCatalogError, 'identity mismatch'):
      compatibility._registered_entry(self.registration, self.manifest.resolve(), other)

  def test_artifact_receipt_metadata_report_and_compiler_tampering_rejected(self):
    for path in (self.model, self.sidecar, self.compiler_report, self.compiler):
      with self.subTest(file=path.name):
        old = path.read_bytes()
        path.write_bytes(old + b'\n')
        report = self.validate()
        self.assertFalse(report['passed'], report)
        self.assertTrue(any('SHA256 mismatch' in item.get('reason', '') for item in report['checks']), report)
        path.write_bytes(old)
    receipt = copy.deepcopy(self.registration)
    receipt['metadata_sha256'] = '0' * 64
    self.receipt.write_text(json.dumps(receipt))
    self.assertFalse(self.validate()['passed'])

  def test_kernel_slot_range_and_access_metadata_rejected_with_bound_rehashed_files(self):
    mutations = [lambda h: h['calls'][0]['args'][0].update(input=100),
                 lambda h: h['calls'][0].update(global_size=[0, 1, 1]),
                 lambda h: h['calls'][0].update(globals=[0] * 17),
                 lambda h: h['calls'][0]['signature'][0].__setitem__(3, [1 << 40]),
                 lambda h: h['calls'][0].update(ins=[999]),
                 lambda h: h['inputs'][0].update(slot=16),
                 lambda h: h['calls'][0].update(lib=[0, 1 << 40])]
    original = copy.deepcopy(self.header)
    for index, mutate in enumerate(mutations):
      with self.subTest(case=index):
        self.header = copy.deepcopy(original)
        mutate(self.header)
        self.save()
        self.assertFalse(self.validate()['passed'])
    self.header = original

  def test_complete_verification_evidence_and_physical_target_remain_required(self):
    original = copy.deepcopy(self.report)
    for mutate in (lambda r: r['backend'].update(emulated=True),
                   lambda r: r['verification'].update(independent_buffers=False),
                   lambda r: r['verification']['runs'][1]['outputs'].pop('next_state_navigation_features_q'),
                   lambda r: r['verification']['runs'][0]['outputs']['outputs'].update(source_finite=False)):
      with self.subTest(mutation=mutate):
        self.report = copy.deepcopy(original)
        mutate(self.report)
        self.compiler_report.write_text(json.dumps(self.report))
        self.metadata['compiler_verification']['report']['sha256'] = digest(self.compiler_report)
        self.save_metadata()
        self.assertFalse(self.validate()['passed'])

  def test_camera_profile_source_geometry_and_state_contract_rejected(self):
    original = copy.deepcopy(self.metadata)
    mutations = [lambda m: m['camera_profile'].update(width=1342),
                 lambda m: m['camera_profile']['source_bindings'].update({'openpilot/system/camerad/cameras/nv12_info.py': '0' * 64}),
                 lambda m: m['output_specs']['next_state_feat_q'].update(shape=[128, 1, 1]),
                 lambda m: m['input_specs']['nav_map'].update(dtype='uint8'),
                 lambda m: m['output_slices']['action'].update(start=2063)]
    for mutate in mutations:
      with self.subTest(mutation=mutate):
        self.metadata = copy.deepcopy(original)
        mutate(self.metadata)
        self.save_metadata()
        self.assertFalse(self.validate()['passed'])

  def test_missing_or_incompatible_fallback_does_not_change_selected_models(self):
    bundle = self.params / 'ModelManager_ActiveBundle'
    original = bundle.read_bytes()
    bundle.write_text('{}')
    self.assertFalse(self.validate()['passed'])
    self.assertEqual(bundle.read_text(), '{}')
    bundle.write_bytes(original)
    (self.models / 'small.sptg').unlink()
    self.assertFalse(self.validate()['passed'])
    self.assertEqual(bundle.read_bytes(), original)

  def test_root_and_owner_mismatch_fail_before_any_receipt_or_model_reads(self):
    with patch.object(os, 'geteuid', return_value=0), patch.object(compatibility, '_json_file', side_effect=AssertionError('No reads')):
      report = self.validate()
    self.assertFalse(report['passed'])
    self.assertIn('without sudo', report['checks'][0]['reason'])

  def test_no_third_party_or_native_imports_under_system_python_no_site(self):
    script = '''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
  if name.split('.')[0] in {'numpy', 'tinygrad', 'capnp', 'requests', 'cereal'} or name.startswith('openpilot.common.params'):
    raise AssertionError('Unexpected native or third-party import: ' + name)
  return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from openpilot.selfdrive.navigation import software_compatibility
print('stdlib-only')
'''
    result = subprocess.run([sys.executable, '-S', '-B', '-c', script], cwd=ROOT, capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip(), 'stdlib-only')

  def test_report_output_protects_inputs_params_checkout_and_symlinks(self):
    report = self.validate()
    output = self.root / 'software-compatibility.json'
    arguments = {'receipt_path': self.receipt, 'checkout': ROOT, 'params_dir': self.params, 'model_root': self.models}
    # The real receipt/package share root in this fixture. Use a separate
    # directory to exercise the same protected package-parent rule as /data.
    with tempfile.TemporaryDirectory() as outside:
      output = Path(outside) / 'software-compatibility.json'
      compatibility.write_compatibility_report(report, output, **arguments)
      self.assertEqual(json.loads(output.read_bytes()), report)
      compatibility.write_compatibility_report(report, output, **arguments)
      self.assertEqual(json.loads(output.read_bytes()), report)
      symlink = Path(outside) / 'symlink.json'
      symlink.symlink_to(self.receipt)
      with self.assertRaisesRegex(ExperimentalCatalogError, 'symlink'):
        compatibility.write_compatibility_report(report, symlink, **arguments)
      existing = Path(outside) / 'unrelated.json'
      existing.write_text('{}')
      with self.assertRaisesRegex(ExperimentalCatalogError, 'not a software compatibility report'):
        compatibility.write_compatibility_report(report, existing, **arguments)
    for destination in (self.receipt, self.manifest, self.sidecar, self.params / 'new.json', ROOT / 'new.json'):
      with self.subTest(output=destination), self.assertRaisesRegex(ExperimentalCatalogError, 'collides'):
        compatibility.write_compatibility_report(report, destination, **arguments)
    with patch.object(os, 'geteuid', return_value=0), self.assertRaisesRegex(ExperimentalCatalogError, 'without sudo'):
      compatibility.write_compatibility_report(report, self.root / 'new.json', **arguments)

  def test_dirty_source_and_unreviewed_integration_fail_without_enabling_driving(self):
    with patch.object(compatibility, 'read_only_git', side_effect=lambda _root, *args: 'changed' if args[0] == 'status' else 'a' * 40):
      report = self.validate()
    self.assertFalse(report['passed'])
    self.assertIn('differs from the reviewed commit', report['checks'][-1]['reason'])
    self.approved['software_integration_reviewed'] = False
    self.registry.write_text(json.dumps({'schema_version': 1, 'candidates': [self.approved]}))
    self.assertFalse(self.validate()['passed'])

  def test_fallback_hash_header_target_and_navigation_free_contract(self):
    path = self.models / 'small.sptg'
    original = path.read_bytes()
    path.write_bytes(original + b'changed')
    report = self.validate()
    self.assertFalse(report['passed'])
    self.assertIn('fallback artifact SHA256 mismatch', report['checks'][-2]['reason'])
    for mutate in (lambda h: h['calls'][0]['target'].update(device='AMD'),
                   lambda h: h['input_names'].append('nav_map')):
      with self.subTest(mutation=mutate):
        header = copy.deepcopy(self.fallback_header)
        mutate(header)
        container(path, header)
        bundle = json.loads((self.params / 'ModelManager_ActiveBundle').read_bytes())
        bundle['models'][0]['artifact']['downloadUri']['sha256'] = digest(path)
        (self.params / 'ModelManager_ActiveBundle').write_text(json.dumps(bundle))
        self.assertFalse(self.validate()['passed'])

  def test_real_branch_warp_layout_no_device_or_kernel_execution(self):
    with patch('pickle.loads', side_effect=AssertionError('No pickle')), patch('pickle.load', side_effect=AssertionError('No pickle')):
      header = compatibility._header(self.warp, self.entry['target'])
      result = compatibility.validate_native_layout(header)
    self.assertEqual(result['programs'], 7)
    self.assertEqual(result['input_output_slots'], 2)

  def test_buffer_views_and_copy_layout_bounds(self):
    header = {'buffers': [{'nbytes': 16, 'dtype': 'dtypes.f32'}], 'inputs': [],
              'calls': [{'kind': 'copy', 'args': [{'buffer': 0, 'offset': 0, 'nbytes': 8, 'dtype': 'dtypes.f32'},
                                                {'buffer': 0, 'offset': 8, 'nbytes': 8, 'dtype': 'dtypes.f32'}]}]}
    self.assertEqual(compatibility.validate_native_layout(header)['copies'], 1)
    header['calls'][0]['args'][0]['offset'] = 12
    with self.assertRaisesRegex(ExperimentalCatalogError, 'exceeds'):
      compatibility.validate_native_layout(header)

  def test_kernel_globals_select_outer_arguments_with_compact_signature_slots(self):
    header = copy.deepcopy(self.header)
    call = header['calls'][0]
    call.update(globals=[2, 0], ins=[0], outs=[2],
                signature=[[None, 0, 'dtypes.u8', [1]], [None, 1, 'dtypes.u8', [1]]])
    self.assertEqual(compatibility.validate_native_layout(header)['programs'], 1)
    mutations = [lambda c: c.update(globals=[2, 2]),
                 lambda c: c.update(globals=[-1, 0]),
                 lambda c: c.update(globals=[len(c['args']), 0]),
                 lambda c: c.update(globals=[True, 0]),
                 lambda c: c['signature'][0].__setitem__(1, 2),  # Valid outer index; invalid compact slot.
                 lambda c: c['signature'][1].__setitem__(1, 0),
                 lambda c: c['signature'][0].__setitem__(3, [1 << 40]),
                 lambda c: c['args'].__setitem__(1, {'input': 100000})]  # Unused arguments still require valid views.
    for index, mutate in enumerate(mutations):
      with self.subTest(mutation=index):
        changed = copy.deepcopy(header)
        mutate(changed['calls'][0])
        with self.assertRaises(ExperimentalCatalogError):
          compatibility.validate_native_layout(changed)


if __name__ == '__main__':
  unittest.main()
