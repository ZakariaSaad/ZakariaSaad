"""Data-only provenance tests using inert SPTG fixtures, never an AMD model.

Positive export fixtures explicitly replace only the unavailable 814 MB public
source-byte identity and Git checkout checks. Artifact/header/report bindings,
all negative report checks and manifest roundtrips execute the real code.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.navigation import export_experimental_model as exporter
from openpilot.selfdrive.navigation.experimental_catalog import (
  DEFAULT_CATALOG, SOURCE_ONNX_SHA256, TINYGRAD_REF, ExperimentalCatalogError, load_experimental_model,
)
from openpilot.selfdrive.navigation.tests import test_experimental_catalog as fixtures


def output_result(tensor, digest):
  return {'shape': tensor['shape'], 'dtype': tensor['dtype'], 'passed': True, 'source_finite': True, 'reloaded_finite': True,
          'max_abs_difference': 0.0, 'source_sha256': digest, 'reloaded_sha256': digest}


class TestExperimentalExport(unittest.TestCase):
  report: dict[str, Any]

  def setUp(self):
    self.fixture = fixtures.TestExperimentalCatalog('runTest')
    self.fixture.setUp()
    self.addCleanup(self.fixture.doCleanups)
    self.fixture.save()
    self.root = self.fixture.root
    self.compiler = self.root / 'compiler.py'
    self.compiler.write_text('# Inert compiler identity fixture; never executed.\n')
    self.source = self.root / 'source.onnx'
    self.source.write_bytes(b'Inert source identity fixture, not an ONNX model')
    self.runtime = self.root / 'tinygrad'
    self.runtime.mkdir()
    self.report_path = self.root / 'verification.json'
    self.output = self.root / 'ready.json'
    outputs = self.fixture.metadata['output_specs']
    results = [{name: output_result(spec, hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest())
                for name, spec in outputs.items()} for seed in (42, 43)]
    target = copy.deepcopy(self.fixture.model_header['calls'][0]['target'])
    self.report = {
      'status': 'passed_source_reload_consistency', 'artifact_promoted': True,
      'artifact_sha256': self.fixture.entry['compiled_artifact']['sha256'], 'source_onnx_sha256': SOURCE_ONNX_SHA256,
      'compiler_sha256': hashlib.sha256(self.compiler.read_bytes()).hexdigest(),
      'tinygrad': {'revision': TINYGRAD_REF, 'dirty': False, 'git_status': ''},
      'backend': {'kind': 'physical_device', 'emulated': False, 'targets': [target]}, 'environment': {'NO_HIPCC': None},
      'verification': {'seeds': [42, 43], 'rtol': 0.001, 'atol': 0.001, 'equal_nan': False,
                       'independent_buffers': True, 'primary_output_changes_with_seed': True,
                       'runs': [{'seed': 42, 'source_repeatability': copy.deepcopy(results[0]), 'outputs': results[0]},
                                {'seed': 43, 'source_repeatability': None, 'outputs': results[1]}]},
    }

  def export(self):
    self.report_path.write_text(json.dumps(self.report))
    return exporter.export_experimental_model(
      self.root / 'model.sptg', self.root / 'warp.sptg', self.report_path, self.source,
      self.compiler, self.runtime, self.output, experimental=True,
    )

  def test_data_only_fixture_roundtrip_binds_real_local_hashes_and_remains_non_actuating(self):
    baseline = DEFAULT_CATALOG.read_bytes()
    with patch.object(exporter, '_source_file') as source_check, patch.object(exporter, '_runtime_checkout') as runtime_check, \
         patch('pickle.load', side_effect=AssertionError('No pickle reads')), \
         patch('pickle.loads', side_effect=AssertionError('No pickle reads')):
      self.assertEqual(self.export(), self.output)
      entry = load_experimental_model(self.output, experimental=True, backend='AMD')
    self.assertTrue(entry.compiler_verification_validated)
    self.assertFalse(entry.normal_activation_allowed)
    self.assertFalse(entry.selector_generation_verified)
    self.assertEqual(entry.selector21_bundle['generation'], 0)
    self.assertGreaterEqual(source_check.call_count, 3)
    self.assertGreaterEqual(runtime_check.call_count, 3)
    self.assertEqual(DEFAULT_CATALOG.read_bytes(), baseline)
    proof = entry.metadata['compiler_verification']
    for key in ('report', 'compiler', 'source_onnx'):
      record = proof[key]
      self.assertEqual(record['sha256'], hashlib.sha256(Path(record['path']).read_bytes()).hexdigest())
    self.assertFalse(entry.metadata['hardware_driving_validation'])

  def test_real_wrong_source_bytes_cannot_be_exported(self):
    with patch.object(exporter, '_runtime_checkout'), self.assertRaisesRegex(ExperimentalCatalogError, 'pinned public model'):
      self.export()
    self.assertFalse(self.output.exists())
    self.assertFalse(self.output.with_name(self.output.name + '.metadata.json').exists())

  def test_signed_zero_reload_matches_the_actual_compiler_comparison(self):
    from openpilot.sunnypilot.modeld_v2.compile_modeld_v2 import compare_outputs
    before, after = np.array([-0.0], np.float32), np.array([0.0], np.float32)
    comparison = compare_outputs({'outputs': before}, {'outputs': after})
    contract = {'outputs': [{'name': 'outputs', 'shape': [1], 'dtype': 'FLOAT'}]}
    self.assertNotEqual(comparison['outputs']['source_sha256'], comparison['outputs']['reloaded_sha256'])
    self.assertEqual(comparison['outputs']['max_abs_difference'], 0)
    exporter._output_comparison(comparison, contract)
    with self.assertRaises(ExperimentalCatalogError):
      exporter._output_comparison(comparison, contract, exact=True)

  def test_failed_emulated_loose_incomplete_and_inconsistent_reports_fail_closed(self):
    cases = [
      lambda r: r.update(status='verified_pending_promotion'),
      lambda r: r.update(artifact_promoted=False),
      lambda r: r.update(artifact_sha256='0' * 64),
      lambda r: r.update(compiler_sha256='0' * 64),
      lambda r: r.update(source_onnx_sha256='0' * 64),
      lambda r: r['tinygrad'].update(dirty=True),
      lambda r: r['tinygrad'].update(revision='0' * 40),
      lambda r: r['backend'].update(emulated=True),
      lambda r: r['backend'].update(kind='host_cpu'),
      lambda r: r['backend']['targets'][0].update(renderer='NULL'),
      lambda r: r['backend']['targets'][0].update(interface='MOCKUSB'),
      lambda r: r['environment'].update(NO_HIPCC='1'),
      lambda r: r['verification'].update(independent_buffers=False),
      lambda r: r['verification'].update(rtol=1.0),
      lambda r: r['verification'].update(equal_nan=True),
      lambda r: r['verification'].update(seeds=[42, 42]),
      lambda r: r['verification']['runs'].pop(),
      lambda r: r['verification']['runs'][0].update(source_repeatability=None),
      lambda r: r['verification']['runs'][0]['outputs']['outputs'].update(source_finite=False),
      lambda r: r['verification']['runs'][0]['outputs']['outputs'].update(passed=False),
      lambda r: r['verification']['runs'][0]['outputs']['outputs'].update(max_abs_difference=0.1),
      lambda r: r['verification']['runs'][0]['outputs']['outputs'].update(shape=[1, 18452]),
      lambda r: r['verification']['runs'][1]['outputs']['outputs'].update(
        source_sha256=r['verification']['runs'][0]['outputs']['outputs']['source_sha256'],
        reloaded_sha256=r['verification']['runs'][0]['outputs']['outputs']['source_sha256']),
    ]
    for index, mutate in enumerate(cases):
      with self.subTest(case=index), patch.object(exporter, '_source_file'), patch.object(exporter, '_runtime_checkout'):
        original = copy.deepcopy(self.report)
        mutate(self.report)
        with self.assertRaises(ExperimentalCatalogError):
          self.export()
        self.assertFalse(self.output.exists())
        self.report = original

  def test_exported_provenance_is_rechecked_on_load_and_files_are_not_overwritten(self):
    with patch.object(exporter, '_source_file'), patch.object(exporter, '_runtime_checkout'):
      self.export()
      first = self.output.read_bytes()
      with self.assertRaisesRegex(ExperimentalCatalogError, 'new local output paths'):
        self.export()
      self.assertEqual(self.output.read_bytes(), first)
      self.report_path.write_text(self.report_path.read_text() + '\n')
      with self.assertRaisesRegex(ExperimentalCatalogError, 'Verification report SHA256 mismatch'):
        load_experimental_model(self.output, experimental=True, backend='AMD')
      self.report_path.write_text(json.dumps(self.report))
      self.compiler.write_text('# Changed compiler bytes\n')
      with self.assertRaisesRegex(ExperimentalCatalogError, 'Compiler source SHA256 mismatch'):
        load_experimental_model(self.output, experimental=True, backend='AMD')

  def test_source_compiler_and_runtime_binding_checks_use_real_read_only_operations(self):
    with patch('subprocess.run', side_effect=[SimpleNamespace(returncode=0, stdout=TINYGRAD_REF + '\n'),
                                             SimpleNamespace(returncode=0, stdout='')]) as git:
      exporter._runtime_checkout(self.runtime)
    self.assertEqual(git.call_args_list[0].args[0], ['git', '-C', str(self.runtime), 'rev-parse', 'HEAD'])
    for revision, status in (('0' * 40, ''), (TINYGRAD_REF, ' M tinygrad/tensor.py')):
      with self.subTest(revision=revision, status=status), \
           patch('subprocess.run', side_effect=[SimpleNamespace(returncode=0, stdout=revision), SimpleNamespace(returncode=0, stdout=status)]), \
           self.assertRaises(ExperimentalCatalogError):
        exporter._runtime_checkout(self.runtime)

  def test_static_fixture_admission_does_not_claim_compiler_verification(self):
    entry = load_experimental_model(self.fixture.manifest, experimental=True, backend='AMD')
    self.assertFalse(entry.compiler_verification_validated)
