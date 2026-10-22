"""Inert binary/JSON admission tests: no weights, kernels, downloads or devices run."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation.experimental_catalog import (
  DEFAULT_CATALOG, ExperimentalCatalogError, SOURCE_ONNX_SHA256, TINYGRAD_REF,
  load_experimental_catalog, load_experimental_model,
)
from openpilot.sunnypilot.models.fetcher import ModelParser

BASELINE = Path(__file__).parent / 'fixtures/chestnut_v28_baseline.json'
BASELINE_SHA256 = 'd2edd5f2d0de92b084b9e7a6f8f34642a934f87588adb5ce129bedd3bda5b46d'
DTYPE = {'FLOAT': 'float32', 'UINT8': 'uint8'}
RAW_DTYPE = {'float32': 'dtypes.f32', 'uint8': 'dtypes.u8'}
NP_DTYPE = {'float32': '<f4', 'uint8': '|u1'}


def tensor_size(spec):
  return math.prod(spec['shape']) * (4 if spec['dtype'] == 'float32' else 1)


def captured(spec):
  return {'data': [0, tensor_size(spec)], 'dtype': NP_DTYPE[spec['dtype']], 'shape': spec['shape']}


def binary_input(slot, name, spec):
  return {'slot': slot, 'name': name, 'dtype': RAW_DTYPE[spec['dtype']],
          'numel': math.prod(spec['shape']), 'device': spec['device']}


def make_container(path, header, target):
  # These bytes contain no executable kernel. They are admission fixtures only;
  # their real digests cannot be mistaken for a compiled public driving model.
  header['calls'] = [{'kind': 'program', 'target': target | {'device': target['backend'], 'renderer': 'LLVM'}, 'lib': [0, 1]}]
  header.setdefault('buffers', [])
  blob_size = max([1] + [row['data'][1] for _, row, _ in header.get('check', {}).get('inputs', [])])
  container = {'parts': [{'key': 'run', 'blob': [0, blob_size], 'header': header}], 'extras': [0, 0]}
  raw = json.dumps(container).encode()
  with path.open('wb') as file:
    file.write(b'SPTGPLN1' + struct.pack('<Q', len(raw)) + raw + bytes(-(16 + len(raw)) % 256))
    file.write(bytes(blob_size))
  return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class TestExperimentalCatalog(unittest.TestCase):
  metadata: dict[str, Any]
  model_header: dict[str, Any]
  warp_header: dict[str, Any]

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.catalog = json.loads(DEFAULT_CATALOG.read_bytes())
    self.entry = self.catalog['entries'][0]
    self.contract = self.entry['contract']
    self.metadata = {
      'schema_version': 1, 'source_onnx_sha256': SOURCE_ONNX_SHA256,
      'tinygrad_ref': TINYGRAD_REF, 'target': self.entry['target'],
      'output_slices': self.contract['output_slices'],
    }
    for kind in ('input', 'output'):
      self.metadata[kind + '_specs'] = {
        row['name']: {'shape': row['shape'], 'dtype': DTYPE[row['dtype']], 'device': 'AMD'}
        for row in self.contract[kind + 's']
      }
    inputs, outputs = self.metadata['input_specs'], self.metadata['output_specs']
    self.model_header = {
      'input_names': sorted(inputs),
      'inputs': [binary_input(i, name, inputs[name]) for i, name in enumerate(sorted(inputs))] +
                [binary_input(len(inputs) + i, None, spec) for i, spec in enumerate(outputs.values())],
      'returns': {'kind': 'NoneType', 'tensors': []},
      'check': {
        'output_container': 'output_buffers',
        'inputs': [[[name], captured(spec), spec['device']] for name, spec in inputs.items()] +
                  [[['output_buffers', name], captured(spec), spec['device']] for name, spec in outputs.items()],
        'outputs': [captured(spec) for spec in outputs.values()],
      },
    }
    self.warp_header = {
      'input_names': ['M_inv', 'input_frame'],
      'inputs': [binary_input(0, 'M_inv', {'shape': [2, 3, 3], 'dtype': 'float32', 'device': 'AMD'}),
                 binary_input(1, 'input_frame', {'shape': [2, 1024], 'dtype': 'uint8', 'device': 'AMD'})],
      'returns': {'kind': 'Tensor', 'tensors': [{'shape': [2, 6, 128, 256], 'dtype': 'dtypes.u8'}]},
    }
    self.entry['enabled'] = True
    self.manifest = self.root / 'manifest.json'

  def save(self, model_target=None, warp_target=None):
    target = self.entry['target']
    self.entry['compiled_artifact'] = make_container(self.root / 'model.sptg', self.model_header, model_target or target)
    self.entry['warp_artifact'] = make_container(self.root / 'warp.sptg', self.warp_header, warp_target or target)
    metadata = self.root / 'compiler.json'
    metadata.write_text(json.dumps(self.metadata))
    self.entry['compiler_metadata'] = {'path': metadata.name, 'sha256': hashlib.sha256(metadata.read_bytes()).hexdigest()}
    self.manifest.write_text(json.dumps(self.catalog))

  def load(self, **kwargs):
    return load_experimental_model(self.manifest, experimental=True, backend='AMD', **kwargs)

  def test_checked_in_template_is_disabled_and_production_parser_sees_no_bundles(self):
    catalog = load_experimental_catalog(DEFAULT_CATALOG, experimental=True)
    self.assertEqual(ModelParser.parse_models(catalog), [])
    self.assertEqual(ModelParser.parse_models({'bundles': [catalog['entries'][0]['selector21_bundle']]})[0].generation, 0)
    with self.assertRaisesRegex(ExperimentalCatalogError, 'disabled'):
      load_experimental_model(DEFAULT_CATALOG, experimental=True, backend='AMD')
    invalid_contexts: list[Any] = [False, 1, 'true', None]
    for enabled in invalid_contexts:
      with self.subTest(context=enabled), self.assertRaisesRegex(ExperimentalCatalogError, 'experimental=True'):
        load_experimental_catalog(DEFAULT_CATALOG, experimental=enabled)

  def test_local_matching_header_metadata_and_real_hashes_admit_diagnostic_inspection_only(self):
    self.save()
    with patch('pickle.load', side_effect=AssertionError('No unpickling')), \
         patch('pickle.loads', side_effect=AssertionError('No unpickling')), \
         patch('requests.get', side_effect=AssertionError('No network')):
      entry = self.load()
    self.assertEqual(entry.artifact_path, self.root / 'model.sptg')
    self.assertEqual(entry.warp_path, self.root / 'warp.sptg')
    self.assertEqual(entry.warp_frame_size, 1024)
    self.assertEqual(entry.input_specs['nav_map'], ((1, 1, 256, 256), 'float32', 'AMD'))
    self.assertEqual(entry.input_specs['navigation'], ((1, 47), 'float32', 'AMD'))
    self.assertFalse(entry.normal_activation_allowed)
    self.assertFalse(entry.selector_generation_verified)
    self.assertEqual(entry.selector21_bundle['generation'], 0)
    parsed = ModelParser.parse_models({'bundles': [entry.selector21_bundle]})
    self.assertEqual(len(parsed), 1)
    self.assertEqual(parsed[0].models[0].artifact.downloadUri.sha256, entry.artifact_sha256)
    self.assertTrue(parsed[0].models[0].artifact.downloadUri.uri.startswith('file://'))
    self.assertNotEqual(entry.artifact_sha256, SOURCE_ONNX_SHA256)
    self.assertEqual(ModelParser.parse_models(json.loads(self.manifest.read_bytes())), [])

  def test_baseline_all_17_chestnut_models_are_byte_preserved_and_parse_unchanged(self):
    before = BASELINE.read_bytes()
    self.assertEqual(hashlib.sha256(before).hexdigest(), BASELINE_SHA256)
    expected = [bundle.to_dict() for bundle in ModelParser.parse_models(json.loads(before))]
    self.assertEqual(len(expected), 17)
    self.save()
    self.load()
    self.assertEqual(BASELINE.read_bytes(), before)
    actual = [bundle.to_dict() for bundle in ModelParser.parse_models(json.loads(BASELINE.read_bytes()))]
    self.assertEqual(actual, expected)

  def test_source_runtime_scope_and_unverified_generation_are_pinned(self):
    mutations = [
      ('source', lambda: self.entry['source_onnx'].update(sha256='0' * 64)),
      ('runtime', lambda: self.entry['runtime'].update(tinygrad_ref='0' * 40)),
      ('scope', lambda: self.catalog.update(scope='production')),
      ('production_root', lambda: self.catalog.update(bundles=[])),
      ('activation', lambda: self.entry.update(normal_activation_allowed=True)),
      ('generation_verified', lambda: self.entry.update(selector_generation_verified=True)),
      ('generation_value', lambda: self.entry['selector21_bundle'].update(generation=12)),
      ('contract', lambda: self.contract['inputs'][4].update(dtype='UINT8')),
    ]
    for name, mutate in mutations:
      with self.subTest(case=name):
        snapshot = copy.deepcopy(self.catalog)
        mutate()
        self.save()
        with self.assertRaises(ExperimentalCatalogError):
          self.load()
        self.catalog = snapshot
        self.entry = self.catalog['entries'][0]
        self.contract = self.entry['contract']

  def test_all_local_file_digests_are_required_and_stream_verified(self):
    for field in ('compiled_artifact', 'warp_artifact', 'compiler_metadata'):
      for wrong in (None, 'f' * 64):
        with self.subTest(field=field, digest=wrong):
          self.save()
          self.entry[field]['sha256'] = wrong
          self.manifest.write_text(json.dumps(self.catalog))
          with self.assertRaisesRegex(ExperimentalCatalogError, 'SHA256'):
            self.load()
    self.save()
    with (self.root / 'model.sptg').open('ab') as file:
      file.write(b'changed')
    with self.assertRaisesRegex(ExperimentalCatalogError, 'SHA256 mismatch'):
      self.load()

  def test_remote_paths_wrong_backend_target_or_metadata_provenance_fail_closed(self):
    self.save()
    self.entry['compiled_artifact']['path'] = 'https://example.invalid/model.sptg'
    self.manifest.write_text(json.dumps(self.catalog))
    with self.assertRaisesRegex(ExperimentalCatalogError, 'local file'):
      self.load()
    self.save()
    with self.assertRaisesRegex(ExperimentalCatalogError, 'AMD gfx1200 USB'):
      load_experimental_model(self.manifest, experimental=True, backend='CPU')
    with self.assertRaisesRegex(ExperimentalCatalogError, 'AMD gfx1200 USB'):
      self.load(expected_arch='gfx1100')
    for field, value in (('source_onnx_sha256', 'f' * 64), ('tinygrad_ref', 'f' * 40), ('target', {})):
      with self.subTest(metadata=field):
        previous = self.metadata[field]
        self.metadata[field] = value
        self.save()
        with self.assertRaisesRegex(ExperimentalCatalogError, 'metadata provenance'):
          self.load()
        self.metadata[field] = previous
    for which in ('model_target', 'warp_target'):
      for field, value in (('backend', 'CPU'), ('arch', 'gfx1100'), ('interface', 'PCI')):
        with self.subTest(artifact=which, target=field):
          self.save(**{which: self.entry['target'] | {field: value}})
          with self.assertRaisesRegex(ExperimentalCatalogError, 'kernel target'):
            self.load()

  def test_tensor_dtype_shape_state_output_slots_and_slices_are_checked(self):
    cases = [
      lambda: self.model_header['inputs'][0].update(dtype='dtypes.u8'),
      lambda: self.model_header['inputs'][0].update(numel=200),
      lambda: self.model_header['inputs'][0].update(slot=1),
      lambda: self.model_header['input_names'].reverse(),
      lambda: self.model_header['inputs'][-1].update(numel=123),
      lambda: self.model_header['check']['inputs'][0][1].update(shape=[1]),
      lambda: self.model_header['check']['inputs'][0][1].update(data=[0, 0]),
      lambda: self.metadata['input_specs']['nav_map'].update(dtype='uint8'),
      lambda: self.metadata['output_specs']['next_state_map_features_q'].update(shape=[32, 1, 64]),
      lambda: self.metadata.update(output_slices={}),
      lambda: self.warp_header['returns']['tensors'][0].update(shape=[1, 6, 128, 256]),
      lambda: self.warp_header['inputs'][0].update(numel=9),
    ]
    for index, mutate in enumerate(cases):
      with self.subTest(case=index):
        snapshots = copy.deepcopy((self.model_header, self.warp_header, self.metadata))
        mutate()
        self.save()
        with self.assertRaises(ExperimentalCatalogError):
          self.load()
        self.model_header, self.warp_header, self.metadata = snapshots

  def test_legacy_pickle_truncated_container_and_unbounded_headers_are_rejected_without_execution(self):
    for content in (b'not-a-native-pickle', b'SPTGPLN1', b'SPTGPLN1' + struct.pack('<Q', 1 << 40),
                    b'SPTGPLN1' + struct.pack('<Q', 12) + b'{}'):
      with self.subTest(content=content):
        self.save()
        artifact = self.root / 'model.sptg'
        artifact.write_bytes(content)
        self.entry['compiled_artifact']['sha256'] = hashlib.sha256(content).hexdigest()
        self.manifest.write_text(json.dumps(self.catalog))
        with self.assertRaises(ExperimentalCatalogError):
          self.load()

  def test_duplicate_json_keys_and_unknown_model_ids_are_rejected(self):
    self.save()
    with self.assertRaisesRegex(ExperimentalCatalogError, 'Unknown experimental model'):
      self.load(model_id='unapproved-model')
    raw = self.manifest.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
    self.manifest.write_text(raw)
    with self.assertRaisesRegex(ExperimentalCatalogError, 'Duplicate JSON key'):
      self.load()
