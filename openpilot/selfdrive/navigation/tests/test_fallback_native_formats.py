"""Production-writer fallback bundles captured from tiny trusted CPU graphs.

The real compiler serializer and bundle writer generate these artifacts. Only
their JSON device/target labels are changed to QCOM for static-parser tests;
the stored CPU machine code is never run as QCOM and proves no GPU behavior.
No driving-model weights, ONNX downloads or vehicle interfaces are involved.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, TinyJit

from openpilot.selfdrive.navigation import software_compatibility as compatibility
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError
from openpilot.sunnypilot.modeld_v2.helpers import load_model, read_parts, write_bundle
from openpilot.sunnypilot.modeld_v2.plain_artifact_compiler import capture_frame, record_kernel_linear, serialize_jit


def serialized_part(function, arrays):
  """Use the same native capture/serializer as compile_modeld.compile_jit."""
  jit = TinyJit(function, prune=True)
  with record_kernel_linear() as recorded:
    for _ in range(2):
      jit(**arrays)
  if len(recorded) != 1:
    raise AssertionError('Expected one trusted CPU kernel capture')
  check, _ = capture_frame(jit, arrays)
  header, blob = serialize_jit(jit, recorded[0], check)
  return header, bytes(blob)


def qcom_part(part):
  header, blob = copy.deepcopy(part[0]), part[1]
  for row in header['buffers'] + header['inputs']:
    if row['device'].startswith('CPU'):
      row['device'] = 'QCOM'
  for call in header['calls']:
    if call['kind'] == 'program':
      call['target'] = {'device': 'QCOM', 'renderer': 'IR3', 'arch': 'a630', 'interface': 'KGSL', 'indices': ''}
  for entry in header['check']['inputs']:
    if entry[2].startswith('CPU'):
      entry[2] = 'QCOM'
  return header, blob


def document(path):
  data = path.read_bytes()
  size = struct.unpack('<Q', data[8:16])[0]
  start = 16 + size
  return json.loads(data[16:start]), data[start + -start % 256:]


def rewrite(path, metadata, blob):
  raw = json.dumps(metadata).encode()
  path.write_bytes(b'SPTGPLN1' + struct.pack('<Q', len(raw)) + raw + bytes(-(16 + len(raw)) % 256) + blob)


class TestProductionFallbackFormats(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # The graph bodies are deliberately tiny; the entrypoint argument names
    # are the actual compile_modeld.py policy/warp compiler calling contracts.
    def policy(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
      return (warped + img_q + big_img_q + feat_q + desire_q + packed_npy_inputs).realize()

    def warp(frame, big_frame, tfm, big_tfm):
      return (frame.cast('float32') + big_frame.cast('float32') + tfm + big_tfm).realize()

    def native(new_img, desire, traffic_convention, action_t):
      return (new_img.cast('float32') + desire + traffic_convention + action_t).realize()

    def split_policy(**kwargs):
      value = policy(**kwargs)
      return value, (value + 1).realize()

    def multi_policy(**kwargs):
      value = policy(**kwargs)
      return value, (value + 1).realize(), (value + 2).realize()

    def list_policy(**kwargs):
      return list(split_policy(**kwargs))

    def inputs(names, *, camera=()):
      return {name: Tensor(np.arange(16, dtype=np.uint8 if name in camera else np.float32), device='CPU').realize() for name in names}

    cls.policy_arrays = inputs(('warped', 'img_q', 'big_img_q', 'feat_q', 'desire_q', 'packed_npy_inputs'))
    cls.policy_cpu_part = serialized_part(policy, cls.policy_arrays)
    cls.policy_part = qcom_part(cls.policy_cpu_part)
    cls.policy_parts = {'supercombo': cls.policy_part,
                        'vision_policy': qcom_part(serialized_part(split_policy, cls.policy_arrays)),
                        'vision_multi_policy': qcom_part(serialized_part(multi_policy, cls.policy_arrays)),
                        'list_return': qcom_part(serialized_part(list_policy, cls.policy_arrays))}
    cls.warp_part = qcom_part(serialized_part(warp, inputs(('frame', 'big_frame', 'tfm', 'big_tfm'), camera=('frame', 'big_frame'))))
    cls.native_part = qcom_part(serialized_part(native, inputs(('new_img', 'desire', 'traffic_convention', 'action_t'), camera=('new_img',))))

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = Path(directory.name)
    self.path = self.root / 'ordinary.sptg'

  def write_legacy(self, *, metadata_kind='vision_policy'):
    # Split vision/policy, multi-policy and legacy supercombo all use one
    # run_policy JIT plus camera-keyed warps in compile_modeld.py. Their model
    # metadata lives in restricted extras, which this validator never unpickles.
    write_bundle(str(self.path), {'run_policy': self.policy_parts[metadata_kind], (1344, 760): self.warp_part, (1928, 1208): self.warp_part},
                 {'metadata': {'fixture_compiler_model_type': metadata_kind}, 'input_devices': {'warp': 'QCOM'}})

  def test_real_writer_tuple_keys_and_supported_legacy_compiler_architectures(self):
    for kind in ('vision_policy', 'vision_multi_policy', 'supercombo', 'list_return'):
      with self.subTest(model_type=kind):
        self.write_legacy(metadata_kind=kind)
        metadata, _ = document(self.path)
        self.assertEqual([row['key'] for row in metadata['parts']], ['run_policy', [1344, 760], [1928, 1208]])
        # Confirm the normal production reader reconstructs exactly these keys.
        parts, _ = read_parts(str(self.path))
        self.assertEqual(set(parts), {'run_policy', (1344, 760), (1928, 1208)})
        with patch('pickle.load', side_effect=AssertionError('No unpickling during static validation')), \
             patch('pickle.loads', side_effect=AssertionError('No unpickling during static validation')), \
             patch('openpilot.sunnypilot.modeld_v2.helpers.PlainJit', side_effect=AssertionError('No native execution')):
          self.assertEqual(len(compatibility._fallback_headers(self.path)), 3)

  def test_native_fused_model_run_entrypoint(self):
    write_bundle(str(self.path), {'run': self.native_part}, {'input_specs': {}})
    self.assertEqual(len(compatibility._fallback_headers(self.path)), 1)

  def test_canonical_duplicate_resolution_and_string_entrypoints_rejected(self):
    self.write_legacy()
    original, blob = document(self.path)
    for index in (0, 1, 2):
      with self.subTest(duplicate_key=original['parts'][index]['key']):
        changed = copy.deepcopy(original)
        changed['parts'].append(copy.deepcopy(changed['parts'][index]))
        rewrite(self.path, changed, blob)
        with self.assertRaisesRegex(ExperimentalCatalogError, '[Dd]uplicate'):
          compatibility._fallback_headers(self.path)

  def test_malformed_camera_keys_rejected_before_python_hashing(self):
    self.write_legacy()
    original, blob = document(self.path)
    for key in ([], [1344], [1344, 760, 1], [True, 760], [1344.0, 760], [0, 760], [-1, 760],
                ['1344', 760], [[1344], 760], {'width': 1344, 'height': 760}, None, 1344, '(1344, 760)', ''):
      with self.subTest(key=key):
        changed = copy.deepcopy(original)
        changed['parts'][1]['key'] = key
        rewrite(self.path, changed, blob)
        with self.assertRaises(ExperimentalCatalogError):
          compatibility._fallback_headers(self.path)

  def test_entrypoint_sets_match_actual_qcom_adapter(self):
    invalid = [
      {'run_policy': self.policy_part},
      {(1344, 760): self.warp_part},
      {'run_vision': self.policy_part, 'run_policy': self.policy_part},
      {'run': self.native_part, 'run_policy': self.policy_part, (1344, 760): self.warp_part},
      {'not_an_entrypoint': self.native_part},
    ]
    for parts in invalid:
      with self.subTest(keys=repr(list(parts))):
        write_bundle(str(self.path), parts)
        with self.assertRaises(ExperimentalCatalogError):
          compatibility._fallback_headers(self.path)

  def test_valid_part_keys_do_not_bypass_targets_or_buffer_bounds(self):
    self.write_legacy()
    original, blob = document(self.path)
    mutations = [
      lambda d: d['parts'][1]['header']['calls'][0]['target'].update(device='AMD'),
      lambda d: d['parts'][1]['header']['calls'][0].update(lib=[0, len(blob) + 1]),
      lambda d: d['parts'][1].update(blob=[0, len(blob) + 1]),
      lambda d: d.update(extras=[0, len(blob) + 1]),
      lambda d: d['parts'][1]['header']['inputs'][0].update(device='AMD'),
      lambda d: d['parts'][1]['header']['calls'][0]['args'].__setitem__(0, {'input': 100000}),
    ]
    for index, mutation in enumerate(mutations):
      with self.subTest(mutation=index):
        changed = copy.deepcopy(original)
        mutation(changed)
        rewrite(self.path, changed, blob)
        with self.assertRaises(ExperimentalCatalogError):
          compatibility._fallback_headers(self.path)

  def test_catalog_digest_still_binds_real_tuple_keyed_bundle(self):
    self.write_legacy()
    params, models = self.root / 'params', self.root / 'models'
    params.mkdir()
    models.mkdir()
    installed = models / self.path.name
    installed.write_bytes(self.path.read_bytes())
    expected = hashlib.sha256(installed.read_bytes()).hexdigest()
    bundle_path = params / 'ModelManager_ActiveBundle'
    bundle = {'minimumSelectorVersion': 21, 'runner': 'tinygrad',
              'models': [{'artifact': {'fileName': installed.name, 'downloadUri': {'sha256': expected}}}]}
    bundle_path.write_text(json.dumps(bundle))
    selected_before = bundle_path.read_bytes()
    self.assertEqual(compatibility._fallback(params, models)['artifact_sha256'], expected)
    installed.write_bytes(installed.read_bytes() + b'corruption')
    with self.assertRaisesRegex(ExperimentalCatalogError, 'SHA256 mismatch'):
      compatibility._fallback(params, models)
    self.assertEqual(bundle_path.read_bytes(), selected_before)

  def test_native_loader_accepts_unused_outer_arguments_excluded_from_kernel_globals(self):
    header, blob = copy.deepcopy(self.policy_cpu_part[0]), self.policy_cpu_part[1]
    program = next(call for call in header['calls'] if call['kind'] == 'program')
    program['args'].append(copy.deepcopy(program['args'][0]))
    self.assertLess(len(program['globals']), len(program['args']))
    write_bundle(str(self.path), {'run': (header, blob)})
    # This executes only the generated tiny CPU graph using the unchanged
    # production loader, proving unused outer call arguments are legitimate.
    loaded = load_model(str(self.path), cache=False, verify=False)['run']
    actual = loaded(**self.policy_arrays).numpy().tolist()
    self.assertEqual(actual, [float(value * 6) for value in range(16)])
    qcom_header, _ = qcom_part((header, blob))
    compatibility.validate_native_layout(qcom_header)


if __name__ == '__main__':
  unittest.main()
