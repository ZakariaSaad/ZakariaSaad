"""Existing warp validation uses JSON/captured data only, no pickle or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation import warp_compatibility as warp
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError, TINYGRAD_REF
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class TestExistingWarp(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.source = warp.installed_warp_path(1344, 760)

  def test_real_branch_warps_match_the_numpy_saved_capture_without_execution(self):
    run = subprocess.run
    def git_only(command, *args, **kwargs):
      self.assertEqual(command[0], 'git', 'Static warp validation must not launch a compiler or diagnostic')
      return run(command, *args, **kwargs)
    with patch('pickle.load', side_effect=AssertionError('No pickle loading')), \
         patch('pickle.loads', side_effect=AssertionError('No pickle loading')), patch('subprocess.run', side_effect=git_only):
      for width, height, frame, allocation in ((1344, 760, 1622016, 2428928), (1928, 1208, 3735552, 4804608)):
        with self.subTest(width=width, height=height):
          path = warp.installed_warp_path(width, height)
          before = (path.stat().st_mtime_ns, path.stat().st_mode, path.stat().st_uid)
          profile = warp.validate_existing_warp(path, camera_width=width, camera_height=height)
          self.assertEqual(profile['artifact_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
          self.assertEqual((profile['width'], profile['height']), (width, height))
          self.assertEqual(profile['frame_size'], frame)
          self.assertEqual(profile['allocation_size'], allocation)
          self.assertEqual(profile['output_shape'], [2, 6, 128, 256])
          self.assertEqual(profile['output_dtype'], 'uint8')
          self.assertEqual(profile['target'], {'backend': 'AMD', 'arch': 'gfx1200', 'interface': 'USB', 'renderer': 'LLVM'})
          self.assertTrue(profile['captured_check_passed'])
          self.assertEqual(profile['captured_reference']['output_bytes'], 393216)
          self.assertEqual(profile['captured_reference']['differing_bytes'], 0)
          self.assertFalse(profile['gpu_execution_performed'])
          self.assertFalse(profile['normal_activation_allowed'])
          self.assertFalse(profile['embedded_compiler_attestation'])
          self.assertEqual(before, (path.stat().st_mtime_ns, path.stat().st_mode, path.stat().st_uid))

  def test_geometry_is_not_inferred_only_from_the_copied_buffer_length(self):
    self.assertEqual(get_nv12_info(1343, 760), get_nv12_info(1344, 760))
    with self.assertRaisesRegex(ExperimentalCatalogError, 'camera geometry'):
      warp.validate_existing_warp(self.source, camera_width=1343)
    for width, height in ((True, 760), (1344, True), (1344, 0), (1928, 760)):
      with self.subTest(width=width, height=height), self.assertRaises(ExperimentalCatalogError):
        warp.camera_layout(width, height)

  def test_the_whole_warp_identity_is_pinned_even_when_header_stays_valid(self):
    changed = self.root / 'renamed.sptg'
    changed.write_bytes(self.source.read_bytes() + b'changed')
    with self.assertRaisesRegex(ExperimentalCatalogError, 'camera-specific branch asset'):
      warp.validate_existing_warp(changed)
    # A filename is not its identity: unchanged bytes may be safely relocated.
    changed.write_bytes(self.source.read_bytes())
    self.assertTrue(warp.validate_existing_warp(changed)['captured_check_passed'])

  def test_pinned_runtime_and_reviewed_recipe_are_required(self):
    for outputs, reason in ((['0' * 40], 'pinned tinygrad'), ([TINYGRAD_REF, ' M tinygrad/tensor.py'], 'clean tinygrad')):
      with self.subTest(outputs=outputs), patch.object(warp, 'read_only_git', side_effect=outputs), \
           self.assertRaisesRegex(ExperimentalCatalogError, reason):
        warp.validate_existing_warp(self.source)
    digest = warp._sha256
    def changed_recipe(path):
      return '0' * 64 if path.name == 'compile_warp.py' and 'tinygrad_repo' in str(path) else digest(path)
    with patch.object(warp, '_sha256', side_effect=changed_recipe), self.assertRaisesRegex(ExperimentalCatalogError, 'warp recipe changed'):
      warp.validate_existing_warp(self.source)

  def mutated_fixture(self, change):
    data = self.source.read_bytes()
    size = struct.unpack_from('<Q', data, 8)[0]
    container = json.loads(data[16:16 + size])
    blob = bytearray(data[16 + size + -(16 + size) % 256:])
    change(container['parts'][0]['header'], blob)
    raw = json.dumps(container).encode()
    output = self.root / 'inert-mutated-warp.sptg'
    output.write_bytes(b'SPTGPLN1' + struct.pack('<Q', len(raw)) + raw + bytes(-(16 + len(raw)) % 256) + blob)
    return output

  def reject_mutation(self, change, reason):
    fixture = self.mutated_fixture(change)
    # Only the byte-identity fixture is substituted to exercise deeper contract
    # rejection. It never becomes a reviewed asset and is never executed.
    with patch.dict(warp._PINNED_WARPS, {(1344, 760): hashlib.sha256(fixture.read_bytes()).hexdigest()}), \
         self.assertRaisesRegex(ExperimentalCatalogError, reason):
      warp.validate_existing_warp(fixture)

  def test_target_dtype_output_and_copied_extent_mismatches_fail_closed(self):
    mutations = [
      (lambda h, _: h['calls'][0]['target'].update(arch='gfx1100'), 'kernel target'),
      (lambda h, _: h['inputs'][0].update(dtype='dtypes.f16'), 'dtype'),
      (lambda h, _: h['inputs'][1].update(numel=h['inputs'][1]['numel'] + 2), 'copied extent'),
      (lambda h, _: h['returns']['tensors'][0].update(shape=[2, 6, 128, 255]), 'shape'),
      (lambda h, _: h.update(input_names=['M_inv', 'M_inv']), 'bindings'),
      (lambda h, _: h['inputs'][1].update(slot=0), 'slots'),
      (lambda h, _: h['check']['inputs'][0][1].update(shape=[1, 3244032]), 'shape'),
    ]
    for change, reason in mutations:
      with self.subTest(reason=reason):
        self.reject_mutation(change, reason)

  def test_saved_output_bytes_are_compared_to_independent_camera_layout_reference(self):
    def change(header, blob):
      blob[header['check']['outputs'][0]['data'][0]] ^= 1
    self.reject_mutation(change, 'NumPy reference')

  def test_nonfinite_saved_transform_is_rejected(self):
    def change(header, blob):
      matrix = next(row[1] for row in header['check']['inputs'] if row[0] == ['M_inv'])
      start = matrix['data'][0]
      blob[start:start + 4] = struct.pack('<f', float('nan'))
    self.reject_mutation(change, 'matrices must be finite')

  def test_persisted_profile_rechecks_geometry_hash_target_and_scope_without_io(self):
    profile = warp.validate_existing_warp(self.source)
    with patch.object(warp, 'read_only_git', side_effect=AssertionError('No Git in a persisted profile check')), \
         patch.object(warp, '_sha256', side_effect=AssertionError('No file hashing in a persisted profile check')):
      warp.validate_camera_profile(profile, profile['artifact_sha256'], profile['frame_size'])
      changes = [
        ('artifact_sha256', '0' * 64), ('tinygrad_ref', '0' * 40), ('width', 1343), ('height', 762),
        ('stride', 1344), ('y_height', 760), ('uv_height', 380), ('frame_size', 2428928), ('allocation_size', 1622016),
        ('camera', {}), ('target', warp.TARGET | {'renderer': 'HIP'}), ('output_dtype', 'float32'),
        ('output_shape', [2, 6, 128, 255]), ('source_bindings', {}), ('captured_check_passed', False),
        ('captured_reference', {}), ('scope', 'production'), ('gpu_execution_performed', True), ('normal_activation_allowed', True),
      ]
      for key, value in changes:
        changed = copy.deepcopy(profile)
        changed[key] = value
        with self.subTest(key=key), self.assertRaises(ExperimentalCatalogError):
          warp.validate_camera_profile(changed, profile['artifact_sha256'], profile['frame_size'])
      with self.assertRaisesRegex(ExperimentalCatalogError, 'warp identity'):
        warp.validate_camera_profile(profile, '0' * 64, profile['frame_size'])
      with self.assertRaisesRegex(ExperimentalCatalogError, 'frame extent'):
        warp.validate_camera_profile(profile, profile['artifact_sha256'], profile['allocation_size'])


if __name__ == '__main__':
  unittest.main()
