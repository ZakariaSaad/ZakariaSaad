"""Native kernel evidence is data-only, bounded, private and integrity checked."""
import hashlib
import json
from pathlib import Path
import pickle
import struct
import tarfile
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from tools import navigation_numerical_export as export


class TestNumericalKernelExport(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.artifact, self.output = self.root / 'model.sptg', self.root / 'evidence.tar.gz'
    self.code = b'AMD_KERNEL_CODE'
    self.weights = b'SECRET_WEIGHT_PAYLOAD'
    self.extras = b'PICKLE_EXTRAS_MUST_NOT_BE_READ_OR_EXECUTED'
    self.container: dict[str, Any] = {'parts': [{'key': 'run', 'blob': [0, len(self.code + self.weights)], 'header': {
      'inputs': [{'name': 'nav_map', 'device': 'NPY'}],
      'buffers': [{'device': 'AMD', 'init': [len(self.code), len(self.code + self.weights)]}],
      'calls': [{'kind': 'program', 'lib': [0, len(self.code)], 'target': {
        'device': 'AMD', 'arch': 'gfx1200', 'renderer': 'LLVM', 'interface': 'USB'}},
        {'kind': 'program', 'lib': [0, len(self.code)], 'target': {
          'device': 'AMD', 'arch': 'gfx1200', 'renderer': 'LLVM', 'interface': 'USB'}}]}}],
      'extras': [len(self.code + self.weights), len(self.code + self.weights + self.extras)]}

  def write(self, raw=None):
    raw = json.dumps(self.container).encode() if raw is None else raw
    payload = b'SPTGPLN1' + struct.pack('<Q', len(raw)) + raw
    payload += bytes(-len(payload) % 256) + self.code + self.weights + self.extras
    self.artifact.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

  def run_export(self):
    return export.export_kernels(self.artifact, self.write(), self.output)

  def test_only_header_kernel_and_summary_exported_and_binaries_deduplicated(self):
    expected = self.write()
    with patch.object(pickle, 'loads', side_effect=AssertionError('Never unpickle')):
      result = export.export_kernels(self.artifact, expected, self.output)
    self.assertEqual(result['artifact_sha256'], expected)
    self.assertEqual(len(result['kernels']), 2)
    self.assertFalse(result['gpu_execution_performed'])
    self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)
    with tarfile.open(self.output) as archive:
      self.assertEqual(len(archive.getmembers()), 3)
      payloads = [archive.extractfile(member).read() for member in archive.getmembers()]
      self.assertIn(self.code, payloads)
      for payload in payloads:
        self.assertNotIn(self.weights, payload)
        self.assertNotIn(self.extras, payload)
      summary_file = archive.extractfile('export.json')
      assert summary_file is not None
      summary = json.load(summary_file)
      self.assertEqual(summary['kernels'][0]['file'], summary['kernels'][1]['file'])

  def test_wrong_hash_refuses_before_writing_any_archive(self):
    self.write()
    with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
      export.export_kernels(self.artifact, '0' * 64, self.output)
    self.assertFalse(self.output.exists())

  def test_duplicate_json_keys_rejected(self):
    expected = self.write(b'{"parts":[],"parts":[]}')
    with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
      export.export_kernels(self.artifact, expected, self.output)

  def test_out_of_bounds_weights_extras_and_libraries_rejected(self):
    for key in ('lib', 'init', 'extras'):
      with self.subTest(key=key):
        original = json.loads(json.dumps(self.container))
        if key == 'extras':
          self.container[key] = [0, 100000]
        elif key == 'init':
          self.container['parts'][0]['header']['buffers'][0][key] = [0, 100000]
        else:
          self.container['parts'][0]['header']['calls'][0][key] = [0, 100000]
        with self.assertRaisesRegex(ValueError, 'blob range'):
          self.run_export()
        self.container = original
    self.assertFalse(self.output.exists())

  def test_wrong_target_rejected(self):
    self.container['parts'][0]['header']['calls'][0]['target']['arch'] = 'gfx1100'
    with self.assertRaisesRegex(ValueError, 'registered AMD'):
      self.run_export()

  def test_library_cannot_export_a_weight_or_extras_range(self):
    self.container['parts'][0]['header']['calls'][0]['lib'] = [len(self.code), len(self.code + self.weights)]
    with self.assertRaisesRegex(ValueError, 'overlaps weights'):
      self.run_export()

  def test_size_bounds_apply_before_copy_and_archive(self):
    expected = self.write()
    with patch.object(export, 'MAX_HEADER', 5), self.assertRaisesRegex(ValueError, 'header exceeds'):
      export.export_kernels(self.artifact, expected, self.output)
    with patch.object(export, 'MAX_EXPORT', len(json.dumps(self.container).encode())), \
         self.assertRaisesRegex(ValueError, 'export limit'):
      export.export_kernels(self.artifact, expected, self.output)
    self.assertFalse(self.output.exists())

  def test_truncated_magic_header_and_payload_rejected(self):
    for payload, reason in ((b'bad', 'SPTGPLN1'), (b'SPTGPLN1', 'header length'),
                            (b'SPTGPLN1' + struct.pack('<Q', 50) + b'{}', 'Truncated native artifact JSON')):
      with self.subTest(reason=reason):
        self.artifact.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, reason):
          export.export_kernels(self.artifact, hashlib.sha256(payload).hexdigest(), self.output)

  def test_mutation_while_exporting_is_rejected(self):
    expected = self.write()
    original = hashlib.file_digest

    def mutate(stream, algorithm):
      digest = original(stream, algorithm)
      with self.artifact.open('ab') as writer:
        writer.write(b'changed')
      return digest

    with patch.object(hashlib, 'file_digest', side_effect=mutate), self.assertRaisesRegex(ValueError, 'changed during'):
      export.export_kernels(self.artifact, expected, self.output)
    self.assertFalse(self.output.exists())

  def test_does_not_replace_existing_files_or_the_artifact(self):
    expected = self.write()
    with self.assertRaisesRegex(ValueError, 'must not replace'):
      export.export_kernels(self.artifact, expected, self.artifact)
    self.output.write_bytes(b'preserved')
    with self.assertRaises(FileExistsError):
      export.export_kernels(self.artifact, expected, self.output)
    self.assertEqual(self.output.read_bytes(), b'preserved')
    self.assertEqual(list(self.root.glob('.nav-kernels-*')), [])

  def test_registered_command_uses_exact_receipt_and_metadata_without_private_fields(self):
    expected = self.write()
    metadata = {'schema_version': 1, 'source_onnx_sha256': 's' * 64, 'target': {'backend': 'AMD'},
                'compiler_verification': {'private_path': '/private/NOT_FOR_EXPORT'}, 'secret': 'NOT_FOR_EXPORT'}
    metadata_raw = json.dumps(metadata).encode()
    (self.root / 'metadata.json').write_bytes(metadata_raw)
    manifest = {'entries': [{'id': 'big-nav-v2', 'compiled_artifact': {'path': 'model.sptg', 'sha256': expected},
                            'compiler_metadata': {'path': 'metadata.json', 'sha256': hashlib.sha256(metadata_raw).hexdigest()}}]}
    (self.root / 'manifest.json').write_text(json.dumps(manifest))
    receipt = {'manifest_path': str(self.root / 'manifest.json')}
    receipt_raw = json.dumps(receipt).encode()
    (self.root / 'receipt.json').write_bytes(receipt_raw)
    registry = {'candidates': [{'id': 'big-nav-v2', 'artifact_sha256': expected,
                               'registration_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
                               'metadata_sha256': hashlib.sha256(metadata_raw).hexdigest()}]}
    (self.root / 'registry.json').write_text(json.dumps(registry))
    with patch.object(export, 'REVIEWED_SHA256', expected), patch.object(export, 'REGISTRY', self.root / 'registry.json'), \
         patch.object(export, 'RECEIPT', self.root / 'receipt.json'):
      export.export_registered(self.output)
      self.assertNotIn(b'NOT_FOR_EXPORT', self.output.read_bytes())
      with tarfile.open(self.output) as archive:
        summary = archive.extractfile('export.json').read()
        self.assertNotIn(b'NOT_FOR_EXPORT', summary)
        self.assertIn(b'source_onnx_sha256', summary)
      (self.root / 'receipt.json').write_text(json.dumps(receipt) + ' ')
      with self.assertRaisesRegex(ValueError, 'receipt SHA256 mismatch'):
        export.export_registered(self.root / 'second.tar.gz')


if __name__ == '__main__':
  unittest.main()
