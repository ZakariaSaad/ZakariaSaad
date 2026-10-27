"""Historical compiler identity is distinct from the exact reviewed execution ABI."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation import corrected_bigv2 as corrected
from openpilot.selfdrive.navigation import export_experimental_model as exporter
from openpilot.selfdrive.navigation.experimental_catalog import TINYGRAD_REF, ExperimentalCatalogError


class TestExecutionRuntime(unittest.TestCase):
  def check(self, revision, digest=corrected.ARTIFACT_SHA256, status=''):
    with TemporaryDirectory() as directory, patch.object(exporter, 'read_only_git', side_effect=[revision, status]):
      return exporter._runtime_checkout(Path(directory), artifact_sha256=digest)

  def test_historical_compiler_remains_exact(self):
    self.assertEqual(TINYGRAD_REF, '69fd4d34d4430036378aba2b43f9d339cdd6ea73')
    self.assertEqual(self.check(TINYGRAD_REF), TINYGRAD_REF)

  def test_reviewed_execution_revision_for_exact_corrected_bytes(self):
    self.assertEqual(self.check(exporter.EXECUTION_TINYGRAD_REF), exporter.EXECUTION_TINYGRAD_REF)

  def test_other_artifact_does_not_inherit_execution_approval(self):
    for digest in (None, '0' * 64, '1d18c42ea4cd95c8bffab4635844b78b8e2d4e25b9de44a9c5ac2e0961371a2e'):
      with self.subTest(digest=digest), self.assertRaisesRegex(ExperimentalCatalogError, 'exact reviewed native artifact'):
        self.check(exporter.EXECUTION_TINYGRAD_REF, digest)

  def test_other_revision_and_dirty_checkout_fail(self):
    with self.assertRaisesRegex(ExperimentalCatalogError, 'revision mismatch'):
      self.check('0' * 40)
    for revision in (TINYGRAD_REF, exporter.EXECUTION_TINYGRAD_REF):
      with self.subTest(revision=revision), self.assertRaisesRegex(ExperimentalCatalogError, 'must be clean'):
        self.check(revision, status=' M tinygrad/runtime/ops_amd.py')
