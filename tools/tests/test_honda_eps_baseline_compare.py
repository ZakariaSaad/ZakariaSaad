from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from tools import honda_eps_baseline_compare as helper
from tools.honda_eps_baseline_compare import BASELINE, compare, normalize_diff, replay_summary, verify_replay_reference


class TestBaselineComparison(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.baseline = Path(self.temp.name) / "baseline"
    self.candidate = Path(self.temp.name) / "candidate"
    self.result: dict[str, Any] = {
      "sha": BASELINE, "submodules": {"panda": "original"}, "python": "3.12.14", "packages": {"numpy": "2.5.4"},
      "unit_exit": 1, "replay_exit": 1, "replay": {"changed": 5, "passed": 61, "errors": 0},
    }
    self.unit: dict[str, Any] = {"collection_errors": [], "records": {"original.test": {"status": "failed", "detail": "AssertionError"}}}
    self.fixtures: dict[str, Any] = {"reference_commit": "original-ref", "files": {"input": {"bytes": 5, "sha256": "fixture-hash"}}}
    for directory in (self.baseline, self.candidate):
      directory.mkdir()
      self.write(directory, "result.json", self.result)
      self.write(directory, "unit.json", self.unit)
      self.write(directory, "fixtures.json", self.fixtures)
      (directory / "normalized-diff.txt").write_text("('change', 'carControl.accel', (1, 2))\n")

  @staticmethod
  def write(directory, name, value):
    (directory / name).write_text(json.dumps(value))

  def test_identical_failures_are_evidence_not_successful_tests(self):
    result = compare(self.baseline, self.candidate)
    self.assertEqual(result["unit_counts"], {"failed": 1})
    self.assertEqual(result["replay"], self.result["replay"])

  def test_normalization_preserves_numerical_data(self):
    text = "*** process: card ***\n\tref: /old\n\tnew: /new\n\t('change', 'x', (1, 2))\n"
    self.assertEqual(normalize_diff(text), "*** process: card ***\n\t('change', 'x', (1, 2))\n")
    self.assertNotEqual(normalize_diff(text), normalize_diff(text.replace("(1, 2)", "(1, 3)")))

  def test_missing_replay_report_is_rejected(self):
    with self.assertRaises(ValueError):
      replay_summary("TEST FAILED")

  def test_replay_reference_must_match_fingerprinted_fixtures(self):
    verify_replay_reference("***** tested against commit ref-a *****\n", "ref-a")
    for text in ("***** tested against commit ref-b *****\n", ""):
      with self.assertRaises(ValueError):
        verify_replay_reference(text, "ref-a")

  def test_cli_resolves_output_before_changing_source_directory(self):
    expected = Path("evidence").resolve()
    command = ["helper", "collect", "--root", str(self.baseline), "--output", "evidence", "--expected-sha", BASELINE]
    with patch.object(helper.sys, "argv", command), patch.object(helper.sys, "path", list(helper.sys.path)), \
         patch.object(helper.os, "chdir"), patch.object(helper, "collect") as collect:
      self.assertEqual(helper.main(), 0)
    self.assertEqual(collect.call_args.args, (self.baseline, expected, BASELINE))

  def test_changed_replay_value_is_rejected(self):
    (self.candidate / "normalized-diff.txt").write_text("('change', 'carControl.accel', (1, 3))\n")
    with self.assertRaises(ValueError):
      compare(self.baseline, self.candidate)

  def test_changed_unit_result_or_trace_is_rejected(self):
    for status, detail in (("error", "AssertionError"), ("failed", "different assertion"), ("passed", "")):
      with self.subTest(status=status, detail=detail):
        changed = deepcopy(self.unit)
        changed["records"]["original.test"] = {"status": status, "detail": detail}
        self.write(self.candidate, "unit.json", changed)
        with self.assertRaises(ValueError):
          compare(self.baseline, self.candidate)

  def test_collection_error_or_missing_tests_is_rejected(self):
    for changed in ({"collection_errors": ["missing import"], "records": self.unit["records"]}, {"collection_errors": [], "records": {}}):
      self.write(self.candidate, "unit.json", changed)
      with self.assertRaises(ValueError):
        compare(self.baseline, self.candidate)

  def test_changed_fixture_bytes_are_rejected(self):
    changed = deepcopy(self.fixtures)
    changed["files"]["input"]["sha256"] = "different-hash"
    self.write(self.candidate, "fixtures.json", changed)
    with self.assertRaises(ValueError):
      compare(self.baseline, self.candidate)

  def test_wrong_base_submodule_dependency_or_partial_replay_is_rejected(self):
    for key, value in (("sha", "wrong"), ("submodules", {}), ("packages", {}), ("replay", {"changed": 5, "passed": 60, "errors": 0})):
      with self.subTest(key=key):
        changed = deepcopy(self.result)
        changed[key] = value
        self.write(self.baseline if key == "sha" else self.candidate, "result.json", changed)
        with self.assertRaises(ValueError):
          compare(self.baseline, self.candidate)
        self.write(self.baseline, "result.json", self.result)
        self.write(self.candidate, "result.json", self.result)

  def prepare_selector21(self):
    original = deepcopy(self.result)
    original["submodules"]["tinygrad_repo"] = "f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae"
    original["packages"]["tinygrad"] = "0.13.0"
    updated = deepcopy(original)
    updated["submodules"]["tinygrad_repo"] = "69fd4d34d4430036378aba2b43f9d339cdd6ea73"
    updated["packages"]["tinygrad"] = "0.14.0"
    self.write(self.baseline, "result.json", original)
    self.write(self.candidate, "result.json", updated)
    return updated

  def test_selector21_allows_only_declared_runtime_and_model_test_migration(self):
    self.prepare_selector21()
    old = deepcopy(self.unit)
    old["records"][next(iter(helper.SELECTOR21_REMOVED_TESTS))] = {"status": "passed", "detail": ""}
    new = deepcopy(self.unit)
    new["records"]["new.model.test"] = {"status": "passed", "detail": ""}
    self.write(self.baseline, "unit.json", old)
    self.write(self.candidate, "unit.json", new)
    self.assertTrue(compare(self.baseline, self.candidate, selector21=True)["selector21"])
    with self.assertRaises(ValueError):
      compare(self.baseline, self.candidate)
    for name, status in (("new.model.test", "failed"), ("original.test", "passed")):
      changed = deepcopy(new)
      changed["records"][name]["status"] = status
      self.write(self.candidate, "unit.json", changed)
      with self.assertRaises(ValueError):
        compare(self.baseline, self.candidate, selector21=True)

  def test_selector21_rejects_other_submodules_and_runtime_revisions(self):
    updated = self.prepare_selector21()
    for module in ("panda", "tinygrad_repo"):
      changed = deepcopy(updated)
      changed["submodules"][module] = "unexpected"
      self.write(self.candidate, "result.json", changed)
      with self.assertRaises(ValueError):
        compare(self.baseline, self.candidate, selector21=True)

  def test_selector21_cannot_remove_unrelated_tests(self):
    self.prepare_selector21()
    self.write(self.candidate, "unit.json", {"collection_errors": [], "records": {"new.test": {"status": "passed", "detail": ""}}})
    with self.assertRaises(ValueError):
      compare(self.baseline, self.candidate, selector21=True)

  def test_only_identical_class_fixture_skips_can_repeat(self):
    record = {"id": "setUpClass (hardware.TestOnroad)", "status": "skipped", "detail": ""}
    result = helper.normalize_unit_records([record, record], Path("/source"))
    self.assertEqual(len(result), 1)
    for changed in ({**record, "status": "error"}, {**record, "detail": "different"}):
      with self.assertRaises(ValueError):
        helper.normalize_unit_records([record, changed], Path("/source"))
    for status in ("passed", "skipped", "failed"):
      method = {"id": "hardware.TestOnroad.test_method", "status": status, "detail": ""}
      with self.assertRaises(ValueError):
        helper.normalize_unit_records([method, method], Path("/source"))

  def test_selector21_only_allows_the_explicit_upstream_legacy_skips(self):
    self.prepare_selector21()
    key = next(iter(helper.SELECTOR21_LEGACY_SKIPS))
    old = deepcopy(self.unit)
    old["records"][key] = {"status": "passed", "detail": ""}
    new = deepcopy(old)
    new["records"][key]["status"] = "skipped"
    self.write(self.baseline, "unit.json", old)
    self.write(self.candidate, "unit.json", new)
    self.assertEqual(compare(self.baseline, self.candidate, selector21=True)["upstream_legacy_compiler_skips"], [key])
    for status in ("failed", "error"):
      changed = deepcopy(new)
      changed["records"][key]["status"] = status
      self.write(self.candidate, "unit.json", changed)
      with self.assertRaises(ValueError):
        compare(self.baseline, self.candidate, selector21=True)


if __name__ == "__main__":
  unittest.main()
