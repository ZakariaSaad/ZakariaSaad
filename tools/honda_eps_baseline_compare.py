"""Compare the original MVL test failures with an isolated controller checkout.

The source checkout is never patched. Its test runner and replay CLI retain their
original comparisons, warning policy, inputs and reference logs. Nonzero exits
are evidence to compare, not a reason to call a failed upstream test successful.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from unittest.mock import patch

BASELINE = "cb15c04e918b759688cb354b1bbc9b3c5a47a65c"
EXTRA_PACKAGES = {"pytest", "iniconfig", "pluggy"}


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(root, *args):
  return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def normalize_diff(text):
  # These two lines contain checkout paths and the current commit in filenames.
  # Every process, segment, field, count and numerical difference is retained.
  return "\n".join(line for line in text.splitlines() if not re.match(r"^\t(?:ref|new): ", line)) + "\n"


def replay_summary(text):
  match = re.search(r"(\d+) changed, (\d+) passed, (\d+) errors", text)
  if match is None:
    raise ValueError("Replay did not produce a complete diff report")
  return dict(zip(("changed", "passed", "errors"), map(int, match.groups()), strict=True))


def verify_replay_reference(text, expected):
  header = f"***** tested against commit {expected} *****"
  if text.splitlines()[0:1] != [header]:
    raise ValueError("Replay used a different reference commit from the fingerprinted fixtures")


def unit_worker(root, output):
  # Only observe the original runner's final report; do not alter collection,
  # execution, warning handling, assertions, captures or its exit status.
  from tools import test_runner
  original_report = test_runner.report

  def report(records, errors, duration_count, elapsed):
    normalized = {r["id"]: {"status": r["status"], "detail": r["detail"].replace(str(root), "<source>")} for r in records}
    if len(normalized) != len(records):
      raise ValueError("Duplicate test records")
    write_json(output, {"records": normalized, "collection_errors": errors})
    return original_report(records, errors, duration_count, elapsed)

  sys.argv = [str(root / "tools/test_runner.py"), "-j2"]
  with patch.object(test_runner, "report", report):
    return test_runner.main()


def fixture_worker(root, output):
  from openpilot.selfdrive.test.process_replay import test_processes as replay
  from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
  from openpilot.tools.lib.filereader import FileReader
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.tools.lib.openpilotci import get_url
  from openpilot.tools.lib.url_file import URLFile

  ref_path = Path(replay.REF_COMMIT_FN)
  ref = ref_path.read_text().strip() if ref_path.exists() else URLFile(replay.BASE_URL + "ref_commit", cache=False).read().decode().strip()
  fixtures = {}

  def record(key, url):
    with FileReader(url) as reader:
      data = reader.read()
    if not data:
      raise ValueError(f"Empty fixture: {key}")
    fixtures[key] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

  for car, segment in replay.segments:
    route, number = segment.rsplit("--", 1)
    record(f"input:{segment}", get_url(route, number, "rlog.zst"))
    for cfg in replay.CONFIGS:
      if cfg.proc_name in replay.EXCLUDED_PROCS:
        continue
      if cfg.proc_name not in ("card", "controlsd", "lagd") and car not in ("HYUNDAI", "TOYOTA"):
        continue
      name = f"{segment}_{cfg.proc_name}_{ref}.zst".replace("|", "_")
      local = Path(replay.FAKEDATA) / name
      record(f"reference:{segment}:{cfg.proc_name}", str(local) if local.exists() else replay.BASE_URL + name)
  for index, url in enumerate(LogReader(TEST_ROUTE).logreader_identifiers):
    record(f"location-scenarios:{TEST_ROUTE}:{index}", url)
  # No raw logs, signed URLs or vehicle message contents are written to evidence.
  write_json(output, {"reference_commit": ref, "files": fixtures})
  return 0


def collect(root, output, expected_sha):
  root, output = root.resolve(), output.resolve()
  output.mkdir(parents=True, exist_ok=True)
  actual = git(root, "rev-parse", "HEAD")
  if actual != expected_sha:
    raise ValueError(f"Wrong source checkout: {actual} != {expected_sha}")
  if git(root, "diff", "HEAD", "--ignore-submodules=untracked", "--"):
    raise ValueError("Tracked source files were modified before testing")
  submodules = {}
  for line in git(root, "ls-tree", "-r", "HEAD").splitlines():
    mode, kind, sha, name = line.split(maxsplit=3)
    if mode == "160000" and kind == "commit":
      submodules[name] = sha
      if git(root / name, "rev-parse", "HEAD") != sha:
        raise ValueError(f"Wrong submodule revision: {name}")
  packages = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
  result = {"sha": actual, "submodules": submodules, "python": platform.python_version(), "packages": packages}
  python = root / ".venv/bin/python"
  helper = Path(__file__).resolve()
  env = {**os.environ, "PYTHONPATH": str(root), "RAYLIB_BACKEND": "headless", "NO_COLOR": "1"}

  def run(name, command):
    with (output / f"{name}.log").open("w") as log:
      process = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1500, check=False)
    print(f"{name}: exit {process.returncode}", flush=True)
    return process.returncode

  if run("fixtures", [str(python), str(helper), "fixtures", "--root", str(root), "--output", str(output / "fixtures.json")]):
    raise ValueError("Could not fingerprint the original test fixtures; see fixtures.log")
  result["unit_exit"] = run("unit", [str(python), str(helper), "unit", "--root", str(root), "--output", str(output / "unit.json")])
  result["replay_exit"] = run("replay", [str(python), "openpilot/selfdrive/test/process_replay/test_processes.py", "-j2"])
  replay_dir = root / "openpilot/selfdrive/test/process_replay"
  raw_diff = (replay_dir / "diff.txt").read_text()
  verify_replay_reference(raw_diff, json.loads((output / "fixtures.json").read_text())["reference_commit"])
  (output / "normalized-diff.txt").write_text(normalize_diff(raw_diff))
  report = (replay_dir / "diff_report.txt").read_text()
  (output / "diff_report.txt").write_text(report)
  result["replay"] = replay_summary(report)
  if git(root, "diff", "HEAD", "--ignore-submodules=untracked", "--"):
    raise ValueError("Tests modified tracked source files")
  write_json(output / "result.json", result)
  print(json.dumps({"sha": actual, "replay": result["replay"], "unit_exit": result["unit_exit"]}), flush=True)


def compare(baseline, candidate):
  def read(directory, name):
    return json.loads((directory / name).read_text())

  before, after = read(baseline, "result.json"), read(candidate, "result.json")
  if before["sha"] != BASELINE:
    raise ValueError("Comparison is not against the exact MVL baseline")
  for key in ("submodules", "python", "replay", "unit_exit", "replay_exit"):
    if before[key] != after[key]:
      raise ValueError(f"Baseline/candidate differ: {key}")
  if before["replay"]["errors"] or sum(before["replay"].values()) != 66:
    raise ValueError("Replay has execution errors or did not run all 66 cases")
  if before["unit_exit"] not in (0, 1) or before["replay_exit"] not in (0, 1):
    raise ValueError("Unexpected test command exit status")
  for name, version in before["packages"].items():
    if after["packages"].get(name) != version:
      raise ValueError(f"Dependency version differs: {name}")
  if set(after["packages"]) - set(before["packages"]) - EXTRA_PACKAGES:
    raise ValueError("Unexpected extra dependencies")
  fixtures = read(baseline, "fixtures.json")
  if fixtures != read(candidate, "fixtures.json") or not fixtures["files"]:
    raise ValueError("Original fixture bytes or replay reference commit differ")
  diff = (baseline / "normalized-diff.txt").read_bytes()
  if not diff or diff != (candidate / "normalized-diff.txt").read_bytes():
    raise ValueError("Replay numerical/field differences differ from untouched MVL")
  units = []
  for directory in (baseline, candidate):
    unit = read(directory, "unit.json")
    if unit["collection_errors"] or not unit["records"]:
      raise ValueError("Unit collection failed or ran no tests")
    if any(r["status"] in ("error", "xpassed") for r in unit["records"].values()):
      raise ValueError("Unit execution error or unexpected success needs review")
    units.append(unit["records"])
  if units[0] != units[1]:
    differing = sorted(k for k in units[0].keys() | units[1].keys() if units[0].get(k) != units[1].get(k))
    raise ValueError(f"Unit outcomes/assertion traces differ: {differing}")
  return {
    "baseline": before["sha"], "candidate": after["sha"], "submodules": before["submodules"],
    "unit_counts": dict(Counter(r["status"] for r in units[0].values())),
    "inherited_unit_failures": {k: r["detail"] for k, r in units[0].items() if r["status"] == "failed"},
    "replay": before["replay"], "replay_reference": fixtures["reference_commit"],
    "identical_fixture_files": len(fixtures["files"]), "normalized_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "result": "Identical unit outcomes and replay differences on untouched MVL and candidate",
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("collect", "unit", "fixtures", "compare"))
  parser.add_argument("--root", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--expected-sha")
  parser.add_argument("--baseline", type=Path)
  parser.add_argument("--candidate", type=Path)
  args = parser.parse_args()
  # Resolve against the caller's directory before entering the source checkout.
  # Artifact upload runs from the workspace and must see the same destination.
  args.output = args.output.resolve()
  if args.mode == "compare":
    result = compare(args.baseline, args.candidate)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
  root = args.root.resolve()
  sys.path.insert(0, str(root))
  os.chdir(root)
  if args.mode == "unit":
    return unit_worker(root, args.output)
  if args.mode == "fixtures":
    return fixture_worker(root, args.output)
  collect(root, args.output, args.expected_sha)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
