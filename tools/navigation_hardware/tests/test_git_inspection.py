"""Git-only regressions for root supervision of an unprivileged owner's checkout.

The real subprocess cases run in a root test container, including with --read-only.
Only disposable fixtures are created; no GPU, service or Git trust/config changes.
"""
import hashlib
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from openpilot.common.git_inspection import GitInspectionError, read_only_git
from openpilot.selfdrive.navigation import export_experimental_model as exporter
from openpilot.selfdrive.navigation.experimental_catalog import ExperimentalCatalogError
from tools.navigation_hardware import safety


class TestGitInspection(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    self.owner = pwd.struct_passwd(('comma', 'x', 1000, 1000, '', '/home/comma', '/bin/bash'))

  def test_root_child_uses_comma_credentials_and_owner_environment(self):
    with patch('os.geteuid', return_value=0), patch('pwd.getpwuid', return_value=self.owner), \
         patch('os.getgrouplist', return_value=[1000, 44]), \
         patch.dict(os.environ, {'HOME': '/root', 'XDG_CONFIG_HOME': '/root/.config'}), \
         patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='abc\n')) as run:
      self.assertEqual(read_only_git(self.root, 'submodule', 'status', '--recursive'), 'abc')
      command, options = run.call_args.args[0], run.call_args.kwargs
      self.assertEqual(command[-3:], ['submodule', 'status', '--recursive'])
      self.assertEqual((options['user'], options['group'], options['extra_groups']), (1000, 1000, [1000, 44]))
      self.assertEqual(options['env']['HOME'], '/home/comma')
      self.assertEqual(options['env']['USER'], 'comma')
      self.assertNotIn('XDG_CONFIG_HOME', options['env'])
      self.assertEqual(options['env']['GIT_OPTIONAL_LOCKS'], '0')
      self.assertEqual(options['cwd'], self.root)
      self.assertIn('core.fsmonitor=false', command)
      self.assertNotIn('safe.directory', ' '.join(command))
      self.assertEqual(os.environ['HOME'], '/root')

  def test_nonroot_cannot_inspect_as_another_owner(self):
    with patch('os.geteuid', return_value=1001), patch('pwd.getpwuid', return_value=self.owner), patch('subprocess.run') as run:
      with self.assertRaisesRegex(GitInspectionError, 'requires repository owner comma'):
        read_only_git(self.root, 'rev-parse', 'HEAD')
      run.assert_not_called()

  def test_writes_and_trust_overrides_are_not_allowed(self):
    for args in (('config', '--global', '--add', 'safe.directory', '*'), ('reset', '--hard'), ('submodule', 'update'), ('fetch',)):
      with self.subTest(args=args), patch('subprocess.run') as run, self.assertRaisesRegex(GitInspectionError, 'Unsupported read-only'):
        read_only_git(self.root, *args)
      run.assert_not_called()

  def test_original_git_stderr_survives_admission_and_export_errors(self):
    stderr = 'fatal: detected dubious ownership in repository at /data/openpilot\noriginal detail\n'
    with patch('subprocess.run', return_value=SimpleNamespace(returncode=128, stdout='', stderr=stderr)):
      with self.assertRaises(safety.HardwareAdmissionError) as failure:
        safety.git(self.root, 'rev-parse', 'HEAD')
      self.assertIn(stderr, str(failure.exception))
      self.assertIn(str(self.root), str(failure.exception))
      self.assertIn('exit 128', str(failure.exception))
      with self.assertRaises(ExperimentalCatalogError) as export_failure:
        exporter._runtime_checkout(self.root)
      self.assertIn(stderr, str(export_failure.exception))

  def test_timeout_stderr_is_not_lost(self):
    with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('git', 15, stderr=b'original timeout detail\n')):
      with self.assertRaises(GitInspectionError) as failure:
        read_only_git(self.root, 'rev-parse', 'HEAD')
    self.assertIn('original timeout detail\n', str(failure.exception))


FIXTURE_SCRIPT = r'''
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
root.mkdir(mode=0o700)
def git(path, *args):
  return subprocess.check_output(['git', '-c', 'user.name=Diagnostic Fixture', '-c', 'user.email=fixture@example.invalid',
                                 '-c', 'protocol.file.allow=always', '-C', str(path), *args], text=True, stderr=subprocess.PIPE).strip()
def repo(name):
  path = root / name
  path.mkdir()
  git(path, 'init', '-q', '-b', 'fixture')
  (path / 'tracked').write_text('fixture\n')
  git(path, 'add', '.')
  git(path, 'commit', '-qm', 'fixture')
  return path
leaf, runtime, checkout = repo('leaf'), repo('runtime'), repo('openpilot')
git(runtime, 'submodule', 'add', '-q', str(leaf), 'nested/leaf')
git(runtime, 'commit', '-qam', 'nested fixture')
git(checkout, 'submodule', 'add', '-q', str(runtime), 'tinygrad_repo')
for name in ('openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py', 'openpilot/sunnypilot/modeld_v2/compile_warp.py',
             'tools/navigation_model_diagnostic.py', 'openpilot/system/hardware/chestnut/firmware_wrapped.bin'):
  path = checkout / name
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b'inert fixture')
path = checkout / 'openpilot/common/hardware/usb.py'
path.parent.mkdir(parents=True)
path.write_text('CHESTNUT_FW_VERSION = "8516b768"\n')
git(checkout, 'add', '.')
git(checkout, 'commit', '-qm', 'source fixture')
git(checkout, 'submodule', 'update', '--init', '--recursive', '-q')
git(checkout, 'config', 'core.fsmonitor', str(root / 'must-not-run'))
'''


@unittest.skipUnless(os.geteuid() == 0, 'Root-only subprocess regression runs in the AGNOS-style test container')
class TestRootSupervisorGit(unittest.TestCase):
  def setUp(self):
    self.owner = pwd.getpwnam('nobody')  # Existing unprivileged fixture identity; device owner is comma.
    self.root = Path(tempfile.gettempdir()) / ('navigation-git-owner-' + uuid.uuid4().hex)
    self.environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_') and key != 'SUDO_UID'}
    self.environment.update(HOME='/root', XDG_CONFIG_HOME='/root/.config', GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_SYSTEM='/dev/null')
    fixture_python = shutil.which('python3')  # System stdlib Python is accessible to the unprivileged fixture owner.
    assert fixture_python is not None
    subprocess.run([fixture_python, '-c', FIXTURE_SCRIPT, str(self.root)], cwd=tempfile.gettempdir(), env=self.environment,
                   user=self.owner.pw_uid, group=self.owner.pw_gid, extra_groups=[], capture_output=True, text=True, check=True, timeout=30)
    self.addCleanup(shutil.rmtree, self.root)
    self.checkout = self.root / 'openpilot'
    self.runtime = self.checkout / 'tinygrad_repo'
    self.env_patch = patch.dict(os.environ, self.environment, clear=True)
    self.env_patch.start()
    self.addCleanup(self.env_patch.stop)

  def snapshot(self):
    return {str(p.relative_to(self.root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns,
                                           p.stat().st_uid, p.stat().st_gid, p.stat().st_mode)
            for p in self.root.rglob('*') if p.is_file()}

  def test_dubious_root_fails_but_owner_inspects_all_nested_repositories_read_only(self):
    raw = subprocess.run(['git', '-C', str(self.checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    self.assertNotEqual(raw.returncode, 0)
    self.assertIn('dubious ownership', raw.stderr)
    before = self.snapshot()
    runtime_sha = safety.git(self.runtime, 'rev-parse', 'HEAD')
    with patch.object(safety, 'TINYGRAD_REF', runtime_sha):
      identity = safety.source_identity(self.checkout)
    self.assertEqual(identity['tinygrad_ref'], runtime_sha)
    self.assertFalse(identity['dirty'])
    self.assertFalse(identity['tinygrad_dirty'])
    self.assertEqual(len(identity['submodules']), 2)
    self.assertIn('tinygrad_repo/nested/leaf', identity['submodules'][1])
    self.assertEqual(len(safety.git(self.runtime / 'nested/leaf', 'rev-parse', 'HEAD')), 40)
    self.assertEqual(before, self.snapshot())  # No index writes, ownership/mode changes or Git config workaround.
    self.assertEqual(os.geteuid(), 0)  # Only Git children dropped privileges.

  def test_compiler_export_and_file_diagnostic_share_owner_aware_inspection(self):
    from openpilot.sunnypilot.modeld_v2 import compile_modeld_v2 as compiler
    from tools import navigation_model_diagnostic as diagnostic
    before = self.snapshot()
    revision = safety.git(self.runtime, 'rev-parse', 'HEAD')
    self.assertEqual(compiler.tinygrad_provenance(self.runtime)['revision'], revision)
    with patch.object(exporter, 'TINYGRAD_REF', revision):
      exporter._runtime_checkout(self.runtime)
    with patch('openpilot.common.basedir.BASEDIR', str(self.checkout)), patch.object(diagnostic, 'TINYGRAD_REF', revision):
      self.assertEqual(diagnostic.verify_runtime(), revision)
    self.assertEqual(before, self.snapshot())

  def test_dirty_checkout_and_wrong_runtime_pin_still_refuse_admission(self):
    revision = safety.git(self.runtime, 'rev-parse', 'HEAD')
    tracked = self.runtime / 'tracked'
    original = tracked.read_bytes()
    tracked.write_bytes(b'changed fixture\n')
    with patch.object(safety, 'TINYGRAD_REF', revision), self.assertRaisesRegex(safety.HardwareAdmissionError, 'clean committed'):
      safety.source_identity(self.checkout)
    tracked.write_bytes(original)
    with patch.object(safety, 'TINYGRAD_REF', '0' * 40), self.assertRaisesRegex(safety.HardwareAdmissionError, 'runtime pin'):
      safety.source_identity(self.checkout)

  @unittest.skipUnless(bool(os.statvfs('/').f_flag & os.ST_RDONLY), 'Read-only-root regression runs with Docker --read-only')
  def test_read_only_root_needs_no_root_gitconfig(self):
    self.assertTrue(os.statvfs('/').f_flag & os.ST_RDONLY)
    self.assertEqual(len(safety.git(self.checkout, 'rev-parse', 'HEAD')), 40)
    self.assertEqual(len(safety.git(self.checkout, 'submodule', 'status', '--recursive').splitlines()), 2)
