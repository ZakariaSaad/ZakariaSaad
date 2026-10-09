"""Real updater Git operations and launcher gate, confined to temporary trees."""
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.system.updated import updated


def git(repo, *args):
  return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT, text=True).strip()


class TestUpdateSwitch(unittest.TestCase):
  def test_fork_branch_staging_and_next_boot_default_preserve_external_data(self):
    with tempfile.TemporaryDirectory() as root:
      root = Path(root)
      seed = root / 'seed'
      seed.mkdir()
      git(seed, 'init', '-b', 'sp-honda-dev-202608')
      git(seed, 'config', 'user.name', 'CI fixture')
      git(seed, 'config', 'user.email', 'fixture@example.invalid')
      (seed / 'source.txt').write_text('MVL working baseline')
      git(seed, 'add', 'source.txt')
      git(seed, 'commit', '-m', 'baseline')
      baseline = git(seed, 'rev-parse', 'HEAD')
      mvl, fork, live, merged, finalized = (root / p for p in ('mvl.git', 'fork.git', 'live', 'merged', 'finalized'))
      subprocess.check_call(['git', 'clone', '--quiet', '--bare', str(seed), str(mvl)])
      git(seed, 'switch', '-c', 'mvl-cinque-v3')
      (seed / 'source.txt').write_text('reviewed custom branch')
      git(seed, 'commit', '-am', 'custom')
      custom_commit = git(seed, 'rev-parse', 'HEAD')
      subprocess.check_call(['git', 'clone', '--quiet', '--bare', str(seed), str(fork)])
      subprocess.check_call(['git', 'clone', '--quiet', str(mvl), str(live)])
      git(live, 'remote', 'rename', 'origin', 'mvl-upstream')
      git(live, 'remote', 'add', 'origin', str(fork))
      shutil.copytree(live, merged)
      params = Params(str(root / 'params'))
      params.put('UpdaterTargetBranch', 'mvl-cinque-v3', block=True)
      params.put('CinqueV3Fault', 'retained diagnosis', block=True)
      params.put_bool('OpenpilotEnabledToggle', False, block=True)
      external_model = root / 'media' / 'models' / 'working.pkl'
      external_model.parent.mkdir(parents=True)
      external_model.write_bytes(b'previously working compiled model')
      with patch.object(updated, 'Params', return_value=params), patch.object(updated, 'BASEDIR', str(live)), \
           patch.object(updated, 'OVERLAY_MERGED', str(merged)), patch.object(updated, 'FINALIZED', str(finalized)), \
           patch.object(updated, 'AGNOS', False):
        updater = updated.Updater()
        updater.check_for_update()
        updater.fetch_update()
        self.assertTrue(updater.update_ready)
        self.assertEqual(git(finalized, 'rev-parse', 'HEAD'), custom_commit)
        self.assertEqual(git(finalized, 'branch', '--show-current'), 'mvl-cinque-v3')
        self.assertEqual(git(finalized, 'remote', 'get-url', 'origin'), str(fork))
        self.assertEqual(git(finalized, 'remote', 'get-url', 'mvl-upstream'), str(mvl))
        self.assertEqual(git(finalized, 'rev-parse', '--abbrev-ref', '@{upstream}'), 'origin/mvl-cinque-v3')
        self.assertEqual(git(live, 'rev-parse', 'HEAD'), baseline)
        self.assertTrue((finalized / '.overlay_consistent').is_file())
        with patch.object(updated, 'BASEDIR', str(finalized)):
          params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
          # UpdaterTargetBranch is deliberately not persistent in MVL: after
          # reboot, the actual checkout branch determines the target.
          self.assertEqual(updater.target_branch, 'mvl-cinque-v3')
      self.assertEqual(params.get('CinqueV3Fault'), 'retained diagnosis')
      self.assertFalse(params.get_bool('CinqueV3Enabled'))
      self.assertFalse(params.get_bool('OpenpilotEnabledToggle'))
      self.assertEqual(external_model.read_bytes(), b'previously working compiled model')

  def launcher_fixture(self, modified, marker=True, backup_exists=False):
    with tempfile.TemporaryDirectory() as root:
      root = Path(root)
      base, staging = root / 'openpilot', root / 'safe_staging'
      base.mkdir()
      staging.mkdir()
      (base / '.git').mkdir()
      (base / '.git' / 'HEAD').write_text('working baseline')
      if marker:
        (base / '.overlay_init').touch()
        timestamp = (base / '.git' / 'HEAD').stat().st_mtime + (-60 if modified else 60)
        os.utime(base / '.overlay_init', (timestamp, timestamp))
      finalized = staging / 'finalized'
      finalized.mkdir()
      (finalized / '.overlay_consistent').touch()
      replacement = finalized / 'launch_chffrplus.sh'
      replacement.write_text('#!/usr/bin/env bash\necho ACTIVATED\n')
      replacement.chmod(0o755)
      if backup_exists:
        (staging / 'old_openpilot').mkdir()
      launch_text = (Path(BASEDIR) / 'launch_chffrplus.sh').read_text()
      gate = launch_text[launch_text.index('function launch {'):launch_text.index('  # handle pythonpath')]
      # Execute the actual overlay gate; omit hardware init and manager launch.
      gate = gate.replace('/data/safe_staging', str(staging))
      script = base / 'launch_chffrplus.sh'
      script.write_text(f'#!/usr/bin/env bash\nDIR={shlex.quote(str(base))}\nSTAGING_ROOT={shlex.quote(str(staging))}\n' +
                        gate + '}\nlaunch\necho KEPT\n')
      script.chmod(0o755)
      result = subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=5)
      self.assertEqual(result.returncode, 0, result.stderr)
      return result.stdout

  def test_launcher_rejects_stale_staging_after_git_branch_or_origin_changes(self):
    output = self.launcher_fixture(modified=True)
    self.assertIn('skipping overlay update installation', output)
    self.assertNotIn('ACTIVATED', output)

  def test_launcher_applies_normal_consistent_update(self):
    self.assertIn('ACTIVATED', self.launcher_fixture(modified=False))

  def test_launcher_does_not_install_finalized_tree_without_init_marker(self):
    self.assertNotIn('ACTIVATED', self.launcher_fixture(modified=False, marker=False))

  def test_launcher_retains_old_backup_instead_of_overwriting_it(self):
    output = self.launcher_fixture(modified=False, backup_exists=True)
    self.assertIn('openpilot backup found, not updating', output)
    self.assertNotIn('ACTIVATED', output)


if __name__ == '__main__':
  unittest.main()
