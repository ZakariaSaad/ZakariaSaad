"""Native imports and manager lifecycle, without starting vehicle processes."""
import importlib
import multiprocessing
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.sunnypilot.modeld_v2.process import V3RecoveryProcess


def harmless_child(*args):
  pass


class TestBootSmoke(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.params = Params(self.tmp.name)

  def test_real_imports_in_fresh_interpreters_both_orders(self):
    modules = ['openpilot.sunnypilot.modeld_v2.modeld', 'openpilot.sunnypilot.modeld_v2.cinque',
               'openpilot.sunnypilot.modeld_v2.cinque_guard', 'openpilot.sunnypilot.modeld_v2.stateful',
               'openpilot.sunnypilot.models.helpers', 'openpilot.system.manager.process_config',
               'openpilot.system.manager.manager', 'openpilot.system.updated.updated']
    for order in (modules, list(reversed(modules))):
      with self.subTest(first=order[0]):
        result = subprocess.run([sys.executable, '-c',
                                'import importlib,sys; [importlib.import_module(m) for m in sys.argv[1:]]', *order],
                                capture_output=True, timeout=45, env=os.environ | {'DEV': 'CPU'})
        self.assertEqual(result.returncode, 0, result.stderr.decode())

  def test_ordinary_startup_does_not_import_experimental_adapter(self):
    script = '''import sys
from types import SimpleNamespace
from openpilot.common.params import Params
from openpilot.sunnypilot.modeld_v2.startup import load_models
load_models(Params(sys.argv[1]), 1344, 760, False, lambda **kw: SimpleNamespace(chestnut=False))
assert 'openpilot.sunnypilot.modeld_v2.cinque' not in sys.modules
'''
    child = subprocess.run([sys.executable, '-c', script, self.tmp.name], capture_output=True, timeout=15)
    self.assertEqual(child.returncode, 0, child.stderr.decode())

  def test_real_modeld_launcher_help(self):
    from openpilot.common.basedir import BASEDIR
    result = subprocess.run(['bash', 'openpilot/sunnypilot/modeld_v2/modeld', '--help'], cwd=BASEDIR,
                            env=os.environ | {'DEV': 'CPU', 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']},
                            capture_output=True, timeout=45)
    self.assertEqual(result.returncode, 0, result.stderr.decode())
    self.assertIn(b'--demo', result.stdout)

  def test_manager_init_defaults_and_persistence(self):
    manager = importlib.import_module('openpilot.system.manager.manager')
    self.params.put('CinqueV3Fault', 'preserve across boot', block=True)
    self.params.put_bool('CinqueV3UseFallback', True, block=True)
    self.params.put_bool('CinqueV3RecoveryPending', True, block=True)
    self.params.put('UpdaterTargetBranch', 'wrong-old-branch', block=True)
    with patch.object(manager, 'Params', return_value=self.params), patch.object(manager, 'save_bootlog'), \
         patch.object(manager, 'register', return_value='CI-NONACTUATING'), patch.object(manager.sentry, 'init'), \
         patch.dict(os.environ):
      manager.manager_init()
    self.assertFalse(self.params.get_bool('CinqueV3Enabled'))
    self.assertFalse(self.params.get_bool('CinqueV3UseFallback'))
    self.assertFalse(self.params.get_bool('CinqueV3RecoveryPending'))
    self.assertEqual(self.params.get('CinqueV3Fault'), 'preserve across boot')
    self.assertIsNone(self.params.get('UpdaterTargetBranch'))

  def test_startup_diagnostic_preserves_fault_and_never_enables_v3(self):
    from tools.mvl_startup_check import check_startup
    self.params.put('CinqueV3Fault', 'persistent hardware diagnosis', block=True)
    with patch('openpilot.selfdrive.modeld.helpers.chestnut_present', return_value=False):
      report = check_startup(self.params, (1344, 760))
      self.assertEqual(report['v3_fault'], 'persistent hardware diagnosis')
      self.assertFalse(report['v3_enabled'])
      with self.assertRaisesRegex(RuntimeError, 'offroad'):
        check_startup(self.params, (1344, 760), warmup=True)
      self.params.put_bool('IsOffroad', True, block=True)
      self.params.put_bool('CinqueV3Enabled', True, block=True)
      with self.assertRaisesRegex(RuntimeError, 'Explicitly disable V3'):
        check_startup(self.params, (1344, 760), warmup=True)
      self.assertEqual(self.params.get('CinqueV3Fault'), 'persistent hardware diagnosis')

  def test_dead_modeld_restarts_only_on_consumable_watchdog_request(self):
    proc = V3RecoveryProcess('ci-modeld', '.', [], lambda *args: True)
    proc.launcher = harmless_child
    self.addCleanup(proc.stop)
    proc.proc = multiprocessing.Process(target=harmless_child)
    proc.proc.start()
    proc.proc.join(3)
    self.assertIsNotNone(proc.proc.exitcode)
    first_pid = proc.proc.pid
    self.params.put('CinqueV3Fault', 'old diagnostic only', block=True)
    with patch('openpilot.sunnypilot.modeld_v2.process.Params', return_value=self.params):
      proc.start()
      self.assertEqual(proc.proc.pid, first_pid)
      self.params.put_bool('CinqueV3RecoveryPending', True, block=True)
      proc.start()
      self.assertNotEqual(proc.proc.pid, first_pid)
      proc.proc.join(3)
      recovered_pid = proc.proc.pid
      for _ in range(5):
        proc.start()
        self.assertEqual(proc.proc.pid, recovered_pid)
      self.assertFalse(self.params.get_bool('CinqueV3RecoveryPending'))
      self.assertFalse(self.params.get_bool('CinqueV3Enabled'))


if __name__ == '__main__':
  unittest.main()
