"""The user CLI shares Settings admission and never operates a GPU or service."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from openpilot.selfdrive.navigation import manage
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled


class RecordingParams:
  def __init__(self, **values):
    self.values = {'NavigationModelDrivingEnabled': False, 'NavigationModelCandidate': 'previous-candidate',
                   'NavigationModelManifest': '/data/navigation-models/registered/manifest.json',
                   'IsOffroad': True, 'IsEngaged': False,
                   'ModelManager_ActiveBundle': {'preserve': 'small'},
                   'ModelManager_ActiveBundleChestnut': {'preserve': 'big'},
                   'ModelRunnerTypeCache': 0, 'NavigationEnabled': True, **values}
    self.writes = []

  def get(self, key, **_kwargs):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value, **_kwargs):
    self.writes.append((key, value))
    self.values[key] = value

  def put_bool(self, key, value, **kwargs):
    self.put(key, bool(value), **kwargs)

  def remove(self, key):
    self.values.pop(key, None)


def status(**updates):
  return {'id': 'big-nav-v2-map-fp32', 'display_name': 'Nav Model Integration Bigv2', 'eligible': False,
          'reason': 'Explicit user authorization for Navigation Model Driving is still required',
          'blockers': ['Explicit user authorization for Navigation Model Driving is still required'],
          'enabled': False, 'registered': True, 'installed': True, 'selected': False, 'approved': False,
          'artifact_path': '/data/nav-bigv2/bigv2-amd-map-fp32.sptg', 'software_compatible': True,
          'artifact_evidence_reviewed': True, 'runtime_compatibility': 'unverified', **updates}


class TestNavigationManage(unittest.TestCase):
  def setUp(self):
    self.params = RecordingParams()
    self.before = copy.deepcopy(self.params.values)
    self.enterContext(patch('os.geteuid', return_value=1000))
    def offroad(params):
      if not params.get_bool('IsOffroad') or params.get_bool('IsEngaged'):
        raise NavigationModelDisabled('Genuine off-road and disengaged state required')
    self.enterContext(patch('openpilot.sunnypilot.models.selection.require_offroad', side_effect=offroad))
    self.enterContext(patch('subprocess.run', side_effect=AssertionError('CLI must not run a compiler, service or diagnostic')))
    self.enterContext(patch('subprocess.Popen', side_effect=AssertionError('CLI must not spawn a compiler, service or diagnostic')))

  def invoke(self, *args):
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
      result = manage.main(list(args), params=self.params)
    return result, output.getvalue() + errors.getvalue()

  def test_status_reports_installed_registered_reviewed_but_unapproved_without_mutation(self):
    with patch.object(manage, 'candidate_status', return_value=status()), patch.object(manage, 'set_driving_enabled') as enable:
      result, output = self.invoke('status')
    self.assertEqual(result, 0)
    self.assertIn('Nav Model Integration Bigv2', output)
    self.assertIn('/data/nav-bigv2/bigv2-amd-map-fp32.sptg', output)
    for label in ('registration', 'compatibility', 'approval', 'selection', 'activation'):
      self.assertIn(label, output.lower())
    self.assertIn('Explicit user authorization', output)
    self.assertEqual(self.params.values, self.before)
    self.assertEqual(self.params.writes, [])
    enable.assert_not_called()

  def test_status_surfaces_all_specific_blockers(self):
    reasons = ['Manifest SHA256 mismatch', 'Ordinary QCOM fallback is missing']
    with patch.object(manage, 'candidate_status', return_value=status(blockers=reasons, reason='; '.join(reasons))):
      result, output = self.invoke('status')
    self.assertEqual(result, 0)
    for reason in reasons:
      self.assertIn(reason, output)
    self.assertEqual(self.params.writes, [])

  def test_enable_uses_shared_admission_and_does_not_write_around_a_refusal(self):
    reason = 'Explicit user authorization for Navigation Model Driving is still required'
    with patch.object(manage, 'set_driving_enabled', side_effect=NavigationModelDisabled(reason)) as enable:
      result, output = self.invoke('enable', 'big-nav-v2-map-fp32')
    self.assertNotEqual(result, 0)
    self.assertIn(reason, output)
    enable.assert_called_once_with(self.params, True, offroad=True, candidate_id='big-nav-v2-map-fp32')
    self.assertEqual(self.params.values, self.before)
    self.assertEqual(self.params.writes, [])

  def test_successful_enable_requests_only_shared_navigation_selection(self):
    with patch.object(manage, 'set_driving_enabled') as enable:
      result, output = self.invoke('enable', 'big-nav-v2-map-fp32')
    self.assertEqual(result, 0)
    self.assertIn('Bigv2', output)
    enable.assert_called_once_with(self.params, True, offroad=True, candidate_id='big-nav-v2-map-fp32')
    # The CLI must not separately mutate a flag, manifest or ordinary bundle.
    self.assertEqual(self.params.writes, [])

  def test_disable_preserves_candidate_manifest_and_ordinary_model_slots(self):
    self.params.values['NavigationModelDrivingEnabled'] = True
    before = copy.deepcopy(self.params.values)
    result, output = self.invoke('disable')
    self.assertEqual(result, 0)
    self.assertIn('off', output.lower())
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    before['NavigationModelDrivingEnabled'] = False
    self.assertEqual({key: self.params.values[key] for key in before}, before)
    self.assertEqual(self.params.values['ModelManager_SelectionState']['state'], 'selected')

  def test_disable_rejects_unsafe_onroad_configuration_change(self):
    self.params.values.update(IsOffroad=False, IsEngaged=True, NavigationModelDrivingEnabled=True)
    with patch.object(manage, 'candidate_status', side_effect=AssertionError('Disable must never need artifact or approval checks')):
      result, _output = self.invoke('disable')
    self.assertEqual(result, 1)
    self.assertTrue(self.params.get_bool('NavigationModelDrivingEnabled'))

  def test_unknown_model_cannot_change_any_selection(self):
    with patch.object(manage, 'set_driving_enabled') as enable:
      with self.assertRaises(SystemExit) as raised:
        self.invoke('enable', 'ordinary-model-or-unknown')
    self.assertNotEqual(raised.exception.code, 0)
    enable.assert_not_called()
    self.assertEqual(self.params.values, self.before)
    self.assertEqual(self.params.writes, [])

  def test_real_cli_refuses_root_before_opening_params(self):
    output = io.StringIO()
    with patch('os.geteuid', return_value=0), contextlib.redirect_stderr(output):
      result = manage.main(['status'])
    self.assertNotEqual(result, 0)
    self.assertIn('comma', output.getvalue())
    self.assertIn('without sudo', output.getvalue())

  def test_admission_io_error_keeps_original_reason_and_selection(self):
    reason = 'Unable to read /data/nav-bigv2/bigv2-amd-map-fp32.sptg: permission denied'
    with patch.object(manage, 'set_driving_enabled', side_effect=OSError(reason)):
      result, output = self.invoke('enable', 'big-nav-v2-map-fp32')
    self.assertNotEqual(result, 0)
    self.assertIn(reason, output)
    self.assertEqual(self.params.values, self.before)
    self.assertEqual(self.params.writes, [])


class TestSystemPythonManage(unittest.TestCase):
  @staticmethod
  def run_system_python(source):
    root = Path(__file__).resolve().parents[4]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
    python = '/usr/bin/python3' if Path('/usr/bin/python3').is_file() else sys.executable
    return subprocess.run([python, '-S', '-B', '-c', source], cwd=root, env=environment,
                          capture_output=True, text=True, timeout=15, check=False)

  def test_status_can_be_imported_without_runtime_python_packages(self):
    """-S excludes site-packages, as a stock SSH system Python may do."""
    source = '''
import sys
from openpilot.selfdrive.navigation import manage
class Params:
  def get(self, key, **kwargs): return None
  def get_bool(self, key): return False
assert manage.main(['status'], params=Params()) == 0
blocked = ('numpy', 'capnp', 'zmq', 'tinygrad', 'openpilot.cereal')
assert not [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
'''
    completed = self.run_system_python(source)
    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

  def test_corrected_selection_persists_and_disable_works_with_system_python(self):
    """Use native Params persistence and normal lifecycle clears, without a GPU."""
    source = '''
import sys
import tempfile
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.navigation import manage
with tempfile.TemporaryDirectory() as directory:
  params = Params(directory)
  params.put('NavigationModelCandidate', 'big-nav-v2-map-fp32', block=True)
  manifest = '/data/navigation-models/inert-corrected-fixture/manifest.json'
  params.put('NavigationModelManifest', manifest, block=True)
  params.put('ModelManager_ActiveBundle', {'preserve': 'ordinary-small'}, block=True)
  params.put('ModelManager_ActiveBundleChestnut', {'preserve': 'ordinary-big'}, block=True)
  params.put_bool('NavigationModelDrivingEnabled', True, block=True)
  del params
  params = Params(directory)
  for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION,
               ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
    params.clear_all(flag)
  assert params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('NavigationModelCandidate') == 'big-nav-v2-map-fp32'
  assert params.get('NavigationModelManifest') == manifest
  assert manage.main(['status'], params=params) == 0
  from unittest.mock import patch
  with patch('openpilot.sunnypilot.models.selection.require_offroad'):
    assert manage.main(['disable'], params=params) == 0
  assert not params.get_bool('NavigationModelDrivingEnabled')
  assert params.get('NavigationModelCandidate') == 'big-nav-v2-map-fp32'
  assert params.get('NavigationModelManifest') == manifest
  assert params.get('ModelManager_ActiveBundle') == {'preserve': 'ordinary-small'}
  assert params.get('ModelManager_ActiveBundleChestnut') == {'preserve': 'ordinary-big'}
  del params
blocked = ('numpy', 'capnp', 'zmq', 'tinygrad', 'openpilot.cereal')
assert not [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
'''
    completed = self.run_system_python(source)
    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class TestCorrectedNavigationManage(unittest.TestCase):
  """Exercise corrected CLI dispatch independently of installed device files."""
  ID = 'big-nav-v2-map-fp32'
  invoke = TestNavigationManage.invoke

  def setUp(self):
    TestNavigationManage.setUp(self)
    self.enterContext(patch.object(manage, 'list_candidates', return_value=[
      {'id': self.ID, 'display_name': 'Nav Model Integration Bigv2 — FP32 map correction'},
    ]))

  def test_register_corrected_dispatches_exact_identity_without_enabling(self):
    receipt = {'manifest_path': '/data/navigation-models/corrected/manifest.json', 'artifact_sha256': '4' * 64}
    corrected = status(id=self.ID, artifact_path='/data/nav-bigv2/bigv2-amd-map-fp32.sptg')
    with patch.object(manage, '_register', return_value=receipt) as register, \
         patch.object(manage, 'candidate_status', return_value=corrected), patch.object(manage, 'set_driving_enabled') as enable:
      result, output = self.invoke('register', self.ID)
    self.assertEqual(result, 0)
    register.assert_called_once_with(self.params, self.ID)
    enable.assert_not_called()
    self.assertIn('remains OFF', output)
    self.assertIn('not approved', output)
    self.assertIn('bigv2-amd-map-fp32.sptg', output)
    self.assertEqual(self.params.writes, [])

  def test_register_refusal_is_reported_without_enable_or_select(self):
    with patch.object(manage, '_register', side_effect=ValueError('Corrected artifact SHA256 mismatch')), \
         patch.object(manage, 'set_driving_enabled') as enable, patch.object(manage, 'select_candidate') as select:
      result, output = self.invoke('register', self.ID)
    self.assertEqual(result, 1)
    self.assertIn('Corrected artifact SHA256 mismatch', output)
    enable.assert_not_called()
    select.assert_not_called()
    self.assertEqual(self.params.writes, [])

  def test_select_corrected_uses_shared_preparation_without_manual_registration_gate(self):
    with patch.object(manage, 'candidate_status') as inspect, \
         patch.object(manage, 'select_candidate') as select, patch.object(manage, 'set_driving_enabled') as enable:
      result, output = self.invoke('select', self.ID)
    self.assertEqual(result, 0)
    inspect.assert_not_called()
    select.assert_called_once_with(self.params, self.ID, offroad=True)
    enable.assert_not_called()
    self.assertIn('requested ON', output)

  def test_onroad_select_never_mutates_selection(self):
    with patch.object(manage, 'select_candidate', side_effect=NavigationModelDisabled('deviceState.started is true')) as select:
      result, output = self.invoke('select', self.ID)
    self.assertEqual(result, 1)
    self.assertIn('deviceState.started', output)
    select.assert_called_once_with(self.params, self.ID, offroad=True)
    self.assertEqual(self.params.writes, [])

  def test_shared_preparation_failure_is_reported(self):
    with patch.object(manage, 'select_candidate', side_effect=NavigationModelDisabled('Corrected artifact SHA256 mismatch')):
      result, output = self.invoke('select', self.ID)
    self.assertEqual(result, 1)
    self.assertIn('Corrected artifact SHA256 mismatch', output)
    self.assertEqual(self.params.writes, [])

  def test_status_addresses_explicit_model_and_defaults_to_selected(self):
    with patch.object(manage, 'candidate_status', return_value=status(id=self.ID)) as inspect:
      self.assertEqual(self.invoke('status', self.ID)[0], 0)
      inspect.assert_called_once_with(self.params, candidate_id=self.ID)
      inspect.reset_mock()
      self.assertEqual(self.invoke('status')[0], 0)
      inspect.assert_called_once_with(self.params, candidate_id=None)

  def test_enable_corrected_cannot_bypass_its_approval(self):
    from openpilot.sunnypilot.models import selection
    with patch.object(selection, 'candidate_status', return_value={'registered': True}), \
         patch('openpilot.selfdrive.navigation.download.prepare_downloaded_model'), \
         patch.object(selection, 'admit_candidate', side_effect=NavigationModelDisabled('Explicit user authorization is required')):
      result, output = self.invoke('enable', self.ID)
    self.assertEqual(result, 1)
    self.assertIn('Explicit user authorization', output)
    self.assertFalse(self.params.get_bool('NavigationModelDrivingEnabled'))
    for key in self.before:
      self.assertEqual(self.params.values[key], self.before[key])


if __name__ == '__main__':
  unittest.main()
