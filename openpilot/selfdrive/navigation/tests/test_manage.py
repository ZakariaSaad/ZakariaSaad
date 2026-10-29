"""Optional CLI uses model selection; registration/status have no activation authority."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation import manage
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub


def status(**updates):
  return {'id': MODEL_ID, 'display_name': 'Nav Model Integration Bigv2', 'eligible': True,
          'reason': '', 'blockers': [], 'registered': True, 'installed': True, 'selected': False,
          'artifact_path': '/data/nav-bigv2/bigv2-amd-map-fp32.sptg', 'software_compatible': True,
          'artifact_evidence_reviewed': True, 'runtime_compatibility': 'reviewed_native_binary_abi_adapter', **updates}


@pytest.fixture
def params():
  return ParamsStub(IsOffroad=True, IsEngaged=False,
                    NavigationModelManifest='/data/navigation-models/registered/manifest.json',
                    ModelManager_ActiveBundle={'preserve': 'small'},
                    ModelManager_ActiveBundleChestnut={'preserve': 'big'}, NavigationEnabled=True)


def invoke(params, *args):
  output, errors = io.StringIO(), io.StringIO()
  with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
    result = manage.main(list(args), params=params)
  return result, output.getvalue() + errors.getvalue()


def test_status_reports_registration_selection_runtime_without_mutation(params):
  before = copy.deepcopy(params.values)
  with patch.object(manage, 'candidate_status', return_value=status()), patch.object(manage, 'select_candidate') as select:
    result, output = invoke(params, 'status')
  assert result == 0
  for label in ('Registration:', 'selection:', 'Runtime:', 'compatibility'):
    assert label in output
  assert 'Branch activation approval' not in output and 'Navigation Model Driving:' not in output
  assert params.values == before
  select.assert_not_called()


def test_status_surfaces_specific_file_blockers(params):
  reasons = ['Manifest SHA256 mismatch', 'Ordinary QCOM fallback is missing']
  with patch.object(manage, 'candidate_status', return_value=status(blockers=reasons, reason='; '.join(reasons))):
    result, output = invoke(params, 'status')
  assert result == 0 and all(reason in output for reason in reasons)


def test_select_dispatches_to_shared_model_service(params):
  before = copy.deepcopy(params.values)
  with patch.object(manage, 'select_candidate') as select, patch('subprocess.Popen', side_effect=AssertionError('No service/diagnostic')):
    result, output = invoke(params, 'select', MODEL_ID)
  assert result == 0 and 'as the Big Model' in output
  select.assert_called_once_with(params, MODEL_ID, offroad=True)
  assert params.values == before


@pytest.mark.parametrize('reason', ['deviceState.started is true', 'Corrected artifact SHA256 mismatch', 'Artifact file unavailable'])
def test_selection_refusal_preserves_reason_and_selection(params, reason):
  before = copy.deepcopy(params.values)
  with patch.object(manage, 'select_candidate', side_effect=NavigationModelDisabled(reason)):
    result, output = invoke(params, 'select', MODEL_ID)
  assert result == 1 and reason in output and params.values == before


def test_register_has_no_selection_side_effect(params):
  receipt = {'manifest_path': '/data/navigation-models/corrected/manifest.json', 'artifact_sha256': '4' * 64}
  with patch.object(manage, '_register', return_value=receipt) as register, \
       patch.object(manage, 'candidate_status', return_value=status()), patch.object(manage, 'select_candidate') as select:
    result, output = invoke(params, 'register', MODEL_ID)
  assert result == 0 and 'Current model selection is unchanged' in output
  register.assert_called_once_with(params, MODEL_ID)
  select.assert_not_called()


def test_register_failure_does_not_select(params):
  before = copy.deepcopy(params.values)
  with patch.object(manage, '_register', side_effect=ValueError('Corrected artifact SHA256 mismatch')), \
       patch.object(manage, 'select_candidate') as select:
    result, output = invoke(params, 'register', MODEL_ID)
  assert result == 1 and 'SHA256 mismatch' in output and params.values == before
  select.assert_not_called()


@pytest.mark.parametrize('args', [('enable', MODEL_ID), ('disable',), ('select', 'big-nav-v2'), ('select', 'unknown')])
def test_obsolete_activation_commands_and_retired_models_are_rejected(params, args):
  before = copy.deepcopy(params.values)
  with pytest.raises(SystemExit) as error:
    invoke(params, *args)
  assert error.value.code != 0 and params.values == before


def test_real_cli_refuses_root_before_opening_params():
  with patch('os.geteuid', return_value=0), contextlib.redirect_stderr(io.StringIO()) as output:
    result = manage.main(['status'])
  assert result == 1 and 'without sudo' in output.getvalue()


def _system_python(source):
  root = Path(__file__).resolve().parents[4]
  environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
  python = '/usr/bin/python3' if Path('/usr/bin/python3').is_file() else sys.executable
  return subprocess.run([python, '-S', '-B', '-c', source], cwd=root, env=environment,
                        capture_output=True, text=True, timeout=15, check=False)


def test_status_imports_without_runtime_packages():
  completed = _system_python('''
import sys
from openpilot.selfdrive.navigation import manage
class Params:
  def get(self, key, **kwargs): return None
  def get_bool(self, key): return False
assert manage.main(['status'], params=Params()) == 0
blocked = ('numpy', 'capnp', 'zmq', 'tinygrad', 'openpilot.cereal')
assert not [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
''')
  assert completed.returncode == 0, completed.stdout + completed.stderr


def test_canonical_selection_survives_native_params_restart_and_status_does_not_mutate():
  completed = _system_python('''
import tempfile
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.navigation import manage
from openpilot.sunnypilot.models.navigation_bundle import navigation_bundle
with tempfile.TemporaryDirectory() as directory:
  params = Params(directory)
  bundle = navigation_bundle()
  params.put('ModelManager_ActiveBundleChestnut', bundle, block=True)
  params.put('ModelManager_ActiveBundle', {'preserve': 'ordinary-small'}, block=True)
  params.put('NavigationModelManifest', '/data/navigation-models/inert/manifest.json', block=True)
  del params
  params = Params(directory)
  for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION,
               ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
    params.clear_all(flag)
  assert params.get('ModelManager_ActiveBundleChestnut') == bundle
  assert params.get('ModelManager_ActiveBundle') == {'preserve': 'ordinary-small'}
  assert manage.main(['status'], params=params) == 0
  assert params.get('ModelManager_ActiveBundleChestnut') == bundle
''')
  assert completed.returncode == 0, completed.stdout + completed.stderr
