"""Predictive boost's real parameter, persistence and SunnyLink RPC contract."""
import ast
import base64
import gzip
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from opendbc.car.structs import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType


def seed_manager_defaults(params):
  # Run only the real default-initialization loop, without launching manager,
  # writing vehicle state, invoking registration, or starting any process.
  tree = ast.parse((Path(BASEDIR) / 'openpilot/system/manager/manager.py').read_text())
  loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and ast.unparse(node.iter) == 'params.all_keys()')
  exec(compile(ast.Module(body=[loop], type_ignores=[]), 'manager_param_defaults', 'exec'), {'params': params})


@pytest.fixture
def device(tmp_path, monkeypatch):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  params = Params()
  seed_manager_defaults(params)
  from openpilot.sunnypilot.sunnylink.athena import sunnylinkd
  monkeypatch.setattr(sunnylinkd, 'params', params)
  return SimpleNamespace(params=params, rpc=sunnylinkd)


def metadata(device):
  return json.loads(gzip.decompress(base64.b64decode(device.rpc.getParamsMetadata())))


def test_default_and_saved_preference_survive_lifecycle(device):
  params = device.params
  assert params.get_type('AccelBoostSP') == ParamKeyType.BOOL
  assert params.get_default_value('AccelBoostSP') is True
  assert params.get_bool('AccelBoostSP') is True
  for flag in (ParamKeyFlag.PERSISTENT, ParamKeyFlag.BACKUP):
    assert b'AccelBoostSP' in params.all_keys(flag)
  for value in (False, True):
    params.put_bool('AccelBoostSP', value, block=True)
    seed_manager_defaults(params)  # A subsequent startup must preserve the choice.
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
      assert Params().get_bool('AccelBoostSP') is value


@pytest.mark.parametrize('longitudinal', [None, False, True])
def test_served_cruise_toggle_uses_existing_longitudinal_capability(device, longitudinal):
  if longitudinal is not None:
    cp = car.CarParams.new_message(brand='honda', openpilotLongitudinalControl=longitudinal)
    device.params.put('CarParamsPersistent', cp.to_bytes(), block=True)
  schema = metadata(device)
  panel = next(panel for panel in schema['panels'] if panel['id'] == 'cruise')
  item = next(item for section in panel['sections'] for item in section['items'] if item['key'] == 'AccelBoostSP')
  assert item['title'] == 'Acceleration Boost'
  assert item['widget'] == 'toggle'
  assert item['enablement'] == [{'type': 'capability', 'field': 'has_longitudinal_control', 'equals': True}]
  assert schema['capabilities']['has_longitudinal_control'] is (longitudinal is True)
  # Availability does not erase the saved choice or alter other settings.
  assert device.params.get_bool('AccelBoostSP') is True


@pytest.mark.parametrize('compressed', [False, True])
def test_remote_toggle_controls_the_runtime_parameter(device, compressed):
  from openpilot.sunnypilot.selfdrive.controls.lib.accel_boost import AccelBoostSP

  boost = AccelBoostSP()
  for value in (False, True, False):
    raw = str(value).lower().encode()
    encoded = base64.b64encode(gzip.compress(raw) if compressed else raw).decode()
    device.rpc.saveParams({'AccelBoostSP': encoded}, compression=compressed)
    response = device.rpc.getParams(['AccelBoostSP'], compression=compressed)
    record = json.loads(response['params'])[0]
    assert record['key'] == 'AccelBoostSP' and record['type'] == int(ParamKeyType.BOOL)
    reply = base64.b64decode(response['AccelBoostSP'])
    decoded = (gzip.decompress(reply) if compressed else reply).decode().lower() in ('true', '1')
    assert decoded is value
    boost.frame = -1  # Poll immediately; production retains its upstream interval.
    boost.update_params()
    assert boost.enabled is value and Params().get_bool('AccelBoostSP') is value
