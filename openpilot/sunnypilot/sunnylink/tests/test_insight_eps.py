"""Real parameter/schema/remote interfaces with an identified synthetic Insight.

Only fresh messaging admission is supplied as an off-road fixture. No firmware,
controllers, vehicle interfaces, GPU or live messaging publishers are opened.
"""
import base64
import gzip
import json
from types import SimpleNamespace

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import FIRMWARE_SHA256, DRIVE_KEY, prepare_drive_config, get_drive_config
from openpilot.sunnypilot.sunnylink import insight_eps
from openpilot.sunnypilot.sunnylink.athena import sunnylinkd
from openpilot.sunnypilot.sunnylink.backups.manager import BackupManagerSP
from openpilot.sunnypilot.sunnylink.capabilities import generate_capabilities


def encoded(value):
  return base64.b64encode(str(value).encode()).decode()


@pytest.fixture
def identified(tmp_path, monkeypatch):
  root = str(tmp_path / 'params')
  monkeypatch.setenv('PARAMS_ROOT', root)
  params = Params(root)
  fingerprint = gen_empty_fingerprint()
  fw = [structs.CarParams.CarFw(ecu=structs.CarParams.Ecu.eps, address=0x18da30f1,
                               fwVersion=b'39990-TXM,A040', brand='honda')]
  CP = CarInterface.get_params(CAR.HONDA_INSIGHT, fingerprint, fw, False, False, False)
  CP_SP = convert_to_capnp(CarInterface.get_params_sp(CP, CAR.HONDA_INSIGHT, fingerprint, fw, False, False, False))
  CP.carFw = [{'ecu': 'eps', 'address': fw[0].address, 'fwVersion': fw[0].fwVersion, 'brand': 'honda'}]
  params.put('CarParamsPersistent', CP.to_bytes(), block=True)
  params.put('CarParamsSPPersistent', CP_SP.to_bytes(), block=True)
  CP.clear_write_flag()
  CP_SP.clear_write_flag()
  params.put_bool('IsOffroad', True, block=True)
  params.put_bool('IsEngaged', False, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', FIRMWARE_SHA256, block=True)
  params.put('HondaEpsFirmwareGeometry', 'james', block=True)
  params.put('HondaEpsFirmwareResponse', 'route32', block=True)
  monkeypatch.setattr(sunnylinkd, 'params', params)
  monkeypatch.setattr(sunnylinkd, 'Params', lambda: params)
  return SimpleNamespace(params=params, CP=CP, CP_SP=CP_SP)


@pytest.fixture
def offroad(monkeypatch):
  calls = []

  def guard(params):
    assert params.get_bool('IsOffroad') and not params.get_bool('IsEngaged')
    calls.append(True)

  monkeypatch.setattr(insight_eps, 'require_offroad', guard)
  return calls


def test_supported_capabilities_and_status_use_real_vehicle_and_confirmation(identified):
  params = identified.params
  before = {key: params.get(key) for key in insight_eps.SETTINGS | insight_eps.PROTECTED}
  caps = generate_capabilities(params)
  assert caps['honda_eps_supported'] and caps['honda_eps_eligible']
  assert not caps['honda_eps_requested'] and not caps['honda_eps_effective']
  assert 'next drive cycle' in caps['honda_eps_status']
  assert 'geometry=james, response=route32' in caps['honda_eps_status']
  assert before == {key: params.get(key) for key in before}


def test_enable_disable_preserve_firmware_calibration_ordinary_choices_and_drive_latch(identified, offroad):
  params, CP, CP_SP = identified.params, identified.CP, identified.CP_SP
  params.put_bool('EnforceTorqueControl', True, block=True)
  params.put_bool('NeuralNetworkLateralControl', True, block=True)
  assert insight_eps.apply_enabled(params, True)['requested']
  config = prepare_drive_config(params, CP, CP_SP)
  snapshot = params.get(DRIVE_KEY)
  assert config is not None and config.geometry == 'james' and config.response == 'route32'
  assert 'preserved' in insight_eps.status(params)['description']
  assert not insight_eps.apply_enabled(params, False)['requested']
  assert get_drive_config(params, CP, CP_SP) == config
  assert params.get(DRIVE_KEY) == snapshot
  assert params.get('HondaEpsFirmwareVerifiedSha256') == FIRMWARE_SHA256
  assert params.get_bool('EnforceTorqueControl') and params.get_bool('NeuralNetworkLateralControl')
  params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
  assert prepare_drive_config(params, CP, CP_SP) is None
  assert len(offroad) == 4


@pytest.mark.parametrize('kind', ['missing_firmware', 'wrong_firmware', 'geometry', 'response', 'wrong_vehicle'])
def test_ineligible_enable_is_visible_and_never_fabricates_confirmation(identified, offroad, kind):
  params = identified.params
  if kind == 'missing_firmware':
    params.remove('HondaEpsFirmwareVerifiedSha256')
  elif kind == 'wrong_firmware':
    params.put('HondaEpsFirmwareVerifiedSha256', '0' * 64, block=True)
  elif kind in ('geometry', 'response'):
    params.put('HondaEpsFirmware' + kind.title(), 'unreviewed', block=True)
  else:
    CP = identified.CP
    CP.carFingerprint = 'HONDA_CIVIC'
    params.put('CarParamsPersistent', CP.to_bytes(), block=True)
  before = {key: params.get(key) for key in insight_eps.SETTINGS | insight_eps.PROTECTED}
  with pytest.raises(ValueError, match='Cannot enable Insight EPS'):
    insight_eps.apply_enabled(params, True)
  assert not insight_eps.status(params)['eligible']
  assert 'Enabling blocked:' in insight_eps.status(params)['description']
  assert before == {key: params.get(key) for key in before}


def test_onroad_requests_cannot_change_any_setting(identified):
  params = identified.params
  params.put_bool('IsOffroad', False, block=True)
  with pytest.raises(ValueError, match='IsOffroad'):
    insight_eps.apply_enabled(params, True)
  assert not params.get_bool(insight_eps.ENABLED)
  assert params.get(DRIVE_KEY) is None


@pytest.mark.parametrize('partial_write', [False, True])
def test_interrupted_settings_batch_restores_previous_settings_without_touching_latch(identified, offroad, monkeypatch, partial_write):
  params = identified.params
  before = {key: params.get(key) for key in insight_eps.SETTINGS | insight_eps.PROTECTED}
  original_put = params.put
  failed = False

  def write(key, value, **kwargs):
    nonlocal failed
    if key == 'HondaEpsFirmwareGeometry' and not failed:
      failed = True
      if partial_write:
        original_put(key, value, **kwargs)
        raise OSError('Synthetic interrupted settings write')
      return
    original_put(key, value, **kwargs)

  monkeypatch.setattr(params, 'put', write)
  with pytest.raises(ValueError, match='persist|interrupted'):
    insight_eps.apply_settings(params, {insight_eps.ENABLED: True, 'HondaEpsFirmwareGeometry': 'mvl'})
  assert before == {key: params.get(key) for key in before}


def test_requested_and_effective_mismatch_explains_next_drive(identified, offroad):
  params, CP, CP_SP = identified.params, identified.CP, identified.CP_SP
  prepare_drive_config(params, CP, CP_SP)  # Ordinary controller latched.
  insight_eps.apply_enabled(params, True)
  params.put('CarParams', CP.to_bytes(), block=True)
  params.put('CarParamsSP', CP_SP.to_bytes(), block=True)
  params.put_bool('IsOffroad', False, block=True)
  status = insight_eps.status(params)
  assert status['requested'] and not status['effective'] and status['pending']
  assert 'Current drive: ordinary controller' in status['description']
  assert 'pending the next drive cycle' in status['description']


def test_actual_remote_metadata_contains_toggle_and_live_status(identified):
  schema = json.loads(gzip.decompress(base64.b64decode(sunnylinkd.getParamsMetadata())))
  steering = next(panel for panel in schema['panels'] if panel['id'] == 'steering')
  section = next(item for item in steering['sections'] if item['id'] == 'insight_eps')
  toggle, info = section['items']
  assert toggle['key'] == insight_eps.ENABLED and toggle['widget'] == 'toggle'
  assert toggle['title'] == 'Insight-specific EPS lateral controller'
  assert info['key'] == insight_eps.STATUS and info['widget'] == 'info' and info['blocked']
  assert schema['capabilities']['honda_eps_eligible']
  response = sunnylinkd.getParams([insight_eps.STATUS])
  assert 'Requested: OFF.' in base64.b64decode(response[insight_eps.STATUS]).decode()


def test_actual_remote_save_uses_existing_parameter_and_rejects_proof_writes(identified, offroad):
  params = identified.params
  sunnylinkd.saveParams({insight_eps.ENABLED: encoded(True)})
  assert params.get_bool(insight_eps.ENABLED)
  assert params.get(DRIVE_KEY) is None
  sunnylinkd.saveParams({'HondaEpsFirmwareVerifiedSha256': encoded('0' * 64), DRIVE_KEY: encoded('{}'), insight_eps.STATUS: encoded('fake')})
  assert params.get('HondaEpsFirmwareVerifiedSha256') == FIRMWARE_SHA256
  assert params.get(DRIVE_KEY) is None
  sunnylinkd.saveParams({insight_eps.ENABLED: encoded(False)})
  assert not params.get_bool(insight_eps.ENABLED)
  assert len(offroad) == 4


def test_backup_restores_settings_through_same_admission_without_proof(identified, offroad):
  params = identified.params
  manager = object.__new__(BackupManagerSP)
  manager.params = params
  manager._apply_config({insight_eps.ENABLED: encoded(True), 'HondaEpsFirmwareGeometry': encoded('mvl'),
                         'HondaEpsFirmwareResponse': encoded('source'), 'HondaEpsFirmwareVerifiedSha256': encoded('0' * 64),
                         DRIVE_KEY: encoded('{}')})
  assert params.get_bool(insight_eps.ENABLED)
  assert params.get('HondaEpsFirmwareGeometry') == 'mvl'
  assert params.get('HondaEpsFirmwareResponse') == 'source'
  assert params.get('HondaEpsFirmwareVerifiedSha256') == FIRMWARE_SHA256 and params.get(DRIVE_KEY) is None
  assert len(offroad) == 2
  backup = {key.decode() for key in params.all_keys(ParamKeyFlag.BACKUP)}
  assert insight_eps.SETTINGS <= backup and not insight_eps.PROTECTED & backup


def test_backup_cannot_enable_without_existing_confirmation(identified, offroad):
  params = identified.params
  params.remove('HondaEpsFirmwareVerifiedSha256')
  manager = object.__new__(BackupManagerSP)
  manager.params = params
  with pytest.raises(ValueError, match='installed-firmware confirmation'):
    manager._apply_config({insight_eps.ENABLED: encoded(True), 'HondaEpsFirmwareVerifiedSha256': encoded(FIRMWARE_SHA256)})
  assert not params.get_bool(insight_eps.ENABLED)
  assert params.get('HondaEpsFirmwareVerifiedSha256') is None
