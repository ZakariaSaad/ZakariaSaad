import json
from pathlib import Path

import pytest

from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.common.params import ParamKeyFlag
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import (
  drive_config_for_model, get_drive_config, is_supported_vehicle, prepare_drive_config,
)


RWD_SHA256 = '7ba3743248d30758f8423bcb2a1c65f0d2d2d3563a830b8268bf448659b28a38'


def authorize(params, geometry='mvl'):
  params.put_bool('HondaEpsFirmwareEnabled', True, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', RWD_SHA256, block=True)
  params.put('HondaEpsFirmwareGeometry', geometry, block=True)


@pytest.mark.parametrize('native_sp', [False, True])
def test_supported_native_insight_and_separate_modified_flag(car_factory, native_sp):
  CP, CP_SP = car_factory(native_sp=native_sp)
  assert CP.flags & HondaFlags.BOSCH
  assert CP_SP.flags & HondaFlagsSP.EPS_MODIFIED
  assert CP.lateralTuning.which() == 'pid'
  assert list(CP.lateralParams.torqueV) == [0, 4096]
  assert is_supported_vehicle(CP, CP_SP)
  CP_SP.flags = int(CP_SP.flags & ~HondaFlagsSP.EPS_MODIFIED)
  CP.flags |= int(HondaFlagsSP.EPS_MODIFIED)
  assert not is_supported_vehicle(CP, CP_SP)


@pytest.mark.parametrize('versions', [(), (b'39990-TXM-A040',), (b'39990-TXM,A030',),
                                    (b'39990-TBA,C020',), (b'39990-TXM,A040', b'39990-TXM-A040')])
def test_unknown_stock_missing_or_conflicting_eps_rejected(car_factory, versions):
  CP, CP_SP = car_factory(versions=versions)
  assert not is_supported_vehicle(CP, CP_SP)


@pytest.mark.parametrize('candidate', [CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_ACCORD])
def test_other_honda_platforms_rejected(car_factory, candidate):
  assert not is_supported_vehicle(*car_factory(candidate=candidate))


def test_changed_transport_mapping_rejected(params, car_factory):
  CP, CP_SP = car_factory()
  authorize(params)
  CP.lateralParams.torqueV = [0, 3840]
  assert not is_supported_vehicle(CP, CP_SP)
  assert prepare_drive_config(params, CP, CP_SP) is None


@pytest.mark.parametrize('tuning', ['torque', 'angle'])
def test_non_native_pid_configuration_rejected(car_factory, tuning):
  CP, CP_SP = car_factory()
  if tuning == 'torque':
    CP.lateralTuning.init('torque')
  else:
    CP.steerControlType = 'angle'
  assert not is_supported_vehicle(CP, CP_SP)


def test_disabled_default_and_unverified_identifier_do_not_enable(params, car_factory):
  CP, CP_SP = car_factory()
  assert not params.get_bool('HondaEpsFirmwareEnabled')
  assert prepare_drive_config(params, CP, CP_SP) is None
  assert get_drive_config(params, CP, CP_SP) is None


@pytest.mark.parametrize('digest', ['', '7ba37432', '0' * 64,
                                  '089a8a7251da04e4bbe693f764c4fe7a02ad6a630afde0fa700925801d9f9728'])
def test_only_complete_corrected_rwd_confirmation_is_accepted(params, car_factory, digest):
  CP, CP_SP = car_factory()
  params.put_bool('HondaEpsFirmwareEnabled', True, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', digest, block=True)
  assert prepare_drive_config(params, CP, CP_SP) is None


@pytest.mark.parametrize('geometry', ['mvl', 'james'])
def test_authorized_geometry_latched_for_drive(params, car_factory, geometry):
  CP, CP_SP = car_factory()
  authorize(params, geometry)
  config = prepare_drive_config(params, CP, CP_SP)
  assert config is not None
  assert config.geometry == geometry
  assert get_drive_config(params, CP, CP_SP) == config
  snapshot = params.get('HondaEpsFirmwareDriveConfig')
  params.put_bool('HondaEpsFirmwareEnabled', False, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', '', block=True)
  params.put('HondaEpsFirmwareGeometry', 'james' if geometry == 'mvl' else 'mvl', block=True)
  assert get_drive_config(params, CP, CP_SP) == config
  assert prepare_drive_config(params, CP, CP_SP) == config
  assert params.get('HondaEpsFirmwareDriveConfig') == snapshot


def test_disabled_snapshot_cannot_be_hot_enabled(params, car_factory):
  CP, CP_SP = car_factory()
  assert prepare_drive_config(params, CP, CP_SP) is None
  authorize(params)
  assert prepare_drive_config(params, CP, CP_SP) is None
  assert get_drive_config(params, CP, CP_SP) is None
  params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
  assert prepare_drive_config(params, CP, CP_SP) is not None


@pytest.mark.parametrize('geometry', ['unknown', 'MVL', 'james\x00'])
def test_invalid_geometry_fails_closed(params, car_factory, geometry):
  authorize(params, geometry)
  with pytest.raises(ValueError):
    prepare_drive_config(params, *car_factory())


@pytest.mark.parametrize('flag', [ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION])
def test_drive_snapshot_cleared_at_restart_and_next_drive(params, car_factory, flag):
  authorize(params)
  assert prepare_drive_config(params, *car_factory()) is not None
  params.clear_all(flag)
  assert params.get('HondaEpsFirmwareDriveConfig') is None
  assert params.get_bool('HondaEpsFirmwareEnabled')
  assert params.get('HondaEpsFirmwareVerifiedSha256') == RWD_SHA256


@pytest.mark.parametrize('snapshot', [None, {}, [], {'enabled': True}, {'geometry': 'james'}, 'not a config', 1])
def test_missing_or_malformed_snapshot_does_not_enable(params, car_factory, snapshot):
  authorize(params)
  if snapshot is not None:
    Path(params.get_param_path('HondaEpsFirmwareDriveConfig')).write_text(json.dumps(snapshot))
  with pytest.raises(ValueError):
    get_drive_config(params, *car_factory())


def test_latched_config_revalidated_against_current_car(params, car_factory):
  authorize(params)
  CP, CP_SP = car_factory()
  assert prepare_drive_config(params, CP, CP_SP) is not None
  CP_SP.flags = int(CP_SP.flags & ~HondaFlagsSP.EPS_MODIFIED)
  with pytest.raises(ValueError):
    get_drive_config(params, CP, CP_SP)
  with pytest.raises(ValueError):
    get_drive_config(params, *car_factory(candidate=CAR.HONDA_ACCORD))


@pytest.mark.parametrize('version', [True, False, 1.0, '1', None, 0, 2])
@pytest.mark.parametrize('enabled', [False, True])
def test_snapshot_version_requires_exact_supported_integer(params, car_factory, version, enabled):
  CP, CP_SP = car_factory()
  snapshot = {'version': version, 'enabled': enabled}
  params.put('HondaEpsFirmwareDriveConfig', snapshot, block=True)
  with pytest.raises(ValueError, match='snapshot'):
    get_drive_config(params, CP, CP_SP)
  # Rejection must happen before the model runner blocks waiting for CP_SP.
  assert params.get('CarParamsSP') is None
  with pytest.raises(ValueError, match='snapshot'):
    drive_config_for_model(params, CP)


@pytest.mark.parametrize('raw', [b'{', b'null', b'', b'not-json'])
def test_corrupted_persisted_json_never_selects_fallback_or_relatches(params, car_factory, raw):
  CP, CP_SP = car_factory()
  authorize(params)
  Path(params.get_param_path('HondaEpsFirmwareDriveConfig')).write_bytes(raw)
  with pytest.raises(ValueError, match='Corrupted'):
    prepare_drive_config(params, CP, CP_SP)
  with pytest.raises(ValueError, match='Corrupted'):
    get_drive_config(params, CP, CP_SP)
  with pytest.raises(ValueError, match='Corrupted'):
    drive_config_for_model(params, CP)


def test_lost_snapshot_after_carparams_publication_cannot_relatch(params, car_factory):
  CP, CP_SP = car_factory()
  authorize(params)
  assert prepare_drive_config(params, CP, CP_SP) is not None
  params.put('CarParams', CP.to_bytes(), block=True)
  params.remove('HondaEpsFirmwareDriveConfig')
  params.put_bool('HondaEpsFirmwareEnabled', False, block=True)
  with pytest.raises(ValueError, match='snapshot'):
    prepare_drive_config(params, CP, CP_SP)
  with pytest.raises(ValueError, match='snapshot'):
    get_drive_config(params, CP, CP_SP)
  with pytest.raises(ValueError, match='snapshot'):
    drive_config_for_model(params, CP)


def test_other_vehicle_retains_fallback_without_firmware_snapshot(params, car_factory):
  CP, CP_SP = car_factory(candidate=CAR.HONDA_ACCORD)
  assert get_drive_config(params, CP, CP_SP) is None
  assert drive_config_for_model(params, CP) is None


def test_response_defaults_to_source_for_original_controller_behavior(params, car_factory):
  authorize(params)
  assert params.get('HondaEpsFirmwareResponse') is None
  assert params.get('HondaEpsFirmwareResponse', return_default=True) == 'source'
  config = prepare_drive_config(params, *car_factory())
  assert config is not None and config.response == 'source'


@pytest.mark.parametrize('response', ['source', 'route32'])
def test_explicit_response_profile_is_latched_with_model_and_controls(params, car_factory, response):
  CP, CP_SP = car_factory()
  authorize(params)
  params.put('HondaEpsFirmwareResponse', response, block=True)
  config = prepare_drive_config(params, CP, CP_SP)
  assert config is not None and config.response == response
  params.put('CarParamsSP', CP_SP.to_bytes(), block=True)
  snapshot = params.get('HondaEpsFirmwareDriveConfig')
  params.put('HondaEpsFirmwareResponse', 'source' if response == 'route32' else 'route32', block=True)
  assert get_drive_config(params, CP, CP_SP) == config
  assert drive_config_for_model(params, CP) == config
  assert prepare_drive_config(params, CP, CP_SP) == config
  assert params.get('HondaEpsFirmwareDriveConfig') == snapshot


@pytest.mark.parametrize('response', ['unknown', 'Route32', 'route_32', 'james', 'source\x00'])
def test_invalid_response_setting_fails_closed(params, car_factory, response):
  authorize(params)
  params.put('HondaEpsFirmwareResponse', response, block=True)
  with pytest.raises(ValueError):
    prepare_drive_config(params, *car_factory())
  assert params.get('HondaEpsFirmwareDriveConfig') is None


@pytest.mark.parametrize('response', [None, 'unknown', 1, [], {}])
def test_invalid_response_in_admitted_snapshot_is_rejected(params, car_factory, response):
  CP, CP_SP = car_factory()
  authorize(params)
  assert prepare_drive_config(params, CP, CP_SP) is not None
  snapshot = params.get('HondaEpsFirmwareDriveConfig')
  snapshot['config']['response'] = response
  params.put('HondaEpsFirmwareDriveConfig', snapshot, block=True)
  with pytest.raises(ValueError):
    get_drive_config(params, CP, CP_SP)


def test_admitted_snapshot_without_response_cannot_silently_change_behavior(params, car_factory):
  CP, CP_SP = car_factory()
  authorize(params)
  assert prepare_drive_config(params, CP, CP_SP) is not None
  snapshot = params.get('HondaEpsFirmwareDriveConfig')
  del snapshot['config']['response']
  params.put('HondaEpsFirmwareDriveConfig', snapshot, block=True)
  with pytest.raises(ValueError, match='Incomplete'):
    get_drive_config(params, CP, CP_SP)
