import ast
import gc
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from openpilot.cereal import custom, log, messaging
from opendbc.car.honda.interface import CarInterface
from opendbc.car.structs import car
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.car import interfaces
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import (
  FIRMWARE_SHA256, drive_config_for_model, get_drive_config, prepare_drive_config,
)
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.timing import prediction_delay


def authorize(params, geometry='mvl', response=None):
  params.put_bool('HondaEpsFirmwareEnabled', True, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', FIRMWARE_SHA256, block=True)
  params.put('HondaEpsFirmwareGeometry', geometry, block=True)
  if response is not None:
    params.put('HondaEpsFirmwareResponse', response, block=True)


@pytest.fixture
def native_controls(params, car_factory, monkeypatch):
  prefix = os.environ['OPENPILOT_PREFIX']
  queue_dir = Path('/dev/shm') / ('msgq_' + prefix)
  queue_dir.mkdir()
  controls_created = []

  def make_controls(enabled=True, geometry='mvl', response=None, **cp_fields):
    CP, CP_SP = car_factory()
    for field, value in cp_fields.items():
      setattr(CP, field, value)
    if enabled:
      authorize(params, geometry, response)
    prepare_drive_config(params, CP, CP_SP)
    params.put('CarParams', CP.to_bytes(), block=True)
    params.put('CarParamsSP', CP_SP.to_bytes(), block=True)
    controls = Controls()
    controls_created.append(controls)
    return controls

  yield make_controls
  for controls in controls_created:
    controls.sm.sock.clear()
    controls.pm.sock.clear()
  controls_created.clear()
  gc.collect()
  shutil.rmtree(queue_dir, ignore_errors=True)


def test_firmware_setup_has_priority_over_torque_and_nn_selection(params, car_factory, monkeypatch):
  CP, CP_SP = car_factory(native_sp=False)
  CI = CarInterface(CP, CP_SP)
  authorize(params)
  params.put_bool('EnforceTorqueControl', True, block=True)
  params.put_bool('NeuralNetworkLateralControl', True, block=True)
  monkeypatch.setattr(interfaces, 'get_nn_model_path', lambda cp: ('', 'MOCK', False))
  monkeypatch.setattr(interfaces.STATSLOGSP, 'raw', Mock())
  interfaces.setup_interfaces(CI, params)
  assert CP.lateralTuning.which() == 'pid'
  assert get_drive_config(params, CP, CP_SP) is not None


def test_disabled_setup_retains_existing_enforce_torque_behavior(params, car_factory, monkeypatch):
  CP, CP_SP = car_factory(native_sp=False)
  CI = CarInterface(CP, CP_SP)
  params.put_bool('EnforceTorqueControl', True, block=True)
  monkeypatch.setattr(interfaces, 'get_nn_model_path', lambda cp: ('', 'MOCK', False))
  monkeypatch.setattr(interfaces.STATSLOGSP, 'raw', Mock())
  interfaces.setup_interfaces(CI, params)
  assert CP.lateralTuning.which() == 'torque'
  assert get_drive_config(params, CP, CP_SP) is None


@pytest.mark.parametrize('geometry', ['mvl', 'james'])
def test_real_controlsd_constructor_selects_latched_geometry(native_controls, params, geometry):
  controls = native_controls(geometry=geometry)
  assert isinstance(controls.LaC, LatControlFirmware)
  assert controls.LaC.geometry_mode == geometry
  assert controls.CP.lateralTuning.which() == 'pid'
  selected = controls.LaC
  params.put_bool('EnforceTorqueControl', True, block=True)
  params.put_bool('HondaEpsFirmwareEnabled', False, block=True)
  params.put('HondaEpsFirmwareGeometry', 'james' if geometry == 'mvl' else 'mvl', block=True)
  controls.get_params_sp(controls.sm)
  assert controls.LaC is selected
  assert selected.geometry_mode == geometry


def test_real_controlsd_constructor_retains_pid_fallback(native_controls):
  controls = native_controls(enabled=False)
  assert isinstance(controls.LaC, LatControlPID)


@pytest.mark.parametrize('response', [None, 'source', 'route32'])
def test_real_controlsd_constructor_selects_latched_response(native_controls, params, response):
  controls = native_controls(response=response)
  selected = controls.LaC
  expected = 'source' if response is None else response
  assert isinstance(selected, LatControlFirmware)
  assert selected.response_profile.name == expected
  params.put('HondaEpsFirmwareResponse', 'source' if expected == 'route32' else 'route32', block=True)
  controls.get_params_sp(controls.sm)
  assert controls.LaC is selected
  assert selected.response_profile.name == expected
  config = get_drive_config(params, controls.CP, controls.CP_SP)
  assert config is not None and config.response == expected


def test_admitted_constructor_failure_prevents_controls_startup(native_controls, monkeypatch):
  def invalid_calibration():
    raise ValueError('corrupted calibration')
  monkeypatch.setattr('openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw.validate_calibration', invalid_calibration)
  with pytest.raises(ValueError, match='corrupted calibration'):
    native_controls()


def test_model_runner_reads_same_native_drive_snapshot(params, car_factory):
  CP, CP_SP = car_factory()
  authorize(params, 'james')
  expected = prepare_drive_config(params, CP, CP_SP)
  params.put('CarParamsSP', CP_SP.to_bytes(), block=True)
  assert drive_config_for_model(params, CP) == expected
  params.put_bool('HondaEpsFirmwareEnabled', False, block=True)
  params.put('HondaEpsFirmwareGeometry', 'mvl', block=True)
  assert drive_config_for_model(params, CP) == expected


def test_disabled_model_runner_does_not_block_for_car_params_sp(params, car_factory):
  CP, CP_SP = car_factory()
  assert prepare_drive_config(params, CP, CP_SP) is None
  assert params.get('CarParamsSP') is None
  assert drive_config_for_model(params, CP) is None


@pytest.mark.parametrize('model_path', ['openpilot/selfdrive/modeld/modeld.py', 'openpilot/sunnypilot/modeld_v2/modeld.py'])
@pytest.mark.parametrize('speed', [3.5, 7.0, 12.0, 20.0, 30.0])
def test_both_model_runners_execute_prediction_schedule_once(model_path, speed):
  # Execute the deployed timing branch without constructing an inference engine
  # or requiring model weights, USB cameras, or a GPU.
  tree = ast.parse((Path(BASEDIR) / model_path).read_text())
  branches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and
              ast.unparse(node.test) == 'firmware_config is not None']
  assert len(branches) == 1
  delay = Mock(wraps=prediction_delay)
  model = SimpleNamespace(lat_delay=0.75)
  namespace = {'firmware_config': object(), 'model': model, 'v_ego': speed,
               'prediction_delay': delay, 'LAT_SMOOTH_SECONDS': 0.0}
  exec(compile(ast.Module(body=[branches[0]], type_ignores=[]), model_path, 'exec'), namespace)
  delay.assert_called_once_with(speed)
  assert model.lat_delay == prediction_delay(speed)


def seed_control_state(controls):
  CS = car.CarState.new_message(vEgo=20.0, canValid=True, steeringAngleDeg=0.0, steeringRateDeg=0.0)
  lp = log.VehicleParameters.new_message(steerRatio=controls.CP.steerRatio, stiffnessFactor=1.0)
  state = log.SelfdriveState.new_message(enabled=True, active=True, state='enabled')
  model = log.ModelDataV2.new_message()
  model.action.desiredCurvature = 0.001
  plan = log.LongitudinalPlan.new_message(aTarget=-0.5, shouldStop=False)
  for service, data in [('carState', CS), ('vehicleParameters', lp), ('selfdriveState', state),
                        ('modelV2', model), ('longitudinalPlan', plan)]:
    controls.sm.data[service] = data.as_reader()
  controls.sm.valid['modelV2'] = True


@pytest.mark.parametrize('geometry', ['mvl', 'james'])
def test_native_controlsd_update_and_log_publish(native_controls, geometry):
  controls = native_controls(geometry=geometry, openpilotLongitudinalControl=True)
  seed_control_state(controls)
  receiver = messaging.sub_sock('controlsState', timeout=1000)
  for _ in range(40):
    command, pid_log = controls.state_control()
  assert command.latActive
  assert command.longActive
  assert abs(command.actuators.torque) <= 3840 / 4096
  assert pid_log.active
  assert pid_log.commandDelay == pytest.approx(0.025)
  assert pid_log.epsFfActive
  assert command.actuators.accel < 0.0
  controls.publish(command, pid_log)
  event = messaging.recv_one(receiver)
  assert event is not None
  assert event.controlsState.lateralControlState.which() == 'pidState'
  published = event.controlsState.lateralControlState.pidState
  assert published.output == pytest.approx(command.actuators.torque)
  assert published.epsFfFeedforward == pytest.approx(pid_log.epsFfFeedforward)
  assert published.commandDelay == pytest.approx(0.025)


@pytest.mark.parametrize('geometry', ['mvl', 'james'])
def test_native_controlsd_override_cuts_request_and_retains_recovery_history(native_controls, geometry):
  controls = native_controls(geometry=geometry, openpilotLongitudinalControl=True)
  seed_control_state(controls)
  for _ in range(200):
    controls.state_control()
  state = controls.sm['carState'].as_builder()
  state.steeringPressed = True
  state.steeringTorque = -1.0 if controls.LaC.core.output >= 0.0 else 1.0
  controls.sm.data['carState'] = state.as_reader()
  for _ in range(5):
    command, _ = controls.state_control()
    assert not command.latActive
    assert command.actuators.torque == 0.0
    assert command.longActive and command.actuators.accel < 0.0
  state.steeringPressed = False
  state.steeringTorque = 0.0
  controls.sm.data['carState'] = state.as_reader()
  command, _ = controls.state_control()
  assert command.latActive and command.longActive
  assert controls.LaC.delivery.ramp == pytest.approx(1. / 150.)
  assert command.actuators.torque == pytest.approx(controls.LaC.core.output * .9375 / 150.)


@pytest.mark.parametrize('difference, limited', [(0.009, False), (0.0095, True), (0.011, True)])
def test_firmware_delivery_limit_threshold_preserves_source_e4_units(native_controls, difference, limited):
  controls = native_controls()
  seed_control_state(controls)
  command, logged = controls.state_control()
  output = car.CarOutput.new_message()
  output.actuatorsOutput.torque = command.actuators.torque - difference
  controls.sm.data['carOutput'] = output.as_reader()
  controls.publish(command, logged)
  assert controls.steer_limited_by_safety is limited
  assert controls.LaC.output_scale * .01 * 4096 == .01 * 3840


def test_process_replay_publishes_disabled_insight_snapshot_before_carparams(params, monkeypatch):
  path = Path(BASEDIR) / 'openpilot/selfdrive/test/process_replay/process_replay.py'
  tree = ast.parse(path.read_text())
  callback = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_car_params_callback')
  original_put = params.put
  def ordered_put(key, value, **kwargs):
    if key in ('CarParams', 'CarParamsSP'):
      assert params.get('HondaEpsFirmwareDriveConfig') == {'version': 1, 'enabled': False}
    return original_put(key, value, **kwargs)
  monkeypatch.setattr(params, 'put', ordered_put)
  namespace: dict[str, Any] = {'Params': lambda: params, 'interfaces': {'HONDA_INSIGHT': CarInterface},
               'prepare_drive_config': prepare_drive_config, 'convert_to_capnp': convert_to_capnp}
  exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), 'exec'), namespace)
  namespace['get_car_params_callback'](None, None, [], 'HONDA_INSIGHT')
  CP = messaging.log_from_bytes(params.get('CarParams'), car.CarParams)
  CP_SP = messaging.log_from_bytes(params.get('CarParamsSP'), custom.CarParamsSP)
  assert get_drive_config(params, CP, CP_SP) is None
  assert drive_config_for_model(params, CP) is None


def test_pid_schema_keeps_existing_ordinals_and_serializes_diagnostics():
  old_fields = ['active', 'steeringAngleDeg', 'steeringRateDeg', 'angleError', 'p', 'i', 'f',
                'output', 'saturated', 'steeringAngleDesiredDeg']
  added = ['epsFfActive', 'epsFfFeedforward', 'epsFfR5', 'epsFfLoad', 'epsFfDesiredRate', 'epsFfWeight', 'commandDelay']
  schema = log.ControlsState.LateralPIDState.schema
  assert [schema.fields[name].proto.ordinal.explicit for name in old_fields + added] == list(range(17))
  event = messaging.new_message('controlsState')
  state = event.controlsState.lateralControlState.init('pidState')
  state.active, state.epsFfActive = True, True
  state.epsFfFeedforward, state.epsFfR5 = 0.3, 1500.0
  state.epsFfLoad, state.epsFfDesiredRate = 200.0, 4.5
  state.epsFfWeight, state.commandDelay = 0.9, 0.175
  with log.Event.from_bytes(event.to_bytes()) as decoded:
    pid = decoded.controlsState.lateralControlState.pidState
    assert pid.active and pid.epsFfActive
    assert pid.epsFfFeedforward == pytest.approx(0.3)
    assert pid.epsFfR5 == 1500.0
    assert pid.epsFfLoad == 200.0
    assert pid.epsFfDesiredRate == 4.5
    assert pid.epsFfWeight == pytest.approx(0.9)
    assert pid.commandDelay == pytest.approx(0.175)
  assert custom.CarParamsSP.schema.fields['flags'].proto.ordinal.explicit == 0
