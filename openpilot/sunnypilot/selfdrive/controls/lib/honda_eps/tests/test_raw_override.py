"""Firmware drives use James's raw threshold; native MVL fallback is unchanged.

Feed packed CAN through the actual CarInterface/CarState parser, then through
firmware control and the unchanged CarController. Other missing vehicle messages
are not synthesized; these component tests do not establish full CAN validity.
"""
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, Mock

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, STEER_THRESHOLD
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.car import card as card_module
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.pandad import can_list_to_can_capnp
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.driver_override import apply_firmware_driver_threshold
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import FIRMWARE_SHA256, prepare_drive_config


def configure(params, cp, sp, response, enabled=True):
  params.put_bool('HondaEpsFirmwareEnabled', enabled, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', FIRMWARE_SHA256, block=True)
  params.put('HondaEpsFirmwareResponse', response, block=True)
  return prepare_drive_config(params, cp, sp)


@pytest.mark.parametrize('response', ['source', 'route32'])
@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('enabled', [False, True])
def test_packed_insight_scoped_threshold_cuts_opposing_request(params, car_factory, response, direction, enabled):
  cp, sp = car_factory(native_sp=False)
  original_cp = cp.to_dict()
  original_sp = convert_to_capnp(sp).to_dict()
  original_thresholds = STEER_THRESHOLD.copy()
  assert STEER_THRESHOLD.get(cp.carFingerprint, 1200) == 1200
  config = configure(params, cp, sp, response, enabled)
  assert (config is not None) == enabled
  ci = CarInterface(cp, sp)
  ci.CC.param_writer.put_many = Mock()
  controller = LatControlFirmware(cp, sp, ci, .01, response_profile=response)
  vm = VehicleModel(cp)
  packer = CANPacker(DBC[cp.carFingerprint][Bus.pt])
  sensor_bus = ci.can_parsers[Bus.pt].bus
  output_parser = CANParser(DBC[cp.carFingerprint][Bus.pt], [('STEERING_CONTROL', 100)], ci.CC.CAN.lkas)
  frame = 0

  def tick(raw_torque):
    nonlocal frame
    frame += 1
    now = frame * 10_000_000
    messages = [
      packer.make_can_msg('STEER_STATUS', sensor_bus,
                         {'STEER_STATUS': 0, 'STEER_CONTROL_ACTIVE': 1, 'STEER_TORQUE_SENSOR': raw_torque}),
      packer.make_can_msg('ENGINE_DATA', sensor_bus, {'XMISSION_SPEED': 54.}),
      packer.make_can_msg('WHEEL_SPEEDS', sensor_bus,
                         {f'WHEEL_SPEED_{wheel}': 54. for wheel in ('FL', 'FR', 'RL', 'RR')}),
    ]
    state, _ = ci.update([(now, messages)])
    native_pressed = bool(state.steeringPressed)
    apply_firmware_driver_threshold(state, config)
    assert state is ci.CS.out
    # Desired curvature opposes the requested sensor-torque sign. Pass the
    # published torque and press boolean to both robust detectors.
    torque, _, _ = controller.update(True, state, vm, SimpleNamespace(roll=0., angleOffsetDeg=0.),
                                     False, direction * .002, None, False, .2)
    command = structs.CarControl.new_message(latActive=not controller.driver_override_active)
    command.actuators.torque = torque
    _, can_messages = ci.apply(command.as_reader(), structs.CarControlSP(), now)
    output_parser.update([(now, can_messages)])
    return state, output_parser.vl['STEERING_CONTROL'], native_pressed

  for _ in range(200):
    state, e4, _ = tick(0.)
  assert state.vEgo == pytest.approx(15.)
  assert not state.steeringPressed and not controller.driver_override_active
  assert e4['STEER_TORQUE_REQUEST'] == 1.
  assert controller.core.output * direction < 0.

  # The parser's native threshold stays 1200. Only an admitted firmware drive
  # publishes the requested James threshold, retaining strict > at 2400.
  for magnitude in (1199, 1200, 1201, 2000, 2399, 2400, 2401, 3000, 2400, 1200, 0):
    state, e4, native_pressed = tick(direction * magnitude)
    assert native_pressed == (magnitude > 1200)
    pressed = magnitude > (2400 if enabled else 1200)
    assert state.steeringTorque == direction * magnitude
    assert bool(state.steeringPressed) == pressed
    assert controller.driver_override_active == pressed
    assert bool(e4['STEER_TORQUE_REQUEST']) == (not pressed)
    if pressed:
      assert e4['STEER_TORQUE'] == 0.

  assert cp.to_dict() == original_cp
  assert convert_to_capnp(sp).to_dict() == original_sp
  assert STEER_THRESHOLD == original_thresholds


@pytest.mark.parametrize('candidate,firmware', [(CAR.HONDA_INSIGHT, b'39990-TXM-A040'),
                                                (CAR.HONDA_INSIGHT, b'39990-TXM,A030'),
                                                (CAR.HONDA_CIVIC_BOSCH, b'39990-TXM,A040')])
def test_unsupported_firmware_or_vehicle_keeps_native_threshold(params, car_factory, candidate, firmware):
  cp, sp = car_factory(candidate=candidate, versions=(firmware,), native_sp=False)
  config = configure(params, cp, sp, 'source')
  assert config is None
  ci = CarInterface(cp, sp)
  packer = CANPacker(DBC[cp.carFingerprint][Bus.pt])
  for frame, raw_torque in enumerate((0, 1200, 1201), start=1):
    message = packer.make_can_msg('STEER_STATUS', ci.can_parsers[Bus.pt].bus,
                                 {'STEER_STATUS': 0, 'STEER_CONTROL_ACTIVE': 1, 'STEER_TORQUE_SENSOR': raw_torque})
    state, _ = ci.update([(frame * 10_000_000, [message])])
    apply_firmware_driver_threshold(state, config)
    assert bool(state.steeringPressed) == (raw_torque > 1200)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('separate_output', [False, True])
def test_real_card_state_update_publishes_latched_threshold_and_updates_actuator_state(
    params, car_factory, monkeypatch, enabled, separate_output):
  cp, sp = car_factory(native_sp=False)
  ci = CarInterface(cp, sp)
  config = configure(params, cp, sp, 'source', enabled)
  packer = CANPacker(DBC[cp.carFingerprint][Bus.pt])
  # Run the deployed card method with native CAN parsing and mock socket/clock
  # plumbing. CI normally aliases its output; also cover a copied CS record.
  if separate_output:
    native_update = ci.update

    def copied_update(can_list):
      state, state_sp = native_update(can_list)
      return state.as_reader().as_builder(), state_sp

    monkeypatch.setattr(ci, 'update', copied_update)
  instance = object.__new__(card_module.Car)
  instance.CI, instance.CP, instance.CP_SP = ci, cp, sp
  instance.firmware_config = config
  instance.can_sock = object()
  instance.RI = Mock()
  instance.RI.update.return_value = None
  instance.sm = MagicMock()
  instance.sm.frame = 1
  instance.sm.all_checks.return_value = False
  instance.sm.__getitem__.side_effect = {'longitudinalPlanSP': object(), 'carControl': SimpleNamespace(enabled=False)}.__getitem__
  instance.v_cruise_helper = Mock(v_cruise_kph=80., v_cruise_cluster_kph=80.)
  instance.is_metric = True
  instance.CC_prev = SimpleNamespace(enabled=False)
  instance.can_rcv_cum_timeout_counter = 0
  instance.pm = Mock()
  instance.last_actuators_output = structs.CarControl.Actuators()
  instance.rk = cast(card_module.Ratekeeper, SimpleNamespace(remaining=0.))
  monkeypatch.setattr(card_module, 'REPLAY', False)
  drain = Mock()
  monkeypatch.setattr(card_module.messaging, 'drain_sock_raw', drain)

  for raw_torque in (0, 1200, 1201, 2400, 2401, -2401, -2400):
    message = packer.make_can_msg('STEER_STATUS', ci.can_parsers[Bus.pt].bus,
                                 {'STEER_STATUS': 0, 'STEER_CONTROL_ACTIVE': 1, 'STEER_TORQUE_SENSOR': raw_torque})
    drain.return_value = [can_list_to_can_capnp([message])]
    state, state_sp, radar = instance.state_update()
    pressed = abs(raw_torque) > (2400 if enabled else 1200)
    assert bool(state.steeringPressed) == pressed
    assert bool(ci.CS.out.steeringPressed) == pressed
    instance.state_publish(state, state_sp, radar)
    published = next(call.args[1].carState for call in reversed(instance.pm.send.call_args_list) if call.args[0] == 'carState')
    assert bool(published.steeringPressed) == pressed
    assert published.steeringTorque == raw_torque
