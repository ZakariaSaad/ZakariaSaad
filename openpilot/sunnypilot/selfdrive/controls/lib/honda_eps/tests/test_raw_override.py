"""The firmware controller preserves MVL's existing Insight raw-torque threshold.

Feed packed CAN through the actual CarInterface/CarState parser, then through
firmware control and the unchanged CarController. Other missing vehicle messages
are not synthesized; these component tests do not establish full CAN validity.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import DBC, STEER_THRESHOLD
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware


@pytest.mark.parametrize('response', ['source', 'route32'])
@pytest.mark.parametrize('direction', [-1, 1])
def test_packed_insight_torque_preserves_1200_boundary_and_cuts_opposing_request(params, car_factory, response, direction):
  cp, sp = car_factory(native_sp=False)
  original_cp = cp.to_dict()
  original_sp = convert_to_capnp(sp).to_dict()
  original_thresholds = STEER_THRESHOLD.copy()
  assert STEER_THRESHOLD.get(cp.carFingerprint, 1200) == 1200
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
    # Desired curvature opposes the requested sensor-torque sign. Pass the
    # parsed torque and press boolean unchanged to both robust detectors.
    torque, _, _ = controller.update(True, state, vm, SimpleNamespace(roll=0., angleOffsetDeg=0.),
                                     False, direction * .002, None, False, .2)
    command = structs.CarControl.new_message(latActive=not controller.driver_override_active)
    command.actuators.torque = torque
    _, can_messages = ci.apply(command.as_reader(), structs.CarControlSP(), now)
    output_parser.update([(now, can_messages)])
    return state, output_parser.vl['STEERING_CONTROL']

  for _ in range(200):
    state, e4 = tick(0.)
  assert state.vEgo == pytest.approx(15.)
  assert not state.steeringPressed and not controller.driver_override_active
  assert e4['STEER_TORQUE_REQUEST'] == 1.
  assert controller.core.output * direction < 0.

  # Equality stays below the native strict-greater-than threshold. In
  # particular, 1201..2400 must already override in this MVL port.
  for magnitude in (1199, 1200, 1201, 2000, 2400, 2401, 3000, 1200, 0):
    state, e4 = tick(direction * magnitude)
    pressed = magnitude > 1200
    assert state.steeringTorque == direction * magnitude
    assert bool(state.steeringPressed) == pressed
    assert controller.driver_override_active == pressed
    assert bool(e4['STEER_TORQUE_REQUEST']) == (not pressed)
    if pressed:
      assert e4['STEER_TORQUE'] == 0.

  assert cp.to_dict() == original_cp
  assert convert_to_capnp(sp).to_dict() == original_sp
  assert STEER_THRESHOLD == original_thresholds
