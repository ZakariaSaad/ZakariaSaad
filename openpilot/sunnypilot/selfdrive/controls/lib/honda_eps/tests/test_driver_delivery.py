"""Driver delivery parity through the source actuator methods and MVL CAN path."""
from types import SimpleNamespace
from unittest.mock import Mock

import math
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import DBC
from opendbc.car.vehicle_model import VehicleModel
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.driver_override import FirmwareDriverDelivery
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.tests.fixtures.james_driver_delivery import JamesDriverDelivery


def source_delivery():
  source = JamesDriverDelivery()
  source.override_ramp = 0.0
  source.steering_pressed_filter_s = 0.0
  source.steering_pressed_robust_prev = False
  source.lat_active_prev = False
  source.last_torque = 0.0
  return source


@pytest.mark.parametrize('fade_down_s', [0., .1])
@pytest.mark.parametrize('pattern', ['opposing', 'same_direction', 'threshold_chatter', 'release_repress', 'short_press'])
def test_delivery_matches_unmodified_james_actuator_methods(pattern, fade_down_s):
  port = FirmwareDriverDelivery(.01, fade_down_s=fade_down_s)
  source = source_delivery()
  live = {'min_steer_speed': 0., 'override_fade_down_s': fade_down_s, 'override_fade_up_s': 1.5,
          'override_torque_scale': 0., 'driver_assist_during_override': False,
          'steer_delta_limiter_enabled': False, 'steer_delta_down': 3., 'steer_delta_up': 3.}
  for frame in range(600):
    active = 10 <= frame < 550
    command = .6 + .05 * math.sin(frame / 25.)
    pressed = 200 <= frame < 260
    torque = .5
    if pattern in ('opposing', 'release_repress', 'short_press'):
      torque = -.5
    if pattern == 'threshold_chatter' and pressed:
      command = .099 if frame % 2 == 0 else .101
    elif pattern == 'release_repress':
      pressed = 200 <= frame < 220 or 221 <= frame < 240 or 250 <= frame < 260
    elif pattern == 'short_press':
      pressed = 200 <= frame < 203 or 205 <= frame < 207 or frame == 350
    cc = SimpleNamespace(latActive=active, actuators=SimpleNamespace(torque=command))
    cs = SimpleNamespace(out=SimpleNamespace(vEgo=15., steeringPressed=pressed, steeringTorque=torque))
    expected, lkas_active, detected = source._update_steering_torque(cc, cs, live)
    if active:
      actual = port.update(command, pressed, torque)
    else:
      port.reset()
      actual = 0.
    assert actual == expected
    assert port.override_active == detected
    assert lkas_active == (active and not port.override_active)
    assert port.ramp == source.override_ramp
    assert port.limited == (abs(command - actual) > .01 if active else False)


def tick(controller, vm, *, pressed=False, torque=0.):
  state = SimpleNamespace(vEgo=15., steeringAngleDeg=0., steeringRateDeg=0., steeringTorque=torque,
                          steeringPressed=pressed)
  return controller.update(True, state, vm, SimpleNamespace(roll=0., angleOffsetDeg=0.),
                           False, -.002, None, False, .2)


def test_wrapper_override_freezes_then_bleeds_integral_without_external_limiting(car_factory):
  cp, sp = car_factory()
  controller = LatControlFirmware(cp, sp, None, .01)
  vm = VehicleModel(cp)
  for _ in range(200):
    tick(controller, vm)
  controller.core.pid.i = .18
  output, _, _ = tick(controller, vm, pressed=True, torque=-1.)
  assert controller.driver_override_active
  assert controller.delivery.limited
  assert output == 0.
  assert controller.core.pid.i == .18
  output, _, _ = tick(controller, vm)
  assert not controller.driver_override_active
  assert controller.delivery.ramp == 1. / 150.
  assert controller.core.pid.i == pytest.approx(.18 * math.exp(-.01 / .5))
  assert output == pytest.approx(controller.core.output * controller.output_scale / 150.)
  controller.reset()
  assert not controller.driver_override_active
  assert not controller.delivery.limited
  assert controller.delivery.ramp == 0.


@pytest.mark.parametrize('openpilot_longitudinal', [False, True])
@pytest.mark.parametrize('response', ['source', 'route32'])
def test_actual_mvl_carcontroller_clears_e4_command_and_request_on_override(params, car_factory, openpilot_longitudinal, response):
  cp, sp = car_factory(native_sp=False)
  cp.openpilotLongitudinalControl = openpilot_longitudinal
  ci = CarInterface(cp, sp)
  cs, _ = ci.update([])
  cs.vEgo = 15.
  ci.CS.steer_control_active = True
  ci.CC.param_writer.put_many = Mock()
  controller = LatControlFirmware(cp, sp, ci, .01, response_profile=response)
  vm = VehicleModel(cp)
  parser = CANParser(DBC[cp.carFingerprint][Bus.pt], [('STEERING_CONTROL', 100)], ci.CC.CAN.lkas)
  for frame in range(235):
    pressed = 200 <= frame < 220
    cs.steeringPressed = pressed
    cs.steeringTorque = -1. if pressed else 0.
    output, _, _ = tick(controller, vm, pressed=pressed, torque=cs.steeringTorque)
    cc = structs.CarControl.new_message(latActive=not controller.driver_override_active)
    cc.actuators.torque = output
    delivered, messages = ci.apply(cc.as_reader(), structs.CarControlSP(), frame * 10_000_000)
    assert any(message[0] == 0xE4 and message[2] == (1 if openpilot_longitudinal else 0) for message in messages)
    parser.update([(frame * 10_000_000, messages)])
    e4 = parser.vl['STEERING_CONTROL']
    if pressed:
      assert controller.driver_override_active
      assert e4['STEER_TORQUE'] == e4['STEER_TORQUE_REQUEST'] == 0.
      if response == 'source':
        assert output == 0.
      elif frame == 200:
        # The recorded profile retains its internal fade-down state, while
        # the actual torque request is disabled immediately in both profiles.
        assert output > 0. and controller.delivery.ramp == pytest.approx(.9)
    else:
      assert e4['STEER_TORQUE_REQUEST'] == 1.
      assert abs(e4['STEER_TORQUE'] + float(delivered.torque) * 4096.) < 1.
    assert abs(e4['STEER_TORQUE']) <= 3840.


@pytest.mark.parametrize('fade_down_s', [-1., 10.1, math.nan, math.inf, -math.inf])
def test_invalid_override_fade_down_rejected(fade_down_s):
  with pytest.raises(ValueError, match='fade-down'):
    FirmwareDriverDelivery(.01, fade_down_s=fade_down_s)
