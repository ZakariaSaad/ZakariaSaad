"""Regression for Bosch inactive/override observations below the set speed.

The production Honda controller already depends on openpilot Params; skip in
standalone opendbc installs without that integration, never mock the controller.
"""
from unittest.mock import Mock

import pytest

pytest.importorskip('openpilot.common.params')

from opendbc.car import structs, gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR


@pytest.mark.parametrize('long_state,gas_pressed,brake_pressed,speed,error', [
  ('off', False, False, 15., 0.),
  ('stopping', False, False, 15., 0.),
  ('pid', True, False, 15., 1.),
  ('pid', False, True, 15., 1.),
  ('pid', False, False, 0., 1.),
  ('pid', False, False, 15., 0.),
  ('pid', False, False, 15., .5),
])
def test_bosch_wind_learning_requires_valid_observation(tmp_path, monkeypatch, long_state, gas_pressed, brake_pressed, speed, error):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  fp = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.HONDA_INSIGHT, fp, [], False, False, False)
  sp = CarInterface.get_params_sp(cp, CAR.HONDA_INSIGHT, fp, [], False, False, False)
  cp.openpilotLongitudinalControl = True
  ci = CarInterface(cp, sp)
  ci.CC.param_writer.put_many = Mock()
  state, _ = ci.update([])
  state.vEgo, state.aEgo = speed, -error
  state.cruiseState.speed = 30.
  state.gasPressed, state.brakePressed = gas_pressed, brake_pressed
  command = structs.CarControl.new_message(longActive=long_state == 'pid')
  command.actuators.longControlState = long_state
  before = ci.CC.windfactor
  # Constructing CAN messages does not transmit them.
  ci.apply(command.as_reader(), structs.CarControlSP(), 0)
  eligible = long_state == 'pid' and not gas_pressed and not brake_pressed and speed > 0 and error != 0
  if eligible:
    assert ci.CC.windfactor > before
  else:
    assert ci.CC.windfactor == before
