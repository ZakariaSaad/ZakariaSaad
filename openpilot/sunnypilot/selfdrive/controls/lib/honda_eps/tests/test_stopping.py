"""Stopping uses the real Honda speed filter, including its zero undershoot."""
import math

import pytest

from opendbc.car.honda.carstate import CarState
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.geometry import curvature_from_angle, desired_angle_no_offset
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware


@pytest.mark.parametrize('mode', ['mvl', 'james'])
def test_honda_speed_filter_undershoot_keeps_stopped_controller_inactive(car_factory, mode):
  CP, CP_SP = car_factory()
  honda_state = CarState(CP, CP_SP)
  model = VehicleModel(CP)
  controller = LatControlFirmware(CP, CP_SP, None, 0.01, geometry_mode=mode)
  state = car.CarState.new_message(standstill=True)
  vehicle_parameters = log.VehicleParameters.new_message()

  # Let the normal filter settle at 1 m/s, then stop at 1 m/s^2 and remain
  # stopped. Raw wheel speed never goes negative; filtered speed can do so.
  for _ in range(200):
    honda_state.update_speed_kf(1.0)
  filtered_speeds = []
  for frame in range(300):
    raw_speed = max(1.0 - frame / 100.0, 0.0)
    state.vEgo, state.aEgo = honda_state.update_speed_kf(raw_speed)
    filtered_speeds.append(state.vEgo)
    angle = desired_angle_no_offset(model, state.vEgo, 0.0, 0.001, mode)
    curvature = curvature_from_angle(model, state.vEgo, 0.0, angle, mode)
    assert math.isfinite(angle) and math.isfinite(curvature)
    if state.vEgo < 0.0:
      assert angle == desired_angle_no_offset(model, 0.0, 0.0, 0.001, mode)
      assert curvature == curvature_from_angle(model, 0.0, 0.0, angle, mode)
    output, desired_angle, logged = controller.update(
      False, state, model, vehicle_parameters, False, 0.001, None, False, 0.2,
    )
    assert output == logged.output == 0.0
    assert math.isfinite(desired_angle)
    assert not logged.active and not logged.epsFfActive
  assert min(filtered_speeds) < -0.001
  assert abs(filtered_speeds[-1]) < 1e-6
