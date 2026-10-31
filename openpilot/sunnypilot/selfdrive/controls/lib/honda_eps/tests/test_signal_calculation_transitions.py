"""Synthetic model -> real controlsd/EPS/transport transitions; no CAN is sent."""
from unittest.mock import Mock

import numpy as np
import pytest

from opendbc.car.honda.values import CarControllerParams
from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_JERK
from openpilot.selfdrive.modeld.modeld import get_action_from_model
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import ANGLE_RATE_LIMIT
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.tests import test_integration
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.timing import command_delay
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.transport import FirmwareTransport

native_controls = test_integration.native_controls


@pytest.mark.parametrize('geometry', ['mvl', 'james'])
@pytest.mark.parametrize('speed', [.5, 1., 2., 3.])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_signal_transitions_keep_existing_curvature_angle_and_transport_limits(native_controls, monkeypatch, geometry, speed, sign):
  controls = native_controls(geometry=geometry)
  test_integration.seed_control_state(controls)
  controls.sm.valid['lateralManeuverPlan'] = False
  controls.lane_centering_enabled = False
  controller = controls.LaC
  reset = Mock(wraps=controller.reset)
  monkeypatch.setattr(controller, 'reset', reset)
  transport = FirmwareTransport(CarControllerParams(controls.CP))
  assert transport.delta_up == transport.delta_down == .03
  previous = log.ModelDataV2.Action(desiredCurvature=sign * .008, desiredAcceleration=-.7)
  output = {'action': np.array([[sign * .3, -.2, 0., 0.]], np.float32)}
  wire = 0.
  max_delta = MAX_LATERAL_JERK * DT_CTRL / max(speed, 1.)**2
  targets = []

  # Left activation, hazards, cancellation, right activation and cancellation.
  sequence = [(False, False)] * 3 + [(True, False)] * 6 + [(True, True)] * 3
  sequence += [(False, False)] * 6 + [(False, True)] * 3 + [(False, False)] * 6
  for left, right in sequence:
    state = controls.sm['carState'].as_builder()
    state.vEgo, state.standstill = speed, False
    state.leftBlinker, state.rightBlinker = left, right
    controls.sm.data['carState'] = state.as_reader()
    action = get_action_from_model(output, previous, .175, .575, speed, turn_signal_active=left or right)
    targets.append(action.desiredCurvature)
    model = controls.sm['modelV2'].as_builder()
    model.action = action
    controls.sm.data['modelV2'] = model.as_reader()
    previous = action

    # Existing 20 Hz model target is consumed by the 100 Hz controls loop.
    for _ in range(5):
      previous_curvature, previous_angle = controls.desired_curvature, controller.prev_angle
      command, pid_log = controls.state_control()
      assert command.latActive and pid_log.active
      assert controls.LaC is controller
      assert abs(command.actuators.curvature - previous_curvature) <= max_delta + 1e-8
      assert abs(command.actuators.curvature) <= MAX_CURVATURE + 1e-8
      assert abs(controller.prev_angle - previous_angle) <= ANGLE_RATE_LIMIT * DT_CTRL + 1e-8
      assert pid_log.commandDelay == pytest.approx(command_delay(speed))
      assert abs(command.actuators.torque) <= 3840 / 4096
      limited = transport.limit(command.actuators.torque, wire, speed)
      assert abs(limited - wire) <= .03 + 1e-8
      assert abs(int(-limited * 4096)) <= 3840
      wire = limited

  reset.assert_not_called()
  if speed == .5:
    # Cancellation restores the hold rule, so the last turn target persists.
    assert targets[0] != targets[3]
    assert targets[11] == targets[12] == targets[-1]
  elif speed < 2.5:
    assert abs(targets[3]) > abs(targets[0])
    assert targets[12] == targets[0]
  else:
    assert len(set(targets)) == 1
