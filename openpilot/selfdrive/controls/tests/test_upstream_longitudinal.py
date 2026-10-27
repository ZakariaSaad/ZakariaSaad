"""Exercise the upstream planner with native Bigv2 publication and legacy settings.

Synthetic output tensors only: no model artifact, GPU, vehicle process or CAN.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.accel_boost import ACCEL_BOOST_MAX, ACCEL_BOOST_PER_OVERRIDE
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, LongitudinalPlanSource
from openpilot.sunnypilot.modeld_v2.constants import Meta, ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.fill_model_msg import PublishState, fill_model_msg
from openpilot.sunnypilot.modeld_v2.modeld import NavigationModelState
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser


class Messages(dict):
  def __init__(self, model, *, speed=20., enabled=True):
    services = ('radarState', 'controlsState', 'carState', 'carControl', 'selfdriveState',
                'carStateSP', 'liveMapDataSP', 'gpsLocationExternal', 'gpsLocation')
    super().__init__((name, getattr(messaging.new_message(name), name)) for name in services)
    self['modelV2'] = model
    self['carState'].vEgo = speed
    self['carState'].vCruise = self['carState'].vCruiseCluster = 100.
    self['carState'].standstill = speed < .01
    self['selfdriveState'].enabled = enabled
    self['carControl'].enabled = enabled
    self['controlsState'].longControlState = 'pid' if enabled else 'off'
    self.valid = dict.fromkeys(self, True)
    self.updated = dict.fromkeys(self, True)
    self.logMonoTime = dict.fromkeys(self, 0)
    self.recv_frame = dict.fromkeys(self, 1)

  def all_checks(self, service_list=None):
    return True


def published_bigv2(accel, *, speed=20.):
  # Same pinned JSON output contract used by native model-adapter regression tests.
  fixtures = Path(__file__).resolve().parents[2] / 'navigation/tests/fixtures/nav_models.json'
  signature = next(model for model in json.loads(fixtures.read_text())['models']
                   if model['tag'] == 'big-nav-model-integration-1d18c42e')
  output = next(item for item in signature['outputs'] if item['name'] == 'outputs')
  raw = np.zeros(tuple(output['shape']), dtype=np.float32)
  slices = {name: slice(bounds['start'], bounds['stop'], bounds['step']) for name, bounds in signature['output_slices'].items()}
  raw[0, slices['action']] = [0., accel, 0., 0.]
  parsed = Parser().parse_outputs({name: raw[:, bounds].copy() for name, bounds in slices.items()})
  parsed['plan'][0, :, Plan.POSITION] = np.column_stack((np.asarray(ModelConstants.T_IDXS) * speed, np.zeros((33, 2))))
  parsed['plan'][0, :, Plan.VELOCITY] = [speed, 0., 0.]
  parsed['plan'][0, :, Plan.ORIENTATION_RATE] = [0., 0., .01]
  action = NavigationModelState.get_action_from_model(None, parsed, log.ModelDataV2.Action(desiredAcceleration=accel), .2, .55, speed)
  base = messaging.new_message('drivingModelData')
  model = messaging.new_message('modelV2')
  fill_model_msg(base, model, parsed, action, PublishState(), 1, 1, 1, 0., 0, 0., True, Meta)
  model.modelV2.big = True
  assert np.isfinite(model.modelV2.action.desiredAcceleration)
  return model.modelV2


def planner(*, speed=20., openpilot_long=True):
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.HONDA_INSIGHT, fingerprint, [], False, False, False)
  cp_sp = CarInterface.get_params_sp(cp, CAR.HONDA_INSIGHT, fingerprint, [], False, False, False)
  cp.openpilotLongitudinalControl = openpilot_long
  return LongitudinalPlanner(cp, cp_sp, init_v=speed)


@pytest.fixture(autouse=True)
def isolated_params(tmp_path, monkeypatch):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))


@pytest.mark.parametrize('old_mode', ('0', '1'))
def test_native_bigv2_braking_ignores_obsolete_mode_files(old_mode):
  params = Params()
  # Emulate files left by an older branch; no now-unregistered key API writes.
  for key in ('ExperimentalMode', 'ExperimentalModeConfirmed', 'DynamicExperimentalControl'):
    Path(params.get_param_path(key)).write_text(old_mode)
  sm = Messages(published_bigv2(-3.))
  control = planner()
  control.update(sm)
  assert control.mpc.source == LongitudinalPlanSource.e2e
  assert control.output_a_target == pytest.approx(-3.)
  assert not hasattr(control, 'dec')
  # The planner no longer needs calibration/steering-angle or deprecated mode fields.
  assert 'vehicleParameters' not in sm


def test_bigv2_stopping_and_ordinary_fallback_share_planner_interface():
  sm = Messages(published_bigv2(-3., speed=.1), speed=.1)
  control = planner(speed=.1)
  control.update(sm)
  assert control.output_should_stop
  assert control.output_a_target < 0
  # The existing fallback's ordinary modelV2 uses the same action contract.
  sm['modelV2'] = published_bigv2(1., speed=.1)
  sm['modelV2'].big = False
  control.update(sm)
  assert not control.output_should_stop
  assert np.isfinite(control.output_a_target)
  assert ACCEL_MIN <= control.output_a_target <= ACCEL_MAX


def test_model_acceleration_cannot_bypass_cruise_or_radar_candidate():
  sm = Messages(published_bigv2(100.))
  control = planner()
  for _ in range(20):
    control.update(sm)
    assert control.mpc.source != LongitudinalPlanSource.e2e
    assert control.output_a_target <= control.a_cruise
    assert ACCEL_MIN <= control.output_a_target <= ACCEL_MAX
  sm['modelV2'] = published_bigv2(-100.)
  control.update(sm)
  assert control.output_a_target == ACCEL_MIN


def test_upstream_boost_cap_braking_scaling_and_disengagement_reset():
  sm = Messages(published_bigv2(0.))
  control = planner()
  for _ in range(50):
    control.update(sm)
  assert control.accel_boost.boost_eligible
  for _ in range(5):
    sm['carState'].gasPressed = False
    control.update(sm)
    before = control.accel_boost.total_boost
    sm['carState'].gasPressed = True
    for _ in range(100):
      control.update(sm)
    assert control.accel_boost.total_boost - before <= ACCEL_BOOST_PER_OVERRIDE + 1e-12
  assert control.accel_boost.total_boost == pytest.approx(ACCEL_BOOST_MAX)
  sm['modelV2'] = published_bigv2(-3.)
  control.update(sm)
  assert control.output_a_target == pytest.approx(-3.)
  sm['selfdriveState'].enabled = False
  sm['controlsState'].longControlState = 'off'
  control.update(sm)
  assert control.accel_boost.total_boost == 0.


def test_stock_longitudinal_mpc_uses_measured_state():
  sm = Messages(published_bigv2(0.))
  sm['carState'].aEgo = -.7
  control = planner(openpilot_long=False)
  original = control.mpc.set_cur_state
  control.mpc.set_cur_state = Mock(wraps=original)
  control.update(sm)
  control.mpc.set_cur_state.assert_called_once_with(sm['carState'].vEgo, sm['carState'].aEgo)


def test_planner_publication_keeps_braking_and_navigation_action_metadata():
  sm = Messages(published_bigv2(-3.))
  sm.logMonoTime['modelV2'] = 12345
  control = planner()
  control.update(sm)
  published = {}
  control.publish(sm, SimpleNamespace(send=lambda name, value: published.update({name: value})))
  result = published['longitudinalPlan'].longitudinalPlan
  assert result.modelMonoTime == 12345
  assert result.allowBrake
  assert result.aTarget == pytest.approx(-3.)
  assert np.isfinite(result.accels).all()
  assert np.isfinite(result.speeds).all()
  assert np.isfinite(result.jerks).all()
