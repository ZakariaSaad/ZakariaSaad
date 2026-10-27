"""Exercise the real sp-steer interface without launching controls or sending CAN."""
from types import SimpleNamespace
from unittest.mock import Mock
import math

import pytest

from opendbc.car import rate_limit
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.lateral_model import HondaLateralModel
from opendbc.car.honda.steer_threshold_learner import HondaSteerThresholdLearner
from openpilot.sunnypilot.selfdrive.car import interfaces
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import FIRMWARE_SHA256, prepare_drive_config
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.transport import FirmwareTransport, FirmwareDriverThreshold, configure_firmware_transport


def test_firmware_transport_matches_previous_final_limit_for_full_sequence():
  adapter = FirmwareTransport(SimpleNamespace(STEER_DELTA_DOWN=3, STEER_DELTA_UP=3))
  last = 0.
  for request in ([.9375] * 120 + [-.9375] * 120 + [.15] * 20 + [0.] * 50):
    output = adapter.update(request, last, True, True, False, 30., .01, .009, 10., 2.)
    assert output == request  # no second plant/feedforward or source→transport conversion
    limited = adapter.limit(output, last, 30.)
    assert limited == rate_limit(request, last, -.03, .03)
    assert adapter.reported_torque(request, output, limited, 1.) == limited
    assert abs(int(-limited * 4096)) <= 3840
    last = limited
  assert adapter.learned_values() == {}


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_firmware_transport_nonfinite_rejected(bad):
  adapter = FirmwareTransport(SimpleNamespace(STEER_DELTA_DOWN=3, STEER_DELTA_UP=3))
  with pytest.raises(ValueError):
    adapter.update(bad, 0., True, True, False, 20., 0., 0., 0., 0.)
  with pytest.raises(ValueError):
    adapter.limit(0., bad, 20.)


def test_firmware_mode_does_not_consume_or_overwrite_ordinary_learning(params, car_factory, monkeypatch):
  cp, sp = car_factory(native_sp=False)
  ci = CarInterface(cp, sp)
  params.put('HondaSteerThresholdFactorParams', 2.5, block=True)
  params.put_bool('EnforceTorqueControl', True, block=True)
  params.put_bool('LateralJerkTorqueController', True, block=True)
  params.put_bool('NeuralNetworkLateralControl', True, block=True)
  params.put_bool('HondaEpsFirmwareEnabled', True, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', FIRMWARE_SHA256, block=True)
  monkeypatch.setattr(interfaces, 'get_nn_model_path', lambda cp: ('', 'MOCK', False))
  monkeypatch.setattr(interfaces.STATSLOGSP, 'raw', Mock())
  interfaces.setup_interfaces(ci, params)
  assert isinstance(ci.CC.lat_model, FirmwareTransport)
  assert isinstance(ci.CS.steer_threshold_learner, FirmwareDriverThreshold)
  for i in range(40):
    ci.CS.steer_threshold_learner.update(3000. if i % 2 else 0., 1200., True, i * .1)
  assert ci.CS.steer_threshold_learner.effective_threshold(1200.) == 2400.
  assert params.get('HondaSteerThresholdFactorParams') == 2.5
  assert params.get_bool('EnforceTorqueControl')
  assert params.get_bool('LateralJerkTorqueController')
  assert params.get_bool('NeuralNetworkLateralControl')
  assert ci.CC.lat_model.learned_values() == {}


def test_ordinary_mode_keeps_actual_upstream_models(params, car_factory):
  cp, sp = car_factory(native_sp=False)
  ci = CarInterface(cp, sp)
  lat_model = ci.CC.lat_model
  threshold = ci.CS.steer_threshold_learner
  assert isinstance(lat_model, HondaLateralModel)
  assert isinstance(threshold, HondaSteerThresholdLearner)
  config = prepare_drive_config(params, cp, sp)
  configure_firmware_transport(ci, config)
  assert config is None
  assert ci.CC.lat_model is lat_model
  assert ci.CS.steer_threshold_learner is threshold
  params.put_bool('HondaEpsFirmwareEnabled', True, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', FIRMWARE_SHA256, block=True)
  configure_firmware_transport(ci, prepare_drive_config(params, cp, sp))
  assert ci.CC.lat_model is lat_model  # no switch until the next drive cycle


@pytest.mark.parametrize('learner_name,cache_key,frame', [('torqued', 'LiveTorqueParameters', 240), ('lagd', 'LiveDelay', 1200)])
@pytest.mark.parametrize('enabled', [False, True])
def test_live_estimators_preserve_ordinary_cache_during_firmware_cycle(params, car_factory, monkeypatch,
                                                                    learner_name, cache_key, frame, enabled):
  import importlib
  from openpilot.selfdrive.car.helpers import convert_to_capnp
  module = importlib.import_module('openpilot.selfdrive.locationd.' + learner_name)
  cp, sp = car_factory(native_sp=False)
  params.put_bool('HondaEpsFirmwareEnabled', enabled, block=True)
  params.put('HondaEpsFirmwareVerifiedSha256', FIRMWARE_SHA256, block=True)
  prepare_drive_config(params, cp, sp)
  params.put('CarParams', cp.to_bytes(), block=True)
  params.put('CarParamsSP', convert_to_capnp(sp).to_bytes(), block=True)
  params.put(cache_key, b'ordinary mode cache', block=True)
  params.put('LagdValueCache', .4321, block=True)

  class Finished(Exception):
    pass

  sm = Mock(frame=frame)
  sm.all_checks.return_value = False
  sm.update.side_effect = [None, Finished()]
  pub = Mock()
  params_spy = Mock(wraps=params)
  monkeypatch.setattr(module, 'Params', Mock(return_value=params_spy))
  monkeypatch.setattr(module, 'config_realtime_process', Mock())
  monkeypatch.setattr(module.messaging, 'SubMaster', Mock(return_value=sm))
  monkeypatch.setattr(module.messaging, 'PubMaster', Mock(return_value=pub))
  with pytest.raises(Finished):
    module.main()
  assert pub.send.call_count == 1  # existing liveness messages still published
  if enabled:
    assert params.get(cache_key) == b'ordinary mode cache'
    assert params.get('LagdValueCache') == .4321
    if learner_name == 'torqued':
      assert not pub.send.call_args.args[1].lateralTorqueParameters.useParams
  else:
    assert any(call.args[0] == cache_key for call in params_spy.put.call_args_list)
