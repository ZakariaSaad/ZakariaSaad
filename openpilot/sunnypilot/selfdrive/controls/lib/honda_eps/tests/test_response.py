"""Response selection is explicit; the default numerical controller is intact."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from opendbc.car.vehicle_model import VehicleModel
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.core import OUTPUT_LPF_TAU
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.latcontrol_fw import LatControlFirmware
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.response import (
  ROUTE32_RESPONSE, SOURCE_RESPONSE, resolve_response_profile,
)


@pytest.mark.parametrize('profile', [SOURCE_RESPONSE, ROUTE32_RESPONSE])
def test_named_response_profiles_are_frozen_and_resolve_exactly(profile):
  assert resolve_response_profile(profile.name) is profile
  with pytest.raises(FrozenInstanceError):
    profile.name = 'changed'


@pytest.mark.parametrize('name', ['unknown', 'Source', 'ROUTE32', '', 'route32\x00', None, 0, True, {}, []])
def test_unknown_or_unnamed_responses_rejected(name, car_factory):
  with pytest.raises(ValueError):
    resolve_response_profile(name)
  with pytest.raises(ValueError):
    cp, sp = car_factory()
    LatControlFirmware(cp, sp, None, .01, response_profile=name)


def test_direct_wrapper_keeps_source_response_default(car_factory):
  cp, sp = car_factory()
  controller = LatControlFirmware(cp, sp, None, .01)
  assert controller.response_profile is SOURCE_RESPONSE
  assert controller.core.output_lpf_tau == OUTPUT_LPF_TAU == (.07, .05, .01)
  assert controller.delivery.fade_down_s == 0.


@pytest.mark.parametrize('profile', [SOURCE_RESPONSE, ROUTE32_RESPONSE])
@pytest.mark.parametrize('speed,band', [(5., 0), (15., 1), (30., 2)])
def test_selected_response_sets_actual_filter_and_survives_reset(car_factory, profile, speed, band):
  cp, sp = car_factory()
  before = cp.to_dict()
  controller = LatControlFirmware(cp, sp, None, .01, response_profile=profile.name)
  state = SimpleNamespace(vEgo=speed, steeringAngleDeg=0., steeringRateDeg=0., steeringTorque=0., steeringPressed=False)
  controller.update(True, state, VehicleModel(cp), SimpleNamespace(roll=0., angleOffsetDeg=0.),
                    False, -.002, None, False, .2)
  assert controller.response_profile is profile
  assert controller.core.output_lpf_tau == profile.output_lpf_tau
  assert controller.core.output_lpf.alpha == pytest.approx(.01 / (.01 + profile.output_lpf_tau[band]))
  assert controller.delivery.fade_down_s == profile.override_fade_down_s
  assert controller.core.override_fade_up_s == 1.5
  assert controller.output_scale == controller.steer_max == .9375
  assert cp.to_dict() == before
  controller.reset()
  assert controller.core.output_lpf_tau == profile.output_lpf_tau
  assert controller.delivery.fade_down_s == profile.override_fade_down_s
