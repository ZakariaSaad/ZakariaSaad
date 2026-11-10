from types import SimpleNamespace

import pytest

from opendbc.car.structs import car
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.car.honda.values import CAR as HONDA_CAR
from openpilot.selfdrive.controls.radard import RadarD


def model_lead(distance: float = 20.0, probability: float = 0.99):
  return SimpleNamespace(
    prob=probability,
    x=[distance + 1.52],
    y=[0.0],
    v=[2.0],
    a=[0.0],
    xStd=[1.0],
    yStd=[1.0],
    vStd=[1.0],
  )


class FakeSubMaster:
  def __init__(self):
    self.seen = {"modelV2": True}
    self.recv_frame = {"radarTracks": 0, "carState": 0}
    self.logMonoTime = {"modelV2": 0, "radarTracks": 0, "carState": 0}
    self.valid = dict.fromkeys(self.logMonoTime, True)
    self.data = {
      "carState": SimpleNamespace(vEgo=8.0, standstill=False),
      "modelV2": SimpleNamespace(
        velocity=SimpleNamespace(x=[8.0]),
        leadsV3=[model_lead(), model_lead(probability=0.0)],
      ),
    }

  def __getitem__(self, key):
    return self.data[key]

  def all_checks(self):
    return all(self.valid.values())

  def advance(self):
    for key in self.recv_frame:
      self.recv_frame[key] += 1
    for key in self.logMonoTime:
      self.logMonoTime[key] += 50_000_000


def radar_data(*, y_rel: float = 0.0, points: bool = True):
  radar = car.RadarData.new_message()
  radar_points = radar.init("points", 1 if points else 0)
  if points:
    radar_points[0].trackId = 7
    radar_points[0].dRel = 20.0
    radar_points[0].yRel = y_rel
    radar_points[0].vRel = -6.0
    radar_points[0].deprecated.measured = True
  return radar


def car_params(fingerprint):
  return SimpleNamespace(
    brand="honda",
    radarUnavailable=False,
    openpilotLongitudinalControl=True,
    carFingerprint=fingerprint,
    radarDelay=0.0,
  )


def test_radard_publishes_strict_only_deprecated_fcw_and_clears_it_on_track_loss():
  radar = RadarD(car_params(HONDA_CAR.HONDA_INSIGHT), SimpleNamespace(flags=HondaFlagsSP.BOSCH_A_LONG))
  sm = FakeSubMaster()

  for _ in range(3):
    sm.advance()
    radar.update(sm, radar_data())
  assert radar.radar_state.leadOne.present
  assert radar.radar_state.leadOne.radar
  assert radar.radar_state.leadOne.deprecated.fcw

  sm.advance()
  radar.update(sm, radar_data(y_rel=1.4))
  assert radar.radar_state.leadOne.present
  assert radar.radar_state.leadOne.radarTrackId == 7
  assert not radar.radar_state.leadOne.deprecated.fcw

  sm.advance()
  radar.update(sm, radar_data(points=False))
  assert radar.radar_state.leadOne.present
  assert not radar.radar_state.leadOne.radar
  assert not radar.radar_state.leadOne.deprecated.fcw


def test_non_bosch_radard_keeps_deprecated_fcw_false():
  radar = RadarD(car_params(HONDA_CAR.HONDA_CIVIC), SimpleNamespace(flags=0))
  sm = FakeSubMaster()
  sm.advance()
  radar.update(sm, radar_data())
  assert radar.radar_state.leadOne.present
  assert radar.radar_state.leadOne.radar
  assert not radar.radar_state.leadOne.deprecated.fcw


def test_vision_acceleration_histories_are_independent_through_actual_radard():
  radar = RadarD(car_params(HONDA_CAR.HONDA_INSIGHT), SimpleNamespace(flags=HondaFlagsSP.BOSCH_A_LONG))
  sm = FakeSubMaster()
  sm.data['modelV2'].leadsV3 = [model_lead(), model_lead()]
  for lead, acceleration in zip(sm.data['modelV2'].leadsV3, (2., -3.), strict=True):
    lead.a = [acceleration]
  for tick in range(20):
    sm.advance()
    radar.update(sm, radar_data(points=False))
    actual = (radar.radar_state.leadOne, radar.radar_state.leadTwo)
    for lead, acceleration in zip(actual, (2., -3.), strict=True):
      assert lead.present and not lead.radar
      assert not lead.deprecated.fcw
      assert lead.aLeadK == pytest.approx(acceleration * (1. - .2 ** (tick + 1)))
  assert radar.bosch_a_policy.vision_states[0] is not radar.bosch_a_policy.vision_states[1]


@pytest.mark.parametrize('probability,authorized', ((.35, False), (.9, False), (.900001, True)))
def test_fcw_requires_strict_probability_not_lead_existence_threshold(probability, authorized):
  radar = RadarD(car_params(HONDA_CAR.HONDA_INSIGHT), SimpleNamespace(flags=HondaFlagsSP.BOSCH_A_LONG))
  sm = FakeSubMaster()
  sm.data['modelV2'].leadsV3[0].prob = probability
  sm.advance()
  radar.update(sm, radar_data())
  assert radar.radar_state.leadOne.deprecated.fcw is authorized


@pytest.mark.parametrize('invalid_service', ('carState', 'modelV2', 'radarTracks'))
def test_invalid_core_input_cannot_publish_fcw_authority(invalid_service):
  radar = RadarD(car_params(HONDA_CAR.HONDA_INSIGHT), SimpleNamespace(flags=HondaFlagsSP.BOSCH_A_LONG))
  sm = FakeSubMaster()
  sm.advance()
  radar.update(sm, radar_data())
  assert radar.radar_state.leadOne.deprecated.fcw
  sm.valid[invalid_service] = False
  sm.advance()
  radar.update(sm, radar_data())
  assert not radar.radar_state_valid
  assert not radar.radar_state.leadOne.deprecated.fcw
  sent = []
  radar.publish(SimpleNamespace(send=lambda service, message: sent.append((service, message.as_reader()))))
  assert len(sent) == 1 and sent[0][0] == 'radarState'
  assert not sent[0][1].valid


@pytest.mark.parametrize('error', ('canError', 'radarUnavailableTemporary'))
def test_radar_error_is_retained_and_does_not_authorize_fcw(error):
  radar = RadarD(car_params(HONDA_CAR.HONDA_INSIGHT), SimpleNamespace(flags=HondaFlagsSP.BOSCH_A_LONG))
  sm = FakeSubMaster()
  sm.advance()
  rr = radar_data()
  setattr(rr.errors, error, True)
  # Even a mistakenly valid envelope cannot turn a decoder fault into FCW authority.
  radar.update(sm, rr)
  assert getattr(radar.radar_state.radarErrors, error)
  assert not radar.radar_state.leadOne.deprecated.fcw


@pytest.mark.parametrize('openpilot_long,flags,unavailable', (
  (True, 0, False), (False, HondaFlagsSP.BOSCH_A_LONG, False),
  (True, HondaFlagsSP.BOSCH_A_LONG, True),
))
def test_unwired_or_stock_insight_retains_generic_radar_update(openpilot_long, flags, unavailable):
  cp = car_params(HONDA_CAR.HONDA_INSIGHT)
  cp.openpilotLongitudinalControl = openpilot_long
  cp.radarUnavailable = unavailable
  radar = RadarD(cp, SimpleNamespace(flags=flags))
  assert radar.bosch_a_policy is None
  assert radar.kalman_params.A[0][1] == .05
  sm = FakeSubMaster()
  for _ in range(2):
    sm.advance()
    radar.update(sm, radar_data())
  assert radar.tracks[7].cnt == 2
