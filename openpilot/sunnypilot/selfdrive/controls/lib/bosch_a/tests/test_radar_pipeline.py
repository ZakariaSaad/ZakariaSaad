import copy
from types import SimpleNamespace

import pytest

from opendbc.car import structs
from opendbc.car.honda.bosch_a_radar import BOSCH_A_FREQ_HZ, BOSCH_A_NUM_SLOTS, BOSCH_A_TRIGGER_MSG
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.tests.test_bosch_a_decoder import sweep as decoder_sweep
from opendbc.car.honda.tests.test_bosch_a_integration import parameters
from opendbc.car.honda.values import CAR
from openpilot.cereal import messaging
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.tests.test_radar_lead_bound import ReplaySubMaster, bosch_a_radard, companion
from openpilot.selfdrive.controls.radard import RadarD


def sweep(cp, index, t_nanos):
  """All 80 IDs, with one valid track in slot 0 and invalid objects elsewhere."""
  frames = []
  for slot in range(BOSCH_A_NUM_SLOTS):
    (stamp, slot_frames), = decoder_sweep(slot, index % 16, 7 if slot == 0 else 15,
      227, 1024, 1 + index * 2, t_nanos, with_aux=True,
      direct_vrel_raw=864, direct_vrel_uncertainty_raw=0)
    frames.extend(frame._replace(src=CanBus(cp).camera) for frame in slot_frames
                  if slot == BOSCH_A_NUM_SLOTS - 1 or frame.address != BOSCH_A_TRIGGER_MSG)
  assert len(frames) == len({frame.address for frame in frames}) == 80
  return [(stamp, frames)]


def source_tick(sm, tick, rr=None, speed=0.0, probability=0.0, distance=10.0, standstill=False,
                *, stamp=None, statuses=None):
  stamp = 1_000_000_000 + tick * 50_000_000 if stamp is None else stamp
  model_event = messaging.new_message('modelV2', valid=True)
  model_event.logMonoTime = stamp
  model = model_event.modelV2
  model.velocity.x = [speed]
  leads = model.init('leadsV3', 2)
  for lead in leads:
    lead.x = [distance + 1.52]
    lead.y = [0.0]
    lead.v = [speed]
    lead.a = [0.0]
    lead.xStd = lead.yStd = lead.vStd = [1.0]
    lead.prob = probability
  car_event = messaging.new_message('carState', valid=True)
  car_event.logMonoTime = stamp
  car_event.carState.vEgo = speed
  car_event.carState.standstill = standstill
  events = [model_event.as_reader(), car_event.as_reader()]
  if rr is not None:
    radar_event = messaging.new_message('radarTracks', valid=not any(rr.errors.to_dict().values()))
    radar_event.logMonoTime = stamp
    radar_event.radarTracks = rr
    events.append(radar_event.as_reader())
    if statuses is not None:
      events.append(companion(stamp, statuses))
  sm.update_msgs(stamp / 1e9, events)


def radar_data(measured=True, relative_speed=0.0, distance=10.0):
  rr = structs.RadarData()
  point = rr.init('points', 1)[0]
  point.trackId = 1
  point.dRel = distance
  point.yRel = 0.0
  point.vRel = relative_speed
  point.deprecated.measured = measured
  return rr


def test_actual_radard_duplicate_and_predicted_point_hold_kalman_measurement():
  sm = ReplaySubMaster(['modelV2', 'carState', 'radarTracks'])
  radar = bosch_a_radard()
  source_tick(sm, 0, radar_data(), speed=10.0)
  radar.update(sm, sm['radarTracks'])
  track = radar.tracks[1]
  assert track.cnt == 1
  original = copy.deepcopy(track.kf.x)
  source_tick(sm, 1, speed=20.0)
  radar.update(sm, sm['radarTracks'])
  assert track.cnt == 1 and track.kf.x == original and not track.measured
  source_tick(sm, 2, radar_data(False, 30.0), speed=20.0)
  radar.update(sm, sm['radarTracks'])
  assert track.cnt == 1 and track.kf.x == original and not track.measured
  source_tick(sm, 3, radar_data(True, 2.0), speed=20.0)
  radar.update(sm, sm['radarTracks'])
  assert track.cnt == 2 and track.kf.x != original and track.measured
  assert radar.kalman_params.A[0][1] == pytest.approx(1.0 / BOSCH_A_FREQ_HZ)


def test_non_honda_radard_keeps_ordinary_duplicate_measurement_behavior():
  sm = ReplaySubMaster(['modelV2', 'carState', 'radarTracks'])
  radar = RadarD(SimpleNamespace(brand='toyota', radarUnavailable=False, carFingerprint='TOYOTA', radarDelay=0.0), SimpleNamespace(flags=0))
  source_tick(sm, 0, radar_data(), speed=10.0)
  radar.update(sm, sm['radarTracks'])
  source_tick(sm, 1, speed=20.0)
  radar.update(sm, sm['radarTracks'])
  assert radar.tracks[1].cnt == 2
  assert radar.kalman_params.A[0][1] == 0.05


@pytest.mark.parametrize('identity', (CAR.HONDA_INSIGHT, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G))
def test_actual_decoder_radard_maturity_stale_retirement_and_recovery(identity):
  cp, cp_sp = parameters(identity, enabled=True)
  decoder = RadarInterface(cp, cp_sp)
  radar = RadarD(cp, cp_sp, cp.radarDelay)
  assert radar.bosch_a_policy is not None
  sm = ReplaySubMaster(['modelV2', 'carState', 'radarTracks', 'carStateSP'])
  period = round(1e9 / BOSCH_A_FREQ_HZ)
  for index in range(4):
    stamp = 1_000_000_000 + index * period
    rr = decoder.update(sweep(cp, index, stamp))
    assert not any(rr.errors.to_dict().values())
    source_tick(sm, index, rr, stamp=stamp, statuses=decoder.point_statuses)
    radar.update(sm, sm['radarTracks'])
    if index < 3:
      assert not radar.radar_state.leadOne.present
  assert radar.radar_state.leadOne.present and radar.radar_state.leadOne.radar
  assert radar.radar_state.leadOne.radarTrackId == 1
  assert radar.tracks[1].objectStatus == 7
  assert radar.radar_state_valid
  stale = decoder.update([(1_500_000_000, [])])
  assert stale.errors.radarUnavailableTemporary and not stale.points
  source_tick(sm, 4, stale, stamp=1_500_000_000, statuses=decoder.point_statuses)
  radar.update(sm, sm['radarTracks'])
  assert not radar.tracks and not radar.radar_state.leadOne.present
  assert not radar.radar_state_valid
  assert radar.radar_state.radarErrors.radarUnavailableTemporary
  for index in range(4):
    stamp = 1_500_000_000 + (index + 1) * period
    rr = decoder.update(sweep(cp, index + 5, stamp))
    source_tick(sm, index + 5, rr, stamp=stamp, statuses=decoder.point_statuses)
    radar.update(sm, sm['radarTracks'])
    if index < 3:
      assert not radar.radar_state.leadOne.present
  assert radar.radar_state_valid
  assert not any(radar.radar_state.radarErrors.to_dict().values())
  assert radar.radar_state.leadOne.present and radar.radar_state.leadOne.radar
