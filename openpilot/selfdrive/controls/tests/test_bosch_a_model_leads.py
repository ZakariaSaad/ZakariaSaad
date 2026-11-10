"""Temporal and geometric qualification before optional native MPC lead paths.

The clock pair and message envelopes are synthetic. No live subscribers or
publishers are created; the model trajectory builder is the production code.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a import human_following as following
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a import model_lead_qualification as qualification

NOW = 10_000_000_000
BOOT_OFFSET = 20_000_000_000


class Messages(dict):
  pass


def model(stamp=NOW-50_000_000):
  times = following.LEAD_T_IDXS_MODEL.tolist()
  return NS(timestampEof=stamp+BOOT_OFFSET-10_000_000, velocity=NS(x=[20.]), leadsV3=[
    NS(prob=.99, x=[51.52+20*t for t in times], y=[0.]*len(times),
       v=[20.]*len(times), t=times) for _ in range(2)
  ])


def messages():
  stamp = NOW-50_000_000
  sm = Messages(modelV2=model(stamp),
                radarState=NS(mdMonoTime=stamp, radarErrors=NS(to_dict=lambda: {'canError': False})),
                carState=NS(canValid=True, canTimeout=False))
  sm.logMonoTime = {'modelV2': stamp, 'radarState': NOW-20_000_000, 'carState': NOW-10_000_000}
  sm.recv_time = {name: (value+1_000_000)/1e9 for name, value in sm.logMonoTime.items()}
  sm.valid = dict.fromkeys(sm, True)
  sm.alive = dict.fromkeys(sm, True)
  sm.freq_ok = dict.fromkeys(sm, True)
  return sm


@pytest.fixture(autouse=True)
def clock_pair(monkeypatch):
  monkeypatch.setattr(qualification, '_model_lead_clock_pair_ns', lambda: (NOW, NOW+BOOT_OFFSET))


def test_newer_car_state_is_allowed_when_each_publication_is_causal_and_fresh():
  sm = messages()
  assert sm.logMonoTime['carState'] > sm.logMonoTime['modelV2']
  qualified = qualification.qualified_model_leads(sm, NOW, qualification.ModelLeadHistory())
  assert qualified is not None
  assert qualified.stamp == sm['radarState'].mdMonoTime
  assert qualified.velocity.x == (20.,)


@pytest.mark.parametrize('field', ['valid', 'alive', 'freq_ok'])
@pytest.mark.parametrize('service', ['modelV2', 'radarState', 'carState'])
def test_missing_core_health_rejects_optional_horizon(field, service):
  sm = messages()
  getattr(sm, field)[service] = False
  assert qualification.qualified_model_leads(sm, NOW) is None


@pytest.mark.parametrize('failure', ['old_publication', 'future_publication', 'old_frame', 'future_frame',
                                      'can_invalid', 'can_timeout', 'radar_error', 'missing_radar_model'])
def test_unqualified_clock_or_vehicle_state_falls_back(failure):
  sm = messages()
  if failure == 'old_publication':
    sm.logMonoTime['carState'] = NOW-151_000_000
  elif failure == 'future_publication':
    sm.logMonoTime['carState'] = NOW+1
  elif failure == 'old_frame':
    sm['modelV2'].timestampEof = NOW+BOOT_OFFSET-151_000_000
  elif failure == 'future_frame':
    sm['modelV2'].timestampEof = NOW+BOOT_OFFSET+1
  elif failure == 'can_invalid':
    sm['carState'].canValid = False
  elif failure == 'can_timeout':
    sm['carState'].canTimeout = True
  elif failure == 'radar_error':
    sm['radarState'].radarErrors = NS(to_dict=lambda: {'canError': True})
  elif failure == 'missing_radar_model':
    sm['radarState'].mdMonoTime = 0
  assert qualification.qualified_model_leads(sm, NOW, qualification.ModelLeadHistory()) is None


def test_history_selects_the_model_used_by_radar_instead_of_current_frame():
  sm = messages()
  history = qualification.ModelLeadHistory()
  first = qualification.qualified_model_leads(sm, NOW, history)
  assert first is not None
  current_stamp = NOW-5_000_000
  sm['modelV2'] = model(current_stamp)
  sm['modelV2'].leadsV3[0].x[0] = 99.
  sm.logMonoTime['modelV2'] = current_stamp
  sm.recv_time['modelV2'] = (current_stamp+1_000_000)/1e9
  selected = qualification.qualified_model_leads(sm, NOW, history)
  assert selected is first
  assert selected.leadsV3[0].x[0] == 51.52
  assert qualification.qualified_model_leads(sm, NOW) is None


def test_clock_pair_failure_falls_back(monkeypatch):
  monkeypatch.setattr(qualification, '_model_lead_clock_pair_ns', lambda: None)
  assert qualification.qualified_model_leads(messages(), NOW) is None


def test_matching_horizon_anchors_current_lead_and_bounds_forward_motion():
  lead = NS(present=True, radar=True, radarTrackId=3, modelProb=.99,
            dRel=50., yRel=0., vLead=20., vRel=0.)
  path = following.build_trajectory(model().leadsV3[0], lead, 20., 20.)
  assert path is not None
  np.testing.assert_allclose(path[0], [50., 20.])
  assert (np.diff(path[:, 0]) >= 0.).all()
  np.testing.assert_allclose(path[:, 0], 50.+20.*following.T_IDXS)


@pytest.mark.parametrize('failure', ['low_probability', 'wrong_range', 'wrong_velocity', 'wrong_lateral',
                                      'wrong_times', 'nonfinite', 'urgent_closing'])
def test_unmatched_or_urgent_horizon_keeps_radar_extrapolation(failure):
  camera = model().leadsV3[0]
  lead = NS(present=True, radar=True, radarTrackId=3, modelProb=.99,
            dRel=50., yRel=0., vLead=20., vRel=0.)
  if failure == 'low_probability':
    camera.prob = .35
  elif failure == 'wrong_range':
    lead.dRel = 90.
  elif failure == 'wrong_velocity':
    lead.vLead = 23.
  elif failure == 'wrong_lateral':
    lead.yRel = 2.
  elif failure == 'wrong_times':
    camera.t = [value+.1 for value in camera.t]
  elif failure == 'nonfinite':
    camera.x[-1] = float('nan')
  elif failure == 'urgent_closing':
    lead.dRel = 10.
    lead.vLead = 10.
    lead.vRel = -10.
    camera.x = [11.52+10*t for t in camera.t]
    camera.v = [10.]*len(camera.t)
  assert following.build_trajectory(camera, lead, 20., 20.) is None
