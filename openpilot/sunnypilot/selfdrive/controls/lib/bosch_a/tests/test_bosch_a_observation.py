"""Actual parser and native-schema observation tests; no sockets or CAN writes."""
from types import SimpleNamespace
from dataclasses import asdict

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.tests.test_bosch_a_radar_interface import DT_NS, sweep
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.interfaces import _initialize_honda
from openpilot.cereal import custom, log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.selfdrive.car import card as card_module
from openpilot.selfdrive.car.card import Car
from openpilot.sunnypilot.selfdrive.car.interfaces import initialize_params
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a.observation import BoschAObservation, HEARTBEAT_NS, SERVICE


class Clock:
  now = 1_000_000_000

  def __call__(self):
    self.now += 100
    return self.now


class Snapshot(dict):
  def __init__(self, now):
    super().__init__(modelV2=log.ModelDataV2.new_message(frameId=42, timestampEof=now - 55_000_000))
    self.seen = self.valid = self.alive = self.freq_ok = {'modelV2': True}
    self.logMonoTime = {'modelV2': now - 5_000_000}
    self.recv_time = {'modelV2': (now - 1_000_000) / 1e9}
    self.frame = 1

  def all_checks(self, services):
    return True


def parameters(enabled=True):
  fp = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.HONDA_INSIGHT, fp, [], False, False, False)
  cp_sp = CarInterface.get_params_sp(cp, CAR.HONDA_INSIGHT, fp, [], False, False, False)
  _initialize_honda(cp, cp_sp, {'HondaEnforceStockLongitudinal': 1,
                               'BoschARadarObservationEnabled': int(enabled)})
  return cp, cp_sp


@pytest.fixture
def harness():
  cp, cp_sp = parameters()
  clock = Clock()
  observer = BoschAObservation(cp, cp_sp, clock)
  state = structs.CarState(vEgo=0., aEgo=0., canValid=True)
  return SimpleNamespace(cp=cp, cp_sp=cp_sp, clock=clock, observer=observer, state=state, sm=Snapshot(clock.now))


def step(h, index, status=7):
  now = 1_000_000_000 + index * DT_NS
  h.clock.now = now + 10_000
  h.observer.update(sweep(h.observer.parser_bus, index, now, status=status), h.state, h.clock.now)
  return h.observer.message(h.state, h.sm, car_state_mono_time=h.clock.now)


def test_actual_decoder_snapshot_schema_hashes_timestamps_and_status_pairing(harness):
  h = harness
  first = step(h, 0)
  assert first.valid and first.boschARadarObservation.hasSample
  assert not first.boschARadarObservation.data.points
  event = step(h, 1, status=9)
  # Serialize/parse the actual new Event union, not a stand-in payload.
  with log.Event.from_bytes(event.to_bytes()) as parsed:
    assert parsed.which() == SERVICE and parsed.valid
    obs = parsed.boschARadarObservation
    assert obs.enabled and obs.hasSample and obs.sampleUpdated and obs.sequence == 2
    assert obs.sampleMonoTime == obs.statusSampleMonoTime == obs.decodeEndMonoTime
    assert obs.egoSpeedUsed == h.state.vEgo and obs.sampleCarStateMonoTime == obs.decodeEndMonoTime
    assert obs.sourceCanFirstMonoTime == obs.sourceCanMonoTime == obs.triggerMonoTime == 1_000_000_000 + DT_NS
    assert obs.sourceCanMonoTime <= obs.batchReceiveMonoTime <= obs.decodeStartMonoTime <= obs.decodeEndMonoTime
    assert obs.decodeEndMonoTime <= obs.publicationMonoTime == parsed.logMonoTime
    assert obs.sampleValid and obs.carStateValid and not obs.stale
    assert obs.statusComplete and obs.statusDiscardedCount == obs.statusMissingCount == 0
    assert [(point.trackId, point.objectStatus) for point in obs.pointStatus] == [(1, 9)]
    point, = obs.data.points
    assert point.trackId == 1 and point.deprecated.measured
    assert point.dRel == pytest.approx(1000 / 16 - 346 / 128)
    assert len(obs.decoderSha256) == len(obs.dbcSha256) == 64
    assert obs.receiver == 'card' and obs.modelSeen and obs.modelValid and obs.modelAlive and obs.modelFreqOk
    assert obs.modelFrameId == 42 and obs.modelTimestampEof == h.sm['modelV2'].timestampEof
    assert obs.modelMonoTime == h.sm.logMonoTime['modelV2']
    assert obs.modelReceiveMonoTime == round(h.sm.recv_time['modelV2'] * 1e9)


def test_service_is_full_rate_log_only_not_qlog_or_control_input():
  service = SERVICE_LIST[SERVICE]
  assert service.should_log and service.decimation is None and service.frequency == 0


def test_stock_radar_and_params_remain_unchanged_while_observer_has_points(harness):
  h = harness
  before = h.cp.to_dict(), asdict(h.cp_sp)
  normal = RadarInterface(h.cp, h.cp_sp)
  assert normal.radar_off_can and normal.bosch_a is None
  results = []
  for index in range(10):
    packets = sweep(h.observer.parser_bus, index, 1_000_000_000 + index * DT_NS)
    stock_result = normal.update(packets)
    if stock_result is not None:
      assert not stock_result.points and not any(stock_result.errors.to_dict().values())
      results.append(stock_result)
    event = step(h, index)
  assert len(results) == 2 and event.boschARadarObservation.data.points
  assert (h.cp.to_dict(), asdict(h.cp_sp)) == before
  assert h.cp.radarUnavailable and not h.cp.openpilotLongitudinalControl and h.cp.pcmCruise


def test_never_seen_heartbeat_is_invalid_and_repetition_does_not_create_a_sample(harness):
  h = harness
  event = h.observer.message(h.state, h.sm)
  obs = event.boschARadarObservation
  assert not event.valid and obs.stale and not obs.hasSample and not obs.sampleUpdated
  assert obs.sequence == obs.sampleMonoTime == obs.statusSampleMonoTime == obs.triggerMonoTime == 0
  assert not obs.statusComplete
  assert h.observer.message(h.state, h.sm) is None
  h.clock.now += HEARTBEAT_NS
  repeated = h.observer.message(h.state, h.sm)
  assert not repeated.valid and repeated.boschARadarObservation.sequence == 0


def test_stale_empty_snapshot_discards_retained_statuses_and_recovers(harness):
  h = harness
  step(h, 0)
  live = step(h, 1)
  assert live.valid and live.boschARadarObservation.data.points
  h.clock.now = 1_400_010_000
  h.observer.update([(1_400_000_000, [])], h.state, h.clock.now)
  event = h.observer.message(h.state, h.sm)
  obs = event.boschARadarObservation
  assert not event.valid and obs.stale and obs.hasSample and obs.sampleUpdated
  assert obs.data.errors.radarUnavailableTemporary
  assert not obs.data.points and not obs.pointStatus and obs.statusComplete
  assert obs.statusDiscardedCount == 1 and obs.statusMissingCount == 0
  old_sequence, old_sample, old_car_state = obs.sequence, obs.sampleMonoTime, obs.sampleCarStateMonoTime
  h.clock.now += HEARTBEAT_NS
  held = h.observer.message(h.state, h.sm).boschARadarObservation
  assert not held.sampleUpdated and held.sequence == old_sequence and held.sampleMonoTime == old_sample
  assert held.sampleCarStateMonoTime == old_car_state
  for index in (21, 22):
    recovered = step(h, index)
  assert recovered.valid and recovered.boschARadarObservation.data.points


def test_invalid_carstate_is_distinct_from_valid_radar_sample(harness):
  h = harness
  h.state.canValid = False
  event = step(h, 0)
  assert not event.valid and event.boschARadarObservation.sampleValid
  assert not event.boschARadarObservation.carStateValid


def test_missing_status_makes_pairing_invalid_even_when_radar_sample_is_valid(harness, monkeypatch):
  h = harness
  step(h, 0)
  real_update = h.observer.decoder.update

  def missing_status(packets):
    result = real_update(packets)
    h.observer.decoder.point_statuses = {}
    return result

  monkeypatch.setattr(h.observer.decoder, 'update', missing_status)
  event = step(h, 1)
  obs = event.boschARadarObservation
  assert not event.valid and obs.sampleValid and not obs.statusComplete
  assert obs.statusMissingCount == 1 and not obs.pointStatus


def test_decoder_exception_is_latched_invalid_without_mutating_authority(harness, monkeypatch):
  h = harness
  before = h.cp.to_dict(), asdict(h.cp_sp)
  step(h, 0)
  step(h, 1)

  def broken(_packets):
    raise RuntimeError('synthetic parser failure')

  monkeypatch.setattr(h.observer.decoder, 'update', broken)
  event = step(h, 2)
  obs = event.boschARadarObservation
  assert not event.valid and not obs.hasSample and obs.stale
  assert obs.sequence == 2 and obs.exceptionCount == 1 and obs.exceptionType == 'RuntimeError'
  assert obs.egoSpeedUsed == 0 and obs.sampleCarStateMonoTime == 0
  assert not obs.data.points and not obs.pointStatus
  assert h.observer.decoder is None
  h.observer.update([], h.state, h.clock.now)
  assert h.observer.exception_count == 1
  assert (h.cp.to_dict(), asdict(h.cp_sp)) == before


def test_unselected_observer_cannot_be_constructed():
  cp, cp_sp = parameters(enabled=False)
  with pytest.raises(ValueError, match='stock-ACC'):
    BoschAObservation(cp, cp_sp)


class Publisher:
  def __init__(self):
    self.sent = []

  def send(self, service, message):
    self.sent.append((service, message.to_bytes()))


def test_actual_card_publisher_separates_observation_from_stock_radar_and_status(harness):
  h = harness
  step(h, 0)
  now = 1_000_000_000 + DT_NS
  h.clock.now = now + 10_000
  h.observer.update(sweep(h.observer.parser_bus, 1, now), h.state, h.clock.now)
  card = object.__new__(Car)
  card.CP, card.CP_SP = h.cp, h.cp_sp
  card.sm = h.sm
  card.pm, card.observation_pm = Publisher(), Publisher()
  card.bosch_a_observer = h.observer
  card.rk = SimpleNamespace(remaining=0.)
  card.can_rcv_cum_timeout_counter = 0
  card.last_actuators_output = structs.CarControl.Actuators()
  card.RI = RadarInterface(h.cp, h.cp_sp)
  card.state_publish(h.state, custom.CarStateSP.new_message(), structs.RadarData())
  assert len(card.observation_pm.sent) == 1 and card.observation_pm.sent[0][0] == SERVICE
  for service, data in card.pm.sent:
    assert service != SERVICE
    with log.Event.from_bytes(data) as msg:
      if service == 'radarTracks':
        assert not msg.radarTracks.points
      if service == 'carStateSP':
        assert msg.carStateSP.radarTracksMonoTime == 0 and not msg.carStateSP.radarPointStatus
      if service == 'carState':
        car_state_stamp = msg.logMonoTime
  with log.Event.from_bytes(card.observation_pm.sent[0][1]) as msg:
    assert msg.valid and msg.boschARadarObservation.data.points
    assert msg.boschARadarObservation.sampleCarStateMonoTime == car_state_stamp


def construct_car(monkeypatch, tmp_path, *, enabled=True, master=True, controller=True, late_icbm=False,
                  fail_observer=False, fail_publisher=False):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  params = Params()
  params.put_bool('OpenpilotEnabledToggle', master, block=True)
  params.put_bool('AlphaLongitudinalEnabled', False, block=True)
  params.put_bool('HondaEnforceStockLongitudinal', True, block=True)
  params.put_bool('BoschARadarObservationEnabled', enabled, block=True)
  cp, cp_sp = parameters(enabled)
  ci = SimpleNamespace(CP=cp, CP_SP=cp_sp, CC=SimpleNamespace() if controller else None,
                       CS=SimpleNamespace())
  calls, errors = [], []

  def no_send(*args, **kwargs):
    raise AssertionError('No message or CAN transmission in constructor test')

  def publisher(services):
    calls.append(('publisher', tuple(services)))
    if services == [SERVICE] and fail_publisher:
      raise RuntimeError('synthetic optional publisher failure')
    return SimpleNamespace(sock={'sendcan': SimpleNamespace(send=no_send)}, send=no_send)

  def observer(final_cp, final_cp_sp):
    calls.append(('observer', bool(final_cp.passive), bool(final_cp_sp.pcmCruiseSpeed)))
    if fail_observer:
      raise RuntimeError('synthetic observer construction failure')
    return BoschAObservation(final_cp, final_cp_sp)

  def setup(*args):
    if late_icbm:
      cp_sp.pcmCruiseSpeed = False

  def radar(final_cp, final_cp_sp):
    calls.append(('radar', bool(final_cp.radarUnavailable)))
    return RadarInterface(final_cp, final_cp_sp)

  monkeypatch.setattr(card_module.messaging, 'sub_sock', lambda *a, **k: object())
  monkeypatch.setattr(card_module.messaging, 'SubMaster', lambda *a, **k: SimpleNamespace())
  monkeypatch.setattr(card_module.messaging, 'PubMaster', publisher)
  monkeypatch.setattr(card_module.messaging, 'recv_one_retry', lambda *a: SimpleNamespace(can=[object()]))
  monkeypatch.setattr(card_module, 'Ratekeeper', lambda *a, **k: SimpleNamespace(remaining=0.))
  monkeypatch.setattr(card_module.sunnypilot_interfaces, 'setup_interfaces', setup)
  monkeypatch.setattr(card_module, 'get_car', lambda *a: ci)
  monkeypatch.setattr(card_module, 'BoschAObservation', observer)
  monkeypatch.setattr(card_module.cloudlog, 'error', lambda *a, **k: errors.append(a))
  monkeypatch.setitem(card_module.interfaces, cp.carFingerprint, SimpleNamespace(RadarInterface=radar))
  return Car(), calls, errors, params


def test_actual_startup_uses_separate_observer_after_final_stock_gate(monkeypatch, tmp_path):
  card, calls, errors, params = construct_car(monkeypatch, tmp_path)
  assert not errors and card.bosch_a_observer is not None
  assert ('observer', False, True) in calls
  assert ('publisher', (SERVICE,)) in calls
  assert card.RI.radar_off_can and card.RI.bosch_a is None
  with structs.car.CarParams.from_bytes(params.get('CarParams')) as saved:
    assert not saved.openpilotLongitudinalControl and saved.pcmCruise and saved.radarUnavailable


@pytest.mark.parametrize('setting', ['disabled', 'master_off', 'no_controller', 'late_icbm'])
def test_final_startup_ineligible_mode_never_constructs_optional_parser_or_publisher(monkeypatch, tmp_path, setting):
  card, calls, errors, _ = construct_car(monkeypatch, tmp_path, enabled=setting != 'disabled',
                                        master=setting != 'master_off', controller=setting != 'no_controller',
                                        late_icbm=setting == 'late_icbm')
  assert card.bosch_a_observer is None and card.observation_pm is None and not errors
  assert not any(call[0] == 'observer' for call in calls)
  assert ('publisher', (SERVICE,)) not in calls
  assert card.RI.radar_off_can and card.RI.bosch_a is None


@pytest.mark.parametrize('failure', ['observer', 'publisher'])
def test_optional_constructor_failure_does_not_fail_card_startup(monkeypatch, tmp_path, failure):
  card, _, errors, params = construct_car(monkeypatch, tmp_path, fail_observer=failure == 'observer',
                                         fail_publisher=failure == 'publisher')
  assert card.bosch_a_observer is None and card.observation_pm is None and len(errors) == 1
  assert not card.CP.openpilotLongitudinalControl and params.get('CarParams')


@pytest.mark.parametrize('failure', ['message', 'send'])
def test_optional_publication_failure_leaves_normal_state_publication_running(harness, monkeypatch, failure):
  h = harness
  card = object.__new__(Car)
  card.CP, card.CP_SP = h.cp, h.cp_sp
  card.sm, card.pm, card.observation_pm = h.sm, Publisher(), Publisher()
  card.rk = SimpleNamespace(remaining=0.)
  card.can_rcv_cum_timeout_counter = 0
  card.last_actuators_output = structs.CarControl.Actuators()
  card.RI = RadarInterface(h.cp, h.cp_sp)
  card.bosch_a_observer = h.observer
  errors = []
  monkeypatch.setattr(card_module.cloudlog, 'error', lambda *a, **k: errors.append(a))

  def broken(*args, **kwargs):
    raise ValueError('synthetic optional serialization/send failure')

  if failure == 'message':
    monkeypatch.setattr(h.observer, 'message', broken)
  else:
    monkeypatch.setattr(card.observation_pm, 'send', broken)
  card.state_publish(h.state, custom.CarStateSP.new_message(), structs.RadarData())
  assert card.bosch_a_observer is None and card.observation_pm is None and len(errors) == 1
  count = len(card.pm.sent)
  card.state_publish(h.state, custom.CarStateSP.new_message(), structs.RadarData())
  assert len(card.pm.sent) == 2 * count and len(errors) == 1


def test_native_default_is_off_and_key_is_forwarded(tmp_path, monkeypatch):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  params = Params()
  assert params.get('BoschARadarObservationEnabled', return_default=True) is False
  settings = {key: value for item in initialize_params(params) for key, value in item.items()}
  assert settings['BoschARadarObservationEnabled'] is False
  params.put_bool('BoschARadarObservationEnabled', True, block=True)
  settings = {key: value for item in initialize_params(params) for key, value in item.items()}
  assert settings['BoschARadarObservationEnabled'] is True


def test_trigger_timestamp_is_not_relabelled_as_last_can_packet_or_model_receipt(harness):
  h = harness
  trigger_ns = 1_000_000_000
  frames = sweep(h.observer.parser_bus, 0, trigger_ns)[0][1]
  assert frames[-2].address == 0x2FF and frames[-1].address == 0x297
  packets = [(trigger_ns, frames[:-1]), (trigger_ns + 5_000_000, frames[-1:])]
  h.clock.now = trigger_ns + 6_000_000
  h.observer.update(packets, h.state, h.clock.now)
  event = h.observer.message(h.state, h.sm)
  obs = event.boschARadarObservation
  assert event.valid
  assert obs.triggerMonoTime == obs.sourceCanFirstMonoTime == trigger_ns
  assert obs.sourceCanMonoTime == trigger_ns + 5_000_000
  assert obs.batchReceiveMonoTime == trigger_ns + 6_000_000
  assert obs.modelReceiveMonoTime != obs.triggerMonoTime


def test_model_receiver_health_is_logged_separately_from_radar_validity(harness):
  h = harness
  h.sm.valid = {'modelV2': False}
  h.sm.alive = {'modelV2': False}
  h.sm.freq_ok = {'modelV2': False}
  event = step(h, 0)
  obs = event.boschARadarObservation
  assert event.valid and obs.sampleValid
  assert obs.modelSeen and not obs.modelValid and not obs.modelAlive and not obs.modelFreqOk


def test_actual_state_update_optional_failure_preserves_stock_state_return(harness, monkeypatch):
  h = harness
  card = object.__new__(Car)
  card.CP, card.CP_SP = h.cp, h.cp_sp
  card.CI = SimpleNamespace(update=lambda packets: (h.state, structs.CarStateSP()))
  card.RI = RadarInterface(h.cp, h.cp_sp)
  card.bosch_a_observer, card.observation_pm = h.observer, Publisher()
  card.can_sock = object()
  card.firmware_config = None
  card.can_rcv_cum_timeout_counter = 0
  card.sm = h.sm
  card.sm['carControl'] = structs.CarControl.new_message()
  card.sm['longitudinalPlanSP'] = custom.LongitudinalPlanSP.new_message()
  card.sm.update = lambda *args: None
  card.CC_prev = structs.CarControl.new_message()
  card.CS_prev = h.state
  card.initial_cruise_speed = 100
  card.is_metric = True
  card.v_cruise_helper = SimpleNamespace(update_speed_limit_assist=lambda *args: None,
                                       update_v_cruise=lambda *args: None,
                                       v_cruise_kph=100., v_cruise_cluster_kph=100.)
  errors = []
  monkeypatch.setattr(card_module.cloudlog, 'error', lambda *a, **k: errors.append(a))
  packet = card_module.messaging.new_message('can', size=0).to_bytes()
  monkeypatch.setattr(card_module.messaging, 'drain_sock_raw', lambda *a, **k: [packet])

  def broken(*args):
    raise RuntimeError('unexpected wrapper failure')

  monkeypatch.setattr(h.observer, 'update', broken)
  state, state_sp, radar = card.state_update()
  assert state is h.state and radar is None
  assert state_sp.schema.node.displayName.endswith('CarStateSP')
  assert card.bosch_a_observer is None and card.observation_pm is None and len(errors) == 1
  assert not card.CP.openpilotLongitudinalControl and card.CP.radarUnavailable
