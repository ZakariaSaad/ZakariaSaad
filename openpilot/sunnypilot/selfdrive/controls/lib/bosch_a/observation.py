"""Receive-only stock-ACC observation, isolated from all control input streams.

Raw CAN and this service share the existing logger clock. The source timestamp
is a software CAN Event timestamp, not an independently timed physical frame.
Model receipt metadata describes card's existing SubMaster, never radard's.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import opendbc
from opendbc.car import structs
from opendbc.car.honda import bosch_a_radar
from opendbc.car.honda.bosch_a_observation import observation_enabled
from openpilot.cereal import messaging


SERVICE = 'boschARadarObservation'
HEARTBEAT_NS = 1_000_000_000
STALE_NS = round(bosch_a_radar.BOSCH_A_STALE_S * 1e9)


class BoschAObservation:
  """Own one parser and immutable result snapshots; never alter CP or RI."""

  def __init__(self, CP, CP_SP, clock=time.monotonic_ns):
    if not observation_enabled(CP, CP_SP):
      raise ValueError('Bosch-A observation requires eligible stock-ACC configuration')
    self.CP, self.CP_SP, self.clock = CP, CP_SP, clock
    self.parser_bus = bosch_a_radar.CanBus(CP).camera
    self.decoder = None
    self.decoder_sha256 = self.dbc_sha256 = ''
    self.sequence = 0
    self.has_sample = self.sample_updated = self.sample_valid = False
    self.status_complete = False
    self.sample_ns = self.source_ns = self.first_source_ns = self.trigger_ns = 0
    self.ego_speed = 0.
    self.sample_car_state_ns = 0
    self.receive_ns = self.decode_start_ns = self.decode_end_ns = 0
    self.last_publish_ns = 0
    self.statuses: dict[int, int] = {}
    self.status_discarded = self.status_missing = 0
    self.data = structs.RadarData().to_dict()
    self.exception_count = 0
    self.exception_type = ''
    self.force_publish = True
    try:
      self.decoder_sha256 = hashlib.sha256(Path(bosch_a_radar.__file__).read_bytes()).hexdigest()
      dbc = Path(opendbc.__file__).parent / 'dbc' / f'{bosch_a_radar.BOSCH_A_DBC_NAME}.dbc'
      self.dbc_sha256 = hashlib.sha256(dbc.read_bytes()).hexdigest()
      self.decoder = bosch_a_radar.BoschARadar(CP, CP_SP)
    except Exception as error:
      self._failed(error)

  def _failed(self, error):
    # Logging must not gain control authority or crash card on parser failure.
    self.exception_count += 1
    self.exception_type = type(error).__name__
    self.has_sample = self.sample_updated = self.sample_valid = False
    self.status_complete = False
    self.sample_ns = self.source_ns = self.first_source_ns = self.trigger_ns = 0
    self.ego_speed = 0.
    self.sample_car_state_ns = 0
    self.receive_ns = self.decode_start_ns = self.decode_end_ns = 0
    self.statuses = {}
    self.status_discarded = self.status_missing = 0
    empty = structs.RadarData()
    empty.errors.radarUnavailableTemporary = True
    self.data = empty.to_dict()
    # Latch the failure until restart rather than run a possibly half-mutated
    # decoder or repeatedly allocate parsers inside the control process.
    self.decoder = None
    self.force_publish = True

  def update(self, can_packets, car_state, receive_ns: int) -> None:
    self.sample_updated = False
    if self.decoder is None or not observation_enabled(self.CP, self.CP_SP):
      return
    try:
      self.decoder.v_ego = car_state.vEgo
      start = self.clock()
      result = self.decoder.update(can_packets)
      end = self.clock()
      if result is None:
        return
      self.sequence += 1
      self.has_sample = self.sample_updated = True
      self.sample_ns = self.decode_end_ns = end
      self.ego_speed = float(car_state.vEgo)
      self.sample_car_state_ns = 0
      self.receive_ns, self.decode_start_ns = receive_ns, start
      self.source_ns = int(self.decoder.rcp._last_update_nanos)
      self.first_source_ns = int(can_packets[0][0]) if can_packets else self.source_ns
      self.trigger_ns = int(self.decoder.rcp.ts_nanos[bosch_a_radar.BOSCH_A_TRIGGER_MSG]['FRAME_IDX'])
      # to_dict copies numeric values instead of retaining mutable decoder points.
      self.data = result.to_dict()
      self.sample_valid = not any(result.errors.to_dict().values())
      point_ids = {int(point.trackId) for point in result.points}
      self.statuses = {track_id: int(status) for track_id, status in self.decoder.point_statuses.items()
                       if track_id in point_ids}
      # Status is the decoder's retained F0 status, which can precede this
      # sweep for a coasted point. Pairing identifies the snapshot, not a new
      # independent physical observation of every status nibble.
      self.status_discarded = len(set(self.decoder.point_statuses) - point_ids)
      self.status_missing = len(point_ids - set(self.statuses))
      self.status_complete = set(self.statuses) == point_ids
    except Exception as error:
      self._failed(error)

  def message(self, car_state, sm, *, car_state_mono_time: int = 0):
    now = self.clock()
    if not (self.sample_updated or self.force_publish or now - self.last_publish_ns >= HEARTBEAT_NS):
      return None
    stale = not self.has_sample or self.trigger_ns <= 0 or not 0 <= now - self.trigger_ns <= STALE_NS
    event = messaging.new_message(SERVICE)
    event.logMonoTime = now
    enabled = observation_enabled(self.CP, self.CP_SP)
    event.valid = bool(enabled and self.has_sample and self.sample_valid and self.status_complete and
                       car_state.canValid and not stale and not self.exception_type)
    obs = event.boschARadarObservation
    obs.enabled = enabled
    obs.hasSample, obs.sampleUpdated, obs.sequence = self.has_sample, self.sample_updated, self.sequence
    obs.sampleMonoTime = obs.statusSampleMonoTime = self.sample_ns
    if self.sample_updated:
      self.sample_car_state_ns = int(car_state_mono_time)
    obs.egoSpeedUsed, obs.sampleCarStateMonoTime = self.ego_speed, self.sample_car_state_ns
    obs.sourceCanMonoTime, obs.sourceCanFirstMonoTime = self.source_ns, self.first_source_ns
    obs.triggerMonoTime, obs.batchReceiveMonoTime = self.trigger_ns, self.receive_ns
    obs.decodeStartMonoTime, obs.decodeEndMonoTime = self.decode_start_ns, self.decode_end_ns
    obs.publicationMonoTime = now
    obs.sampleValid, obs.carStateValid, obs.stale = self.sample_valid, bool(car_state.canValid), stale
    obs.data = self.data
    rows = obs.init('pointStatus', len(self.statuses))
    for row, (track_id, status) in zip(rows, sorted(self.statuses.items()), strict=True):
      row.trackId, row.objectStatus = track_id, status
    obs.statusComplete = self.status_complete
    obs.statusDiscardedCount, obs.statusMissingCount = self.status_discarded, self.status_missing
    obs.exceptionCount, obs.exceptionType = self.exception_count, self.exception_type
    obs.decoderSha256, obs.dbcSha256 = self.decoder_sha256, self.dbc_sha256
    obs.parserBus = self.parser_bus
    obs.receiver = 'card'
    obs.modelSeen = bool(sm.seen.get('modelV2', False))
    obs.modelMonoTime = int(sm.logMonoTime.get('modelV2', 0))
    obs.modelReceiveMonoTime = round(sm.recv_time.get('modelV2', 0.) * 1e9)
    obs.modelValid = bool(sm.valid.get('modelV2', False))
    obs.modelAlive = bool(sm.alive.get('modelV2', False))
    obs.modelFreqOk = bool(sm.freq_ok.get('modelV2', False))
    if obs.modelSeen:
      obs.modelFrameId = int(sm['modelV2'].frameId)
      obs.modelTimestampEof = int(sm['modelV2'].timestampEof)
    self.last_publish_ns, self.force_publish = now, False
    self.sample_updated = False
    return event
