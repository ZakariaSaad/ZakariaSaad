"""Synthetic local-log validation; no CAN transmission or remote reads."""
import bz2
import json
import shutil
import subprocess

import pytest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.interfaces import _initialize_honda
from openpilot.cereal import messaging
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.tools.bosch_a import check_observation_log as checker


def event(name, stamp, size=None):
  result = messaging.new_message(name, size)
  result.logMonoTime, result.valid = stamp, True
  return result


def complete_events(count=2):
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.HONDA_INSIGHT, fingerprint, [], False, False, False)
  sp = CarInterface.get_params_sp(cp, CAR.HONDA_INSIGHT, fingerprint, [], False, False, False)
  _initialize_honda(cp, sp, {'HondaEnforceStockLongitudinal': 1, 'BoschALongitudinalEnabled': 0,
                            'BoschARadarObservationEnabled': 1})
  messages = [event('carParams', 1), event('carParamsSP', 2)]
  messages[0].carParams = cp
  messages[1].carParamsSP = convert_to_capnp(sp)
  hashes = checker.source_hashes()
  for number in range(count):
    source = 10_000_000_000 + number * 70_000_000
    frame_id, eof = 17 + number, source - 30_000_000
    model = event('modelV2', source - 10_000_000)
    model.modelV2.frameId, model.modelV2.timestampEof = frame_id, eof
    camera = event('narrowRoadCameraState', eof + 1)
    camera.narrowRoadCameraState.frameId, camera.narrowRoadCameraState.timestampEof = frame_id, eof
    index = event('qNarrowRoadEncodeIdx', eof + 2)
    index.qNarrowRoadEncodeIdx.frameId, index.qNarrowRoadEncodeIdx.timestampEof = frame_id, eof
    index.qNarrowRoadEncodeIdx.segmentNum, index.qNarrowRoadEncodeIdx.segmentId = 0, number
    raw = event('can', source, 1)
    raw.can[0].address, raw.can[0].src, raw.can[0].dat = 0x2FF, 2, b'\x00' * 8
    observation = event('boschARadarObservation', source + 5_000_000)
    obs = observation.boschARadarObservation
    obs.enabled = obs.hasSample = obs.sampleUpdated = obs.sampleValid = obs.carStateValid = obs.statusComplete = True
    obs.sequence = number + 1
    obs.sampleMonoTime = obs.statusSampleMonoTime = obs.decodeEndMonoTime = source + 4_000_000
    obs.sourceCanMonoTime = obs.sourceCanFirstMonoTime = obs.triggerMonoTime = source
    obs.batchReceiveMonoTime, obs.decodeStartMonoTime = source + 2_000_000, source + 3_000_000
    obs.publicationMonoTime = observation.logMonoTime
    obs.sampleCarStateMonoTime, obs.egoSpeedUsed = source + 4_500_000, 0.
    obs.data.points = [{'trackId': 7, 'dRel': 10., 'yRel': 0., 'vRel': 0.,
                        'deprecated': {'measured': True, 'aRel': float('nan'), 'yvRel': float('nan')}}]
    obs.pointStatus = [{'trackId': 7, 'objectStatus': 7}]
    obs.decoderSha256, obs.dbcSha256 = hashes['decoder'], hashes['dbc']
    obs.parserBus = 2
    obs.modelSeen = obs.modelValid = obs.modelAlive = obs.modelFreqOk = True
    obs.modelMonoTime, obs.modelReceiveMonoTime = model.logMonoTime, source + 1_000_000
    obs.modelFrameId, obs.modelTimestampEof = frame_id, eof
    obs.receiver = 'card'
    tracks = event('radarTracks', source + 6_000_000)
    state = event('carState', source + 4_500_000)
    state.carState.standstill, state.carState.vEgo, state.carState.gearShifter = True, 0., 'park'
    controls = event('carControl', source + 7_000_000)
    outgoing = event('sendcan', source + 8_000_000, 1)
    outgoing.sendcan[0].address, outgoing.sendcan[0].src, outgoing.sendcan[0].dat = 0xE4, 0, b'\x00' * 5
    messages.extend([model, camera, index, raw, observation, tracks, state, controls, outgoing])
  return messages


def observations(events):
  return [item for item in events if item.which() == 'boschARadarObservation']


def check(events, **kwargs):
  return checker.inspect_events(events, videos={0: {'frames': 2}}, min_healthy_seconds=0., **kwargs)


def test_complete_capture_pairs_points_model_camera_and_raw_trigger():
  result = check(complete_events(), require_measured_points=True)
  assert result['result'] == 'pass'
  assert result['unique_samples'] == result['valid_updated_samples'] == 2
  assert result['measured_point_records'] == 2 and result['coasted_point_records'] == 0
  assert result['sample_cadence']['mean_hz'] == pytest.approx(1 / .07)
  assert result['exact_model_camera_index_frames'] == result['model_frames_in_supplied_video'] == 2
  assert result['forbidden_send_intents'] == []


@pytest.mark.parametrize('field', ['statusSampleMonoTime', 'publicationMonoTime', 'triggerMonoTime', 'decoderSha256'])
def test_mispaired_or_wrong_source_metadata_does_not_pass(field):
  events = complete_events()
  obs = observations(events)[0].boschARadarObservation
  setattr(obs, field, 'not-the-source-hash' if field == 'decoderSha256' else 3)
  assert check(events)['result'] != 'pass'


def test_stale_status_identity_is_rejected():
  events = complete_events()
  observations(events)[0].boschARadarObservation.pointStatus[0].trackId = 99
  result = check(events)
  assert result['result'] == 'fail'
  assert any('stale IDs' in error for error in result['errors'])


def test_unknown_acceleration_nan_is_permitted_but_geometry_nan_is_not():
  events = complete_events()
  assert check(events)['result'] == 'pass'
  observations(events)[0].boschARadarObservation.data.points[0].dRel = float('nan')
  assert check(events)['result'] == 'fail'


def test_empty_scene_is_not_a_decoder_failure_and_target_check_is_explicit():
  events = complete_events()
  for item in observations(events):
    item.boschARadarObservation.data.points = []
    item.boschARadarObservation.pointStatus = []
  assert check(events)['result'] == 'pass'
  result = check(events, require_measured_points=True)
  assert result['result'] == 'incomplete'
  assert not result['errors']


def test_coasted_samples_do_not_count_as_measured():
  events = complete_events()
  for item in observations(events):
    item.boschARadarObservation.data.points[0].deprecated.measured = False
  result = check(events, require_measured_points=True)
  assert result['measured_point_records'] == 0 and result['coasted_point_records'] == 2
  assert result['result'] == 'incomplete'


@pytest.mark.parametrize('service', ['modelV2', 'can', 'carParams', 'carParamsSP', 'qNarrowRoadEncodeIdx', 'carControl', 'carState',
                                   'sendcan', 'radarTracks'])
def test_missing_required_full_log_stream_is_incomplete(service):
  events = [item for item in complete_events() if item.which() != service]
  assert check(events)['result'] == 'incomplete'


def test_stock_received_acc_is_not_mistaken_for_an_outgoing_command():
  events = complete_events()
  raw = next(item for item in events if item.which() == 'can')
  raw.can = [raw.can[0].to_dict(), {'address': 0x1DF, 'src': 1, 'dat': b'\x00' * 8}]
  assert check(events)['result'] == 'pass'


@pytest.mark.parametrize('address,data', [(0x1DF, b'\x00' * 8), (0x1EF, b'\x00' * 8),
                                         (0x18DAB0F1, bytes.fromhex('0328830300000000'))])
def test_forbidden_send_intents_fail(address, data):
  events = complete_events()
  tx = next(item for item in events if item.which() == 'sendcan').sendcan[0]
  tx.address, tx.dat = address, data
  result = check(events)
  assert result['result'] == 'fail'
  assert len(result['forbidden_send_intents']) == 1


def test_normal_diagnostic_identification_is_not_a_disable_request():
  events = complete_events()
  tx = next(item for item in events if item.which() == 'sendcan').sendcan[0]
  tx.address, tx.dat = 0x18DAB0F1, bytes.fromhex('0322f19000000000')
  assert check(events)['result'] == 'pass'


@pytest.mark.parametrize('kind', ['active', 'radar_points', 'longitudinal_configuration'])
def test_control_authority_or_observation_leak_fails(kind):
  events = complete_events()
  if kind == 'active':
    next(item for item in events if item.which() == 'carControl').carControl.longActive = True
  elif kind == 'radar_points':
    next(item for item in events if item.which() == 'radarTracks').radarTracks.points = [{'trackId': 7, 'dRel': 10.}]
  else:
    next(item for item in events if item.which() == 'carParams').carParams.openpilotLongitudinalControl = True
  assert check(events)['result'] == 'fail'


def test_video_index_must_address_a_decoded_frame():
  events = complete_events()
  result = checker.inspect_events(events, videos={0: {'frames': 1}})
  assert result['result'] == 'fail'


def test_missing_video_indexes_or_moving_capture_are_incomplete():
  events = complete_events()
  assert checker.inspect_events(events, videos={0: {'frames': 3}})['result'] == 'incomplete'
  next(item for item in events if item.which() == 'carState').carState.standstill = False
  result = check(events)
  assert result['result'] == 'incomplete' and not result['parked_state']['all_standstill']


def test_post_sample_exception_can_clear_snapshot_without_resetting_sequence():
  events = complete_events()
  prior = observations(events)[-1]
  heartbeat = event('boschARadarObservation', prior.logMonoTime + 1_000_000_000)
  heartbeat.boschARadarObservation = prior.boschARadarObservation.to_dict()
  obs = heartbeat.boschARadarObservation
  heartbeat.valid = False
  obs.hasSample = obs.sampleUpdated = obs.sampleValid = obs.statusComplete = False
  obs.stale, obs.exceptionCount, obs.exceptionType = True, 1, 'ValueError'
  for field in ('sampleMonoTime', 'statusSampleMonoTime', 'sourceCanMonoTime', 'sourceCanFirstMonoTime',
                'triggerMonoTime', 'batchReceiveMonoTime', 'decodeStartMonoTime', 'decodeEndMonoTime', 'sampleCarStateMonoTime'):
    setattr(obs, field, 0)
  obs.publicationMonoTime = heartbeat.logMonoTime
  obs.data.points, obs.pointStatus = [], []
  obs.data.errors.radarUnavailableTemporary = True
  events.append(heartbeat)
  result = check(events)
  assert not result['errors']
  assert result['decoder_error_publications']['observer:ValueError'] == 1
  assert result['notes']


@pytest.mark.parametrize('value', ['https://connect.comma.ai/private/route', 'cd:/route', 'file:///tmp/rlog'])
def test_remote_or_service_identifiers_are_rejected(value):
  with pytest.raises(ValueError, match='local'):
    checker.local_file(value)


def test_real_local_logreader_round_trip_and_no_qlog_fallback(tmp_path, monkeypatch):
  rlog = tmp_path / 'rlog.bz2'
  rlog.write_bytes(bz2.compress(b''.join(item.to_bytes() for item in complete_events())))
  video = tmp_path / 'qcamera.ts'
  video.write_bytes(b'synthetic-video-metadata-test')
  monkeypatch.setattr(checker, 'probe_video', lambda path: {'frames': 2, 'path': str(path)})
  output = tmp_path / 'report.json'
  assert checker.main([str(rlog), '--qcamera', f'0={video}', '--require-measured-points', '--min-healthy-seconds', '0',
                       '--output', str(output)]) == 0
  report = json.loads(output.read_text())
  assert report['rlogs'][0]['sha256'] == checker.sha256(rlog)
  qlog = tmp_path / 'qlog.bz2'
  qlog.write_bytes(rlog.read_bytes())
  assert checker.main([str(qlog), '--output', str(output)]) == 2
  assert 'full rlogs' in json.loads(output.read_text())['incomplete'][0]


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='Local ffmpeg tools unavailable')
def test_actual_video_probe_counts_local_synthetic_frames(tmp_path):
  path = tmp_path / 'qcamera.ts'
  subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=32x32:r=20',
                  '-frames:v', '20', '-c:v', 'mpeg2video', '-f', 'mpegts', str(path)], check=True)
  metadata = checker.probe_video(path)
  assert metadata['frames'] == 20 and metadata['width'] == metadata['height'] == 32
  assert metadata['sha256'] == checker.sha256(path)


def test_default_health_window_rejects_two_good_samples_and_accepts_sustained_parked_capture():
  short = checker.inspect_events(complete_events(), videos={0: {'frames': 2}})
  assert short['result'] == 'incomplete' and short['longest_healthy_window_seconds'] == pytest.approx(.07)
  full = checker.inspect_events(complete_events(160), videos={0: {'frames': 160}}, require_measured_points=True)
  assert full['result'] == 'pass' and full['longest_healthy_window_seconds'] == pytest.approx(159 * .07)


def test_two_good_samples_followed_by_persistent_failure_cannot_pass_default_check():
  events = complete_events(160)
  for item in observations(events)[2:]:
    obs = item.boschARadarObservation
    item.valid, obs.sampleValid = False, False
    obs.data.errors.canError = True
  result = checker.inspect_events(events, videos={0: {'frames': 160}})
  assert result['result'] == 'incomplete' and result['longest_healthy_window_seconds'] == pytest.approx(.07)


def test_fabricated_stale_flag_is_rejected_even_when_outer_valid_matches_it():
  events = complete_events()
  item = observations(events)[0]
  item.valid, item.boschARadarObservation.stale = False, True
  result = check(events)
  assert result['result'] == 'fail'
  assert any('stale metadata' in error for error in result['errors'])


def test_measured_points_only_in_invalid_samples_do_not_satisfy_target_check():
  events = complete_events()
  for item in observations(events):
    item.valid, item.boschARadarObservation.sampleValid = False, False
    item.boschARadarObservation.data.errors.canError = True
  result = check(events, require_measured_points=True)
  assert result['result'] == 'incomplete'
  assert result['measured_point_records'] == 2 and result['valid_measured_point_records'] == 0


@pytest.mark.parametrize('field,value', [('sampleCarStateMonoTime', 123), ('egoSpeedUsed', 2.)])
def test_missing_or_wrong_exact_car_state_join_does_not_pass(field, value):
  events = complete_events()
  setattr(observations(events)[0].boschARadarObservation, field, value)
  assert check(events)['result'] != 'pass'


def test_sequence_skip_reports_lost_observation_publication():
  events = complete_events()
  observations(events)[1].boschARadarObservation.sequence = 3
  result = check(events)
  assert result['result'] == 'incomplete' and result['skipped_sample_sequences'] == [[2, 2]]


@pytest.mark.parametrize('kind', ['narrowRoadCameraState', 'qNarrowRoadEncodeIdx', 'can'])
def test_invalid_camera_index_or_referenced_can_cannot_qualify_recording(kind):
  events = complete_events()
  for item in events:
    if item.which() == kind:
      item.valid = False
  result = check(events)
  assert result['result'] == 'incomplete'
  assert result['invalid_events'][kind] == 2
  assert result['longest_healthy_window_seconds'] == 0


@pytest.mark.parametrize('destination', ['rlog', 'video', 'symlink', 'hardlink', 'malformed_video'])
def test_output_collision_preserves_all_inputs_and_reports_to_stdout(tmp_path, monkeypatch, capsys, destination):
  rlog, video = tmp_path / 'rlog.bz2', tmp_path / 'qcamera.ts'
  original_log, original_video = b'preserved rlog bytes', b'preserved video bytes'
  rlog.write_bytes(original_log)
  video.write_bytes(original_video)
  if destination in ('symlink', 'hardlink'):
    output = tmp_path / 'summary.json'
    if destination == 'symlink':
      output.symlink_to(video)
    else:
      output.hardlink_to(rlog)
  else:
    output = rlog if destination == 'rlog' else video

  def must_not_read(*args, **kwargs):
    raise AssertionError('Input/output aliases must be rejected before reading or probing inputs')

  monkeypatch.setattr(checker, 'LogReader', must_not_read)
  monkeypatch.setattr(checker, 'probe_video', must_not_read)
  video_arg = str(video) if destination == 'malformed_video' else f'0={video}'
  assert checker.main([str(rlog), '--qcamera', video_arg, '--output', str(output)]) == 2
  report = json.loads(capsys.readouterr().out)
  assert report['result'] == 'incomplete' and 'aliases an input' in report['incomplete'][0]
  assert rlog.read_bytes() == original_log and video.read_bytes() == original_video
