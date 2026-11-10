#!/usr/bin/env python3
"""Check a completed local parked recording; never contact a vehicle or service.

PASS establishes the listed recording-integrity checks, not calibration, physical
actuation, closed-loop suitability or every possible longitudinal CAN invariant.
Use the same source revision as the producer and supply complete rlogs/video.
"""
from __future__ import annotations

import argparse
import capnp
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import warnings

from opendbc.can.dbc import DBC
from opendbc.car import Bus
from opendbc.car.honda import bosch_a_radar
from opendbc.car.honda.bosch_a_observation import observation_enabled
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import CAR, DBC as HONDA_DBC
from openpilot.tools.lib.logreader import LogReader


def local_file(value: str) -> Path:
  if '://' in value or value.startswith('cd:/'):
    raise ValueError('Only existing local files are accepted; no URLs or route identifiers')
  path = Path(value).expanduser().resolve(strict=True)
  if not path.is_file():
    raise ValueError(f'Not a regular file: {path}')
  return path


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def source_hashes() -> dict[str, str]:
  parser = Path(bosch_a_radar.__file__).resolve()
  dbc = parser.parents[2] / 'dbc' / 'honda_bosch_a_radar.dbc'
  return {'decoder': sha256(parser), 'dbc': sha256(dbc)}


def probe_video(path: Path) -> dict:
  result = subprocess.run([
    'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
    '-show_entries', 'stream=codec_name,width,height,nb_read_frames', '-of', 'json', str(path),
  ], check=True, capture_output=True, text=True, timeout=120)
  streams = json.loads(result.stdout)['streams']
  if len(streams) != 1:
    raise ValueError('Expected one readable video stream')
  stream = streams[0]
  return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path),
          'frames': int(stream['nb_read_frames']), 'width': int(stream['width']),
          'height': int(stream['height']), 'codec': stream['codec_name']}


def _cadence(stamps: list[int]) -> dict:
  intervals = [(b - a) / 1e9 for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
  return {'count': len(stamps), 'median_interval_s': statistics.median(intervals) if intervals else None,
          'max_interval_s': max(intervals) if intervals else None,
          'mean_hz': (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else None}


def inspect_events(events, *, videos: dict[int, dict] | None = None, require_measured_points=False, min_healthy_seconds=10.) -> dict:
  """Inspect already read Events. Video metadata comes only from local ffprobe."""
  videos = videos or {}
  errors, incomplete, notes = set(), set(), set()
  counts, decoder_errors, invalid_events = Counter(), Counter(), Counter()
  cp_records, sp_records, observations = [], [], []
  can_records, valid_can_records, models, camera_frames, indexes, valid_indexes = {}, {}, {}, set(), {}, {}
  cc_count = 0
  valid_car_states = []
  car_states = {}
  forbidden = []
  dbc = DBC(HONDA_DBC[CAR.HONDA_INSIGHT][Bus.pt])
  acc_ids = {dbc.name_to_msg[name].address for name in ('ACC_CONTROL', 'ACC_CONTROL_ON')}
  for event in events:
    kind = event.which()
    counts[kind] += 1
    if not event.valid:
      invalid_events[kind] += 1
    stamp = int(event.logMonoTime)
    if kind == 'carParams':
      cp_records.append((stamp, event.carParams, bool(event.valid)))
    elif kind == 'carParamsSP':
      sp_records.append((stamp, event.carParamsSP, bool(event.valid)))
    elif kind == 'boschARadarObservation':
      observations.append((stamp, event.boschARadarObservation.to_dict(), bool(event.valid)))
    elif kind == 'can':
      can_records[stamp] = {(int(frame.address), int(frame.src)) for frame in event.can}
      if event.valid:
        valid_can_records[stamp] = can_records[stamp]
    elif kind == 'modelV2':
      models[stamp] = (int(event.modelV2.frameId), int(event.modelV2.timestampEof), bool(event.valid))
    elif kind == 'narrowRoadCameraState':
      if event.valid:
        camera_frames.add((int(event.narrowRoadCameraState.frameId), int(event.narrowRoadCameraState.timestampEof)))
    elif kind == 'qNarrowRoadEncodeIdx':
      index = event.qNarrowRoadEncodeIdx
      indexes[(int(index.frameId), int(index.timestampEof))] = (int(index.segmentNum), int(index.segmentId))
      if event.valid:
        valid_indexes[(int(index.frameId), int(index.timestampEof))] = (int(index.segmentNum), int(index.segmentId))
    elif kind == 'carControl':
      cc_count += 1
      if event.carControl.longActive:
        errors.add('carControl.longActive became true in the stock-only observation recording')
    elif kind == 'carState':
      state = event.carState
      car_states[stamp] = (bool(event.valid), bool(state.standstill), float(state.vEgo), float(state.cumLagMs))
      if event.valid:
        valid_car_states.append((bool(state.standstill), float(state.vEgo), str(state.gearShifter), float(state.cumLagMs)))
    elif kind == 'radarTracks' and len(event.radarTracks.points):
      errors.add('Normal radarTracks contains points; observation data must remain isolated')
    elif kind == 'carStateSP' and (len(event.carStateSP.radarPointStatus) or event.carStateSP.radarTracksMonoTime):
      errors.add('Control-path Bosch-A status carrier is populated in observation-only mode')
    elif kind == 'sendcan':
      for frame in event.sendcan:
        address, data = int(frame.address), bytes(frame.dat)
        if address in acc_ids:
          forbidden.append({'ns': stamp, 'address': hex(address), 'bus': int(frame.src), 'reason': 'openpilot ACC command'})
        # Honda radar disable is an ISO-TP single-frame UDS CommunicationControl
        # request. Identification/firmware reads and ordinary button messages
        # are not mislabelled as radar-disable commands.
        if address == 0x18DAB0F1 and len(data) >= 4 and data[0] >> 4 == 0:
          if data[1] == 0x28 and data[2] & 0x7F in (1, 2, 3):
            forbidden.append({'ns': stamp, 'address': hex(address), 'bus': int(frame.src), 'reason': 'radar communication disable'})

  if forbidden:
    errors.add('Forbidden longitudinal/disable send intents are present')
  if not cp_records or not sp_records:
    incomplete.add('Missing logged carParams/carParamsSP; include the complete parked segment containing configuration')
  else:
    for _, cp, cp_valid in cp_records:
      for _, sp, sp_valid in sp_records:
        if not cp_valid or not sp_valid or not observation_enabled(cp, sp):
          errors.add('Logged configuration is not the strict selected stock-ACC observation preset')
  expected_bus = CanBus(cp_records[0][1]).camera if cp_records else None
  if not cc_count:
    incomplete.add('No carControl records to check longitudinal authority')
  if not valid_car_states:
    incomplete.add('No valid carState records to check the parked condition')
  elif not all(row[0] for row in valid_car_states):
    incomplete.add('Valid carState records report non-standstill; this is not an entirely parked capture')
  if not counts['sendcan']:
    incomplete.add('No sendcan records; absence of forbidden commands cannot be checked')
  if not counts['radarTracks']:
    incomplete.add('No normal radarTracks records to check observation/control isolation')
  if not can_records:
    incomplete.add('No full-rate raw CAN records; qlog alone is insufficient')
  if not observations:
    incomplete.add('No boschARadarObservation events')

  expected = source_hashes()
  sequence_samples, valid_samples, matched_models, useful_context_frames = {}, [], set(), set()
  measured = coasted = valid_measured = healthy_model_contexts = 0
  previous_sequence = 0
  last_updated_sequence = None
  last_sample_signature = None
  source_missing, trigger_missing = set(), set()
  healthy_updates, skipped_sequences, decode_durations_ms = [], [], []
  for stamp, obs, valid in sorted(observations, key=lambda row: row[0]):
    sample = bool(obs.get('hasSample', False))
    sequence = int(obs.get('sequence', 0))
    points = obs.get('data', {}).get('points', [])
    statuses = obs.get('pointStatus', [])
    sample_stamp = int(obs.get('sampleMonoTime', 0))
    trigger = int(obs.get('triggerMonoTime', 0))
    sample_state = None
    sample_source_ok = False
    healthy_model_frame = None
    if not obs.get('enabled') or obs.get('receiver') != 'card':
      errors.add('Observation enabled/receiver metadata does not match its contract')
    if int(obs.get('publicationMonoTime', 0)) != stamp:
      errors.add('Observation publicationMonoTime does not match its Event timestamp')
    if obs.get('decoderSha256') != expected['decoder'] or obs.get('dbcSha256') != expected['dbc']:
      errors.add('Recorded parser/DBC hashes differ from the checker source revision')
    if expected_bus is not None and int(obs.get('parserBus', -1)) != expected_bus:
      errors.add('Recorded parser bus differs from the Insight camera-side bus')
    if sequence < previous_sequence:
      errors.add('Observation sequence regressed; do not mix independent startups in one check')
    previous_sequence = sequence
    expected_stale = not sample or trigger <= 0 or not 0 <= stamp - trigger <= round(bosch_a_radar.BOSCH_A_STALE_S * 1e9)
    if bool(obs.get('stale')) != expected_stale:
      errors.add('stale metadata disagrees with publication/trigger age and the parser stale boundary')
    expected_valid = (obs.get('enabled', False) and sample and obs.get('sampleValid', False) and
                      obs.get('statusComplete', False) and obs.get('carStateValid', False) and
                      not obs.get('stale', False) and not obs.get('exceptionType'))
    if valid != bool(expected_valid):
      errors.add('Event.valid disagrees with recorded sample/input/stale/exception metadata')
    for name, value in obs.get('data', {}).get('errors', {}).items():
      if value:
        decoder_errors[name] += 1
    if obs.get('exceptionType'):
      decoder_errors['observer:' + obs['exceptionType']] += 1
      incomplete.add('Observer exception is latched; this startup is not ready for another route')
    if sample:
      if sequence <= 0 or sample_stamp <= 0:
        errors.add('A sample lacks a positive sequence or sample timestamp')
      if int(obs.get('statusSampleMonoTime', 0)) != sample_stamp:
        errors.add('Point status is not paired to the exact decoded sample timestamp')
      point_ids = [int(p['trackId']) for p in points]
      status_ids = [int(p['trackId']) for p in statuses]
      if len(set(point_ids)) != len(point_ids) or len(set(status_ids)) != len(status_ids):
        errors.add('Duplicate decoded point or status identities')
      complete = set(point_ids) == set(status_ids)
      if set(status_ids) - set(point_ids) or bool(obs.get('statusComplete')) != complete:
        errors.add('Decoded point/status identity metadata is inconsistent or contains stale IDs')
      if int(obs.get('statusMissingCount', 0)) != len(set(point_ids) - set(status_ids)):
        errors.add('statusMissingCount disagrees with point/status identities')
      if not complete:
        decoder_errors['status_incomplete'] += 1
      if any(not 0 <= int(row['objectStatus']) <= 15 for row in statuses):
        errors.add('Object status lies outside the raw four-bit field')
      # Unknown aRel/yvRel are deliberately NaN in the pinned decoder's
      # deprecated group; they are not failed geometry measurements.
      if any(not math.isfinite(float(p.get(name, 0))) for p in points for name in ('dRel', 'yRel', 'vRel')):
        errors.add('Decoded point contains non-finite geometry or velocity')
      if bool(obs.get('sampleValid')) == any(obs.get('data', {}).get('errors', {}).values()):
        errors.add('sampleValid disagrees with decoder error flags')
      timing = [int(obs.get(name, 0)) for name in ('batchReceiveMonoTime', 'decodeStartMonoTime', 'decodeEndMonoTime')]
      if not 0 < timing[0] <= timing[1] <= timing[2] == sample_stamp <= stamp:
        errors.add('Receive/decode/sample/publication timestamps are inconsistent')
      first, source = int(obs.get('sourceCanFirstMonoTime', 0)), int(obs.get('sourceCanMonoTime', 0))
      if not 0 < first <= source <= timing[0]:
        errors.add('Source CAN batch timestamps are inconsistent with receipt')
      if source not in valid_can_records or first not in valid_can_records:
        source_missing.add(sequence)
      trigger_ok = trigger > 0 and (bosch_a_radar.BOSCH_A_TRIGGER_MSG, int(obs.get('parserBus', -1))) in valid_can_records.get(trigger, set())
      if valid and not trigger_ok:
        trigger_missing.add(sequence)
      sample_source_ok = source in valid_can_records and first in valid_can_records and trigger_ok
      state_stamp = int(obs.get('sampleCarStateMonoTime', 0))
      sample_state = car_states.get(state_stamp)
      if sample_state is None:
        incomplete.add('A decoded sample lacks its exact logged carState publication')
      else:
        if float(obs.get('egoSpeedUsed', float('nan'))) != sample_state[2]:
          errors.add('egoSpeedUsed differs from the exact recorded sample carState')
        if obs.get('sampleUpdated') and bool(obs.get('carStateValid')) != sample_state[0]:
          errors.add('New sample carStateValid differs from the exact logged carState validity')
        if not sample_stamp <= state_stamp <= stamp:
          errors.add('Sample carState publication is outside decode/publication ordering')
      signature = (sample_stamp, json.dumps(obs.get('data', {}), sort_keys=True), json.dumps(statuses, sort_keys=True),
                   int(obs.get('statusSampleMonoTime', 0)), first, source, trigger)
      if obs.get('sampleUpdated'):
        if sequence in sequence_samples:
          errors.add('A decoded sequence was marked updated more than once')
        sequence_samples[sequence] = sample_stamp
        if last_updated_sequence is not None and sequence > last_updated_sequence + 1:
          skipped_sequences.append([last_updated_sequence + 1, sequence - 1])
        last_updated_sequence = sequence
        decode_durations_ms.append((timing[2] - timing[1]) / 1e6)
        measured += sum(bool(p.get('deprecated', {}).get('measured')) for p in points)
        coasted += sum(not p.get('deprecated', {}).get('measured', False) for p in points)
        if valid:
          valid_samples.append(sample_stamp)
          valid_measured += sum(bool(p.get('deprecated', {}).get('measured')) for p in points)
        last_sample_signature = (sequence, signature)
      elif last_sample_signature is not None and last_sample_signature != (sequence, signature):
        errors.add('A held sample changed without sampleUpdated')
    else:
      # A latched decoder exception clears the snapshot but retains sequence
      # history, so a nonzero sequence alone is legitimate after failure.
      if points or statuses or sample_stamp or obs.get('sampleUpdated') or (sequence and not obs.get('exceptionType')):
        errors.add('No-sample heartbeat contains decoded sample data')
      last_sample_signature = None

    if obs.get('modelSeen'):
      model_stamp = int(obs.get('modelMonoTime', 0))
      frame = (int(obs.get('modelFrameId', 0)), int(obs.get('modelTimestampEof', 0)))
      logged = models.get(model_stamp)
      if logged is None:
        incomplete.add('An observation references modelV2 outside the supplied full rlogs')
      elif frame != logged[:2] or bool(obs.get('modelValid')) != logged[2]:
        errors.add('Observed model frame/EOF/validity differs from the exact logged model publication')
      else:
        matched_models.add(frame)
        if valid and obs.get('modelValid') and obs.get('modelAlive') and obs.get('modelFreqOk'):
          useful_context_frames.add(frame)
          healthy_model_frame = frame
      receive = int(obs.get('modelReceiveMonoTime', 0))
      if not 0 < model_stamp <= receive <= stamp:
        errors.add('Card model receive/publication timestamps are inconsistent')
      healthy_model_contexts += bool(obs.get('modelValid') and obs.get('modelAlive') and obs.get('modelFreqOk'))

    if obs.get('sampleUpdated'):
      camera_ok = healthy_model_frame in camera_frames and healthy_model_frame in valid_indexes
      if camera_ok:
        segment, ordinal = valid_indexes[healthy_model_frame]
        camera_ok = segment in videos and ordinal < videos[segment]['frames']
      healthy_updates.append((sample_stamp, sequence, bool(valid and sample_source_ok and camera_ok and sample_state and
                                                         sample_state[0] and sample_state[1] and not obs.get('exceptionType'))))

  if source_missing:
    incomplete.add('Some decoded samples reference CAN batches absent or invalid in supplied rlogs')
  if trigger_missing:
    incomplete.add('Some valid samples lack a valid-envelope trigger frame on the declared parser bus')
  if skipped_sequences:
    incomplete.add('Decoded sequence gaps show missing observation publications in the supplied recording')
  if len(valid_samples) < 2:
    incomplete.add('Fewer than two valid decoded samples; useful stream/cadence not established')
  if require_measured_points and valid_measured == 0:
    incomplete.add('No measured point for the requested known-target check; empty scenes are not automatically decoder faults')
  if not healthy_model_contexts:
    incomplete.add('No healthy card-received model context was recorded')
  if decoder_errors:
    notes.add('Decoder/observer errors were recorded; inspect their timing and burden before requesting another route')
  if observations and not observations[-1][2]:
    notes.add('The final observation is invalid; confirm this is expected shutdown rather than ongoing capture failure')

  matched_camera = {frame for frame in useful_context_frames if frame in camera_frames and frame in valid_indexes}
  if not matched_camera:
    incomplete.add('No exact model frame/EOF match to both narrowRoadCameraState and qcamera encode index')
  if not videos:
    incomplete.add('No matching local qcamera video supplied; recording usefulness is not fully checked')
  video_matches = 0
  for segment, metadata in videos.items():
    ordinals = {ordinal for _, (seg, ordinal) in indexes.items() if seg == segment}
    if not ordinals:
      incomplete.add(f'No qcamera index records for supplied video segment {segment}')
    elif max(ordinals) >= metadata['frames']:
      errors.add(f'qcamera segment {segment} index exceeds decoded frame count')
    elif len(ordinals) != metadata['frames']:
      incomplete.add(f'qcamera segment {segment} lacks index records for some decoded frames; include adjacent rlogs if needed')
    if metadata['frames'] <= 0:
      errors.add(f'qcamera segment {segment} has no decoded video frames')
    for frame in matched_camera:
      seg, ordinal = valid_indexes[frame]
      video_matches += seg == segment and ordinal < metadata['frames']
  if videos and not video_matches:
    incomplete.add('No observed model frame maps into a supplied decoded qcamera file')

  start = previous = previous_seq = None
  longest_healthy_s = 0.
  longest_window = None
  for stamp, sequence, healthy in healthy_updates:
    if not healthy:
      start = previous = previous_seq = None
      continue
    if previous is None or sequence != previous_seq + 1 or not 0 < stamp - previous <= round(bosch_a_radar.BOSCH_A_STALE_S * 1e9):
      start = stamp
    duration = (stamp - start) / 1e9
    if duration > longest_healthy_s:
      longest_healthy_s, longest_window = duration, [start, stamp]
    previous, previous_seq = stamp, sequence
  if longest_healthy_s < min_healthy_seconds:
    incomplete.add(f'No contiguous healthy parked radar/model/camera window lasting {min_healthy_seconds:g} seconds')

  result = 'fail' if errors else ('incomplete' if incomplete else 'pass')
  return {'result': result, 'errors': sorted(errors), 'incomplete': sorted(incomplete), 'notes': sorted(notes),
          'events': dict(counts), 'invalid_events': dict(invalid_events),
          'parser_hashes_expected': expected, 'parser_bus_expected': expected_bus,
          'observations': len(observations), 'unique_samples': len(sequence_samples), 'valid_updated_samples': len(valid_samples),
          'sample_cadence': _cadence(list(sequence_samples.values())), 'valid_sample_cadence': _cadence(valid_samples),
          'measured_point_records': measured, 'coasted_point_records': coasted, 'decoder_error_publications': dict(decoder_errors),
          'valid_measured_point_records': valid_measured,
          'min_healthy_seconds': min_healthy_seconds, 'longest_healthy_window_seconds': longest_healthy_s,
          'longest_healthy_window_mono_ns': longest_window, 'skipped_sample_sequences': skipped_sequences,
          'decoder_call_ms': {'min': min(decode_durations_ms, default=None), 'max': max(decode_durations_ms, default=None),
                              'median': statistics.median(decode_durations_ms) if decode_durations_ms else None},
          'parked_state': {'valid_records': len(valid_car_states),
                           'all_standstill': bool(valid_car_states) and all(row[0] for row in valid_car_states),
                           'max_abs_v_ego_mps': max((abs(row[1]) for row in valid_car_states), default=None),
                           'gears': dict(Counter(row[2] for row in valid_car_states)),
                           'cum_lag_ms_min': min((row[3] for row in valid_car_states), default=None),
                           'cum_lag_ms_max': max((row[3] for row in valid_car_states), default=None),
                           'positive_cum_lag_records': sum(row[3] > 0 for row in valid_car_states)},
          'healthy_model_contexts': healthy_model_contexts, 'exact_model_frames': len(matched_models),
          'exact_model_camera_index_frames': len(matched_camera), 'model_frames_in_supplied_video': video_matches,
          'missing_source_sample_sequences': sorted(source_missing), 'missing_trigger_sample_sequences': sorted(trigger_missing),
          'forbidden_send_intents': forbidden, 'videos': videos,
          'limits': ['Host/log, receive and publication times are not physical radar acquisition timestamps.',
                     'Receiver health describes card, not a future radard consumer.',
                     'No forbidden send intents in this capture is not cross-run command byte equivalence or ECU acknowledgement.',
                     'PASS is a recording-integrity smoke check; inspect errors/gaps and scene usefulness before another route.',
                     'No surveyed range origin, physical braking, target association truth or closed-loop validation is established.']}


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('rlog', nargs='+', help='Existing local full rlog[.bz2|.zst] files; no route URLs')
  parser.add_argument('--qcamera', action='append', default=[], metavar='SEGMENT=FILE', help='Local matching qcamera.ts, repeat per segment')
  parser.add_argument('--require-measured-points', action='store_true', help='Require measured points when a known stationary target is in view')
  parser.add_argument('--min-healthy-seconds', type=float, default=10.,
                      help='Contiguous healthy parked capture required (default10s diagnostic criterion, not a calibration/safety threshold)')
  parser.add_argument('--output', type=Path, help='Write the JSON summary; defaults to stdout')
  args = parser.parse_args(argv)
  output_checked = args.output is None
  try:
    if args.output is not None:
      output = args.output.expanduser().resolve()
      # A malformed bare video path is still an input to preserve when the
      # later argument-format check reports its error.
      input_names = args.rlog + [value.split('=', 1)[-1] for value in args.qcamera]
      for name in input_names:
        source = Path(name).expanduser().resolve()
        if output == source or (output.exists() and source.exists() and output.samefile(source)):
          raise ValueError('Output path aliases an input rlog or qcamera file; summary is printed to stdout without writing')
      # Errors before this point must also go to stdout: an unchecked output
      # path must never overwrite an input while reporting another failure.
      output_checked = True
    if not math.isfinite(args.min_healthy_seconds) or args.min_healthy_seconds < 0:
      raise ValueError('--min-healthy-seconds must be finite and nonnegative')
    paths = [local_file(name) for name in args.rlog]
    if any('qlog' in path.name.lower() for path in paths):
      raise ValueError('Use full rlogs; qlog decimation removes required raw CAN/model evidence')
    videos = {}
    for value in args.qcamera:
      segment, path = value.split('=', 1)
      number = int(segment)
      if number < 0 or number in videos:
        raise ValueError('Each nonnegative video segment must be supplied once')
      videos[number] = probe_video(local_file(path))
    with warnings.catch_warnings(record=True) as captured:
      warnings.simplefilter('always')
      report = inspect_events(LogReader([str(path) for path in paths], sources=[]), videos=videos,
                              require_measured_points=args.require_measured_points, min_healthy_seconds=args.min_healthy_seconds)
    if captured:
      report['incomplete'].extend('Log reader warning: ' + str(item.message) for item in captured)
      if report['result'] != 'fail':
        report['result'] = 'incomplete'
    report['rlogs'] = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in paths]
  except (OSError, ValueError, KeyError, capnp.KjException, subprocess.SubprocessError) as exc:
    report = {'result': 'incomplete', 'errors': [], 'incomplete': [str(exc)]}
  text = json.dumps(report, indent=2, sort_keys=True) + '\n'
  if args.output and output_checked:
    args.output.write_text(text)
  else:
    print(text, end='')
  return {'pass': 0, 'fail': 1, 'incomplete': 2}[report['result']]


if __name__ == '__main__':
  raise SystemExit(main())
