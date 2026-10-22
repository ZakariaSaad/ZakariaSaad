#!/usr/bin/env python3
"""Prepare local recorded cameras for offline model sensitivity experiments.

This does not reconstruct an original model process replay. Camera captures,
calibration and vehicle telemetry come from the log; neutral desire and the
target Honda action timing are explicitly reconstructed experimental inputs.
Absent recorded navigation is never replaced by a claimed real route.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.timing import prediction_delay
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def padded_nv12(frame: np.ndarray, width: int, height: int) -> np.ndarray:
  """Copy decoded visible NV12 planes into the pinned native warp's input layout."""
  if width <= 0 or height <= 0 or width % 2 or height % 2:
    raise ValueError('NV12 dimensions must be positive and even')
  source = np.asarray(frame)
  if source.dtype != np.uint8 or source.size != width * height * 3 // 2:
    raise ValueError('Decoded NV12 frame has an unexpected dtype or length')
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  packed = np.zeros((y_height + uv_height, stride), dtype=np.uint8)
  source = source.reshape(height * 3 // 2, width)
  packed[:height, :width] = source[:height]
  packed[y_height:y_height + height // 2, :width] = source[height:]
  return packed.reshape(-1)


def match_camera_frames(narrow: list[dict], wide: list[dict], *, tolerance_ns=10_000_000) -> list[tuple[dict, dict]]:
  """Pair by physical capture time; equal frame IDs/file offsets need not align."""
  wide = sorted(wide, key=lambda frame: frame['timestamp_sof'])
  times = [frame['timestamp_sof'] for frame in wide]
  if not wide or tolerance_ns < 0:
    raise ValueError('Missing wide-camera indices or invalid pairing tolerance')
  pairs = []
  previous = -1
  for camera in narrow:
    index = bisect_left(times, camera['timestamp_sof'])
    candidates = [i for i in (index - 1, index) if 0 <= i < len(wide)]
    nearest = min(candidates, key=lambda i: abs(times[i] - camera['timestamp_sof']))
    if nearest <= previous or abs(times[nearest] - camera['timestamp_sof']) > tolerance_ns:
      raise ValueError('Camera captures are missing, reused, out of order or outside synchronization tolerance')
    pairs.append((camera, wide[nearest]))
    previous = nearest
  return pairs


def latest_context(events: Sequence[tuple[int, object]], timestamp_ns: int, *, max_age_ns=None):
  """Last causally available telemetry, rejecting a missing or stale source."""
  index = bisect_right([event[0] for event in events], timestamp_ns) - 1
  if index < 0 or (max_age_ns is not None and timestamp_ns - events[index][0] >= max_age_ns):
    raise ValueError('No sufficiently fresh causal telemetry for this capture')
  return events[index]


def decode_frames(path: Path, indices: list[int], width: int, height: int) -> dict[int, np.ndarray]:
  if not indices or indices != sorted(set(indices)) or min(indices) < 0:
    raise ValueError('Decoded indices must be distinct, increasing and nonnegative')
  selection = '+'.join(f'eq(n\\,{index})' for index in indices)
  raw = subprocess.check_output([
    'ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-vf', f'select={selection}',
    '-vsync', '0', '-frames:v', str(len(indices)), '-f', 'rawvideo', '-pix_fmt', 'nv12', 'pipe:1',
  ])
  size = width * height * 3 // 2
  if len(raw) != len(indices) * size:
    raise ValueError('Decoded frame count or dimensions do not match the camera indices')
  frames = np.frombuffer(raw, np.uint8).reshape(len(indices), size)
  return {index: padded_nv12(frame, width, height) for index, frame in zip(indices, frames, strict=True)}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def prepare(rlog: Path, narrow_camera: Path, wide_camera: Path, *, segment: int, start_index: int, count: int,
            scenario: str, width=1344, height=760) -> tuple[dict[str, np.ndarray], dict]:
  """Bounded local-recording adapter. No network, model execution or messaging."""
  from openpilot.tools.lib.logreader import LogReader

  if count < 1 or count > 128 or start_index < 0:
    raise ValueError('Choose 1..128 frames and a nonnegative encoded start index')
  histories = {name: [] for name in ('carState', 'extrinsicsCalibration', 'driverMonitoringState', 'deviceState', 'narrowRoadCameraState',
                                    'gpsLocation', 'gpsLocationExternal')}
  indices = {'narrowRoadEncodeIdx': [], 'wideRoadEncodeIdx': []}
  params, source_commit, cp = {}, None, None
  recorded_destination_present = False
  nav_counts = {'navInstruction': 0, 'navRoute': 0}
  for event in LogReader(str(rlog)):
    name = event.which()
    if name == 'initData':
      source_commit = event.initData.gitCommit
      params = {entry.key: bytes(entry.value).decode(errors='replace') for entry in event.initData.params.entries
                if entry.key in ('CameraOffset', 'LatSmoothSeconds', 'LongSmoothSeconds')}
      recorded_destination_present = any(entry.key == 'NavDestination' and bytes(entry.value) for entry in event.initData.params.entries)
    elif name in indices:
      value = getattr(event, name)
      if value.segmentNum == segment:
        indices[name].append({'frame_id': int(value.frameId), 'segment_id': int(value.segmentId),
                              'timestamp_sof': int(value.timestampSof), 'timestamp_eof': int(value.timestampEof)})
    elif name == 'carParams':
      cp = event.carParams
    elif name in histories:
      histories[name].append((int(event.logMonoTime), event))
    elif name in nav_counts:
      nav_counts[name] += 1
  for events in histories.values():
    events.sort(key=lambda event: event[0])
  if cp is None:
    raise ValueError('Recorded CarParams is required for the target action timing')
  if not np.isfinite(cp.longitudinalActuatorDelay) or cp.longitudinalActuatorDelay < 0:
    raise ValueError('Invalid recorded longitudinal actuator delay')
  if float(params.get('CameraOffset', 0)) != 0:
    raise ValueError('Nonzero recorded camera offsets require a separately validated reconstruction')
  narrow = sorted((frame for frame in indices['narrowRoadEncodeIdx'] if start_index <= frame['segment_id'] < start_index + count),
                  key=lambda frame: frame['segment_id'])
  if [frame['segment_id'] for frame in narrow] != list(range(start_index, start_index + count)):
    raise ValueError('Missing encoded narrow-camera frames')
  pairs = match_camera_frames(narrow, indices['wideRoadEncodeIdx'])
  if any(b['timestamp_sof'] <= a['timestamp_sof'] for a, b in zip(narrow[:-1], narrow[1:], strict=True)):
    raise ValueError('Narrow-camera capture timestamps must increase')
  decoded = decode_frames(narrow_camera, [a['segment_id'] for a, _ in pairs], width, height)
  decoded_big = decode_frames(wide_camera, [b['segment_id'] for _, b in pairs], width, height)
  arrays: dict[str, list] = {name: [] for name in ('img', 'big_img', 'M_img', 'M_big_img', 'intrinsics_img', 'intrinsics_big_img',
                                                      'action_t', 'traffic_convention', 'camera_timestamp_ns', 'big_camera_timestamp_ns',
                                                      'camera_timestamp_eof_ns', 'big_camera_timestamp_eof_ns', 'camera_pair_skew_ns',
                                                      'frame_id', 'big_frame_id', 'narrow_segment_id', 'wide_segment_id',
                                                      'v_ego', 'steering_angle_deg', 'rpy_calib', 'calibration_source_ns', 'gps_source_ns', 'gps_valid')}
  camera_model = None
  for main, extra in pairs:
    timestamp = main['timestamp_sof']
    _, cs_event = latest_context(histories['carState'], timestamp, max_age_ns=200_000_000)
    calibration_ns, calibration = latest_context(histories['extrinsicsCalibration'], timestamp, max_age_ns=2_000_000_000)
    _, dm = latest_context(histories['driverMonitoringState'], timestamp, max_age_ns=2_000_000_000)
    _, device = latest_context(histories['deviceState'], timestamp, max_age_ns=2_000_000_000)
    _, camera = latest_context(histories['narrowRoadCameraState'], timestamp, max_age_ns=2_000_000_000)
    if not all(event.valid for event in (cs_event, calibration, dm, device, camera)):
      raise ValueError('Invalid recorded context cannot establish a model input')
    if str(calibration.extrinsicsCalibration.calStatus) != 'calibrated':
      raise ValueError('Recorded camera calibration is not established')
    camera_model = (str(device.deviceState.deviceType), str(camera.narrowRoadCameraState.sensor))
    config = DEVICE_CAMERAS[camera_model]
    if config.narrow_road.size != (width, height) or config.wide_road.size != (width, height):
      raise ValueError('Recorded camera identity does not match the decoded frame dimensions')
    rpy = np.asarray(calibration.extrinsicsCalibration.rpyCalib, np.float32)
    speed = float(cs_event.carState.vEgo)
    if rpy.shape != (3,) or not np.isfinite(rpy).all() or not np.isfinite(speed) or speed < 0:
      raise ValueError('Invalid recorded calibration or speed')
    gps_sources = []
    for source in ('gpsLocation', 'gpsLocationExternal'):
      try:
        receive_ns, fix = latest_context(histories[source], timestamp, max_age_ns=2_000_000_000)
        if fix.valid and getattr(fix, source).hasFix:
          gps_sources.append(receive_ns)
      except ValueError:
        pass
    values = {
      'img': decoded[main['segment_id']], 'big_img': decoded_big[extra['segment_id']],
      'M_img': get_warp_matrix(rpy, config.narrow_road.intrinsics, False).astype(np.float32),
      'M_big_img': get_warp_matrix(rpy, config.wide_road.intrinsics, True).astype(np.float32),
      'intrinsics_img': config.narrow_road.intrinsics.astype(np.float32),
      'intrinsics_big_img': config.wide_road.intrinsics.astype(np.float32),
      'action_t': np.array([prediction_delay(speed) + DT_MDL * 1.5, cp.longitudinalActuatorDelay + DT_MDL * 1.5], np.float32),
      'traffic_convention': np.array([not dm.driverMonitoringState.isRHD, dm.driverMonitoringState.isRHD], np.float32),
      'camera_timestamp_ns': timestamp, 'big_camera_timestamp_ns': extra['timestamp_sof'],
      'camera_timestamp_eof_ns': main['timestamp_eof'], 'big_camera_timestamp_eof_ns': extra['timestamp_eof'],
      'camera_pair_skew_ns': timestamp - extra['timestamp_sof'], 'frame_id': main['frame_id'], 'big_frame_id': extra['frame_id'],
      'narrow_segment_id': main['segment_id'], 'wide_segment_id': extra['segment_id'], 'v_ego': speed,
      'steering_angle_deg': float(cs_event.carState.steeringAngleDeg), 'rpy_calib': rpy, 'calibration_source_ns': calibration_ns,
      'gps_source_ns': max(gps_sources, default=0), 'gps_valid': bool(gps_sources),
    }
    for name, value in values.items():
      arrays[name].append(value)
  output = {name: np.asarray(values) for name, values in arrays.items()}
  output |= {
    'desire': np.zeros((count, 8), np.float32), 'nav_map': np.zeros((count, 256, 256), np.uint8),
    'navigation': np.zeros((count, 47), np.float32), 'nav_valid': np.zeros(count, np.bool_),
    'route_generation': np.zeros(count, np.int64), 'map_source_ns': np.zeros(count, np.int64),
    'instruction_source_ns': np.zeros(count, np.int64), 'scenario': np.full(count, scenario),
  }
  manifest = {
    'scope': 'recorded-camera sensitivity preparation; not original model process replay or recorded navigation parity',
    'source_git_commit': source_commit, 'segment': segment, 'start_encoded_index': start_index, 'frames': count,
    'sources': {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in
                (('rlog', rlog), ('narrow_camera', narrow_camera), ('wide_camera', wide_camera))},
    'camera': {'identity': camera_model, 'width': width, 'height': height, 'nv12_layout': get_nv12_info(width, height),
               'padding': 'zero-filled decoder padding; visible camera pixels unchanged',
               'pairing': 'physical capture SOF, at most10ms; each stream uses its own encoded segmentId'},
    'transforms': 'logged causal rpyCalib + pinned target DEVICE_CAMERAS/get_warp_matrix, zero camera offset',
    'desire': 'neutral experimental input; original input desire is not reconstructed from predicted desireState',
    'action_t': {'method': 'target Honda prediction_delay(vEgo)+0.075, recordedCP.longitudinalActuatorDelay+0.075',
                 'lat_smoothing_seconds': 0, 'long_smoothing_seconds': 0, 'original_recorded_input': False},
    'recorded_smoothing_settings': params,
    'target_runtime_sources': {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path) for path in
                               (Path(__file__).resolve().parents[1] / 'openpilot/common/transformations/camera.py',
                                Path(__file__).resolve().parents[1] / 'openpilot/common/transformations/model.py',
                                Path(__file__).resolve().parents[1] / 'openpilot/sunnypilot/selfdrive/controls/lib/honda_eps/timing.py')},
    'navigation': {'recorded_message_counts': nav_counts, 'recorded_map_pixels': False, 'recorded_destination': recorded_destination_present,
                   'payload': 'unavailable; zero-filled, invalid; any injected nav is synthetic and must be labeled separately'},
  }
  output['recording_manifest'] = np.full(count, json.dumps(manifest, sort_keys=True))
  return output, manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--rlog', type=Path, required=True)
  parser.add_argument('--narrow-camera', type=Path, required=True)
  parser.add_argument('--wide-camera', type=Path, required=True)
  parser.add_argument('--segment', type=int, required=True)
  parser.add_argument('--start-index', type=int, required=True)
  parser.add_argument('--count', type=int, default=8)
  parser.add_argument('--scenario', required=True, help='Truthful scene label; this does not establish recorded navigation intent')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--manifest', type=Path, required=True)
  args = parser.parse_args()
  arrays, manifest = prepare(args.rlog, args.narrow_camera, args.wide_camera, segment=args.segment,
                             start_index=args.start_index, count=args.count, scenario=args.scenario)
  np.savez_compressed(args.output, allow_pickle=False, **arrays)
  args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
  print(json.dumps({'recording': str(args.output), 'manifest': str(args.manifest), 'frames': args.count,
                    'recorded_navigation_valid': False}))


if __name__ == '__main__':
  main()
