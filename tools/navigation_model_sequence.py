#!/usr/bin/env python3
"""Prepare an explicitly synthetic long camera sequence for file-only diagnostics.

Repeating cameras is never recorded route replay or a learned driving-quality
test. Without the opt-in, a sufficient contiguous recording is copied unchanged.
Navigation stays invalid in this file until the diagnostic's synthetic flag is
explicitly supplied; no real GPS or route metadata is fabricated here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_sequence(recording, output, *, frames=164, allow_repeated_cameras=False):
  recording, output = Path(recording).resolve(strict=True), Path(output).resolve()
  manifest = output.with_suffix(output.suffix + '.manifest.json')
  if output == recording or output.exists() or manifest.exists() or not 164 <= frames <= 512:
    raise ValueError('Use a new output path and 164..512 frames')
  with np.load(recording, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
  count = len(arrays['img'])
  if not count or any(value.ndim == 0 or len(value) != count for value in arrays.values()):
    raise ValueError('Every input array must share a nonempty leading frame count')
  for name in ('camera_timestamp_ns', 'big_camera_timestamp_ns'):
    times = arrays[name]
    if times.dtype.kind not in 'iu' or times.shape != (count,) or any(not 0 < int(t) < (1 << 63) for t in times):
      raise ValueError('Positive physical capture timestamps are required')
    if np.any(np.diff(times.astype(np.int64)) <= 0):
      raise ValueError('Source camera captures must be ordered and distinct')
  repeated = count < frames
  inherited_repeat = bool(np.asarray(arrays.get('synthetic_repeated_camera', False)).any())
  if (repeated or inherited_repeat) and not allow_repeated_cameras:
    raise ValueError('Insufficient contiguous cameras; repeated-camera tests require explicit acknowledgment')
  indices = np.arange(frames) % count
  result = {name: value[indices].copy() for name, value in arrays.items()}
  if repeated:
    result['recorded_camera_timestamp_ns'] = result['camera_timestamp_ns'].copy()
    result['recorded_big_camera_timestamp_ns'] = result['big_camera_timestamp_ns'].copy()
    skew = result['big_camera_timestamp_ns'].astype(np.int64) - result['camera_timestamp_ns'].astype(np.int64)
    start = int(arrays['camera_timestamp_ns'][0])
    narrow = np.arange(frames, dtype=np.int64) * 50_000_000 + start
    result['camera_timestamp_ns'], result['big_camera_timestamp_ns'] = narrow, narrow + skew
    # Original physical GPS does not belong to the reconstructed capture clock.
    for name in ('gps_source_ns', 'instruction_source_ns', 'map_source_ns', 'map_render_ns',
                 'route_generation', 'instruction_generation', 'map_generation'):
      result[name] = np.zeros(frames, np.int64)
    for name in ('destination_key', 'instruction_destination_key', 'map_destination_key'):
      result[name] = np.full(frames, '')
    result['gps_valid'] = result['nav_valid'] = np.zeros(frames, bool)
  result['source_frame_index'] = indices.astype(np.int32)
  result['synthetic_repeated_camera'] = np.full(frames, repeated or inherited_repeat, bool)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('xb') as destination:
    np.savez_compressed(destination, allow_pickle=False, **result)
  with recording.open('rb') as source:
    source_sha = hashlib.file_digest(source, 'sha256').hexdigest()
  with output.open('rb') as destination:
    output_sha = hashlib.file_digest(destination, 'sha256').hexdigest()
  provenance = {'source_sha256': source_sha, 'output_sha256': output_sha, 'source_frames': count, 'frames': frames,
                'camera_sequence': 'synthetic repeated clip at reconstructed20Hz' if repeated or inherited_repeat else 'recorded contiguous cameras',
                'navigation_synthesized_by_this_tool': False, 'driving_quality_validation': False,
                'source_frame_indices': indices.tolist()}
  with manifest.open('x') as destination:
    destination.write(json.dumps(provenance, indent=2) + '\n')
  return provenance


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--recording', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=164)
  parser.add_argument('--allow-repeated-cameras', action='store_true')
  args = parser.parse_args()
  print(json.dumps(prepare_sequence(args.recording, args.output, frames=args.frames,
                                    allow_repeated_cameras=args.allow_repeated_cameras), indent=2))


if __name__ == '__main__':
  main()
