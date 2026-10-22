#!/usr/bin/env python3
"""File-only CPU/Chestnut navigation inference; no prediction publishers or actuation.

Recorded cameras and simulated navigation are labeled separately. AMD admission
requires a ready local experimental catalog; the shipped template is disabled.
This tool never downloads models, updates firmware, starts manager or sends CAN.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.navigation.experimental_catalog import TINYGRAD_REF, load_experimental_model
from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled, diagnostic_session
from openpilot.selfdrive.navigation.temporal import NavigationContext, NavigationTransitions, context_reason
from tools.navigation_model_offline import load_recording, require_cpu_artifact, validate_warp_specs

ARMS = ('neutral', 'map_only', 'maneuver_only', 'both')


def digest(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_runtime():
  from openpilot.common.basedir import BASEDIR
  root = Path(BASEDIR) / 'tinygrad_repo'
  revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
  dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip()
  if revision != TINYGRAD_REF or dirty:
    raise ValueError('Diagnostic requires the unchanged pinned tinygrad checkout')
  return revision


@contextmanager
def no_vehicle_io():
  """Deny publishers and live subscribers even if a future caller enters main()."""
  import openpilot.cereal.messaging as messaging
  import openpilot.cereal.visionipc as cereal_vision
  import msgq.visionipc as native_vision
  import msgq
  from openpilot.sunnypilot.modeld_v2 import modeld

  def denied(*args, **kwargs):
    raise NavigationModelDisabled('Diagnostic process cannot publish or connect to live vehicle streams')

  with ExitStack() as stack:
    for owner, name in ((messaging, 'PubMaster'), (messaging, 'pub_sock'), (messaging, 'SubMaster'), (messaging, 'sub_sock'),
                        (modeld, 'PubMaster'), (modeld, 'SubMaster'), (modeld, 'VisionIpcClient'),
                        (native_vision, 'VisionIpcClient')):
      stack.enter_context(patch.object(owner, name, denied))
    if hasattr(cereal_vision, 'VisionIpcClient'):
      stack.enter_context(patch.object(cereal_vision, 'VisionIpcClient', denied))
    for owner in (msgq, messaging):
      for name in ('PubSocket', 'SubSocket', 'pub_sock', 'sub_sock'):
        if hasattr(owner, name):
          stack.enter_context(patch.object(owner, name, denied))
    yield


def measurements():
  from tinygrad.helpers import GlobalCounters
  return {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
          'tinygrad_logical_allocated_bytes': GlobalCounters.mem_used}


def prepare_contexts(path, frames, *, synthetic_suite=False):
  with np.load(path, allow_pickle=False) as data:
    arrays = {key: data[key] for key in data.files}
  count = len(frames)
  if count < 1:
    raise ValueError('Diagnostic needs at least one recorded camera frame')

  def scalar(name, kind, default):
    if name not in arrays:
      return np.full(count, default)
    value = arrays[name]
    if value.ndim != 1 or value.shape[0] < count:
      raise ValueError(f'{name} must contain one scalar per selected camera frame')
    value = value[:count]
    if kind == 'boolean':
      if value.dtype != np.bool_:
        raise ValueError(f'{name} must be explicitly boolean')
    elif kind == 'integer':
      if value.dtype.kind not in 'iu' or any(not 0 <= int(v) < (1 << 64) for v in value):
        raise ValueError(f'{name} must contain nonnegative UInt64 integers')
    elif value.dtype.kind != 'U':
      raise ValueError(f'{name} must contain text strings')
    return value

  if 'camera_timestamp_ns' not in arrays or 'big_camera_timestamp_ns' not in arrays:
    raise ValueError('Diagnostic needs physical narrow/wide capture timestamps')
  for key in ('camera_timestamp_ns', 'big_camera_timestamp_ns'):
    value = scalar(key, 'integer', 0)
    if any(not 0 < int(v) < (1 << 63) for v in value):
      raise ValueError('Camera capture times must be positive, integer and strictly increasing')
    arrays[key] = value.astype(np.int64)
    if np.any(np.diff(arrays[key]) <= 0):
      raise ValueError('Camera capture times must be positive, integer and strictly increasing')
  if np.any(np.abs(arrays['camera_timestamp_ns'][:count] - arrays['big_camera_timestamp_ns'][:count]) > 10_000_000):
    raise ValueError('Narrow/wide capture skew exceeds 10ms')
  values = {name: scalar(name, 'boolean', False) for name in ('nav_valid', 'gps_valid')}
  integer_fields = ('route_generation', 'instruction_generation', 'map_generation', 'frame_id', 'metadata_frame_id',
                    'gps_source_ns', 'instruction_source_ns', 'map_source_ns', 'map_render_ns')
  values.update({name: scalar(name, 'integer', 0) for name in integer_fields})
  values.update({name: scalar(name, 'text', '') for name in ('destination_key', 'instruction_destination_key', 'map_destination_key')})
  values['scenario'] = scalar('scenario', 'text', 'prepared_navigation')
  for name in ('frame_id', 'metadata_frame_id'):
    if any(int(value) >= (1 << 32) for value in values[name]):
      raise ValueError(f'{name} must contain UInt32 frame identifiers')
  if not synthetic_suite and values['nav_valid'].any():
    required = (*integer_fields, 'destination_key', 'instruction_destination_key', 'map_destination_key', 'gps_valid')
    missing = sorted(name for name in required if name not in arrays)
    if missing:
      raise ValueError(f'Valid prepared navigation requires independent provenance arrays: {missing}')
  contexts, scenarios = [], []
  for i, frame in enumerate(frames):
    camera = int(arrays['camera_timestamp_ns'][i])
    valid = bool(values['nav_valid'][i])
    generation, instruction_generation, map_generation = (int(values[name][i]) for name in
                                                          ('route_generation', 'instruction_generation', 'map_generation'))
    key, instruction_key, map_key = (str(values[name][i]) for name in
                                    ('destination_key', 'instruction_destination_key', 'map_destination_key'))
    frame_id, metadata_frame_id = int(values['frame_id'][i]), int(values['metadata_frame_id'][i])
    gps = int(values['gps_source_ns'][i])
    gps_valid = bool(values['gps_valid'][i])
    instruction = int(values['instruction_source_ns'][i])
    source = int(values['map_source_ns'][i])
    rendered = int(values['map_render_ns'][i])
    scenario = str(values['scenario'][i])
    if synthetic_suite:
      # Explicitly simulated route/map/timing, over unchanged recorded cameras.
      # This is never described as a recorded GPS/route/map observation.
      generation, key = 1, hashlib.sha256(b'simulated-route-a').hexdigest()
      gps = instruction = source = rendered = camera
      gps_valid = valid = True
      map_generation = generation
      scenario = ('intersection_right', 'intersection_left', 'highway_exit', 'consecutive_right',
                  'consecutive_left', 'reroute_old_map', 'missing_gps', 'stale_map', 'route_reacquired')[min(i, 8)]
      command = 20 if i == 2 else 15 if i in (1, 4) else 10
      pixels = np.zeros((256, 256), np.uint8)
      pixels[20:230, 125:132] = 255
      pixels[97:104, 30:230] = 255
      pixels[100:193, 127:130] = 128
      pixels[99:102, 48:129 if command == 15 else 210] = 128
      vector = np.zeros(47, np.float32)
      vector[command], vector[-1] = 1, 0.5 if i == 4 else 1.0
      frame['navigation'] = {'nav_map': pixels, 'navigation': vector}
      if i >= 5:
        generation, key = 2, hashlib.sha256(b'simulated-route-b').hexdigest()
        map_generation = 1 if i == 5 else 2
      if i == 6:
        gps_valid = False
      if i == 7:
        source = camera - 3_000_000_000
      instruction_generation, instruction_key, map_key = generation, key, key
      frame_id = metadata_frame_id = i
    context = NavigationContext(
      camera_ns=camera, gps_ns=gps, instruction_ns=instruction, map_source_ns=source, map_render_ns=rendered,
      generation=generation, destination_key=key, instruction_generation=instruction_generation, instruction_destination_key=instruction_key,
      map_generation=map_generation, map_destination_key=map_key, frame_id=frame_id, metadata_frame_id=metadata_frame_id,
      gps_valid=gps_valid, instruction_valid=valid, map_valid=valid)
    contexts.append(context)
    scenarios.append(scenario)
  return contexts, scenarios


def run_diagnostic(factory, frames, contexts, scenarios, output, *, arms=ARMS, fallback_factory=None):
  """Independent arms, explicit resets, failures/skips visible, no actuator conversion."""
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  records, predictions = [], {}
  with (output / 'events.jsonl').open('w') as journal:
    def emit(event):
      records.append(event)
      journal.write(json.dumps(event, allow_nan=False) + '\n')
      journal.flush()
      print(json.dumps({key: event[key] for key in ('event', 'arm', 'frame', 'reason') if key in event}), flush=True)

    for arm in arms:
      predictions[arm] = {}
      model = None
      try:
        model = factory()
        model.warmup()
      except Exception as error:
        if model is not None:
          try:
            model.reset_state('initialization failure')
          except Exception:
            pass
        emit({'event': 'failure', 'arm': arm, 'frame': -1, 'model_role': 'primary',
              'reason': 'initialization: ' + str(error), **measurements()})
        continue
      transitions = NavigationTransitions()
      fallback = None
      primary_failed = False
      previous_role = 'primary'
      for i, (frame, context, scenario) in enumerate(zip(frames, contexts, scenarios, strict=True)):
        status = context_reason(context)
        decision = transitions.update(context.identity, status == 'valid', context.camera_ns, status)
        event = {'event': 'frame', 'arm': arm, 'frame': i, 'scenario': scenario, 'camera_ns': context.camera_ns,
                 'generation': context.generation, 'reason': status, 'reset': decision.reset,
                 'reset_reason': decision.reason if decision.reset else None, 'model_role': 'primary'}
        if decision.reset:
          model.reset_state(decision.reason)
        active = model
        if not decision.infer or primary_failed:
          if fallback_factory is None:
            emit(event | {'event': 'skipped', 'reason': 'primary_quarantined' if primary_failed else status, **measurements()})
            continue
          try:
            if fallback is None:
              fallback = fallback_factory()
              if any(key in getattr(fallback, 'numpy_inputs', {}) for key in ('nav_map', 'navigation')):
                raise ValueError('Diagnostic fallback must be a non-navigation model')
              fallback.warmup()
            if previous_role != 'non_navigation_fallback' or decision.reset:
              fallback.reset_state('invalid navigation or primary failure: ' + status)
          except Exception as error:
            fallback = None
            emit(event | {'event': 'failure', 'model_role': 'non_navigation_fallback', 'reason': str(error), **measurements()})
            continue
          active = fallback
          event['model_role'] = 'non_navigation_fallback'
        previous_role = event['model_role']
        inputs = {key: np.asarray(value).copy() for key, value in frame['inputs'].items()}
        if decision.infer and active is model:
          for name in ('nav_map', 'navigation'):
            if name in frame['navigation'] and (arm == 'both' or (name == 'nav_map' and arm == 'map_only') or
                                               (name == 'navigation' and arm == 'maneuver_only')):
              inputs[name] = np.asarray(frame['navigation'][name]).copy()
        started = time.perf_counter()
        try:
          raw = active.run(frame['bufs'], frame['transforms'], inputs)
          inference_seconds = time.perf_counter()-started
          parsed = {key: np.asarray(value).copy() for key, value in raw.items()}
          if not all(np.isfinite(value).all() for value in parsed.values()):
            raise ValueError('Non-finite model prediction')
          if event['model_role'] == 'primary':
            predictions[arm][i] = parsed
          folder = output / arm
          folder.mkdir(exist_ok=True)
          path = folder / f'{i:06d}.npz'
          np.savez_compressed(path, allow_pickle=False, **parsed)
          emit(event | {'inference_seconds': inference_seconds, 'frame_total_seconds': time.perf_counter()-started,
                        'state_reset_count': getattr(active, 'state_reset_count', None), 'prediction_file': str(path.relative_to(output)),
                        'prediction_sha256': digest(path), **measurements()})
        except Exception as error:
          try:
            active.reset_state('inference failure')
          except Exception:
            pass
          emit(event | {'event': 'failure', 'reason': str(error), 'inference_seconds': time.perf_counter()-started, **measurements()})
          if active is model:
            primary_failed = True
            if fallback_factory is not None:
              try:
                fallback = fallback_factory()
                if any(key in getattr(fallback, 'numpy_inputs', {}) for key in ('nav_map', 'navigation')):
                  raise ValueError('Diagnostic fallback must be a non-navigation model')
                fallback.warmup()
                fallback.reset_state('primary inference failed')
                base_inputs = {key: np.asarray(value).copy() for key, value in frame['inputs'].items()}
                started = time.perf_counter()
                raw = fallback.run(frame['bufs'], frame['transforms'], base_inputs)
                inference_seconds = time.perf_counter()-started
                parsed = {key: np.asarray(value).copy() for key, value in raw.items()}
                if not all(np.isfinite(value).all() for value in parsed.values()):
                  raise ValueError('Non-finite fallback prediction')
                folder = output / arm
                folder.mkdir(exist_ok=True)
                path = folder / f'{i:06d}-fallback.npz'
                np.savez_compressed(path, allow_pickle=False, **parsed)
                emit(event | {'model_role': 'non_navigation_fallback', 'reason': 'primary inference failed',
                              'inference_seconds': inference_seconds, 'frame_total_seconds': time.perf_counter()-started,
                              'prediction_file': str(path.relative_to(output)),
                              'prediction_sha256': digest(path), **measurements()})
                previous_role = 'non_navigation_fallback'
              except Exception as fallback_error:
                fallback = None
                emit(event | {'event': 'failure', 'model_role': 'non_navigation_fallback', 'reason': str(fallback_error), **measurements()})
          else:
            fallback = None
      del model, fallback
  comparisons = {}
  for arm in arms:
    if arm == 'neutral':
      continue
    differences = {}
    matched = 0
    for index in predictions.get('neutral', {}).keys() & predictions[arm].keys():
      before, after = predictions['neutral'][index], predictions[arm][index]
      if before.keys() != after.keys() or any(before[k].shape != after[k].shape for k in before):
        raise ValueError('Prediction schema changed between independent arms')
      matched += 1
      for name in before:
        differences[name] = max(differences.get(name, 0.0), float(np.abs(after[name]-before[name]).max()) if before[name].size else 0.0)
    comparisons[arm] = {'paired_primary_frames': matched, 'max_abs_difference': differences}
  summary = {'frames_per_arm': len(frames), 'arms': list(arms), 'comparisons': comparisons,
             'failures': sum(event['event'] == 'failure' for event in records),
             'skipped': sum(event['event'] == 'skipped' for event in records),
             'completed_predictions': sum(event['event'] == 'frame' for event in records),
             'scope': 'non-actuating file-only sensitivity; no vehicle or driving-quality conclusion'}
  (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--recording', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--backend', choices=('CPU', 'AMD'), required=True)
  parser.add_argument('--artifact', type=Path, help='Trusted CPU native artifact')
  parser.add_argument('--warp-artifact', type=Path, help='Trusted CPU warp')
  parser.add_argument('--fallback-artifact', type=Path, help='Optional trusted CPU non-navigation artifact; no production selection')
  parser.add_argument('--experimental-catalog', type=Path, help='Ready local AMD diagnostic manifest')
  parser.add_argument('--synthetic-suite', action='store_true', help='Explicitly simulate navigation contexts over recorded cameras')
  parser.add_argument('--max-frames', type=int, default=8)
  parser.add_argument('--width', type=int, default=1344)
  parser.add_argument('--height', type=int, default=760)
  args = parser.parse_args()
  if os.environ.get('NAV_MODEL_DIAGNOSTIC') != '1' or args.max_frames < 1:
    parser.error('NAV_MODEL_DIAGNOSTIC=1 and positive frame count are required')
  verify_runtime()
  if any(os.environ.get(name, '0') not in ('', '0') for name in ('MOCKGPU', 'MOCKKFD', 'MOCKUSB', 'ASM_CALL')):
    parser.error('This diagnostic entrypoint does not certify emulated/no-op GPU execution')
  if args.backend == 'AMD':
    if args.fallback_artifact is not None:
      parser.error('AMD fallback needs a separately validated catalog implementation; CPU fallback option is not accepted')
    if args.experimental_catalog is None:
      parser.error('AMD requires a ready local experimental catalog')
    entry = load_experimental_model(args.experimental_catalog, experimental=True, backend='AMD')
    if not entry.compiler_verification_validated:
      parser.error('AMD inference requires an exported compiler-verified local manifest')
    artifact, warp = entry.artifact_path, entry.warp_path
    specs = entry.input_specs
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    stride, y_height, uv_height, _ = get_nv12_info(args.width, args.height)
    if entry.warp_frame_size != stride*(y_height+uv_height):
      parser.error('Experimental AMD warp does not match requested camera copied extent')
  else:
    if args.artifact is None or args.warp_artifact is None:
      parser.error('CPU requires explicit driving and warp artifacts')
    artifact, warp = args.artifact.resolve(strict=True), args.warp_artifact.resolve(strict=True)
    _, extras = require_cpu_artifact(artifact)
    warp_parts, warp_extras = require_cpu_artifact(warp)
    validate_warp_specs(extras, warp_parts['run'][0], warp_extras, args.width, args.height)
    specs = extras['input_specs']
  fallback_path = None
  if args.fallback_artifact is not None:
    fallback_path = args.fallback_artifact.resolve(strict=True)
    _, fallback_extras = require_cpu_artifact(fallback_path)
    if any(name in fallback_extras.get('input_specs', {}) for name in ('nav_map', 'navigation')):
      parser.error('Fallback artifact must not accept navigation inputs')
    validate_warp_specs(fallback_extras, warp_parts['run'][0], warp_extras, args.width, args.height)
    for name in ('desire', 'desire_pulse'):
      if name in specs or name in fallback_extras['input_specs']:
        primary_spec = specs.get(name)
        fallback_spec = fallback_extras['input_specs'].get(name)
        if primary_spec is None or fallback_spec is None or tuple(primary_spec[0]) != tuple(fallback_spec[0]) or primary_spec[1] != fallback_spec[1]:
          parser.error('Diagnostic fallback must share the primary desire input contract')
  desire_key = 'desire' if 'desire' in specs else 'desire_pulse'
  frames = load_recording(args.recording, desire_key)[:args.max_frames]
  contexts, scenarios = prepare_contexts(args.recording, frames, synthetic_suite=args.synthetic_suite)
  from openpilot.common.params import Params
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.sunnypilot.modeld_v2.modeld import ModelState
  with patch.dict(os.environ, {'COMBINED_MODEL_PKL': str(artifact)}), OpenpilotPrefix():
    Params().put_bool('NavigationModelInputsEnabled', True, block=True)
    with diagnostic_session(args.backend), no_vehicle_io():
      def factory():
        return ModelState(args.width, args.height, chestnut=args.backend == 'AMD', warp_path=str(warp), diagnostic=True)
      def fallback_factory():
        with patch.dict(os.environ, {'COMBINED_MODEL_PKL': str(fallback_path)}):
          return ModelState(args.width, args.height, chestnut=False, warp_path=str(warp), diagnostic=True)
      summary = run_diagnostic(factory, frames, contexts, scenarios, args.output,
                               fallback_factory=fallback_factory if fallback_path is not None else None)
  provenance = {'recording_sha256': digest(args.recording), 'artifact_sha256': digest(artifact), 'warp_sha256': digest(warp),
                'tinygrad_ref': TINYGRAD_REF, 'backend': args.backend, 'navigation_context': 'synthetic' if args.synthetic_suite else 'prepared',
                'normal_prediction_publishing': False, 'vehicle_io': False, 'firmware_operations': False}
  (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
  if summary['failures'] or not summary['completed_predictions']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
