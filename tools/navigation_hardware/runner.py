"""Reproducible, bounded physical hardware workflow; no production process control."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import zipfile

import numpy as np

from openpilot.selfdrive.navigation.experimental_catalog import SOURCE_ONNX_SHA256
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from tools.navigation_hardware.comparison import compare_results
from tools.navigation_hardware.safety import (
  HardwareAdmissionError, preflight, process_snapshot, read, separate_paths, sha256, source_identity, telemetry,
)


@dataclass(frozen=True)
class Step:
  name: str
  stage: str
  backend: str
  arguments: tuple[str, ...]
  timeout: int


def recording_identity(path, width, height, long_frames):
  path = Path(path).resolve(strict=True)
  shapes = {}
  with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    if len(names) > 256 or len(names) != len(set(names)) or sum(item.file_size for item in archive.infolist()) > 2 << 30:
      raise HardwareAdmissionError('Prepared recording exceeds bounded archive size/schema limits')
    for member in names:
      if not member.endswith('.npy'):
        raise HardwareAdmissionError('Prepared recording must contain only NumPy arrays')
      name = member[:-4]
      with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
          shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
          shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
          raise HardwareAdmissionError('Unsupported recording array header')
        shapes[name] = (shape, dtype)
        if dtype.hasobject:
          raise HardwareAdmissionError('Object/pickle recording arrays are prohibited')
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  frame_size = stride * (y_height + uv_height)
  count = shapes['img'][0][0]
  if not long_frames <= count <= 512 or any(shapes[name] != ((count, frame_size), np.dtype(np.uint8)) for name in ('img', 'big_img')):
    raise HardwareAdmissionError('Prepared recording lacks enough matching uint8 camera frames for the requested long run')
  if any(not shape or shape[0] != count for shape, _ in shapes.values()):
    raise HardwareAdmissionError('Every prepared array must share the camera frame count')
  with np.load(path, allow_pickle=False) as archive:
    for name, tails in {'M_img': ((3, 3),), 'M_big_img': ((3, 3),), 'desire': ((8,),),
                        'traffic_convention': ((2,), (1, 2)), 'action_t': ((2,), (1, 2))}.items():
      array = archive[name]
      if array.shape[1:] not in tails or array.dtype != np.float32 or not np.isfinite(array).all():
        raise HardwareAdmissionError(f'Prepared auxiliary input shape/dtype/finite check failed: {name}')
    camera_times = []
    for name in ('camera_timestamp_ns', 'big_camera_timestamp_ns'):
      timestamps = archive[name]
      if timestamps.shape != (count,) or timestamps.dtype.kind not in 'iu' or np.any(timestamps == 0) \
         or np.any(timestamps >= (1 << 63)) or np.any(np.diff(timestamps.astype(np.int64)) <= 0):
        raise HardwareAdmissionError('Prepared camera timestamps must be valid and strictly increasing')
      camera_times.append(timestamps.astype(np.int64))
    if np.any(np.abs(camera_times[0] - camera_times[1]) > 10_000_000):
      raise HardwareAdmissionError('Prepared narrow/wide capture skew exceeds10ms')
    distinct = {}
    for name in ('frame_id', 'big_frame_id'):
      if name in archive:
        distinct[name] = len(np.unique(archive[name])) == count
    repeated = False
    if 'synthetic_repeated_camera' in archive:
      flags = archive['synthetic_repeated_camera']
      if flags.shape != (count,) or flags.dtype != np.bool_:
        raise HardwareAdmissionError('Synthetic camera provenance must be explicitly boolean per frame')
      repeated = bool(flags.any())
  return {'path': str(path), 'sha256': sha256(path), 'frames': count, 'width': width, 'height': height,
          'copied_frame_bytes': frame_size, 'distinct_recorded_frame_ids': distinct,
          'synthetic_repeated_camera': repeated,
          'caveat': 'Timestamp validity does not certify real route/GPS provenance; synthetic/repeated camera flags remain explicit'}


def build_plan(checkout, workspace, onnx, recording, *, width=1344, height=760, smoke_frames=2, comparison_frames=4,
               long_frames=164, synthetic_navigation=False, allow_repeated_cameras=False, comparison_atol=1e-3, comparison_rtol=1e-3):
  checkout, workspace = separate_paths(checkout, workspace)
  onnx, recording = Path(onnx).resolve(strict=True), Path(recording).resolve(strict=True)
  if width <= 0 or height <= 0 or width % 2 or height % 2:
    raise HardwareAdmissionError('Camera dimensions must be positive and even')
  if sha256(onnx) != SOURCE_ONNX_SHA256:
    raise HardwareAdmissionError('Bigv2 source ONNX does not match the pinned official digest')
  if not 1 <= smoke_frames <= 8 or not 1 <= comparison_frames <= 8 or not 164 <= long_frames <= 512:
    raise HardwareAdmissionError('Use smoke/comparison1..8 frames and long164..512 frames')
  if not 0 <= comparison_atol <= 1e-3 or not 0 <= comparison_rtol <= 1e-3:
    raise HardwareAdmissionError('Comparison tolerances must be explicit and no looser than 1e-3')
  identity = recording_identity(recording, width, height, long_frames)
  if not allow_repeated_cameras and (identity['synthetic_repeated_camera'] or not identity['distinct_recorded_frame_ids']
                                     or not all(identity['distinct_recorded_frame_ids'].values())):
    raise HardwareAdmissionError('Distinct recorded camera frame IDs are required unless repeated cameras are explicitly acknowledged')
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  frame = f'{width},{height},{stride},{y_height},{uv_height},{identity["copied_frame_bytes"]}'
  artifacts = workspace / 'artifacts'
  model = {backend: artifacts / f'bigv2-{backend.lower()}.sptg' for backend in ('CPU', 'AMD')}
  warp = {backend: artifacts / f'warp-{backend.lower()}.sptg' for backend in ('CPU', 'AMD')}
  manifest = artifacts / 'amd-ready.json'
  steps = []
  for backend in ('CPU', 'AMD'):
    steps.append(Step(f'compile-{backend.lower()}', 'compile', backend,
                      (str(onnx), str(model[backend]), '--device-input', '*', '--benchmark-runs', '0', '--verification-runs', '2'), 3600))
    steps.append(Step(f'warp-{backend.lower()}', 'warp', backend,
                      ('--frame', frame, '--warp-to', '512x256', '--output', str(warp[backend])), 600))
  steps.append(Step('export-amd', 'export', 'CPU',
                    ('--experimental', '--artifact', str(model['AMD']), '--warp-artifact', str(warp['AMD']),
                     '--verification', str(model['AMD']) + '.verification.json', '--source-onnx', str(onnx),
                     '--compiler', str(checkout / 'openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py'),
                     '--tinygrad-root', str(checkout / 'tinygrad_repo'), '--output-manifest', str(manifest)), 180))
  common = ('--recording', str(recording), '--width', str(width), '--height', str(height))
  for name, backend, count in (('gpu-smoke', 'AMD', smoke_frames), ('cpu-comparison', 'CPU', comparison_frames),
                               ('gpu-comparison', 'AMD', comparison_frames), ('gpu-long', 'AMD', long_frames)):
    selection = ('--experimental-catalog', str(manifest)) if backend == 'AMD' else (
      '--artifact', str(model['CPU']), '--warp-artifact', str(warp['CPU']))
    navigation_flags = ('--synthetic-long-sequence',) if name == 'gpu-long' else ('--synthetic-suite',)
    state_flags = ('--capture-recurrent-state',) if name in ('cpu-comparison', 'gpu-comparison') else ()
    steps.append(Step(name, 'diagnostic', backend, common + (navigation_flags if synthetic_navigation else ()) +
                     state_flags +
                     ('--backend', backend, '--max-frames', str(count),
                     '--output', str(workspace / name)) + selection, 3600 if name == 'cpu-comparison' else 1800))
  return {'schema_version': 1, 'checkout': str(checkout), 'workspace': str(workspace), 'onnx': str(onnx),
          'source_onnx_sha256': SOURCE_ONNX_SHA256, 'recording': identity, 'synthetic_navigation': synthetic_navigation,
          'repeated_cameras_explicitly_allowed': allow_repeated_cameras, 'normal_prediction_publishers': False,
          'comparison_atol': comparison_atol, 'comparison_rtol': comparison_rtol,
          'persistent_firmware_operations': False, 'memory_limit_bytes': 4 << 30,
          'steps': [{'name': step.name, 'stage': step.stage, 'backend': step.backend, 'arguments': list(step.arguments), 'timeout': step.timeout}
                    for step in steps]}


def worker_command(plan, step):
  checkout, workspace = Path(plan['checkout']), Path(plan['workspace'])
  return [sys.executable, '-m', 'tools.navigation_hardware.worker', '--stage', step['stage'], '--backend', step['backend'],
          '--checkout', str(checkout), '--identity-output', str(workspace / 'identities' / (step['name'] + '.json')), '--', *step['arguments']]


def descendants(pid, proc_root=Path('/proc')):
  result, pending = {pid}, [pid]
  while pending:
    current = pending.pop()
    children = read(Path(proc_root) / str(current) / 'task' / str(current) / 'children')
    for item in children.split():
      child = int(item)
      if child not in result:
        result.add(child)
        pending.append(child)
  return result


def resident_bytes(pids, proc_root=Path('/proc')):
  total = 0
  for pid in pids:
    for line in read(Path(proc_root) / str(pid) / 'status').splitlines():
      if line.startswith('VmRSS:'):
        total += int(line.split()[1]) * 1024
  return total


def stop_own_child(process):
  # The process was created with its own session. No production PID is targeted.
  if process.poll() is None:
    try:
      os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
      return
    try:
      process.wait(timeout=5)
    except subprocess.TimeoutExpired:
      os.killpg(process.pid, signal.SIGKILL)
      process.wait(timeout=5)


def run_step(plan, step, device_node):
  checkout, workspace = Path(plan['checkout']), Path(plan['workspace'])
  command = worker_command(plan, step)
  environment = os.environ.copy()
  for name in ('MOCKGPU', 'MOCKKFD', 'MOCKUSB', 'ASM_CALL', 'NO_HIPCC', 'BEAM', 'NOOPT', 'TC', 'TC_OPT', 'TC_MIN_GLOBALS', 'COMBINED_MODEL_PKL'):
    environment.pop(name, None)
  environment.update(DEV='USB+AMD:LLVM' if step['backend'] == 'AMD' else 'CPU:LLVM', NAV_HARDWARE_NON_ACTUATING='1',
                     NAV_MODEL_DIAGNOSTIC='1', NAV_MODEL_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                     TINYGRAD_PATH=str(checkout / 'tinygrad_repo'), XDG_CACHE_HOME=str(workspace / 'cache'),
                     CACHEDB=str(workspace / 'cache/tinygrad/cache.db'),
                     PYTHONPATH=':'.join(str(path) for path in (checkout, checkout / 'tinygrad_repo', checkout / 'msgq_repo',
                                                              checkout / 'opendbc_repo', checkout / 'rednose_repo', checkout / 'panda')))
  started, peak, reason = time.monotonic(), 0, None
  with (workspace / 'logs' / (step['name'] + '.log')).open('w') as log, \
       (workspace / 'telemetry' / (step['name'] + '.jsonl')).open('w') as measurements:
    process = subprocess.Popen(command, cwd=checkout, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
      while process.poll() is None:
        allowed = descendants(process.pid)
        state = process_snapshot(device_node=device_node, allowed_pids=allowed)
        rss = resident_bytes(allowed | {os.getpid()})
        peak = max(peak, rss)
        sample = telemetry(pid=process.pid)
        measurements.write(json.dumps(sample | {'package_resident_bytes': rss}) + '\n')
        measurements.flush()
        if state['conflicts'] or state['inaccessible_pids']:
          reason = 'Production/device ownership conflict or loss of process visibility'
        elif len(sample['usb']) != 1 or sample['usb'][0].get('device_node') != device_node:
          reason = 'Chestnut disconnected or re-enumerated; start a new inspected run'
        elif rss > 4 << 30:
          reason = 'Package resident memory exceeded 4 GiB'
        elif time.monotonic() - started > step['timeout']:
          reason = 'Bounded stage timeout'
        if reason:
          stop_own_child(process)
          break
        time.sleep(0.5)
      return {'name': step['name'], 'command': command, 'environment': {key: environment[key] for key in (
        'DEV', 'TINYGRAD_PATH', 'XDG_CACHE_HOME', 'PYTHONPATH', 'NAV_HARDWARE_NON_ACTUATING', 'NAV_MODEL_DIAGNOSTIC')},
        'exit_code': process.wait(), 'elapsed_seconds': time.monotonic() - started, 'peak_package_rss_bytes': peak, 'abort_reason': reason}
    finally:
      stop_own_child(process)


def collect_results(workspace):
  workspace = Path(workspace).resolve(strict=True)
  plan = json.loads((workspace / 'plan.json').read_text())
  if plan.get('schema_version') != 1 or Path(plan.get('workspace', '')).resolve() != workspace \
     or plan.get('normal_prediction_publishers') is not False:
    raise HardwareAdmissionError('Archive only a workspace created by this hardware package')
  inventory, paths = [], []
  folders = {'identities', 'logs', 'telemetry', 'gpu-smoke', 'cpu-comparison', 'gpu-comparison', 'gpu-long'}
  root_files = {'plan.json', 'preflight.json', 'results.json', 'cpu-gpu-comparison.json', 'long-sequence-validation.json'}
  for path in sorted(workspace.rglob('*')):
    relative = path.relative_to(workspace)
    if (relative.parts[0] not in folders and str(relative) not in root_files) or path.is_symlink() or not path.is_file():
      continue
    paths.append((path, str(relative)))
    inventory.append({'path': str(relative), 'size_bytes': path.stat().st_size, 'sha256': sha256(path)})
  for path in (workspace / 'artifacts').glob('*.json'):
    if path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 << 20:
      name = 'identities/artifact-reports/' + path.name
      paths.append((path, name))
      inventory.append({'path': name, 'size_bytes': path.stat().st_size, 'sha256': sha256(path)})
  manifest = workspace / 'archive-inventory.json'
  manifest.write_text(json.dumps({'files': inventory, 'excluded': ['artifacts', 'cache', 'raw input recordings'],
                                  'contains_physical_hardware_pass_claim': False}, indent=2) + '\n')
  with tarfile.open(workspace / 'results.tar.gz', 'w:gz') as archive:
    for path, name in paths + [(manifest, manifest.name)]:
      archive.add(path, arcname=name, recursive=False)
  return workspace / 'results.tar.gz'


def validate_long_run(root):
  summary = json.loads((Path(root) / 'summary.json').read_text())
  if summary['frames_per_arm'] < 164 or summary['failures']:
    raise HardwareAdmissionError('Long GPU sequence did not complete required frames without failures')
  events = [json.loads(line) for line in (Path(root) / 'events.jsonl').read_text().splitlines()]
  arms = {}
  for arm in summary['arms']:
    consecutive, maximum, post_history_reset, recovery = 0, 0, False, False
    for event in (value for value in events if value.get('arm') == arm and value.get('frame', -1) >= 0):
      valid = event['event'] == 'frame' and event.get('model_role') == 'primary' and event.get('reason') == 'valid'
      consecutive = consecutive + 1 if valid and not event.get('reset') else int(valid)
      maximum = max(maximum, consecutive)
      if event['frame'] >= 128 and event.get('reset'):
        post_history_reset = True
      if post_history_reset and valid:
        recovery = True
    arms[arm] = {'maximum_consecutive_valid_primary_frames': maximum, 'post_history_reset_observed': post_history_reset,
                 'recovery_observed': recovery}
    if maximum < 132 or not post_history_reset or not recovery:
      raise HardwareAdmissionError('Long sequence must fill 132 history frames and exercise a later reset/recovery for every arm')
  return {'arms': arms, 'driving_fallback_validated': False,
          'scope': 'Long recurrent-history and diagnostic transition execution; invalid navigation may be explicitly skipped'}


def run_plan(plan):
  checkout, workspace = Path(plan['checkout']), Path(plan['workspace'])
  if workspace.exists():
    raise HardwareAdmissionError('Use a new workspace for each physical hardware run')
  inspection = preflight(checkout, workspace)
  if not inspection['admitted']:
    raise HardwareAdmissionError('; '.join(inspection['errors']))
  for name in ('artifacts', 'identities', 'logs', 'telemetry', 'cache'):
    (workspace / name).mkdir(parents=True, exist_ok=True)
  (workspace / 'preflight.json').write_text(json.dumps(inspection, indent=2) + '\n')
  (workspace / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
  result = {'status': 'running', 'source_commit': inspection['source']['commit'], 'stages': [], 'physical_validation_claim': False}
  try:
    for step in plan['steps']:
      current = preflight(checkout, workspace)
      if not current['admitted'] or current['source']['commit'] != result['source_commit'] \
         or current['usb'][0]['device_node'] != inspection['usb'][0]['device_node']:
        raise HardwareAdmissionError('Hardware/source admission changed before the next stage')
      stage = run_step(plan, step, inspection['usb'][0]['device_node'])
      result['stages'].append(stage)
      (workspace / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
      if stage['exit_code'] != 0 or stage['abort_reason']:
        raise HardwareAdmissionError(f'Stage failed: {step["name"]}; see its log')
    if source_identity(checkout)['commit'] != result['source_commit']:
      raise HardwareAdmissionError('Source commit changed during hardware execution')
    comparison = compare_results(workspace / 'cpu-comparison', workspace / 'gpu-comparison',
                                 atol=plan['comparison_atol'], rtol=plan['comparison_rtol'],
                                 expected_recording_sha256=plan['recording']['sha256'])
    (workspace / 'cpu-gpu-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    if not comparison['passed']:
      raise HardwareAdmissionError('CPU/GPU numerical comparison failed; do not loosen tolerances without investigation')
    long_result = validate_long_run(workspace / 'gpu-long')
    (workspace / 'long-sequence-validation.json').write_text(json.dumps(long_result, indent=2) + '\n')
    result['status'] = 'completed_file_only_hardware_checks'
  except BaseException as error:
    result.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
  finally:
    result['ending_source_commit'] = source_identity(checkout, require_clean=False)['commit']
    (workspace / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    collect_results(workspace)
  return workspace / 'results.tar.gz'
