"""Equivalent-file-input CPU/GPU output comparisons; no quality/safety verdict."""
import json
from pathlib import Path

import numpy as np

from tools.navigation_hardware.safety import HardwareAdmissionError, sha256


def compare_results(cpu_root, gpu_root, *, atol=1e-3, rtol=1e-3, expected_recording_sha256=None):
  if not 0 <= atol <= 1e-3 or not 0 <= rtol <= 1e-3:
    raise HardwareAdmissionError('Use explicit comparison tolerances no looser than 1e-3')
  roots = [Path(cpu_root).resolve(strict=True), Path(gpu_root).resolve(strict=True)]
  provenance = [json.loads((root / 'provenance.json').read_text()) for root in roots]
  if provenance[0]['recording_sha256'] != provenance[1]['recording_sha256']:
    raise HardwareAdmissionError('CPU/GPU recordings differ; prepared inputs are not equivalent')
  if expected_recording_sha256 is not None and provenance[0]['recording_sha256'] != expected_recording_sha256:
    raise HardwareAdmissionError('Compared predictions do not identify the planned recording')
  if provenance[0]['backend'] != 'CPU' or provenance[1]['backend'] != 'AMD':
    raise HardwareAdmissionError('Comparison requires independent CPU and physical AMD runs')
  journals = [[json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()] for root in roots]
  if any(event['event'] == 'failure' for journal in journals for event in journal):
    raise HardwareAdmissionError('Inference failure prevents CPU/GPU comparison acceptance')
  indexed = [{(event['arm'], event['frame']): event for event in journal if event.get('frame', -1) >= 0} for journal in journals]
  if indexed[0].keys() != indexed[1].keys():
    raise HardwareAdmissionError('CPU/GPU frame/arm sets differ')
  comparisons, skipped, paired, passed = {}, 0, 0, True
  for key in sorted(indexed[0]):
    events = [index[key] for index in indexed]
    fields = ('event', 'camera_ns', 'generation', 'reason', 'reset', 'reset_reason', 'model_role', 'scenario')
    if any(events[0].get(field) != events[1].get(field) for field in fields):
      raise HardwareAdmissionError('CPU/GPU temporal contexts or recurrent reset schedules differ')
    if events[0]['event'] == 'skipped':
      skipped += 1
      continue
    if events[0]['event'] != 'frame':
      raise HardwareAdmissionError('Unsupported diagnostic event')
    arrays = []
    for root, event in zip(roots, events, strict=True):
      path = (root / event['prediction_file']).resolve(strict=True)
      if not path.is_relative_to(root) or sha256(path) != event['prediction_sha256']:
        raise HardwareAdmissionError('Prediction path/digest is invalid')
      with np.load(path, allow_pickle=False) as archive:
        arrays.append({name: archive[name] for name in archive.files})
    if arrays[0].keys() != arrays[1].keys():
      raise HardwareAdmissionError('CPU/GPU prediction schemas differ')
    paired += 1
    for name, before in arrays[0].items():
      after = arrays[1][name]
      if before.shape != after.shape or before.dtype != after.dtype or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise HardwareAdmissionError(f'CPU/GPU tensor shape/dtype/finite check failed: {name}')
      difference = np.abs(after.astype(np.float64) - before.astype(np.float64))
      matched = bool(np.allclose(before, after, atol=atol, rtol=rtol, equal_nan=False))
      values = comparisons.setdefault(name, {'max_abs_difference': 0.0, 'compared_values': 0, 'passed': True})
      values['max_abs_difference'] = max(values['max_abs_difference'], float(difference.max()) if difference.size else 0.0)
      values['compared_values'] += int(before.size)
      values['passed'] &= matched
      passed &= matched
  if paired == 0:
    raise HardwareAdmissionError('No equivalent inference outputs were compared')
  return {'passed': passed, 'paired_predictions': paired, 'equivalent_policy_skips': skipped, 'outputs': comparisons,
          'atol': atol, 'rtol': rtol, 'equal_nan': False, 'recording_sha256': provenance[0]['recording_sha256'],
          'scope': 'Equivalent prepared cameras/transforms/navigation and reset schedules; numerical consistency only, no model-quality verdict'}
