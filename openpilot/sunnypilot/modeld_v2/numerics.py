"""Bounded failure-only summaries of already available host arrays.

Never read GPU state, record camera/map pixels, dump prediction vectors, or log
route identities. These observations locate a numerical failure without changing
inference, recurrent buffers, model selection or the existing rejection guard.
"""
import numpy as np


_HOST_INPUTS = frozenset(('tfm', 'big_tfm', 'action_t', 'lateral_control_params', 'traffic_convention',
                          'desire', 'desire_pulse', 'nav_map', 'navigation'))
_CONTEXT_INTS = frozenset(('frame_id', 'camera_sof_ns', 'camera_eof_ns', 'camera_extra_skew_ns', 'run_count'))
_CONTEXT_REASONS = frozenset(('map_reason', 'maneuver_reason'))


def _stats(array: np.ndarray) -> dict:
  finite = np.isfinite(array)
  return {'shape': list(array.shape), 'dtype': str(array.dtype), 'total': int(array.size), 'finite': int(finite.sum()),
          'nan': int(np.isnan(array).sum()), 'posinf': int(np.isposinf(array).sum()), 'neginf': int(np.isneginf(array).sum())}


def nonfinite_details(output: np.ndarray, output_slices: dict, host_inputs: dict, *, context: dict | None = None,
                      inference_ms: float | None = None) -> dict:
  """Inspect CPU arrays only; the caller retains the original failure behavior."""
  if not isinstance(output, np.ndarray):
    raise TypeError('Numerical failure summary requires an already copied NumPy output')
  vector = output.reshape(-1)
  invalid = ~np.isfinite(vector)
  sections = {}
  for name, selection in list(output_slices.items())[:512]:
    if isinstance(name, str) and isinstance(selection, slice) and invalid[selection].any():
      sections[name] = _stats(vector[selection]) | {'slice': list(selection.indices(vector.size))}
  inputs = {name: _stats(value) for name, value in host_inputs.items()
            if name in _HOST_INPUTS and isinstance(value, np.ndarray)}
  safe_context = {}
  for name, value in (context or {}).items():
    if name in _CONTEXT_INTS and type(value) is int:
      safe_context[name] = value
    elif name in _CONTEXT_REASONS and isinstance(value, str):
      safe_context[name] = value[:160]
  details = {'output_stats': _stats(output), 'first_invalid_indices': np.flatnonzero(invalid)[:8].tolist(),
             'invalid_output_sections': sections, 'host_input_stats': inputs, 'context': safe_context,
             'recurrent_state_finiteness': 'not_observed_no_extra_device_reads'}
  action_t = host_inputs.get('action_t')
  if isinstance(action_t, np.ndarray) and action_t.size <= 8:
    details['action_t'] = [float(value) if np.isfinite(value) else None for value in action_t.flat]
  if inference_ms is not None and np.isfinite(inference_ms):
    details['inference_ms'] = float(inference_ms)
  return details
