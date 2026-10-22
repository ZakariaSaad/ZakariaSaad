"""Navigation-conditioned model handover; ordinary model inference is the fallback.

Priming is an explicit eligibility policy, not proof of learned driving quality.
The candidate remains branch-gated until its physical transition behavior is reviewed.
"""
import numpy as np

from openpilot.selfdrive.navigation.activation import validate_runtime_permit
from openpilot.selfdrive.navigation.model_inputs import NavigationInputs, NavigationModelDisabled


class NavigationDrivingSession:
  def __init__(self, permit, candidate, ordinary, params, reader=None):
    self.permit, self.candidate, self.ordinary = permit, candidate, ordinary
    self.params = params
    self.reader = reader if reader is not None else NavigationInputs(params, driving_permit=permit)
    self.valid_frames = 0
    self.inputs = {}
    self.failed = False
    self.reason = 'waiting_for_valid_navigation'
    self._runtime_state = None
    self._failure_reported = False

  def _report_failure(self):
    if not self._failure_reported:
      self.params.put_bool('ChestnutActive', False)
      self._failure_reported = True

  def publish_runtime_state(self, model):
    role = 'failed' if self.failed else 'candidate' if model is self.candidate else 'ordinary_small'
    state = {'candidate_id': self.permit.model_id, 'role': role, 'reason': self.reason,
             'active_source': 'chestnut' if model is self.candidate else 'qcom'}
    if state != self._runtime_state:
      self.params.put('NavigationModelRuntimeState', state)
      self._runtime_state = state

  def prepare(self, sm, speed, camera_ns):
    if self.failed:
      return self.ordinary
    try:
      validate_runtime_permit(self.permit, params=self.params)
      inputs = self.reader.read(sm, speed, camera_ns)
    except Exception as error:
      self.invalidate('navigation permission or input failure: ' + str(error))
      return self.ordinary
    if self.reader.reset_required:
      self.invalidate(getattr(self.reader, 'reset_reason', None) or self.reader.reason)
    if not inputs:
      self.invalidate(getattr(self.reader, 'input_reason', self.reader.reason))
      return self.ordinary
    self.inputs = inputs
    self.reason = 'priming_navigation_history' if self.valid_frames < self.permit.minimum_valid_frames else 'navigation_active'
    return self.candidate if self.valid_frames >= self.permit.minimum_valid_frames else self.ordinary

  def invalidate(self, reason):
    if self.valid_frames or self.inputs:
      try:
        self.candidate.reset_state(reason)
      except Exception as error:
        self.failed = True
        self._report_failure()
        reason += ': candidate state reset failed: ' + str(error)
    self.valid_frames = 0
    self.inputs = {}
    self.reason = reason

  def fail(self, reason, ordinary=None):
    self.invalidate(reason)
    self.failed = True
    self._report_failure()
    if ordinary is not None:
      self.ordinary = ordinary

  def shadow(self, bufs, transforms, inputs, desire_key, speed, base_latency, longitudinal_delay, frame_delay, action_delay):
    if self.failed or not self.inputs or self.valid_frames >= self.permit.minimum_valid_frames:
      return
    candidate = self.candidate
    auxiliary = {key: np.asarray(value).copy() for key, value in inputs.items() if key not in (desire_key, 'action_t', 'lateral_control_params')}
    auxiliary[candidate.desire_key] = np.asarray(inputs[desire_key]).copy()
    candidate.lat_delay = base_latency
    auxiliary['action_t'] = np.array([base_latency + candidate.LAT_SMOOTH_SECONDS + frame_delay + action_delay,
                                     longitudinal_delay + candidate.LONG_SMOOTH_SECONDS + frame_delay + action_delay], np.float32)
    auxiliary['lateral_control_params'] = np.array([speed, base_latency + candidate.LAT_SMOOTH_SECONDS], np.float32)
    auxiliary.update({key: value.copy() for key, value in self.inputs.items()})
    try:
      result = candidate.run(bufs, transforms, auxiliary)
      if result is None or not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Navigation shadow inference returned invalid outputs')
      self.valid_frames += 1
    except NavigationModelDisabled as error:
      self.invalidate('navigation permission changed: ' + str(error))
    except Exception as error:
      self.fail('navigation shadow inference failed: ' + str(error))

  def run_ordinary(self, bufs, transforms, inputs, desire_key, speed, base_latency, longitudinal_delay, frame_delay, action_delay):
    """Maintain the configured small model's history while candidate outputs are used."""
    ordinary = self.ordinary
    auxiliary = {key: np.asarray(value).copy() for key, value in inputs.items()
                 if key not in (desire_key, 'action_t', 'lateral_control_params', 'nav_map', 'navigation')}
    auxiliary[ordinary.desire_key] = np.asarray(inputs[desire_key]).copy()
    ordinary.lat_delay = base_latency
    auxiliary['action_t'] = np.array([base_latency + ordinary.LAT_SMOOTH_SECONDS + frame_delay + action_delay,
                                     longitudinal_delay + ordinary.LONG_SMOOTH_SECONDS + frame_delay + action_delay], np.float32)
    auxiliary['lateral_control_params'] = np.array([speed, base_latency + ordinary.LAT_SMOOTH_SECONDS], np.float32)
    try:
      result = ordinary.run(bufs, transforms, auxiliary)
      if result is None or not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Ordinary fallback inference returned invalid outputs')
      return result
    except Exception:
      self.fail('ordinary small-model fallback inference failed')
      raise
