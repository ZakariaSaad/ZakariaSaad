"""Experimental navigation-model inputs for explicit offline/file-only diagnostics.

Navigation Alpha's display toggle never authorizes navigation-conditioned driving.
Production model daemons never publish navigation-conditioned predictions.
"""

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np

from openpilot.common.hardware import PC
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.helpers import navigation_input, valhalla_type
from openpilot.selfdrive.navigation.generation import destination_key, gps_source
from openpilot.selfdrive.navigation.temporal import NavigationContext, NavigationTransitions, context_reason

NAV_INPUT_NAMES = ('nav_map', 'navigation')
MAP_SHAPE = (1, 1, 256, 256)
MAP_RECEIPT_TIMEOUT = 0.5
SOURCE_TIMEOUT = 2.0  # GPS may publish at 1 Hz; this is distinct from map delivery age.
SOURCE_FUTURE_TOLERANCE = 0.2  # camera/GPS capture skew; requires hardware-independent replay clock alignment.


def driving_maneuver_reason(maneuvers, traveled):
  """Unknown/missing guidance is a production fallback, never a fabricated zero turn.

  Known depart/arrival commands <=6 and intentionally omitted merge commands
  retain the reference's zero-vector semantics; they are not missing metadata.
  """
  if not maneuvers:
    return 'missing_maneuver_metadata'
  upcoming = False
  try:
    for maneuver in maneuvers:
      distance = maneuver.distance - traveled
      if not np.isfinite(distance):
        return 'invalid_maneuver_metadata'
      if distance < 0:
        continue
      upcoming = True
      if maneuver.type == 'merge':
        continue
      command = valhalla_type(maneuver.type, maneuver.modifier)
      if command is None:
        return 'unsupported_upcoming_maneuver'
      if command > 6:
        return 'valid'
  except (AttributeError, TypeError, ValueError):
    return 'invalid_maneuver_metadata'
  return 'valid' if upcoming else 'no_upcoming_maneuver_metadata'


class NavigationModelDisabled(ValueError):
  pass


@dataclass(frozen=True)
class _DiagnosticSession:
  backend: str
  token: object


_DIAGNOSTIC_SESSION: ContextVar[_DiagnosticSession | None] = ContextVar('navigation_diagnostic_session', default=None)


def diagnostic_session_token():
  session = _DIAGNOSTIC_SESSION.get()
  return session.token if session is not None else None


def diagnostic_session_backend() -> str | None:
  session = _DIAGNOSTIC_SESSION.get()
  return session.backend if session is not None else None


def diagnostic_session_active(params=None) -> bool:
  if _DIAGNOSTIC_SESSION.get() is None or os.environ.get('NAV_MODEL_DIAGNOSTIC') != '1':
    return False
  params = params if params is not None else Params()
  return params.get_bool('NavigationModelInputsEnabled')


@contextmanager
def diagnostic_session(backend: str):
  """Permit file-only diagnostic ModelState use; production publication stays denied.

  Each session has a fresh identity. A model cannot retain recurrent state into
  another session, and withdrawing either opt-in revokes permission per frame.
  The dedicated diagnostic entrypoint owns the no-publication boundary.
  """
  if backend not in ('CPU', 'AMD') or os.environ.get('NAV_MODEL_DIAGNOSTIC') != '1':
    raise NavigationModelDisabled('Explicit NAV_MODEL_DIAGNOSTIC=1 and CPU/AMD diagnostic backend required')
  token = _DIAGNOSTIC_SESSION.set(_DiagnosticSession(backend, object()))
  try:
    if not diagnostic_session_active():
      raise NavigationModelDisabled('Diagnostic navigation permission is disabled')
    yield
  finally:
    _DIAGNOSTIC_SESSION.reset(token)


def offline_navigation_enabled(params=None) -> bool:
  if not PC or os.environ.get('NAV_MODEL_OFFLINE') != '1':
    return False
  params = params if params is not None else Params()
  return params.get_bool('NavigationModelInputsEnabled')


def validate_model_inputs(input_specs, params=None, *, diagnostic=False, driving_permit=None) -> bool:
  """Validate typed native specs before warmup/run; ordinary models are unchanged."""
  takes_nav = any(name in input_specs for name in NAV_INPUT_NAMES)
  if diagnostic:
    if not diagnostic_session_active(params):
      raise NavigationModelDisabled('Explicit non-actuating diagnostic session required')
    backend = diagnostic_session_backend()
    if any(spec[2].split(':', 1)[0] not in (backend, 'NPY') for spec in input_specs.values()):
      raise NavigationModelDisabled('Model input devices do not match the diagnostic session backend')
  if not takes_nav:
    return False
  if driving_permit is not None:
    if diagnostic:
      raise NavigationModelDisabled('Diagnostic and driving admission cannot be combined')
    from openpilot.selfdrive.navigation.activation import validate_runtime_permit
    validate_runtime_permit(driving_permit, input_specs, params)
  elif not diagnostic and not offline_navigation_enabled(params):
    raise NavigationModelDisabled('Navigation-aware models are restricted to opted-in PC offline/replay tests')
  contracts = {
    'nav_map': ((MAP_SHAPE,), (np.dtype('uint8'), np.dtype('float32'))),
    'navigation': (((47,), (1, 47)), (np.dtype('float32'),)),
    'state_map_q': (((32, 1, 256, 256),), (np.dtype('uint8'),)),
    'state_nav_q': (((32, 47),), (np.dtype('float32'),)),
    'state_map_features_q': (((128, 1, 64),), (np.dtype('float32'),)),
    'state_navigation_features_q': (((128, 1, 64),), (np.dtype('float32'),)),
  }
  for name, (shapes, dtypes) in contracts.items():
    if name not in input_specs:
      continue
    shape, dtype, _ = input_specs[name]
    if tuple(shape) not in shapes or np.dtype(dtype) not in dtypes:
      raise ValueError(f'Unsupported navigation model input contract: {name} {shape} {dtype}')
  return True


class NavigationInputs:
  """Associate exact IPC frames with route-tagged metadata and physical sources.

  Accepted frames may be held briefly while asynchronous metadata catches up.
  A caller must honor reset_required before feeding inputs to a recurrent model;
  invalid data is skipped rather than asserted to be a trained zero fallback.
  """
  def __init__(self, params=None, client=None, clock=time.monotonic, *, diagnostic=False, driving_permit=None):
    self.params = params if params is not None else Params()
    self._client = client
    self._clock = clock
    self.diagnostic = diagnostic
    self.driving_permit = driving_permit
    self._map = None
    self._map_received = 0.0
    self._map_source_ns = 0
    self._map_render_ns = 0
    self._map_frame_id = 0
    self._map_identity = None
    self._map_metadata_received = 0.0
    self._pending = OrderedDict()
    self._last_connect = -1.0
    self._destination = None
    self._failed = False
    self.transitions = NavigationTransitions()
    self.reset_required = False
    self.reset_reason = None
    self.reason = 'not_started'
    self.input_reason = 'not_started'
    self.route_generation = 0
    self.context = None
    self._generation_floor = 0
    self._instruction_source_floor = 0
    self._instruction_event_floor = 0
    self._gps_source_floor = 0
    self._render_floor = 0
    self._map_source_floor = 0
    self._last_frame_id = None
    self._producer_restart = False

  def _clear(self):
    self._map = None
    self._map_received = 0.0
    self._map_source_ns = 0
    self._map_render_ns = 0
    self._map_identity = None
    self._map_metadata_received = 0.0
    self._pending.clear()

  def _finish(self, inputs, reason, identity, camera_ns):
    decision = self.transitions.update(identity, bool(inputs), camera_ns, reason)
    self.reset_required = decision.reset
    self.reset_reason = decision.reason if decision.reset else None
    self.reason = decision.reason if not decision.infer else reason
    if self._producer_restart:
      self.reset_required = True
      self.reason = 'map_producer_restart'
      self.reset_reason = 'map_producer_restart'
      self._producer_restart = False
    self.route_generation = identity[0]
    self.input_reason = reason
    return inputs if decision.infer else {}

  def read(self, sm, v_ego: float, camera_timestamp_ns: int) -> dict:
    identity = (0, '')
    self.context = None
    try:
      key = destination_key(self.params.get('NavDestination'))
      identity = (0, key)
      inputs, reason, identity = self._read(sm, v_ego, camera_timestamp_ns, key)
    except Exception:
      if not self._failed:
        cloudlog.exception('navigation_model_input_preparation_failed')
      self._failed = True
      self._clear()
      inputs, reason = {}, 'preparation_failure'
    return self._finish(inputs, reason, identity, camera_timestamp_ns)

  def _read(self, sm, v_ego, camera_ns, key):
    now = self._clock()
    if key != self._destination:
      self._clear()
      self._destination = key
    if self.driving_permit is not None:
      from openpilot.selfdrive.navigation.activation import validate_runtime_permit
      validate_runtime_permit(self.driving_permit, params=self.params)
      permitted = True
    else:
      permitted = diagnostic_session_active(self.params) if self.diagnostic else offline_navigation_enabled(self.params)
    if not permitted or not self.params.get_bool('NavigationEnabled') or not key:
      self._clear()
      return {}, 'disabled_or_no_destination', (0, key)
    instruction = sm['navInstruction']
    identity = (int(instruction.routeGeneration), key)
    if instruction.destinationKey != key or identity[0] <= 0:
      self._clear()
      return {}, 'instruction_route_mismatch', identity
    if identity[0] < self._generation_floor:
      self._clear()
      return {}, 'route_generation_regressed', identity
    if identity[0] > self._generation_floor:
      self._generation_floor = identity[0]
      self._clear()
    if self._map_identity is not None and self._map_identity != identity:
      self._clear()
    source = gps_source(sm, now=now)
    if source is None or not sm.valid['navInstruction']:
      self._clear()
      return {}, 'missing_gps_or_instruction', identity
    instruction_event = int(sm.logMonoTime['navInstruction'])
    instruction_source = int(instruction.gpsSourceMonoTime)
    gps_timestamp = int(sm.logMonoTime[source])
    if (instruction_event <= 0 or instruction_event < self._instruction_event_floor or
        instruction_source < self._instruction_source_floor or gps_timestamp < self._gps_source_floor):
      self._clear()
      return {}, 'instruction_or_gps_timestamp_regressed', identity
    self._instruction_event_floor = instruction_event
    self._instruction_source_floor = instruction_source
    self._gps_source_floor = gps_timestamp
    if not np.isfinite(v_ego) or v_ego < 0:
      self._clear()
      return {}, 'invalid_speed', identity
    if self._client is None:
      from msgq.visionipc import VisionIpcClient
      from openpilot.cereal.visionipc import VisionStreamType
      self._client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
    if self._client.is_connected() or (now - self._last_connect >= 0.2 and self._connect(now)):
      if (frame := self._client.recv(0)) is not None:
        signature = (int(self._client.frame_id), int(self._client.timestamp_sof), int(self._client.timestamp_eof))
        if frame.width != 256 or frame.height != 256 or signature[1] <= 0 or signature[2] <= 0:
          self._clear()
          return {}, 'invalid_map_frame', identity
        pixels = np.frombuffer(frame.data, dtype=np.uint8, count=256 * 256).reshape(256, 256).copy()
        self._pending[signature] = (pixels, now)
        while len(self._pending) > 8:
          self._pending.popitem(last=False)
    metadata = sm['navModelStateSP']
    map_identity = (int(metadata.routeGeneration), metadata.destinationKey)
    if sm.valid['navModelStateSP'] and metadata.destinationKey == key and map_identity[0] > self._generation_floor:
      self._generation_floor = map_identity[0]
    if not sm.valid['navModelStateSP'] or map_identity != identity:
      self._clear()
      return {}, 'map_route_mismatch_or_invalid', identity
    signature = (int(metadata.frameId), int(metadata.locationMonoTime), int(metadata.renderMonoTime))
    if signature in self._pending:
      if signature[2] < self._render_floor or signature[1] < self._map_source_floor:
        self._clear()
        return {}, 'map_timestamp_regressed', identity
      if signature[2] == self._render_floor and self._last_frame_id is not None:
        self._pending.pop(signature)
      else:
        if self._last_frame_id is not None and signature[0] < self._last_frame_id:
          rollover = self._last_frame_id >= 0xf0000000 and signature[0] < 0x10000000
          self._producer_restart = not rollover
        self._last_frame_id = signature[0]
        self._render_floor = signature[2]
        self._map_source_floor = signature[1]
        self._map, self._map_received = self._pending.pop(signature)
        self._map_frame_id, self._map_source_ns, self._map_render_ns = signature
        self._map_identity = map_identity
        self._map_metadata_received = sm.recv_time['navModelStateSP']
    if self._map is None or self._map_identity is None:
      return {}, 'waiting_for_matching_map', identity
    # A held frame remains associated with its own producer metadata. A newer
    # metadata event is not allowed to relabel that older image.
    context = NavigationContext(
      camera_ns=camera_ns, gps_ns=int(sm.logMonoTime[source]), instruction_ns=int(instruction.gpsSourceMonoTime),
      map_source_ns=self._map_source_ns, map_render_ns=self._map_render_ns,
      generation=identity[0], destination_key=key, instruction_generation=int(instruction.routeGeneration),
      instruction_destination_key=instruction.destinationKey, map_generation=self._map_identity[0],
      map_destination_key=self._map_identity[1], frame_id=self._map_frame_id, metadata_frame_id=self._map_frame_id,
      gps_valid=True, instruction_valid=sm.valid['navInstruction'], map_valid=True,
      map_age=now-self._map_received, instruction_age=now-sm.recv_time['navInstruction'],
      metadata_age=now-self._map_metadata_received)
    self.context = context
    reason = context_reason(context)
    if reason != 'valid':
      self._clear()
      return {}, reason, identity
    traveled = v_ego * max(0.0, (camera_ns - context.instruction_ns) / 1e9)
    if self.driving_permit is not None:
      reason = driving_maneuver_reason(instruction.allManeuvers, traveled)
      if reason != 'valid':
        self._clear()
        return {}, reason, identity
    vector = navigation_input(instruction.allManeuvers, traveled)
    if vector.shape != (47,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
      self._clear()
      return {}, 'invalid_maneuver_vector', identity
    self._failed = False
    return {'nav_map': self._map, 'navigation': vector}, 'valid', identity

  def _connect(self, now):
    self._last_connect = now
    return self._client.connect(False)
