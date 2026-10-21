"""Experimental navigation-model inputs, restricted to explicit PC offline testing.

Navigation Alpha's display toggle never authorizes navigation-conditioned driving.
No compatible model may be run on comma hardware through this integration.
"""

import os
import time

import numpy as np

from openpilot.common.hardware import PC
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.helpers import navigation_input

NAV_INPUT_NAMES = ('nav_map', 'navigation')
MAP_SHAPE = (1, 1, 256, 256)
MAP_RECEIPT_TIMEOUT = 0.5
SOURCE_TIMEOUT = 2.0  # GPS may publish at 1 Hz; this is distinct from map delivery age.
SOURCE_FUTURE_TOLERANCE = 0.2  # camera/GPS capture skew; requires hardware-independent replay clock alignment.


class NavigationModelDisabled(ValueError):
  pass


def offline_navigation_enabled(params=None) -> bool:
  if not PC or os.environ.get('NAV_MODEL_OFFLINE') != '1':
    return False
  params = params if params is not None else Params()
  return params.get_bool('NavigationModelInputsEnabled')


def validate_model_inputs(input_specs, params=None) -> bool:
  """Validate typed native specs before warmup/run; ordinary models are unchanged."""
  takes_nav = any(name in input_specs for name in NAV_INPUT_NAMES)
  if not takes_nav:
    return False
  if not offline_navigation_enabled(params):
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
  """Read optional model inputs without allowing bad navigation data to kill modeld.

  Recurrent model state is not reset here. Missing input selects the existing zero
  default during offline tests; this is not evidence of a trained safe fallback.
  """

  def __init__(self, params=None, client=None, clock=time.monotonic):
    self.params = params if params is not None else Params()
    self._client = client
    self._clock = clock
    self._map = None
    self._map_received = 0.0
    self._map_source_ns = 0
    self._last_connect = -1.0
    self._destination = None
    self._failed = False

  def _clear(self):
    self._map = None
    self._map_received = 0.0
    self._map_source_ns = 0

  def read(self, sm, v_ego: float, camera_timestamp_ns: int) -> dict:
    try:
      return self._read(sm, v_ego, camera_timestamp_ns)
    except Exception:
      if not self._failed:
        cloudlog.exception('navigation_model_input_preparation_failed')
      self._failed = True
      self._clear()
      return {}

  def _read(self, sm, v_ego, camera_timestamp_ns):
    now = self._clock()
    destination = self.params.get('NavDestination')
    if destination != self._destination:
      self._clear()
      self._destination = destination
    if not offline_navigation_enabled(self.params) or not self.params.get_bool('NavigationEnabled') or destination is None:
      self._clear()
      return {}
    gps_ok = any(sm.valid[s] and sm[s].hasFix and 0 <= now - sm.recv_time[s] < SOURCE_TIMEOUT for s in ('gpsLocation', 'gpsLocationExternal'))
    instruction_age = now - sm.recv_time['navInstruction']
    if not gps_ok or not sm.valid['navInstruction'] or not 0 <= instruction_age < SOURCE_TIMEOUT:
      self._clear()
      return {}
    if not np.isfinite(v_ego) or v_ego < 0:
      self._clear()
      return {}
    if self._client is None:
      from msgq.visionipc import VisionIpcClient
      from openpilot.cereal.visionipc import VisionStreamType

      self._client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
    if self._client.is_connected() or (now - self._last_connect >= 0.2 and self._connect(now)):
      if (frame := self._client.recv(0)) is not None:
        self._clear()
        if frame.width != 256 or frame.height != 256 or self._client.timestamp_sof <= 0:
          return {}
        self._map = np.frombuffer(frame.data, dtype=np.uint8, count=256 * 256).reshape(256, 256).copy()
        self._map_received = now
        self._map_source_ns = self._client.timestamp_sof
    source_age = (camera_timestamp_ns - self._map_source_ns) / 1e9
    if (
      self._map is None
      or camera_timestamp_ns <= 0
      or not 0 <= now - self._map_received < MAP_RECEIPT_TIMEOUT
      or not -SOURCE_FUTURE_TOLERANCE <= source_age < SOURCE_TIMEOUT
    ):
      self._clear()
      return {}
    vector = navigation_input(sm['navInstruction'].allManeuvers, v_ego * instruction_age)
    if vector.shape != (47,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
      self._clear()
      return {}
    self._failed = False
    return {'nav_map': self._map, 'navigation': vector}

  def _connect(self, now):
    self._last_connect = now
    return self._client.connect(False)
