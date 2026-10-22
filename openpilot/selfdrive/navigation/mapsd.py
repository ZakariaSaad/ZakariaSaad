#!/usr/bin/env python3
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.generation import destination_key, gps_source
from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import SIZE, MapRenderer
from openpilot.selfdrive.navigation.route import RouteTracker
from openpilot.selfdrive.navigation.tiles import RemoteTiles

RETRY_SECONDS = 10


class MapsD:
  def __init__(self, params=None):
    self.params = params
    self.renderer = None
    self.tracker = None
    self.route = np.empty((0, 2))
    self.route_generation = 0
    self.route_destination_key = ''
    self.prefetched = False
    self.retry_at = 0.0
    self._render_generation = 0
    self._render_destination_key = ''
    self._render_source_ns = 0
    self._route_event_ns = 0
    self._highest_route_generation = 0

  def _update_route(self, sm):
    message = sm['navRoute']
    route = np.array([(c.longitude, c.latitude) for c in message.coordinates], dtype=np.float64).reshape((-1, 2))
    generation, key = int(message.routeGeneration), str(message.destinationKey)
    event_ns = int(sm.logMonoTime['navRoute'])
    # Route identities increase across normal navd restarts and are immutable.
    # Delivery order must not roll the renderer back or resurrect a cleared route.
    if (generation > 0 and event_ns <= 0) or event_ns < self._route_event_ns or (generation > 0 and generation < self._highest_route_generation):
      return
    if generation > 0 and generation == self._highest_route_generation:
      if self.route_generation == 0 or key != self.route_destination_key or not np.array_equal(route, self.route):
        return
    self._route_event_ns = event_ns
    self._highest_route_generation = max(self._highest_route_generation, generation)
    if not sm.valid['navRoute']:
      route, generation, key = np.empty((0, 2)), 0, ''
    if generation != self.route_generation or key != self.route_destination_key or not np.array_equal(route, self.route):
      self.route = route
      self.route_generation, self.route_destination_key = generation, key
      usable = (len(route) >= 2 and np.isfinite(route).all()
                and (np.abs(route[:, 0]) <= 180).all() and (np.abs(route[:, 1]) <= 90).all())
      self.tracker = RouteTracker(route) if usable else None
      self.prefetched = False

  def update(self, sm) -> tuple[np.ndarray, int]:
    """Return a map plus its physical GPS timestamp; every unusable frame has a zero timestamp."""
    self._render_generation = 0
    self._render_destination_key = ''
    self._render_source_ns = 0
    if sm.updated['navRoute']:
      self._update_route(sm)
    now = time.monotonic()
    source = gps_source(sm, check_source_time=self.params is not None)
    # The service injects live Params. Standalone rendering callers may omit it for legacy/log viewing.
    current_key = destination_key(self.params.get('NavDestination')) if self.params is not None else self.route_destination_key
    route_current = bool(current_key and current_key == self.route_destination_key) or (self.params is None and self.route_generation == 0)
    try:
      if self.tracker is not None and route_current and now >= self.retry_at:
        if self.renderer is None:
          self.renderer = MapRenderer(PMTiles(RemoteTiles()))
        if not self.prefetched:
          # Match upstream's route prefetch while preserving the existing bounded tile cache.
          self.renderer.prefetch(self.route)
          self.prefetched = True
        source = gps_source(sm, check_source_time=self.params is not None)
        if source is not None:
          now = time.monotonic()
          gps = sm[source]
          source_ns = int(sm.logMonoTime[source])
          # Official nav-model-integration pose: route-relative fix plus distance driven since its receipt.
          speed = float(gps.speed)
          if not np.isfinite(speed):
            return np.zeros((SIZE, SIZE), dtype=np.uint8), 0
          ahead = max(speed, 0.) * (now - sm.recv_time[source])
          pose = self.tracker.pose(self.tracker.project(gps.longitude, gps.latitude) + ahead)
          image, valid = self.renderer.render(*pose, self.route)
          # A destination may have changed during blocking tile prefetch; discard the completed old frame.
          if self.params is not None and destination_key(self.params.get('NavDestination')) != self.route_destination_key:
            return np.zeros((SIZE, SIZE), dtype=np.uint8), 0
          if valid and gps_source(sm, check_source_time=self.params is not None) == source and int(sm.logMonoTime[source]) == source_ns:
            self._render_generation = self.route_generation
            self._render_destination_key = self.route_destination_key
            self._render_source_ns = source_ns
            return image, self._render_source_ns
    except Exception:  # a failed fetch, or a corrupt cached tile
      cloudlog.exception('mapsd.render_failed')
      self.close()
      self.retry_at = now + RETRY_SECONDS
    return np.zeros((SIZE, SIZE), dtype=np.uint8), 0

  def state_message(self, frame_id: int, source_ns: int, render_ns: int):
    """Metadata for exactly one VisionIPC send, with its original frame/timestamp semantics intact."""
    valid = bool(source_ns > 0 and render_ns >= source_ns and source_ns == self._render_source_ns
                 and self._render_generation > 0 and self._render_destination_key)
    if self.params is not None and destination_key(self.params.get('NavDestination')) != self._render_destination_key:
      valid = False
    msg = messaging.new_message('navModelStateSP', valid=valid)
    msg.navModelStateSP.frameId = frame_id
    msg.navModelStateSP.routeGeneration = self._render_generation if valid else 0
    msg.navModelStateSP.destinationKey = self._render_destination_key if valid else ''
    msg.navModelStateSP.locationMonoTime = source_ns if valid else 0
    msg.navModelStateSP.renderMonoTime = render_ns
    return msg

  def close(self):
    if self.renderer is not None:
      self.renderer.archive.close()
    self.renderer = None
    self.prefetched = False


def main():
  rk = Ratekeeper(20.0)
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal', 'navRoute'])
  pm = messaging.PubMaster(['navModelStateSP'])
  server = VisionIpcServer('mapsd')
  server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_MAP, 4, SIZE, SIZE, SIZE * SIZE, SIZE, SIZE * SIZE)
  server.start_listener()
  mapsd = MapsD(Params())
  try:
    while True:
      sm.update(0)
      image, timestamp = mapsd.update(sm)
      frame_id = rk.frame & 0xffffffff
      render_ns = time.monotonic_ns()
      server.send(VisionStreamType.VISION_STREAM_MAP, image.tobytes(), frame_id, timestamp, render_ns)
      pm.send('navModelStateSP', mapsd.state_message(frame_id, timestamp, render_ns))
      rk.keep_time()
  finally:
    mapsd.close()


if __name__ == '__main__':
  main()
