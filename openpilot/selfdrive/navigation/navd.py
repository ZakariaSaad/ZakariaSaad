#!/usr/bin/env python3
import os

import numpy as np
import requests

import openpilot.cereal.messaging as messaging
from openpilot.common.api.comma_connect import API_HOST, CommaConnectApi as Api
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.navigation.generation import destination_key, gps_source, next_generation
from openpilot.selfdrive.navigation.route import RouteTracker, along_path, local_meters

REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10
REROUTE_COUNTER_MIN = 3
ROUTE_RESEND_PERIOD = 10  # updates (seconds); a restarted mapsd gets the route within this
RECOMPUTE_HOLD = 10  # updates (seconds) between route requests


def set_banner(instruction, banners, distance_to_maneuver):
  """Fill the banner fields from the banner to show now: the last one still ahead of the ego, else the first."""
  if not banners:
    return
  banner = next((b for b in reversed(banners) if distance_to_maneuver < b['distanceAlongGeometry']), banners[0])
  instruction.showFull = distance_to_maneuver < banner['distanceAlongGeometry']
  primary = banner['primary']
  instruction.maneuverPrimaryText = primary.get('text') or ''
  instruction.maneuverType = primary.get('type') or ''
  instruction.maneuverModifier = primary.get('modifier') or ''
  if banner.get('secondary') is not None:
    instruction.maneuverSecondaryText = banner['secondary']['text']


class RouteEngine:
  def __init__(self, sm, pm, params=None):
    self.sm = sm
    self.pm = pm
    self.params = params if params is not None else Params()

    self.last_position = None  # (longitude, latitude)
    self.last_bearing = None

    self.nav_destination = None
    self.step_idx = None
    self.route = None
    self.route_geometry = None  # an (n, 2) array of (longitude, latitude) per step
    self.route_tracker = None
    self.maneuver_distances = None
    self.route_generation = 0
    self.route_destination_key = ''
    self._last_generation = 0

    self.recompute_hold = 0
    self.reroute_counter = 0

    self.mapbox_token = os.environ.get("MAPBOX_TOKEN")
    self.api = Api(self.params.get("DongleId")) if self.mapbox_token is None else None
    self.mapbox_host = "https://api.mapbox.com" if self.mapbox_token is not None else API_HOST.rstrip('/') + '/v1/nav'

  def update(self):
    self.sm.update(0)

    source = gps_source(self.sm)
    # Keep legacy log-view/display positioning when no physical sample is
    # admissible, but never use that exception to request a new model route.
    display_source = source or gps_source(self.sm, check_source_time=False)
    if display_source is not None:
      gps = self.sm[display_source]
      self.last_position, self.last_bearing = (gps.longitude, gps.latitude), gps.bearingDeg
    route = self.route
    try:
      self.recompute_route(gps_ok=source is not None)
      self.send_instruction()
    except Exception:
      cloudlog.exception("navd.failed_to_compute")
    # mapsd dedupes repeats, so a periodic resend is how a restarted mapsd gets the route
    if self.route is not route or self.sm.frame % ROUTE_RESEND_PERIOD == 0:
      self.send_route()

  def recompute_route(self, gps_ok: bool):
    destination = self.params.get("NavDestination")
    requested_key = destination_key(destination)
    if not requested_key:
      self.clear_route()
      self.recompute_hold = 0
      return

    # A new destination invalidates the old route immediately, even during retry hold or GPS loss.
    if self.route is not None and requested_key != self.route_destination_key:
      self.clear_route()
      self.recompute_hold = 0
    if self.last_position is None or not gps_ok:
      return
    new_destination = (destination['longitude'], destination['latitude'])
    self.recompute_hold -= 1
    if (self.should_recompute() or new_destination != self.nav_destination) and self.recompute_hold <= 0:
      self.recompute_hold = RECOMPUTE_HOLD
      self.calculate_route(new_destination)
      self.reroute_counter = 0

  def calculate_route(self, destination):
    assert self.last_position is not None
    cloudlog.warning(f"Calculating route {self.last_position} -> {destination}")
    requested_key = destination_key({'longitude': destination[0], 'latitude': destination[1]})
    if not requested_key:
      self.clear_route()
      return

    # Preserve the visible route while the request blocks, but revoke its model
    # snapshot immediately, including when rerouting to the same destination.
    self.send_instruction(model_ready=False)

    lang = self.params.get('LanguageSetting')
    if lang is not None:
      lang = lang.replace('main_', '')

    token = self.mapbox_token
    if token is None:
      token = self.api.get_token()

    params = {
      'access_token': token,
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
      'language': lang,
    }
    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90;"

    coords_str = ';'.join(f'{lon},{lat}' for lon, lat in (self.last_position, destination))
    url = self.mapbox_host + '/directions/v5/mapbox/driving-traffic/' + coords_str
    try:
      resp = requests.get(url, params=params, timeout=10)
      if resp.status_code != 200:
        cloudlog.event("API request failed", status_code=resp.status_code, text=resp.text, error=True)
      resp.raise_for_status()

      r = resp.json()
      # A destination can change while the synchronous HTTP request is in flight.
      if destination_key(self.params.get('NavDestination')) != requested_key:
        self.clear_route()
        self.recompute_hold = 0
        return
      if len(r['routes']):
        route = [dict(step) for step in r['routes'][0]['legs'][0]['steps']]
        geometry = [np.array(step['geometry']['coordinates'], dtype=np.float64).reshape((-1, 2)) for step in route]
        if not geometry or any(len(path) < 2 or not np.isfinite(path).all()
                               or (np.abs(path[:, 0]) > 180).any() or (np.abs(path[:, 1]) > 90).any() for path in geometry):
          raise ValueError('Invalid navigation route geometry')
        for step in route:
          step['distance'], step['duration'] = float(step['distance']), float(step['duration'])
          if not np.isfinite([step['distance'], step['duration']]).all() or min(step['distance'], step['duration']) < 0:
            raise ValueError('Invalid navigation route distance or duration')
          if not isinstance(step.get('bannerInstructions'), list):
            raise ValueError('Invalid navigation route banners')
        # Commit the route atomically only after the full response is usable.
        tracker = RouteTracker(np.concatenate(geometry))
        maneuver_distances = tracker.distance[np.cumsum([0] + [len(path) for path in geometry[:-1]])]
        self.route, self.route_geometry = route, geometry
        self.route_tracker, self.maneuver_distances = tracker, maneuver_distances
        self.step_idx = 0
        self.nav_destination = destination
        self.route_destination_key = requested_key
        self._last_generation = next_generation(self._last_generation)
        self.route_generation = self._last_generation
      else:
        cloudlog.warning("Got empty route response")
        self.clear_route()

    except (requests.exceptions.RequestException, ValueError, TypeError, KeyError, IndexError, OverflowError):
      cloudlog.exception("failed to get usable route")
      self.clear_route()

  def send_instruction(self, *, model_ready: bool = True):
    msg = messaging.new_message('navInstruction', valid=True)
    msg.navInstruction.routeGeneration = self.route_generation
    msg.navInstruction.destinationKey = self.route_destination_key
    source = gps_source(self.sm) if model_ready else None
    gps_source_ns = int(self.sm.logMonoTime[source]) if source is not None else 0
    msg.navInstruction.gpsSourceMonoTime = max(gps_source_ns, 0)

    if self.step_idx is None:
      msg.valid = False
      self.pm.send('navInstruction', msg)
      return

    assert self.route is not None and self.route_geometry is not None and self.last_position is not None
    assert self.route_tracker is not None and self.maneuver_distances is not None
    step = self.route[self.step_idx]
    along_geometry, _ = along_path(self.route_geometry[self.step_idx], self.last_position)
    distance_to_maneuver_along_geometry = step['distance'] - along_geometry

    # Banner instructions are for the following maneuver step, don't use empty last step
    banner_step = step
    if not len(banner_step['bannerInstructions']) and self.step_idx == len(self.route) - 1:
      banner_step = self.route[max(self.step_idx - 1, 0)]

    # Current instruction
    msg.navInstruction.maneuverDistance = distance_to_maneuver_along_geometry
    set_banner(msg.navInstruction, banner_step['bannerInstructions'], distance_to_maneuver_along_geometry)

    # Maneuvers not yet passed, for the model's navigation input
    maneuvers = []
    if gps_source_ns > 0:
      # The model distance and timestamp must refer to the same physical fix.
      # Legacy display positioning can select another source or retain an old
      # position; it must not advance the model tracker's monotonic progress.
      gps = self.sm[source]
      progress = self.route_tracker.project(gps.longitude, gps.latitude)
      for route_step, distance in zip(self.route, self.maneuver_distances, strict=True):
        if distance < progress:
          continue
        maneuver = route_step.get('maneuver')
        # An incomplete response must not invent a model command; retain the UI instruction.
        if not isinstance(maneuver, dict):
          maneuvers = []
          break
        kind, modifier = maneuver.get('type'), maneuver.get('modifier', '')
        if not isinstance(kind, str) or not isinstance(modifier, str):
          maneuvers = []
          break
        maneuvers.append({'distance': float(distance - progress), 'type': kind, 'modifier': modifier})
    msg.navInstruction.allManeuvers = maneuvers

    # Total remaining distance and time: the rest of this step, then every step after it
    remaining = 1.0 - along_geometry / max(step['distance'], 1)
    future = self.route[self.step_idx + 1:]
    msg.navInstruction.distanceRemaining = step['distance'] * remaining + sum(s['distance'] for s in future)
    msg.navInstruction.timeRemaining = step['duration'] * remaining + sum(s['duration'] for s in future)

    self.pm.send('navInstruction', msg)

    # Transition to next route segment
    if source is not None and distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD:
      if self.step_idx + 1 < len(self.route):
        self.step_idx += 1
        self.recompute_hold = 0
      else:
        cloudlog.warning("Destination reached")

        # Clear route if driving away from destination
        if np.linalg.norm(local_meters(self.last_position, self.nav_destination)) > REROUTE_DISTANCE:
          self.params.remove("NavDestination")
          self.clear_route()

  def send_route(self):
    coords = np.concatenate(self.route_geometry) if self.route_geometry is not None else np.empty((0, 2))
    msg = messaging.new_message('navRoute', valid=True)
    msg.navRoute.coordinates = [{'longitude': lon, 'latitude': lat} for lon, lat in coords.tolist()]
    msg.navRoute.routeGeneration = self.route_generation
    msg.navRoute.destinationKey = self.route_destination_key
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
    self.route_geometry = None
    self.route_tracker = None
    self.maneuver_distances = None
    self.route_generation = 0
    self.route_destination_key = ''
    self.step_idx = None
    self.nav_destination = None

  def should_recompute(self):
    if self.step_idx is None:
      return True

    assert self.route is not None and self.route_geometry is not None and self.last_position is not None
    # Don't recompute in last segment, assume destination is reached
    if self.step_idx == len(self.route) - 1:
      return False

    # TODO: Check for going wrong way in segment
    if along_path(self.route_geometry[self.step_idx], self.last_position)[1] > REROUTE_DISTANCE:
      self.reroute_counter += 1
    else:
      self.reroute_counter = 0
    return self.reroute_counter > REROUTE_COUNTER_MIN


def main():
  pm = messaging.PubMaster(['navInstruction', 'navRoute'])
  sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])

  rk = Ratekeeper(1.0)
  route_engine = RouteEngine(sm, pm)
  while True:
    route_engine.update()
    rk.keep_time()


if __name__ == "__main__":
  main()
