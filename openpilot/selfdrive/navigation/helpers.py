import time
from typing import Any

import numpy as np


def latest_gps_source(sm) -> str | None:
  """The service with the freshest usable GPS fix, or None."""
  now = time.monotonic()
  sources = [s for s in ('gpsLocation', 'gpsLocationExternal')
             if sm.valid[s] and sm[s].hasFix and 0 <= now - sm.recv_time[s] < 2]
  return max(sources, key=lambda s: sm.logMonoTime[s]) if sources else None


# convert mapbox steps to valhalla enums the model was trained on
VALHALLA_COMMAND_COUNT = 46
VALHALLA_SIDED_TYPES = {  # (straight, right, left)
  'depart': (1, 2, 3), 'arrive': (4, 5, 6),
  'on ramp': (17, 18, 19), 'off ramp': (20, 20, 21), 'fork': (22, 23, 24),
  'roundabout': (26, 26, 26), 'rotary': (26, 26, 26), 'roundabout turn': (26, 26, 26),
  'exit roundabout': (27, 27, 27), 'exit rotary': (27, 27, 27), 'notification': (28, 28, 28),
}
VALHALLA_TURN_TYPES = {'straight': 8, 'slight right': 9, 'right': 10, 'sharp right': 11, 'uturn': 13,
                       'sharp left': 14, 'left': 15, 'slight left': 16}  # turn, continue, new name, end of road


SIMPLE_MANEUVER_TYPES = {'turn', 'continue', 'new name', 'end of road'}


def valhalla_type(maneuver_type: str, modifier: str) -> int | None:
  if maneuver_type == 'merge':
    return None  # Preserve the official intentionally omitted merge command.
  if maneuver_type in VALHALLA_SIDED_TYPES:
    if modifier not in ('', 'straight', 'slight right', 'right', 'sharp right', 'slight left', 'left', 'sharp left', 'uturn'):
      return None
    straight, right, left = VALHALLA_SIDED_TYPES[maneuver_type]
    return right if 'right' in modifier else left if 'left' in modifier else straight
  if maneuver_type not in SIMPLE_MANEUVER_TYPES:
    return None
  return VALHALLA_TURN_TYPES.get(modifier, 8 if not modifier and maneuver_type in ('continue', 'new name') else None)


def navigation_input(maneuvers: Any, traveled: float) -> np.ndarray:
  navigation = np.zeros(VALHALLA_COMMAND_COUNT + 1, dtype=np.float32)
  if not np.isfinite(traveled) or traveled < 0:
    return navigation
  for maneuver in maneuvers:
    distance = maneuver.distance - traveled
    if not np.isfinite(distance):
      return navigation
    if distance < 0:
      continue
    command = valhalla_type(maneuver.type, maneuver.modifier)
    if command is None:
      if maneuver.type == 'merge':
        continue
      return navigation  # Unknown upcoming instructions must not become a fabricated straight or a later turn.
    if command > 6:
      navigation[command], navigation[-1] = 1, distance / 100
      break
  return navigation
