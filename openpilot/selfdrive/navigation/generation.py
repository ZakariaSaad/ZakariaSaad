"""Route identities shared by navigation producers and experimental model readers."""
import hashlib
import json
import math
import time
from collections.abc import Mapping

UINT64_MAX = (1 << 64) - 1


def destination_key(destination) -> str:
  """Identity of the routing coordinates only; labels and numeric JSON spelling do not change it."""
  if not isinstance(destination, Mapping):
    return ''
  values = []
  for name, limit in (('longitude', 180), ('latitude', 90)):
    value = destination.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      return ''
    try:
      coordinate = float(value)
    except (OverflowError, ValueError):
      return ''
    if not math.isfinite(coordinate) or not -limit <= coordinate <= limit:
      return ''
    values.append(0.0 if coordinate == 0 else coordinate)
  canonical = json.dumps(values, separators=(',', ':'), allow_nan=False)
  return hashlib.sha256(canonical.encode('ascii')).hexdigest()


def next_generation(previous: int = 0, clock=time.monotonic_ns) -> int:
  """Allocate a positive route identity; monotonic time also separates normal daemon restarts."""
  value = max(int(clock()), int(previous) + 1, 1)
  if value > UINT64_MAX:
    raise OverflowError('Navigation route generation exceeds UInt64')
  return value


def gps_source(sm, now: float | None = None) -> str | None:
  """A fresh fix with valid coordinates and a physical source timestamp, never only a live subscription."""
  now = time.monotonic() if now is None else now
  candidates = []
  for source in ('gpsLocation', 'gpsLocationExternal'):
    gps = sm[source]
    if (sm.valid[source] and gps.hasFix and 0 <= now - sm.recv_time[source] < 2
        and math.isfinite(gps.longitude) and math.isfinite(gps.latitude)
        and -180 <= gps.longitude <= 180 and -90 <= gps.latitude <= 90 and int(sm.logMonoTime[source]) > 0):
      candidates.append(source)
  return max(candidates, key=lambda source: sm.logMonoTime[source]) if candidates else None
