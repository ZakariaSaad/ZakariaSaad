"""Fresh sustained vehicle turn-signal state for model action calculations."""
import time

from openpilot.cereal.services import SERVICE_LIST


def fresh_turn_signal_active(sm, *, now: float | None = None) -> bool:
  # carState blinkers are already held/debounced by the vehicle interface.
  # Use SubMaster's existing 10-period freshness budget, including publisher
  # age so replayed/queued old messages cannot select the original calculation.
  if not (sm.seen.get('carState', False) and sm.valid.get('carState', False) and sm.alive.get('carState', False)):
    return False
  timestamp = sm.logMonoTime.get('carState', 0)
  now_ns = time.monotonic_ns() if now is None else int(now * 1e9)
  if timestamp <= 0 or not 0 <= now_ns - timestamp < int(1e9 * 10. / SERVICE_LIST['carState'].frequency):
    return False
  car_state = sm['carState']
  return bool(car_state.leftBlinker or car_state.rightBlinker)
