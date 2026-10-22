"""Read-only activation guard; software off-road state is independent of ignition.

Control services normally stop off road, so fresh manager state confirms their
absence instead of requiring messages from stopped publishers. No publishers,
vehicle interfaces, GPU or service-control operations are created here.
"""
import time

from openpilot.selfdrive.navigation.errors import NavigationModelDisabled

MAX_STATE_AGE = 2.5
CONTROL_PROCESSES = frozenset(('controlsd', 'selfdrived', 'card', 'modeld', 'modeld_tinygrad', 'joystickd'))
REQUIRED_PROCESSES = frozenset(('controlsd', 'selfdrived', 'card', 'modeld', 'modeld_tinygrad'))
SERVICES = ('deviceState', 'managerState', 'selfdriveState', 'carControl')


def offroad_reason(params, sm, now=None) -> str:
  """Return an exact blocker for an already received native messaging snapshot."""
  if not params.get_bool('IsOffroad'):
    return 'IsOffroad is false; Navigation Model Driving can be enabled only off road'
  if params.get_bool('IsEngaged'):
    return 'IsEngaged is true; disengage before enabling Navigation Model Driving'
  now = time.monotonic() if now is None else now

  def fresh(service):
    timestamp = sm.logMonoTime.get(service, 0) / 1e9
    return sm.seen.get(service, False) and sm.valid.get(service, False) and timestamp > 0 and -0.1 <= now - timestamp <= MAX_STATE_AGE

  if not fresh('deviceState'):
    return 'A fresh, valid deviceState is required to confirm genuine off-road status'
  if sm['deviceState'].started:
    return 'deviceState.started is true; the comma is on road'
  if not fresh('managerState'):
    return 'A fresh, valid managerState is required to confirm vehicle-control processes are stopped'
  processes = list(sm['managerState'].processes)
  names = [process.name for process in processes]
  if len(names) != len(set(names)) or not REQUIRED_PROCESSES.issubset(names):
    return 'managerState does not identify all required vehicle-control and model processes'
  running = sorted(process.name for process in processes if process.name in CONTROL_PROCESSES and
                   (process.running or process.shouldBeRunning))
  if running:
    return 'Vehicle-control/model processes are running or starting: ' + ', '.join(running)
  if fresh('selfdriveState') and (sm['selfdriveState'].enabled or sm['selfdriveState'].active):
    return 'Fresh selfdriveState reports enabled or active controls'
  if fresh('carControl') and (sm['carControl'].latActive or sm['carControl'].longActive):
    return 'Fresh carControl reports active lateral or longitudinal control'
  return ''


def require_offroad(params) -> None:
  """Receive fresh state using existing messaging; refuse if it cannot be read."""
  if not params.get_bool('IsOffroad') or params.get_bool('IsEngaged'):
    raise NavigationModelDisabled('Enabling requires IsOffroad=true and IsEngaged=false')
  try:
    from openpilot.cereal import messaging

    sm = messaging.SubMaster(list(SERVICES), poll='deviceState')
    deadline = time.monotonic() + MAX_STATE_AGE
    while time.monotonic() < deadline:
      sm.update(100)
      if sm.seen['deviceState'] and sm.seen['managerState']:
        break
    reason = offroad_reason(params, sm)
  except (ImportError, OSError, RuntimeError, AttributeError, KeyError, TypeError, ValueError) as error:
    raise NavigationModelDisabled(f'Cannot read fresh off-road/control state: {error}') from error
  if reason:
    raise NavigationModelDisabled(reason)
