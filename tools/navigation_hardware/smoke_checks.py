"""Read-only admission for installed comma4 maintenance diagnostics."""
import struct
import time
from pathlib import Path

from tools.navigation_hardware.safety import HardwareAdmissionError, read

GIB = 1 << 30
CONTROL_PROCESSES = frozenset(('controlsd', 'selfdrived', 'card'))


def offroad_params(params):
  """Read real settings; never set off-road, engagement or model selection."""
  result = {'is_offroad': params.get_bool('IsOffroad'), 'is_engaged': params.get_bool('IsEngaged'),
            'navigation_driving_requested': params.get_bool('NavigationModelDrivingEnabled')}
  if result['is_offroad'] is not True or result['is_engaged'] is not False or result['navigation_driving_requested']:
    raise HardwareAdmissionError('Software off-road/engagement settings changed; diagnostic refused without changing any Params')
  return result


def require_offroad(snapshot):
  if not snapshot.get('fresh') or snapshot.get('device_type') != 'mici':
    raise HardwareAdmissionError('Need fresh deviceState, pandaStates and managerState from the running comma4; no status is spoofed')
  if snapshot.get('started') is not False or snapshot.get('is_offroad') is not True or snapshot.get('is_engaged') is not False:
    raise HardwareAdmissionError('comma4 is not genuinely off-road/disengaged. Park and wait for the off-road screen; do not spoof Params')
  processes = snapshot.get('control_processes', {})
  if set(processes) != CONTROL_PROCESSES or any(processes.values()):
    raise HardwareAdmissionError('Fresh managerState must confirm all vehicle-control processes are stopped')
  for name, state in snapshot.get('control_states', {}).items():
    if state.get('fresh') and (not state.get('valid') or state.get('active')):
      raise HardwareAdmissionError('Fresh ' + name + ' is active or invalid; diagnostic refused')
  pandas = snapshot.get('pandas', [])
  if len(pandas) != 1 or pandas[0].get('type') != 'cuatro':
    raise HardwareAdmissionError('This smoke profile requires the single internal comma4 panda')
  for panda in pandas:
    if panda['controls_allowed']:
      raise HardwareAdmissionError('Panda controls are allowed; diagnostic refused')
  # Ignition is recorded, not treated as an off-road definition. Genuine
  # software off-road state and stopped controls are required even when it is ON.


def live_offroad_snapshot(timeout=8):
  from openpilot.cereal import messaging
  from openpilot.common.hardware import HARDWARE
  from openpilot.common.params import Params

  required = ('deviceState', 'pandaStates', 'managerState')
  optional = ('selfdriveState', 'carControl')
  sm = messaging.SubMaster([*required, *optional])
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    sm.update(100)
    now = time.monotonic()
    if all(sm.seen[name] and sm.valid[name] and 0 <= now - sm.recv_time[name] < 1.5 for name in required):
      params = Params()
      result = {'fresh': True, 'device_type': HARDWARE.get_device_type(), 'started': bool(sm['deviceState'].started),
                'is_offroad': params.get_bool('IsOffroad'), 'is_engaged': params.get_bool('IsEngaged'),
                'navigation_driving_requested': params.get_bool('NavigationModelDrivingEnabled'),
                'comma_power_draw_w': float(sm['deviceState'].powerDrawW),
                'source_event_ns': {name: sm.logMonoTime[name] for name in required},
                'control_processes': {p.name: bool(p.running) for p in sm['managerState'].processes if p.name in CONTROL_PROCESSES},
                'control_states': {},
                'pandas': [{'type': str(p.pandaType), 'ignition_line': bool(p.ignitionLine), 'ignition_can': bool(p.ignitionCan),
                            'controls_allowed': bool(p.controlsAllowed)} for p in sm['pandaStates']]}
      source_clocks = {'deviceState': time.monotonic_ns(), 'managerState': time.monotonic_ns(),
                       'pandaStates': time.clock_gettime_ns(time.CLOCK_BOOTTIME)}
      if any(not 0 <= source_clocks[name] - result['source_event_ns'][name] < 2_000_000_000 for name in source_clocks):
        raise HardwareAdmissionError('Off-road telemetry source timestamps are stale or in an unexpected clock domain')
      for name in optional:
        seen = bool(sm.seen[name])
        fresh = seen and 0 <= now - sm.recv_time[name] < 1.5 and 0 <= time.monotonic_ns() - sm.logMonoTime[name] < 2_000_000_000
        message = sm[name]
        active = (message.enabled or message.active or str(message.state) != 'disabled') if name == 'selfdriveState' \
                 else (message.enabled or message.latActive or message.longActive)
        result['control_states'][name] = {'seen': seen, 'fresh': fresh, 'valid': bool(sm.valid[name]), 'active': bool(active)}
      offroad_params(params)
      require_offroad(result)
      return result
  raise HardwareAdmissionError('Fresh off-road telemetry unavailable. Keep the original comma service running for admission')


class PandaSafetyReader:
  """SPI health reads only. Never construct Panda(), which changes CAN settings."""
  def __init__(self):
    from panda import Panda
    from panda.python.spi import PandaSpiHandle
    self.panda = Panda
    self.handle = PandaSpiHandle()
    try:
      version = self.handle.get_protocol_version()
      if len(version) < 15 or version[13] != 0xcc or version[14] != self.handle.PROTOCOL_VERSION:
        raise HardwareAdmissionError('Internal panda application/SPI protocol mismatch; no reset or flashing is attempted')
      versions = self.handle.controlRead(Panda.REQUEST_IN, 0xdd, 0, 0, 8, timeout=100)
      if len(versions) != 8 or struct.unpack('<II', versions)[0] != Panda.HEALTH_PACKET_VERSION:
        raise HardwareAdmissionError('Internal panda health version mismatch; no firmware operation is attempted')
    except BaseException:
      self.close()
      raise

  def sample(self):
    p = self.panda
    data = self.handle.controlRead(p.REQUEST_IN, 0xd2, 0, 0, p.HEALTH_STRUCT.size, timeout=100)
    values = p.HEALTH_STRUCT.unpack(data)
    flags = values[8]
    result = {'ignition_line': bool(flags & p.HEALTH_FLAG_IGNITION_LINE), 'ignition_can': bool(flags & p.HEALTH_FLAG_IGNITION_CAN),
              'controls_allowed': bool(flags & p.HEALTH_FLAG_CONTROLS_ALLOWED) or bool(values[21] & 3),
              'safety_mode': values[10], 'harness_status': values[9], 'voltage_mv': values[1], 'faults': values[7]}
    # silent=0, noOutput=19. No safety mode or heartbeat writes are performed.
    if result['controls_allowed'] or result['safety_mode'] not in (0, 19):
      raise HardwareAdmissionError('Panda allows controls or an active safety mode; stopping diagnostic immediately')
    if result['harness_status'] not in (1, 2) or result['faults'] or result['voltage_mv'] < 11000:
      raise HardwareAdmissionError('Internal panda/harness power or fault check failed; inspect report')
    return result

  def close(self):
    self.handle.close()


def bridge_power(handle):
  voltage, current, fault = struct.unpack('<Hh?', bytes(handle.controlRead(0xc0, 0xc0, 0, 0, 5, timeout=100)))
  watts = voltage * current / 1e6
  # Existing powered threshold is5V.162W is a conservative90% budget of the user's180W outlet, not an OEM rating test.
  if fault or voltage < 5000 or current < 0 or watts >= 162:
    raise HardwareAdmissionError(f'Chestnut power unavailable/faulted or above162W diagnostic budget: {voltage}mV, {current}mA, fault={fault}')
  return {'voltage_mv': voltage, 'current_ma': current, 'observed_watts': watts, 'supply_fault': fault,
          'outlet_rating_user_supplied_watts': 180, 'diagnostic_abort_watts': 162, 'peak_transient_headroom_verified': False}


def resources(free_bytes, available_bytes, *, model=False, preparing=False):
  needed_disk = 4 * GIB if preparing else (512 << 20 if model else 256 << 20)
  needed_memory = 5 * GIB if model or preparing else GIB
  if free_bytes < needed_disk or available_bytes < needed_memory:
    raise HardwareAdmissionError(f'Insufficient resources: need {needed_disk / GIB:g}GiB free storage and ' +
                                 f'{needed_memory / GIB:g}GiB MemAvailable for this step; ' +
                                 f'observed {free_bytes / GIB:.2f}GiB and {available_bytes / GIB:.2f}GiB. No files/settings are removed')
  return {'free_bytes': free_bytes, 'available_bytes': available_bytes, 'minimum_disk_bytes': needed_disk,
          'minimum_memory_bytes': needed_memory, 'worker_rss_limit_bytes': 4 * GIB if model or preparing else 512 << 20}


def memory_available(proc_root=Path('/proc')):
  fields = dict(line.split(':', 1) for line in read(proc_root / 'meminfo').splitlines() if ':' in line)
  return int(fields['MemAvailable'].split()[0]) * 1024


def thermal_guard(sys_root=Path('/sys')):
  temperatures = {}
  for path in (sys_root / 'class/thermal').glob('thermal_zone*/temp'):
    value = read(path)
    if value:
      temperature = float(value) / 1000
      temperatures[read(path.with_name('type'), path.parent.name)] = temperature
      if temperature >= 85:
        raise HardwareAdmissionError('Device thermal sensor reached85C maintenance limit; compiler/inference stopped')
  if not temperatures:
    raise HardwareAdmissionError('Device thermal monitoring is unavailable')
  return temperatures
