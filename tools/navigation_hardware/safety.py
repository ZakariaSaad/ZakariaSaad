"""Read-only hardware/process admission. Importing this module opens no devices."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import time

from openpilot.common.git_inspection import GitInspectionError, read_only_git
from openpilot.selfdrive.navigation.experimental_catalog import TINYGRAD_REF

LIVE_IDS = {('add1', '0001'), ('3801', '0001')}
ROM_IDS = {('174c', '2463'), ('174c', '2464')}
PRODUCTION = {'manager', 'modeld', 'camerad', 'hardwared', 'controlsd', 'selfdrived', 'card', 'pandad', 'boardd'}
PROTECTED = ('/data/openpilot', '/data/params', '/data/models', '/data/modeld', '/data/model_cache')


class HardwareAdmissionError(ValueError):
  pass


def sha256(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, default=''):
  try:
    return Path(path).read_text().strip()
  except FileNotFoundError:
    return default


def git(checkout, *args):
  try:
    return read_only_git(checkout, *args)
  except GitInspectionError as error:
    raise HardwareAdmissionError(str(error)) from error


def separate_paths(checkout, workspace, *, protected=PROTECTED):
  checkout, workspace = Path(checkout).resolve(strict=True), Path(workspace).resolve()
  for raw in protected:
    active = Path(raw).resolve()
    if any(path == active or path.is_relative_to(active) for path in (checkout, workspace)):
      raise HardwareAdmissionError('Use a separate checkout and workspace outside production installation/settings/model caches')
  if workspace == checkout or workspace.is_relative_to(checkout):
    raise HardwareAdmissionError('Workspace must be outside the source checkout')
  if (checkout / 'tools/navigation_model_diagnostic.py').is_file() is False:
    raise HardwareAdmissionError('Checkout does not contain the current diagnostic implementation')
  return checkout, workspace


def source_identity(checkout, *, require_clean=True):
  checkout = Path(checkout)
  revision = git(checkout, 'rev-parse', 'HEAD')
  dirty = git(checkout, 'status', '--porcelain', '--untracked-files=all')
  tinygrad = checkout / 'tinygrad_repo'
  runtime = git(tinygrad, 'rev-parse', 'HEAD')
  runtime_dirty = git(tinygrad, 'status', '--porcelain', '--untracked-files=all')
  if require_clean and (dirty or runtime_dirty):
    raise HardwareAdmissionError('Hardware execution requires a clean committed checkout and runtime')
  if runtime != TINYGRAD_REF:
    raise HardwareAdmissionError('Unexpected tinygrad runtime pin')
  usb_source = checkout / 'openpilot/common/hardware/usb.py'
  text = usb_source.read_text()
  version = re.search(r'CHESTNUT_FW_VERSION\s*=\s*["\']([^"\']+)', text)
  if version is None:
    raise HardwareAdmissionError('Cannot establish the branch firmware product profile')
  components = ('openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py', 'openpilot/sunnypilot/modeld_v2/compile_warp.py',
                'tools/navigation_model_diagnostic.py', 'openpilot/common/hardware/usb.py',
                'openpilot/system/hardware/chestnut/firmware_wrapped.bin')
  return {'commit': revision, 'branch': git(checkout, 'branch', '--show-current'), 'dirty': bool(dirty),
          'tinygrad_ref': runtime, 'tinygrad_dirty': bool(runtime_dirty), 'expected_usb_product': f'custom {version[1]}-CLEAN',
          'component_sha256': {name: sha256(checkout / name) for name in components},
          'submodules': git(checkout, 'submodule', 'status', '--recursive').splitlines()}


def usb_devices(sys_root=Path('/sys')):
  devices = []
  for path in sorted((Path(sys_root) / 'bus/usb/devices').glob('*')):
    vendor, product_id = read(path / 'idVendor').lower(), read(path / 'idProduct').lower()
    if (vendor, product_id) not in LIVE_IDS | ROM_IDS:
      continue
    names = ('product', 'manufacturer', 'busnum', 'devnum', 'speed', 'bcdDevice', 'bMaxPower')
    value = {name: read(path / name) for name in names}
    value.update(sysfs_path=str(path), vendor_id=vendor, product_id=product_id, rom_mode=(vendor, product_id) in ROM_IDS)
    serial = read(path / 'serial')
    value['serial_sha256'] = hashlib.sha256(serial.encode()).hexdigest() if serial else None
    try:
      value['device_node'] = f"/dev/bus/usb/{int(value['busnum']):03d}/{int(value['devnum']):03d}"
    except ValueError:
      value['device_node'] = None
    devices.append(value)
  return devices


def gpu_identity(sys_root=Path('/sys')):
  """Host PCI evidence does not establish the architecture behind a USB bridge."""
  found = []
  for path in sorted((Path(sys_root) / 'bus/pci/devices').glob('*')):
    if read(path / 'vendor').lower() == '0x1002':
      identifier = read(path / 'device').lower()
      found.append({'sysfs_path': str(path), 'device_id': identifier,
                    'architecture': 'gfx1200' if identifier == '0x7590' else 'unknown'})
  return {'host_pci_devices': found, 'chestnut_architecture': 'unknown_until_explicit_compute_initialization',
          'required_discovered_architecture': 'gfx1200'}


def process_snapshot(proc_root=Path('/proc'), *, device_node=None, allowed_pids=()):
  allowed = set(allowed_pids)
  conflicts, inaccessible = [], []
  for directory in Path(proc_root).iterdir():
    if not directory.name.isdigit() or int(directory.name) == os.getpid():
      continue
    pid = int(directory.name)
    try:
      comm = read(directory / 'comm')
      tokens = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\x00')
      names = {comm.removesuffix('.py')} | {Path(token).name.removesuffix('.py') for token in tokens if token}
      modules = {token.rsplit('.', 1)[-1] for token in tokens if token.startswith(('openpilot.', 'selfdrive.', 'system.'))}
      blocked = (names | modules) & PRODUCTION
      owners = []
      for fd in (() if pid in allowed else (directory / 'fd').iterdir()):
        try:
          target = os.readlink(fd)
        except FileNotFoundError:
          continue
        if target == device_node or target == '/dev/kfd' or target.startswith('/dev/dri/'):
          owners.append(target)
      if blocked or owners:
        conflicts.append({'pid': pid, 'process': comm, 'production_roles': sorted(blocked), 'device_fds': sorted(set(owners))})
    except (FileNotFoundError, ProcessLookupError):
      continue
    except PermissionError:
      inaccessible.append(pid)
  return {'conflicts': conflicts, 'inaccessible_pids': inaccessible}


def preflight(checkout, workspace, *, sys_root=Path('/sys'), proc_root=Path('/proc'), require_clean=True, allowed_pids=()):
  checkout, workspace = separate_paths(checkout, workspace)
  identity = source_identity(checkout, require_clean=require_clean)
  devices = usb_devices(sys_root)
  errors = []
  if len(devices) != 1:
    errors.append(f'Expected exactly one Chestnut bridge; observed {len(devices)}')
  device = devices[0] if len(devices) == 1 else {}
  if device.get('rom_mode'):
    errors.append('Chestnut is in ROM/bootloader mode; no recovery or flashing is performed')
  if device.get('product') != identity['expected_usb_product']:
    errors.append('USB product differs from this branch profile; compatibility/migration needs review, never automatic flashing')
  if not device.get('device_node'):
    errors.append('USB bus/device identity is unavailable')
  existing_parent = workspace.parent
  while not existing_parent.exists():
    existing_parent = existing_parent.parent
  free = shutil.disk_usage(existing_parent).free
  if free < 8 << 30:
    errors.append('At least 8 GiB free workspace storage is required; no files are deleted automatically')
  processes = process_snapshot(proc_root, device_node=device.get('device_node'), allowed_pids=allowed_pids)
  if processes['conflicts']:
    errors.append('Production or competing GPU/USB processes are running; package will not stop them')
  if processes['inaccessible_pids']:
    errors.append('Process ownership cannot be inspected completely; use an adequately privileged inspection session')
  return {'scope': 'read_only_sysfs_proc_git', 'wall_time_ns': time.time_ns(), 'source': identity, 'usb': devices,
          'gpu': gpu_identity(sys_root), 'processes': processes, 'admitted': not errors, 'errors': errors,
          'workspace_free_bytes': free, 'persistent_firmware_operations': False, 'production_processes_modified': False}


def telemetry(sys_root=Path('/sys'), proc_root=Path('/proc'), *, pid=None):
  sys_root, proc_root = Path(sys_root), Path(proc_root)
  sensors = {}
  for directory in (sys_root / 'class/hwmon').glob('hwmon*'):
    for pattern in ('temp*_input', 'power*_average', 'power*_input', 'in*_input', 'curr*_input'):
      for path in directory.glob(pattern):
        sensors[str(path.relative_to(sys_root))] = {'raw': read(path), 'label': read(path.with_name(path.name.replace('_input', '_label'))),
                                                  'hwmon': read(directory / 'name')}
  memory = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in read(proc_root / 'meminfo').splitlines() if ':' in line}
  status = read(proc_root / str(pid) / 'status') if pid else ''
  rss = next((line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('VmRSS:')), None)
  return {'monotonic_ns': time.monotonic_ns(), 'sensors': sensors, 'memory': memory, 'child_rss': rss,
          'usb': usb_devices(sys_root), 'bridge_voltage_temperature': 'unavailable unless exposed by sysfs; no bridge monitor/publisher started'}
