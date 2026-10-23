"""Keep the opt-in corrected-model compiler off road, without changing services.

Only the parent subscribes to existing software state. The compiler child keeps
tinygrad's ordinary exclusive USB lock; neither process publishes vehicle data.
The original model, registration and Params are never written by this helper.
"""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from types import FunctionType

from openpilot.selfdrive.navigation.offroad import MAX_STATE_AGE, SERVICES, offroad_reason

MONITOR_FD = '_OPENPILOT_COMPILE_MONITOR_FD'
_monitored_pid = None


class OffroadCompileError(RuntimeError):
  pass


def _child_admission():
  """Accept only a live inherited parent pipe, and die if that parent disappears."""
  global _monitored_pid
  try:
    fd = int(os.environ.pop(MONITOR_FD))
    if fd < 3 or not stat.S_ISFIFO(os.fstat(fd).st_mode):
      raise ValueError('monitor descriptor is not a private pipe')
    proof = json.loads(os.read(fd, 128))
    os.close(fd)
    parent = proof['parent_pid']
    if type(parent) is not int or parent != os.getppid():
      raise ValueError('monitor parent differs')
    # Linux/AGNOS: cover SIGKILL and SSH loss, not just the parent's finally block.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
      raise OSError(ctypes.get_errno(), 'Cannot install compiler parent-death protection')
    if os.getppid() != parent:
      raise ValueError('monitor exited during compiler admission')
  except (KeyError, ValueError, TypeError, OSError, AttributeError) as error:
    raise OffroadCompileError(f'Invalid off-road compiler monitor: {error}') from error

  def cancelled(signum, frame):
    raise OffroadCompileError('Off-road compilation cancelled; existing model and registration were not changed')

  signal.signal(signal.SIGTERM, cancelled)
  signal.signal(signal.SIGHUP, cancelled)
  _monitored_pid = os.getpid()


def require_compile_monitor():
  """The opt-in correction may run on an installed comma only under the monitor."""
  if Path('/AGNOS').is_file() and _monitored_pid != os.getpid():
    raise OffroadCompileError('Corrected-model compilation on AGNOS requires --offroad; keep comma.service running')


def _stop_child(child):
  if child.poll() is None:
    try:
      os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass
    try:
      child.wait(timeout=2)
    except subprocess.TimeoutExpired:
      try:
        os.killpg(child.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
  child.wait()


def _watch_child(child, params, sm):
  while child.poll() is None:
    sm.update(100)
    if reason := offroad_reason(params, sm):
      raise OffroadCompileError('Compilation stopped: ' + reason)
  # An exit concurrent with an on-road transition must not be reported as an
  # off-road verification pass. The compiler never registers its output.
  sm.update(0)
  if reason := offroad_reason(params, sm):
    raise OffroadCompileError('Compilation admission withdrawn: ' + reason)
  return child.returncode


def offroad_compile(argv: list[str]) -> int | None:
  """Call before opening a GPU. None continues main(); an int is the child exit."""
  if '--offroad' not in argv:
    if MONITOR_FD in os.environ:
      raise OffroadCompileError('Inherited compiler monitor requires --offroad')
    return None
  if MONITOR_FD in os.environ:
    _child_admission()
    return None
  if '--nav-map-precision-fix' not in argv:
    raise OffroadCompileError('--offroad is only for the targeted --nav-map-precision-fix build')

  from openpilot.common.params import Params
  from openpilot.cereal import messaging

  params = Params()
  sm = messaging.SubMaster(list(SERVICES), poll='deviceState')
  deadline = time.monotonic() + MAX_STATE_AGE
  while time.monotonic() < deadline:
    sm.update(100)
    if sm.seen['deviceState'] and sm.seen['managerState']:
      break
  if reason := offroad_reason(params, sm):
    raise OffroadCompileError('Compilation refused: ' + reason)

  read_fd, write_fd = os.pipe()
  child = None
  try:
    os.write(write_fd, json.dumps({'parent_pid': os.getpid()}).encode())
    command = [sys.executable, '-B', '-m', 'openpilot.sunnypilot.modeld_v2.compile_modeld_v2', *argv]
    child = subprocess.Popen(command, env=os.environ | {MONITOR_FD: str(read_fd)}, pass_fds=(read_fd,), start_new_session=True)
    os.close(read_fd)
    read_fd = -1
    print('Compiling off road; keep the car parked. comma.service and model selections remain unchanged.', flush=True)
    return _watch_child(child, params, sm)
  finally:
    if child is not None:
      _stop_child(child)
    if read_fd != -1:
      os.close(read_fd)
    os.close(write_fd)


@contextmanager
def cached_gpu_firmware():
  """Use pinned tinygrad's normal local/cache microcode path, never the network.

  This scoped child-only hook changes fetch_fw's fetch dependency, not urllib or
  the normal running service. GPU microcode loading is not bridge flashing.
  """
  import tinygrad.helpers as helpers
  import tinygrad.runtime.support.am.amdev as amdev
  from openpilot.sunnypilot.modeld_v2.helpers import patch_fetch_fw

  patch_fetch_fw()

  def cached_fetch(url, *, subdir, sha256):
    if subdir != 'fw' or not isinstance(sha256, str) or len(sha256) != 64:
      raise OffroadCompileError('Unexpected GPU microcode cache request')
    root = Path('/raid/downloads') if Path('/etc/tinybox-release').is_file() else Path(helpers.cache_dir) / 'downloads'
    path = root / subdir / hashlib.md5(url.encode()).hexdigest()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
      raise OffroadCompileError(f'Required pinned GPU microcode is not available in the local verified cache: {Path(url).name}; ' +
                                'this compiler does not download firmware')
    return path

  original = amdev.fetch_fw
  # Reuse the pinned implementation, including its /lib/firmware .zst handling
  # and digest verification, with only the network-capable fallback replaced.
  amdev.fetch_fw = FunctionType(original.__code__, original.__globals__ | {'fetch': cached_fetch},
                               original.__name__, original.__defaults__, original.__closure__)
  try:
    yield
  finally:
    amdev.fetch_fw = original
