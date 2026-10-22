"""One-command installed comma4 smoke test; systemd owns cleanup after SSH loss."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid

# Use the installed pinned repositories, never a globally installed tinygrad.
CHECKOUT = Path(__file__).resolve().parents[2]
for component in reversed(('tinygrad_repo', 'msgq_repo', 'opendbc_repo', 'rednose_repo')):
  sys.path.insert(0, str(CHECKOUT / component))

from tools.navigation_hardware import smoke_checks as checks
from tools.navigation_hardware.runner import descendants, resident_bytes, stop_own_child
from tools.navigation_hardware.safety import HardwareAdmissionError, process_snapshot, sha256, source_identity, usb_devices

ROOT = Path('/data/navigation-smoke')
UNIT = 'navigation-chestnut-smoke'
SOURCE_SIZE = 814591106
ONNX_URL = 'https://github.com/happymaj11r/openpilot-models/releases/download/big-nav-model-integration-1d18c42e/big_driving_supercombo.onnx'


def write_json(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2) + '\n')
  temporary.replace(path)


def systemctl(*args, check=True):
  return subprocess.run(['systemctl', *args], check=check, capture_output=True, text=True, timeout=45)


def service_active():
  return systemctl('is-active', '--quiet', 'comma.service', check=False).returncode == 0


def installed_identity():
  if CHECKOUT != Path('/data/openpilot').resolve() or not Path('/AGNOS').is_file():
    raise HardwareAdmissionError('Run this command in the updated /data/openpilot installation on your comma4')
  identity = source_identity(CHECKOUT)
  if identity['branch'] != 'mvl-honda-eps-selector21-chestnut-v28':
    raise HardwareAdmissionError('Installed branch differs from the requested v28 branch; no branch switch is attempted')
  return identity


def usb_profile(identity):
  devices = usb_devices()
  if len(devices) != 1 or devices[0]['rom_mode'] or devices[0]['product'] != identity['expected_usb_product'] \
     or not devices[0]['device_node'] or float(devices[0]['speed'] or '0') < 5000:
    observed = [{key: device.get(key) for key in ('product', 'vendor_id', 'product_id', 'speed', 'rom_mode')} for device in devices]
    raise HardwareAdmissionError('Need one SuperSpeed Chestnut with expected firmware product ' + identity['expected_usb_product'] +
                                 '. Observed: ' + json.dumps(observed) + '. Check USB/power; no firmware change or recovery is performed')
  return devices[0]


def initial_checks():
  identity = installed_identity()
  snapshot = checks.live_offroad_snapshot()
  if not service_active():
    raise HardwareAdmissionError('Normal comma.service must be active for fresh off-road admission')
  return {'source': identity, 'offroad': snapshot, 'usb': usb_profile(identity), 'original_service_active': True,
          'activation_approved': False, 'selected_models_changed': False, 'firmware_changed': False}


def stage_download_links(cache):
  """Reuse installed runtime downloads via individual read links; never its writable DB."""
  target = cache / 'tinygrad/downloads'
  target.mkdir(parents=True, exist_ok=True)
  linked = 0
  for source in (Path('/home/comma/.cache/tinygrad/downloads'), Path('/root/.cache/tinygrad/downloads')):
    if not source.is_dir():
      continue
    for path in source.rglob('*'):
      if linked >= 4096:
        break
      if path.is_file() and not path.is_symlink():
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
          destination.symlink_to(path)
          linked += 1
  return linked


def worker_environment(cache):
  environment = os.environ.copy()
  for name in ('MOCKGPU', 'MOCKKFD', 'MOCKUSB', 'ASM_CALL', 'NO_HIPCC', 'BEAM', 'NOOPT', 'TC', 'TC_OPT',
               'TC_MIN_GLOBALS', 'COMBINED_MODEL_PKL', 'DEV', 'AMD', 'AMD_LLVM', 'AMD_IFACE', 'HCQ_RUNTIME_DEV'):
    environment.pop(name, None)
  environment.update(DEV='USB+AMD:LLVM', GMMU='0', NAV_SMOKE_WORKER='1', PYTHONDONTWRITEBYTECODE='1',
                     XDG_CACHE_HOME=str(cache), CACHEDB=str(cache / 'tinygrad/smoke-cache.db'), DISABLE_HTTP_CACHE='0',
                     TINYGRAD_PATH=str(CHECKOUT / 'tinygrad_repo'),
                     PYTHONPATH=':'.join(str(p) for p in (CHECKOUT / 'tinygrad_repo', CHECKOUT, CHECKOUT / 'msgq_repo',
                                                        CHECKOUT / 'opendbc_repo', CHECKOUT / 'rednose_repo')))
  return environment


class MaintenanceGuard:
  def __init__(self, identity, device):
    self.identity, self.device = identity, device
    self.panda = None
    self.context = None
    self.handle = None

  def __enter__(self):
    from openpilot.common.params import Params
    self.params = Params()
    self.sample_processes()
    self.panda = checks.PandaSafetyReader()
    try:
      import usb1
      self.context = usb1.USBContext()
      self.context.open()
      self.handle = self.context.openByVendorIDAndProductID(int(self.device['vendor_id'], 16), int(self.device['product_id'], 16))
      if self.handle is None:
        raise HardwareAdmissionError('Chestnut disappeared before read-only power inspection')
      self.sample()
      return self
    except BaseException:
      self.__exit__(None, None, None)
      raise

  def sample_processes(self, allowed=()):
    state = process_snapshot(device_node=self.device['device_node'], allowed_pids=allowed)
    if state['conflicts'] or state['inaccessible_pids']:
      raise HardwareAdmissionError('Another production/GPU process is active or process ownership is not inspectable')
    # No competing user of the internal panda SPI bus may run during maintenance.
    for directory in Path('/proc').iterdir():
      if not directory.name.isdigit() or int(directory.name) == os.getpid():
        continue
      try:
        if any(os.readlink(fd) == '/dev/spidev0.0' for fd in (directory / 'fd').iterdir()):
          raise HardwareAdmissionError('Another process owns the internal panda SPI interface')
      except (FileNotFoundError, ProcessLookupError):
        continue

  def sample(self, allowed=()):
    # Publishers are stopped during exclusive maintenance. Do not pretend the
    # admission deviceState is refreshed: monitor Params, process ownership and
    # read-only physical control enforcement throughout this bounded window.
    settings = checks.offroad_params(self.params)
    self.sample_processes(allowed)
    current = usb_profile(self.identity)
    if current['device_node'] != self.device['device_node'] or current['serial_sha256'] != self.device['serial_sha256']:
      raise HardwareAdmissionError('Chestnut lost power or re-enumerated; diagnostic stopped')
    return {'software_settings': settings, 'panda': self.panda.sample(), 'chestnut_power': checks.bridge_power(self.handle),
            'temperatures_c': checks.thermal_guard(), 'mem_available_bytes': checks.memory_available(), 'monotonic': time.monotonic()}

  def __exit__(self, *_):
    if self.handle is not None:
      self.handle.close()
    if self.context is not None:
      self.context.close()
    if self.panda is not None:
      self.panda.close()


def run_worker(action, workspace, cache, guard, *, artifact=None, onnx=None, allow_runtime_downloads=False):
  preparing = action == 'compile'
  budget = checks.resources(shutil.disk_usage(cache).free, checks.memory_available(), model=action != 'probe', preparing=preparing)
  command = [sys.executable, '-m', 'tools.navigation_hardware.smoke_worker', action, '--output', str(workspace / (action + '.json'))]
  if artifact is not None:
    command += ['--artifact', str(artifact)]
  if onnx is not None:
    command += ['--onnx', str(onnx)]
  seconds = 1800 if preparing else 60 if action == 'probe' else 90
  started, peak = time.monotonic(), 0
  with (workspace / (action + '.log')).open('w') as log, (workspace / (action + '-telemetry.jsonl')).open('w') as telemetry:
    environment = worker_environment(cache)
    environment['NAV_SMOKE_ALLOW_RUNTIME_DOWNLOADS'] = '1' if allow_runtime_downloads else '0'
    process = subprocess.Popen(command, cwd=CHECKOUT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
      while process.poll() is None:
        sample = guard.sample(allowed=descendants(process.pid))
        sample['package_rss_bytes'] = resident_bytes(descendants(process.pid) | {os.getpid()})
        peak = max(peak, sample['package_rss_bytes'])
        telemetry.write(json.dumps(sample) + '\n')
        telemetry.flush()
        if peak > budget['worker_rss_limit_bytes'] or sample['mem_available_bytes'] < (512 << 20) \
           or shutil.disk_usage(cache).free < (256 << 20):
          raise HardwareAdmissionError('Memory/storage watchdog stopped the diagnostic before resource exhaustion')
        if time.monotonic() - started > seconds:
          raise HardwareAdmissionError(f'{action} exceeded its {seconds}s budget; no unverified artifact is admitted')
        time.sleep(.5)
      if process.returncode != 0:
        report_file = workspace / (action + '.json')
        detail = json.loads(report_file.read_text()).get('error') if report_file.is_file() else 'worker exited without a report'
        raise HardwareAdmissionError(f'{action} failed: {detail}; see {action}.log in results archive')
      guard.sample()
      result = json.loads((workspace / (action + '.json')).read_text())
      if result.get('passed') is not True:
        raise HardwareAdmissionError(action + ' did not produce a successful report')
      return result | {'command': command, 'budget': budget, 'elapsed_total_seconds': time.monotonic() - started, 'peak_package_rss_bytes': peak}
    finally:
      stop_own_child(process)


def obtain_onnx(cache, guard):
  import requests
  from openpilot.selfdrive.navigation.experimental_catalog import SOURCE_ONNX_SHA256
  output = cache / 'bigv2.onnx'
  if output.is_file():
    if output.stat().st_size == SOURCE_SIZE and sha256(output) == SOURCE_ONNX_SHA256:
      return output
    raise HardwareAdmissionError('Cached ONNX identity is wrong; retained for review, never silently overwritten')
  checks.resources(shutil.disk_usage(cache).free, checks.memory_available(), preparing=True)
  # An installed matching source can be used without a download. Existing model files are read-only.
  for path in (CHECKOUT / 'openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx',):
    if path.is_file() and path.stat().st_size == SOURCE_SIZE and sha256(path) == SOURCE_ONNX_SHA256:
      output.symlink_to(path)
      return output
  partial = cache / 'bigv2.onnx.partial'
  started, count = time.monotonic(), 0
  try:
    with requests.get(ONNX_URL, stream=True, timeout=(10, 10)) as response:
      response.raise_for_status()
      with partial.open('wb') as stream:
        for chunk in response.iter_content(1 << 20):
          guard.sample()
          if time.monotonic() - started > 900 or count + len(chunk) > SOURCE_SIZE or shutil.disk_usage(cache).free < checks.GIB:
            raise HardwareAdmissionError('ONNX download exceeded time/size/storage budget')
          stream.write(chunk)
          count += len(chunk)
    if count != SOURCE_SIZE or sha256(partial) != SOURCE_ONNX_SHA256:
      raise HardwareAdmissionError('Downloaded ONNX size/SHA256 differs from pinned Bigv2')
    partial.replace(output)
    return output
  except requests.RequestException as error:
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    raise HardwareAdmissionError(f'Pinned Bigv2 download failed: {type(error).__name__}, HTTP {status}; no redirected URL/token retained') from None
  finally:
    partial.unlink(missing_ok=True)


def readiness(result):
  return {'normal_activation_allowed': False, 'smoke_sufficient_for_approval': False,
          'observed': {'gpu_probe_passed': result.get('probe', {}).get('passed') is True,
                       'bigv2_smoke_passed': result.get('infer', {}).get('passed') is True},
          'remaining_before_activation': [
            'Verified native Bigv2 AMD model, matching camera warp and exact package hashes',
            'Native inference/recurrent outputs and equivalent-input CPU/GPU comparison',
            'Camera upload, timestamp association, route/GPS validity and recurrent reset integration',
            'Candidate plus warm QCOM fallback loading, timing and failure handover',
            'Automatic settings-based package preparation and explicit user activation authorization'],
          'optional_development_checks': ['Long synthetic sequences, endurance, thermal and power characterization',
                                         'Extended recorded-route and model-quality comparisons'],
          'scope': 'Software deployment requirements; this smoke is not driving approval or hardware certification'}


def job(workspace, mode):
  result = json.loads((workspace / 'report.json').read_text())
  try:
    fresh = initial_checks()  # Recheck before stopping the real service, not a cached admission decision.
    if fresh['source']['commit'] != result['source']['commit']:
      raise HardwareAdmissionError('Installed source changed after admission')
    result.update(status='stopping_service')
    write_json(workspace / 'report.json', result)
    systemctl('stop', 'comma.service')
    if service_active():
      raise HardwareAdmissionError('Normal service did not stop; GPU execution refused')
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    result['reused_runtime_download_links'] = stage_download_links(cache)
    artifact, onnx = cache / 'bigv2-amd.sptg', cache / 'bigv2.onnx'
    if mode == 'prepare' and not artifact.exists():
      checks.resources(shutil.disk_usage(cache).free, checks.memory_available(), preparing=True)
    with MaintenanceGuard(result['source'], fresh['usb']) as guard:
      result['stage'] = 'probing_gpu'
      write_json(workspace / 'report.json', result)
      result['probe'] = run_worker('probe', workspace, cache, guard, allow_runtime_downloads=mode == 'prepare')
      if mode == 'prepare' and not artifact.exists():
        result['stage'] = 'downloading_required_source'
        write_json(workspace / 'report.json', result)
        onnx = obtain_onnx(cache, guard)
        result['stage'] = 'compiling_verified_bigv2'
        write_json(workspace / 'report.json', result)
        result['compile'] = run_worker('compile', workspace, cache, guard, artifact=artifact, onnx=onnx)
        for public_file in (artifact, Path(str(artifact) + '.verification.json')):
          if not public_file.is_symlink():
            public_file.chmod(0o644)  # Public model data can later be reused by an approved UI preparer.
      if artifact.is_file():
        result['stage'] = 'running_synthetic_bigv2'
        write_json(workspace / 'report.json', result)
        result['infer'] = run_worker('infer', workspace, cache, guard, artifact=artifact, onnx=onnx if onnx.is_file() else None)
        result['status'] = 'bigv2_smoke_passed_not_activation_approval'
      else:
        result['status'] = 'gpu_probe_passed_bigv2_not_tested'
        result['blocker'] = ('Software preparation required: no verified native Bigv2 AMD artifact is installed in the diagnostic cache. ' +
                             'No model download or model compilation was started. Send this archive for compatible package preparation')
        result['artifact_requirement'] = {'format': 'SPTGPLN1', 'device': 'AMD', 'renderer': 'LLVM', 'arch': 'gfx1200', 'interface': 'USB',
                                          'tinygrad_ref': result['source'].get('tinygrad_ref'),
                                          'compiler_verification_required': True,
                                          'official_tinyjit_pkl_compatible': False}
    if source_identity(CHECKOUT) != result['source']:
      raise HardwareAdmissionError('Installed source/runtime changed during maintenance')
  except BaseException as error:
    result.update(status='failed', error=f'{type(error).__name__}: {error}')
  finally:
    result['activation_readiness'] = readiness(result)
    write_json(workspace / 'report.json', result)
  return 0 if result['status'] != 'failed' else 2


def archive(workspace):
  destination = workspace / 'results.tar.gz'
  temporary_archive = workspace / 'results.tar.gz.partial'
  with tarfile.open(temporary_archive, 'w:gz') as bundle:
    for path in sorted(workspace.iterdir()):
      if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.jsonl', '.log'):
        bundle.add(path, arcname=path.name)
  temporary_archive.replace(destination)
  latest = ROOT / 'latest-results.tar.gz'
  temporary = latest.with_suffix('.tmp')
  shutil.copyfile(destination, temporary)
  temporary.chmod(0o644)
  temporary.replace(latest)
  return latest


def finalize(workspace):
  """systemd ExecStopPost: runs after failure, timeout, cancellation or SSH loss."""
  result = json.loads((workspace / 'report.json').read_text())
  try:
    if result.get('original_service_active'):
      if not service_active():
        owners = process_snapshot(device_node=result.get('usb', {}).get('device_node'))
        if owners['conflicts'] or owners['inaccessible_pids']:
          raise HardwareAdmissionError('Remaining device/process owner prevents safe service restoration; inspect status before restarting')
        devices = usb_devices()
        if any(device['rom_mode'] or device['product'] != result['source']['expected_usb_product'] for device in devices):
          raise HardwareAdmissionError('Unexpected/ROM Chestnut firmware product: restoration held to avoid normal startup flashing; '
                                       + 'disconnect Chestnut USB before restoring comma.service and send this report')
      systemctl('start', 'comma.service')
      result['service_restored'] = service_active()
      if not result['service_restored']:
        raise HardwareAdmissionError('Original service did not return to active state')
    if result.get('status') not in ('failed', 'bigv2_smoke_passed_not_activation_approval', 'gpu_probe_passed_bigv2_not_tested'):
      result.update(status='aborted', error='Maintenance ended before completion (cancel, timeout, OOM or worker termination)')
  except BaseException as error:
    result.update(status='restoration_failed', restoration_error=str(error))
  finally:
    result['activation_readiness'] = readiness(result)
    write_json(workspace / 'report.json', result)
    archive(workspace)
    write_json(workspace / 'completion.json', {'service_restored': result.get('service_restored', False), 'status': result['status']})
  return 0 if result.get('service_restored') else 2


def unit_command(workspace, mode):
  finish = f'{sys.executable} -m tools.navigation_hardware.smoke --finalize {workspace}'
  memory_cap = '5G' if mode == 'prepare' or (ROOT / 'cache/bigv2-amd.sptg').exists() else '1G'
  return ['systemd-run', '--unit=' + UNIT, '--collect', '--property=Type=exec', '--property=RuntimeMaxSec=' + ('3300' if mode == 'prepare' else '240'),
          '--property=TimeoutStopSec=90', '--property=KillMode=control-group', '--property=MemoryMax=' + memory_cap, '--property=MemorySwapMax=0',
          '--property=CPUQuota=200%', '--property=OOMPolicy=stop',
          '--property=LimitCORE=0', '--property=WorkingDirectory=' + str(CHECKOUT), '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
          '--property=ExecStopPost=' + finish, sys.executable, '-m', 'tools.navigation_hardware.smoke',
          '--job', str(workspace), '--mode', mode]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.epilog = ('Default smoke never downloads models or compiles Bigv2. prepare is an optional, potentially long developer step; ' +
                   'it is not required for the short GPU check and is not recommended for a battery-powered vehicle session.')
  parser.add_argument('action', nargs='?', choices=('check', 'smoke', 'prepare', 'status', 'cancel'), default='smoke')
  parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
  parser.add_argument('--finalize', type=Path, help=argparse.SUPPRESS)
  parser.add_argument('--mode', choices=('smoke', 'prepare'), default='smoke', help=argparse.SUPPRESS)
  args = parser.parse_args()
  if os.geteuid() != 0:
    parser.exit(2, 'Use sudo so all process/device owners can be checked and the original service restored\n')
  if args.job or args.finalize:
    property_name = 'MainPID' if args.job else 'ControlPID'
    owner = systemctl('show', UNIT + '.service', '--property=' + property_name, '--value').stdout.strip()
    if owner != str(os.getpid()):
      parser.exit(2, 'Internal maintenance entrypoints require their registered systemd supervisor\n')
    workspace = (args.job or args.finalize).resolve(strict=True)
    if workspace.parent != ROOT.resolve() or not workspace.name.startswith('run-'):
      parser.error('Invalid internal maintenance workspace')
    if args.job:
      def interrupted(signum, frame):
        raise InterruptedError('Maintenance cancellation')
      signal.signal(signal.SIGTERM, interrupted)
      raise SystemExit(job(workspace, args.mode))
    raise SystemExit(finalize(workspace))
  if args.action in ('status', 'cancel'):
    if args.action == 'cancel':
      systemctl('stop', UNIT + '.service', check=False)
    current = ROOT / 'current-run.json'
    if current.is_file():
      workspace = Path(json.loads(current.read_text())['workspace'])
      report = json.loads((workspace / 'report.json').read_text())
      print(json.dumps({key: report[key] for key in ('status', 'stage', 'service_restored', 'error', 'restoration_error') if key in report}, indent=2))
    print(systemctl('status', UNIT + '.service', '--no-pager', check=False).stdout)
    print('Results: /data/navigation-smoke/latest-results.tar.gz')
    return
  ROOT.mkdir(mode=0o755, exist_ok=True)
  if systemctl('is-active', '--quiet', UNIT + '.service', check=False).returncode == 0:
    parser.exit(2, 'A diagnostic is already running; use status or cancel\n')
  workspace = ROOT / ('run-' + datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:6])
  workspace.mkdir(mode=0o700)
  write_json(ROOT / 'current-run.json', {'workspace': str(workspace)})
  try:
    result = initial_checks()
    result['status'] = 'admitted'
    result['resources'] = checks.resources(shutil.disk_usage(ROOT).free, checks.memory_available())
  except (HardwareAdmissionError, OSError, subprocess.SubprocessError, ValueError) as error:
    report = {'status': 'admission_failed', 'error': str(error), 'original_service_active': False,
              'activation_readiness': readiness({}), 'service_was_not_stopped': True}
    write_json(workspace / 'report.json', report)
    archive(workspace)
    parser.exit(2, str(error) + '\nReport: /data/navigation-smoke/latest-results.tar.gz\n')
  write_json(workspace / 'report.json', result)
  archive(workspace)  # A crash/power loss must not leave a previous successful run as the latest result.
  if args.action == 'check':
    print('Off-road/USB checks passed. GPU has not been opened. Results: /data/navigation-smoke/latest-results.tar.gz')
    return
  try:
    subprocess.run(unit_command(workspace, args.action), check=True, timeout=30)
  except (OSError, subprocess.SubprocessError) as error:
    # A registration timeout is ambiguous: the supervisor might have started.
    # Retain the original cleanup receipt; never claim the service was untouched.
    write_json(workspace / 'launcher-error.json', {'error': str(error)})
    archive(workspace)
    parser.exit(2, 'Supervisor registration failed/uncertain. Run status; if needed sudo systemctl start comma.service. Report retained.\n')
  print('Maintenance running independently of SSH. Original comma service will be restored automatically.', flush=True)
  print('Cancel safely: sudo .venv/bin/python -m tools.navigation_hardware.smoke cancel', flush=True)
  try:
    deadline = time.monotonic() + (3420 if args.action == 'prepare' else 360)
    previous_stage = None
    while time.monotonic() < deadline:
      current = json.loads((workspace / 'report.json').read_text())
      if current.get('stage') != previous_stage:
        previous_stage = current.get('stage')
        print('Stage: ' + str(previous_stage), flush=True)
      if (workspace / 'completion.json').is_file():
        report = json.loads((workspace / 'report.json').read_text())
        print(json.dumps({'status': report['status'], 'service_restored': report.get('service_restored'),
                          'error': report.get('error') or report.get('restoration_error'), 'blocker': report.get('blocker'),
                          'next_command': report.get('next_command'),
                          'timing_ms': report.get('infer', {}).get('inference', {}).get('timing_ms'),
                          'results': '/data/navigation-smoke/latest-results.tar.gz'}, indent=2))
        if report['status'] in ('failed', 'aborted', 'restoration_failed'):
          raise SystemExit(2)
        return
      time.sleep(2)
    print('Supervisor result not yet available; use status. Do not drive until normal comma.service is active.')
  except KeyboardInterrupt:
    print('\nSSH viewer stopped; supervised maintenance still owns cleanup. Use status or cancel after reconnecting.')
  except (OSError, ValueError, subprocess.SubprocessError) as error:
    print('Viewer error: ' + str(error) + '. Maintenance cleanup is owned by systemd; reconnect and use status.')


if __name__ == '__main__':
  main()
