"""Host-only tests of actual admission, maintenance cleanup, fixtures and native buffers."""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.navigation_hardware import smoke, smoke_checks as checks, smoke_worker as worker
from tools.navigation_hardware.safety import HardwareAdmissionError


class TestSmokeChecks(unittest.TestCase):
  def snapshot(self):
    return {'fresh': True, 'device_type': 'mici', 'started': False, 'is_offroad': True, 'is_engaged': False,
            'control_processes': dict.fromkeys(checks.CONTROL_PROCESSES, False),
            'control_states': {},
            'pandas': [{'type': 'cuatro', 'ignition_line': False, 'ignition_can': False, 'controls_allowed': False}]}

  def test_software_offroad_is_required_and_ignition_alone_is_not_a_veto(self):
    value = self.snapshot()
    checks.require_offroad(value)
    for field, invalid in (('fresh', False), ('device_type', 'tici'), ('started', True), ('is_offroad', False), ('is_engaged', True)):
      with self.subTest(field=field), self.assertRaises(HardwareAdmissionError):
        checks.require_offroad(value | {field: invalid})
    for field in ('ignition_line', 'ignition_can'):
      with self.subTest(field=field):
        changed = copy.deepcopy(value)
        changed['pandas'][0][field] = True
        checks.require_offroad(changed)
    changed = copy.deepcopy(value)
    changed['pandas'][0]['controls_allowed'] = True
    with self.assertRaisesRegex(HardwareAdmissionError, 'controls are allowed'):
      checks.require_offroad(changed)
    for state in ({}, dict.fromkeys(checks.CONTROL_PROCESSES, True)):
      with self.subTest(state=state), self.assertRaisesRegex(HardwareAdmissionError, 'processes'):
        checks.require_offroad(value | {'control_processes': state})
    for state in ({'fresh': True, 'valid': True, 'active': True}, {'fresh': True, 'valid': False, 'active': False}):
      with self.subTest(state=state), self.assertRaisesRegex(HardwareAdmissionError, 'carControl'):
        checks.require_offroad(value | {'control_states': {'carControl': state}})
    for pandas in ([], value['pandas'] * 2, [value['pandas'][0] | {'type': 'unknown'}]):
      with self.subTest(pandas=pandas), self.assertRaises(HardwareAdmissionError):
        checks.require_offroad(value | {'pandas': pandas})

  def test_physical_health_only_reads_never_constructs_panda_or_sets_can(self):
    from panda import Panda
    fields = [0] * 22
    fields[1], fields[9], fields[10] = 12000, 1, 19
    fields[20] = 65
    class ReadHandle:
      PROTOCOL_VERSION = 2
      def __init__(self):
        self.calls = []
        self.closed = False
      def get_protocol_version(self):
        return b'\0' * 13 + b'\xcc\x02'
      def controlRead(self, request_type, request, value, index, length, timeout):
        self.calls.append(request)
        if request == 0xdd:
          return struct.pack('<II', Panda.HEALTH_PACKET_VERSION, 0)
        if request == 0xd2:
          return Panda.HEALTH_STRUCT.pack(*fields)
        raise AssertionError('Unexpected control request')
      def close(self):
        self.closed = True
    handle = ReadHandle()
    with patch('panda.python.spi.PandaSpiHandle', return_value=handle), patch.object(Panda, '__init__', side_effect=AssertionError('No Panda constructor')):
      reader = checks.PandaSafetyReader()
      self.assertEqual(reader.sample()['safety_mode'], 19)
      for flags in (1, 2, 3):
        fields[8] = flags  # Physical ignition can be ON while software stays off-road.
        reader.sample()
      for flags, mode, controls in ((4, 19, 0), (0, 20, 0), (0, 19, 1), (0, 19, 2)):
        fields[8], fields[10], fields[21] = flags, mode, controls
        with self.subTest(flags=flags, mode=mode, controls=controls), self.assertRaises(HardwareAdmissionError):
          reader.sample()
      reader.close()
    self.assertTrue(handle.closed)
    self.assertEqual(set(handle.calls), {0xdd, 0xd2})

  def test_maintenance_params_are_rechecked_without_writing_them(self):
    values = {'IsOffroad': True, 'IsEngaged': False, 'NavigationModelDrivingEnabled': False}
    params = SimpleNamespace(get_bool=values.get)
    self.assertTrue(checks.offroad_params(params)['is_offroad'])
    for name, invalid in (('IsOffroad', False), ('IsEngaged', True), ('NavigationModelDrivingEnabled', True)):
      with self.subTest(name=name):
        original = values[name]
        values[name] = invalid
        with self.assertRaises(HardwareAdmissionError):
          checks.offroad_params(params)
        values[name] = original

  def test_power_and_resource_limits_are_checked_without_larger_hardware(self):
    handle = SimpleNamespace(controlRead=lambda *args, **kwargs: struct.pack('<Hh?', 12000, 1500, False))
    self.assertEqual(checks.bridge_power(handle)['observed_watts'], 18)
    for voltage, current, fault in ((0, 0, False), (12000, 15000, False), (12000, -1, False), (12000, 1000, True)):
      handle.controlRead = lambda *args, v=voltage, c=current, f=fault, **kwargs: struct.pack('<Hh?', v, c, f)
      with self.subTest(voltage=voltage, current=current, fault=fault), self.assertRaises(HardwareAdmissionError):
        checks.bridge_power(handle)
    self.assertEqual(checks.resources(512 << 20, checks.GIB)['worker_rss_limit_bytes'], 512 << 20)
    checks.resources(4 * checks.GIB, 5 * checks.GIB, preparing=True)
    for disk, ram in ((checks.GIB, 5 * checks.GIB), (4 * checks.GIB, 4 * checks.GIB)):
      with self.subTest(disk=disk, ram=ram), self.assertRaises(HardwareAdmissionError):
        checks.resources(disk, ram, preparing=True)

  def test_usb_product_and_disconnect_never_trigger_firmware_recovery(self):
    identity = {'expected_usb_product': 'custom 8516b768-CLEAN'}
    good = {'product': identity['expected_usb_product'], 'rom_mode': False, 'device_node': '/dev/bus/usb/001/002', 'speed': '10000'}
    with patch.object(smoke, 'usb_devices', return_value=[good]):
      self.assertEqual(smoke.usb_profile(identity), good)
    for devices in ([], [good, good], [good | {'rom_mode': True}], [good | {'speed': '480'}], [good | {'product': 'custom ed4e39b7-CLEAN'}]):
      with self.subTest(devices=devices), patch.object(smoke, 'usb_devices', return_value=devices), self.assertRaises(HardwareAdmissionError):
        smoke.usb_profile(identity)
    with patch.object(smoke, 'usb_devices', return_value=[good | {'product': 'custom ed4e39b7-CLEAN'}]), \
         self.assertRaisesRegex(HardwareAdmissionError, 'Observed:.*custom ed4e39b7-CLEAN'):
      smoke.usb_profile(identity)


class TestMaintenance(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    self.workspace = self.root / 'run-test'
    self.workspace.mkdir()
    self.identity = {'commit': 'a' * 40}
    self.receipt = {'source': self.identity, 'original_service_active': True, 'usb': {'device_node': 'test'}, 'status': 'admitted'}
    smoke.write_json(self.workspace / 'report.json', self.receipt)
    self.root_patch = patch.object(smoke, 'ROOT', self.root)
    self.root_patch.start()
    self.addCleanup(self.root_patch.stop)

  def test_transient_unit_owns_cleanup_after_timeout_oom_and_ssh_loss(self):
    command = smoke.unit_command(self.workspace, 'smoke')
    self.assertTrue(any(arg.startswith('--property=ExecStopPost=') and '--finalize' in arg for arg in command))
    self.assertIn('--property=KillMode=control-group', command)
    self.assertIn('--property=OOMPolicy=stop', command)
    self.assertIn('--property=RuntimeMaxSec=240', command)
    self.assertIn('--property=MemoryMax=1G', command)
    self.assertIn('--property=MemoryMax=5G', smoke.unit_command(self.workspace, 'prepare'))
    self.assertNotIn('bash', command)
    self.assertNotIn('--scope', command)  # Detached service survives the SSH viewer.
    smoke.write_json(self.workspace / 'report.json', self.receipt | {'status': 'stopping_service'})
    with patch.object(smoke, 'systemctl') as ctl, patch.object(smoke, 'service_active', return_value=True):
      self.assertEqual(smoke.finalize(self.workspace), 0)
      ctl.assert_called_once_with('start', 'comma.service')
    report = json.loads((self.workspace / 'report.json').read_text())
    self.assertEqual(report['status'], 'aborted')
    self.assertTrue(report['service_restored'])
    self.assertTrue((self.workspace / 'completion.json').is_file())
    with tarfile.open(self.root / 'latest-results.tar.gz') as archive:
      self.assertIn('report.json', archive.getnames())
      member = archive.extractfile('report.json')
      assert member is not None
      self.assertEqual(json.load(member)['status'], 'aborted')

  def test_restoration_failure_is_explicit_and_still_archived(self):
    with patch.object(smoke, 'systemctl', side_effect=subprocess.CalledProcessError(1, ['systemctl'])), \
         patch.object(smoke, 'service_active', return_value=False):
      self.assertEqual(smoke.finalize(self.workspace), 2)
    report = json.loads((self.workspace / 'report.json').read_text())
    self.assertEqual(report['status'], 'restoration_failed')
    self.assertTrue((self.root / 'latest-results.tar.gz').is_file())

  def test_leftover_device_owner_prevents_restart_over_a_hung_worker(self):
    with patch.object(smoke, 'service_active', return_value=False), patch.object(smoke, 'systemctl') as ctl, \
         patch.object(smoke, 'process_snapshot', return_value={'conflicts': [{'pid': 99}], 'inaccessible_pids': []}):
      self.assertEqual(smoke.finalize(self.workspace), 2)
      ctl.assert_not_called()
    report = json.loads((self.workspace / 'report.json').read_text())
    self.assertEqual(report['status'], 'restoration_failed')
    self.assertIn('owner', report['restoration_error'])

  def test_rom_or_changed_firmware_holds_restoration_without_flashing(self):
    self.identity['expected_usb_product'] = 'custom 8516b768-CLEAN'
    smoke.write_json(self.workspace / 'report.json', self.receipt)
    with patch.object(smoke, 'service_active', return_value=False), patch.object(smoke, 'systemctl') as ctl, \
         patch.object(smoke, 'process_snapshot', return_value={'conflicts': [], 'inaccessible_pids': []}), \
         patch.object(smoke, 'usb_devices', return_value=[{'rom_mode': True, 'product': 'AS2462'}]):
      self.assertEqual(smoke.finalize(self.workspace), 2)
      ctl.assert_not_called()
    self.assertIn('firmware product', json.loads((self.workspace / 'report.json').read_text())['restoration_error'])

  def test_disconnected_chestnut_allows_original_service_fallback_restoration(self):
    with patch.object(smoke, 'service_active', side_effect=[False, True]), patch.object(smoke, 'systemctl') as ctl, \
         patch.object(smoke, 'process_snapshot', return_value={'conflicts': [], 'inaccessible_pids': []}), \
         patch.object(smoke, 'usb_devices', return_value=[]):
      self.assertEqual(smoke.finalize(self.workspace), 0)
      ctl.assert_called_once_with('start', 'comma.service')
    self.assertTrue(json.loads((self.workspace / 'report.json').read_text())['service_restored'])

  def test_no_download_or_cpu_compilation_when_artifact_is_missing(self):
    guard = SimpleNamespace(__enter__=lambda self: self, __exit__=lambda *args: None)
    class Guard:
      def __enter__(self):return guard
      def __exit__(self, *args):return None
    with patch.object(smoke, 'initial_checks', return_value=self.receipt), \
         patch.object(smoke, 'systemctl'), patch.object(smoke, 'service_active', return_value=False), \
         patch.object(smoke, 'stage_download_links', return_value=0), patch.object(smoke, 'MaintenanceGuard', return_value=Guard()), \
         patch.object(smoke, 'run_worker', return_value={'passed': True}) as worker, \
         patch.object(smoke, 'obtain_onnx', side_effect=AssertionError('No implicit model download')), \
         patch.object(smoke, 'source_identity', return_value=self.identity):
      self.assertEqual(smoke.job(self.workspace, 'smoke'), 0)
      self.assertEqual([call.args[0] for call in worker.call_args_list], ['probe'])
    report = json.loads((self.workspace / 'report.json').read_text())
    self.assertEqual(report['status'], 'gpu_probe_passed_bigv2_not_tested')
    self.assertFalse(report['activation_readiness']['normal_activation_allowed'])
    self.assertNotIn('next_command', report)
    self.assertIn('Software preparation required', report['blocker'])

  def test_cached_verified_artifact_can_be_tested_without_onnx_download(self):
    class Guard:
      def __enter__(self):return self
      def __exit__(self, *args):return None
    cache = self.root / 'cache'
    cache.mkdir()
    (cache / 'bigv2-amd.sptg').write_bytes(b'inert artifact fixture; verified by real worker in separate tests')
    with patch.object(smoke, 'initial_checks', return_value=self.receipt), patch.object(smoke, 'systemctl'), \
         patch.object(smoke, 'service_active', return_value=False), patch.object(smoke, 'stage_download_links', return_value=0), \
         patch.object(smoke, 'MaintenanceGuard', return_value=Guard()), patch.object(smoke, 'run_worker', return_value={'passed': True}) as worker, \
         patch.object(smoke, 'obtain_onnx', side_effect=AssertionError('No source download for smoke')), \
         patch.object(smoke, 'source_identity', return_value=self.identity):
      self.assertEqual(smoke.job(self.workspace, 'smoke'), 0)
    self.assertEqual([call.args[0] for call in worker.call_args_list], ['probe', 'infer'])
    self.assertIsNone(worker.call_args.kwargs['onnx'])

  def test_explicit_prepare_uses_only_native_amd_compiler_then_smoke(self):
    class Guard:
      def __enter__(self):return self
      def __exit__(self, *args):return None
    cache = self.root / 'cache'
    def source(*args):
      path = cache / 'bigv2.onnx'
      path.write_bytes(b'inert source fixture')
      return path
    def compute(action, *args, **kwargs):
      if action == 'compile':
        (cache / 'bigv2-amd.sptg').write_bytes(b'inert artifact fixture')
        (cache / 'bigv2-amd.sptg.verification.json').write_text('{}')
      return {'passed': True}
    with patch.object(smoke, 'initial_checks', return_value=self.receipt), patch.object(smoke, 'systemctl'), \
         patch.object(smoke, 'service_active', return_value=False), patch.object(smoke, 'stage_download_links', return_value=0), \
         patch.object(smoke, 'MaintenanceGuard', return_value=Guard()), patch.object(checks, 'resources'), \
         patch.object(smoke, 'obtain_onnx', side_effect=source), patch.object(smoke, 'run_worker', side_effect=compute) as worker, \
         patch.object(smoke, 'source_identity', return_value=self.identity):
      self.assertEqual(smoke.job(self.workspace, 'prepare'), 0)
    self.assertEqual([call.args[0] for call in worker.call_args_list], ['probe', 'compile', 'infer'])
    report = json.loads((self.workspace / 'report.json').read_text())
    self.assertEqual(report['status'], 'bigv2_smoke_passed_not_activation_approval')
    self.assertFalse(report['activation_readiness']['normal_activation_allowed'])
    self.assertIn('optional_development_checks', report['activation_readiness'])

  def test_worker_failure_retains_cleanup_receipt_and_cannot_approve_activation(self):
    with patch.object(smoke, 'initial_checks', side_effect=HardwareAdmissionError('ignition switched on')), patch.object(smoke, 'systemctl') as ctl:
      self.assertEqual(smoke.job(self.workspace, 'smoke'), 2)
      ctl.assert_not_called()
    report = json.loads((self.workspace / 'report.json').read_text())
    self.assertEqual(report['status'], 'failed')
    self.assertTrue(report['original_service_active'])
    self.assertFalse(report['activation_readiness']['smoke_sufficient_for_approval'])
    with patch.object(smoke, 'systemctl'), patch.object(smoke, 'service_active', return_value=True):
      smoke.finalize(self.workspace)
    self.assertTrue(json.loads((self.workspace / 'report.json').read_text())['service_restored'])

  def test_watchdog_kills_only_its_own_worker_on_controls_or_power_loss(self):
    guard = SimpleNamespace(sample=lambda **kwargs: (_ for _ in ()).throw(HardwareAdmissionError('power lost')))
    with patch.object(smoke, 'worker_environment', return_value=os.environ.copy()), \
         patch.object(checks, 'resources', return_value={'worker_rss_limit_bytes': 4 * checks.GIB}), \
         patch.object(smoke, 'descendants', side_effect=lambda pid: {pid}), \
         patch.object(smoke, 'stop_own_child', wraps=smoke.stop_own_child) as stop:
      # Substitute an inert real subprocess, keeping actual process-group termination behavior.
      real_popen = subprocess.Popen
      with patch.object(smoke.subprocess, 'Popen') as spawn:
        spawn.side_effect = lambda *args, **kwargs: real_popen([sys_executable(), '-c', 'import time;time.sleep(30)'], **kwargs)
        with self.assertRaisesRegex(HardwareAdmissionError, 'power lost'):
          smoke.run_worker('probe', self.workspace, self.root, guard)
      self.assertEqual(stop.call_count, 1)
      self.assertIsNotNone(stop.call_args.args[0].returncode)

  def test_model_weights_and_cache_are_excluded_from_archive(self):
    (self.workspace / 'weights.sptg').write_bytes(b'not a model')
    (self.workspace / 'cache').mkdir()
    (self.workspace / 'cache/private.json').write_text('{}')
    smoke.archive(self.workspace)
    with tarfile.open(self.root / 'latest-results.tar.gz') as archive:
      self.assertEqual(archive.getnames(), ['report.json'])


def sys_executable():
  import sys
  return sys.executable


class TestSyntheticInference(unittest.TestCase):
  def contract(self):
    return json.loads((smoke.CHECKOUT / 'openpilot/selfdrive/navigation/experimental_models.json').read_text())['entries'][0]['contract']

  def specs(self):
    return {kind + '_specs': {r['name']: (tuple(r['shape']), {'FLOAT': 'float32', 'UINT8': 'uint8'}[r['dtype']], 'AMD')
                             for r in self.contract()[kind + 's']} for kind in ('input', 'output')}

  def test_built_in_arrays_use_exact_shapes_dtypes_raw_map_and_no_recordings(self):
    loaded = self.specs()
    worker.validate_loaded_contract(loaded, self.contract())
    for arm in ('neutral', 'map_only', 'maneuver_only', 'both'):
      arrays = worker.synthetic_inputs(loaded['input_specs'], 1, arm)
      self.assertEqual(arrays['new_img'].shape, (2, 6, 128, 256))
      self.assertEqual(arrays['new_img'].dtype, np.uint8)
      self.assertEqual(arrays['nav_map'].dtype, np.float32)
      self.assertEqual(float(arrays['nav_map'].max()), 255 if arm in ('map_only', 'both') else 0)
      self.assertEqual(np.count_nonzero(arrays['navigation']), 2 if arm in ('maneuver_only', 'both') else 0)
    for kind, name, wrong in (('input_specs', 'nav_map', ((1, 1, 256, 256), 'uint8', 'AMD')),
                               ('output_specs', 'outputs', ((1, 18580), 'float32', 'CPU'))):
      changed = copy.deepcopy(loaded)
      changed[kind][name] = wrong
      with self.subTest(name=name), self.assertRaises(ValueError):
        worker.validate_loaded_contract(changed, self.contract())

  def test_native_tensor_state_aliases_really_advance_and_reject_nan(self):
    from tinygrad import Tensor
    loaded: dict = {kind: {name: (shape, dtype, 'CPU') for name, (shape, dtype, _) in specs.items()} for kind, specs in self.specs().items()}
    def run(output_buffers, **inputs):
      from tinygrad.engine.jit import _prepare_jit_inputs
      slots, _, _, _ = _prepare_jit_inputs((), dict(output_buffers=output_buffers, **inputs))
      self.assertEqual(len(slots), 17, 'Five distinct state-output views must preserve the captured native JIT slots')
      image = inputs['state_img_q'].numpy().copy()
      image[:, :-1] = image[:, 1:]
      image[:, -1] = inputs['new_img'].numpy()
      output_buffers['next_state_img_q'].assign(Tensor(image, device='CPU')).realize()
      value = float(inputs['new_img'].numpy().flat[0])
      output_buffers['outputs'].assign(value).realize()
    loaded['run'] = run
    result = worker.infer_sequence(loaded, SimpleNamespace(synchronize=lambda: None))
    self.assertEqual(len(result['predictions']), 8)
    self.assertEqual(len(result['recurrent_pairs']), 5)
    loaded['run'] = lambda output_buffers, **kwargs: output_buffers['outputs'].assign(float('nan')).realize()
    with self.assertRaisesRegex(ValueError, 'finite'):
      worker.infer_sequence(loaded, SimpleNamespace(synchronize=lambda: None))

  def test_actual_static_amd_proof_and_header_bindings_are_required_before_loading(self):
    from openpilot.selfdrive.navigation.tests.test_experimental_export import TestExperimentalExport
    from openpilot.selfdrive.navigation.experimental_catalog import SOURCE_ONNX_SHA256, ExperimentalCatalogError
    fixture = TestExperimentalExport('runTest')
    fixture.setUp()
    self.addCleanup(fixture.doCleanups)
    artifact = fixture.root / 'model.sptg'
    report = Path(str(artifact) + '.verification.json')
    report.write_text(json.dumps(fixture.report))
    original_sha = worker.sha256
    # Only unavailable814MB ONNX bytes are substituted; header/report/digests execute real checks.
    with patch.object(worker, 'sha256', side_effect=lambda p: SOURCE_ONNX_SHA256 if Path(p) == fixture.source else original_sha(p)):
      proof = worker.verified_artifact(artifact, fixture.source, fixture.compiler)
      self.assertEqual(proof['artifact_sha256'], original_sha(artifact))
      self.assertTrue(proof['source_onnx_bytes_rechecked'])
      cached_proof = worker.verified_artifact(artifact, None, fixture.compiler)
      self.assertFalse(cached_proof['source_onnx_bytes_rechecked'])
      self.assertEqual(proof['source_onnx_sha256'], cached_proof['source_onnx_sha256'])
      failed = copy.deepcopy(fixture.report)
      failed['backend']['kind'] = 'host_cpu'
      report.write_text(json.dumps(failed))
      with self.assertRaises(ExperimentalCatalogError):
        worker.verified_artifact(artifact, fixture.source, fixture.compiler)
      report.write_text(json.dumps(fixture.report))
      artifact.write_bytes(artifact.read_bytes() + b'tampered')
      with self.assertRaises(ExperimentalCatalogError):
        worker.verified_artifact(artifact, fixture.source, fixture.compiler)

  def test_gpu_worker_cannot_publish_subscribe_import_controllers_or_download_in_smoke(self):
    import importlib.util
    import urllib.request
    from openpilot.cereal import messaging
    with worker.isolated_io(allow_downloads=False):
      with self.assertRaises(RuntimeError):
        messaging.PubMaster(['modelV2'])
      with self.assertRaises(RuntimeError):
        messaging.SubMaster(['deviceState'])
      with self.assertRaisesRegex(RuntimeError, 'not cached'):
        urllib.request.urlopen('https://example.invalid/weight')
      # find_spec is exercised directly to avoid depending on modules loaded by other tests.
      guard = next(g for g in __import__('sys').meta_path if type(g).__name__ == 'NoControllers')
      with self.assertRaises(RuntimeError):
        guard.find_spec('openpilot.selfdrive.controls.controlsd', None)
      with self.assertRaises(RuntimeError):
        guard.find_spec('panda.python', None)
      self.assertIsNone(guard.find_spec('numpy', None))
      self.assertTrue(importlib.util.find_spec('numpy'))


class TestNativeOffroadAdmission(unittest.TestCase):
  def exercise(self, *, stale=False, running_control=False, active_control=False, ignition=True):
    import threading
    import time
    from openpilot.cereal import messaging
    from openpilot.common.hardware import HARDWARE
    from openpilot.common.params import Params
    from openpilot.common.prefix import OpenpilotPrefix
    with OpenpilotPrefix():
      params = Params()
      params.put_bool('IsOffroad', True, block=True)
      params.put_bool('IsEngaged', False, block=True)
      finished = threading.Event()
      def publish():
        pm = messaging.PubMaster(['deviceState', 'pandaStates', 'managerState', 'carControl', 'selfdriveState'])
        while not finished.is_set():
          device = messaging.new_message('deviceState')
          device.valid = True
          device.deviceState.started = False
          device.deviceState.powerDrawW = 9
          device.logMonoTime = time.monotonic_ns() - (3_000_000_000 if stale else 0)
          pandas = messaging.new_message('pandaStates', 1)
          pandas.valid = True
          pandas.pandaStates[0].pandaType = 'cuatro'
          pandas.pandaStates[0].ignitionLine = ignition
          pandas.pandaStates[0].ignitionCan = ignition
          pandas.logMonoTime = time.clock_gettime_ns(time.CLOCK_BOOTTIME) - (3_000_000_000 if stale else 0)
          manager = messaging.new_message('managerState', valid=True)
          manager.managerState.processes = [{'name': name, 'running': running_control} for name in sorted(checks.CONTROL_PROCESSES)]
          control = messaging.new_message('carControl', valid=True)
          control.carControl.enabled = active_control
          control.carControl.latActive = active_control
          state = messaging.new_message('selfdriveState', valid=True)
          state.selfdriveState.enabled = active_control
          state.selfdriveState.active = active_control
          state.selfdriveState.state = 'enabled' if active_control else 'disabled'
          pm.send('managerState', manager)
          pm.send('carControl', control)
          pm.send('selfdriveState', state)
          pm.send('deviceState', device)
          pm.send('pandaStates', pandas)
          finished.wait(.05)
      thread = threading.Thread(target=publish, daemon=True)
      thread.start()
      try:
        with patch.object(HARDWARE, 'get_device_type', return_value='mici'):
          return checks.live_offroad_snapshot(timeout=3)
      finally:
        finished.set()
        thread.join(3)

  def test_native_msgq_and_params_admit_only_fresh_genuine_offroad_controls(self):
    result = self.exercise()
    self.assertTrue(result['is_offroad'])
    self.assertTrue(result['pandas'][0]['ignition_line'])
    self.assertFalse(result['control_states']['carControl']['active'])
    with self.assertRaisesRegex(HardwareAdmissionError, 'timestamps'):
      self.exercise(stale=True)
    with self.assertRaisesRegex(HardwareAdmissionError, 'processes'):
      self.exercise(running_control=True)
    with self.assertRaisesRegex(HardwareAdmissionError, 'active'):
      self.exercise(active_control=True)


if __name__ == '__main__':
  unittest.main()
