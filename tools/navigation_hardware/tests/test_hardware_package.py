"""Host-only integration tests over fake sysfs/proc and inert prepared arrays."""
import copy
import json
import io
from pathlib import Path
from types import SimpleNamespace
import signal
from contextlib import redirect_stdout
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.navigation_hardware import __main__ as cli, comparison, runner, safety, worker
from tools.navigation_hardware.safety import HardwareAdmissionError


class TestHardwarePackage(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.checkout = self.root / 'checkout'
    (self.checkout / 'tools').mkdir(parents=True)
    (self.checkout / 'tools/navigation_model_diagnostic.py').touch()
    self.workspace = self.root / 'results'
    self.sys = self.root / 'sys'
    self.proc = self.root / 'proc'
    (self.sys / 'bus/usb/devices').mkdir(parents=True)
    self.proc.mkdir()
    self.usb = self.sys / 'bus/usb/devices/1-2'
    self.usb.mkdir()
    for name, value in {'idVendor': 'add1', 'idProduct': '0001', 'product': 'custom 8516b768-CLEAN',
                        'busnum': '1', 'devnum': '2', 'speed': '10000', 'serial': 'private-device-serial'}.items():
      (self.usb / name).write_text(value)
    self.identity = {'commit': 'a' * 40, 'tinygrad_ref': '69fd4d34d4430036378aba2b43f9d339cdd6ea73',
                     'expected_usb_product': 'custom 8516b768-CLEAN'}

  def process(self, pid, name='python', args=(), fds=()):
    directory = self.proc / str(pid)
    directory.mkdir()
    (directory / 'comm').write_text(name)
    (directory / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in args))
    (directory / 'fd').mkdir()
    for index, target in enumerate(fds):
      (directory / 'fd' / str(index)).symlink_to(target)
    return directory

  def inspect(self):
    with patch.object(safety, 'source_identity', return_value=self.identity), \
         patch('shutil.disk_usage', return_value=SimpleNamespace(free=16 << 30)):
      return safety.preflight(self.checkout, self.workspace, sys_root=self.sys, proc_root=self.proc)

  def test_sysfs_proc_inspection_is_read_only_and_does_not_invent_usb_gpu_architecture(self):
    before = {str(path): path.read_bytes() for path in self.sys.rglob('*') if path.is_file()}
    report = self.inspect()
    self.assertTrue(report['admitted'])
    self.assertEqual(report['usb'][0]['device_node'], '/dev/bus/usb/001/002')
    self.assertNotIn('serial', report['usb'][0])
    self.assertEqual(report['gpu']['chestnut_architecture'], 'unknown_until_explicit_compute_initialization')
    self.assertEqual(before, {str(path): path.read_bytes() for path in self.sys.rglob('*') if path.is_file()})
    self.assertFalse(report['persistent_firmware_operations'])

  def test_production_daemons_and_any_competing_gpu_or_selected_bridge_owner_are_rejected(self):
    for index, role in enumerate(safety.PRODUCTION):
      with self.subTest(role=role):
        directory = self.process(1000 + index, args=('python', f'/data/openpilot/{role}.py'))
        self.assertFalse(self.inspect()['admitted'])
        for path in (directory / 'fd').iterdir():
          path.unlink()
        (directory / 'fd').rmdir()
        (directory / 'comm').unlink()
        (directory / 'cmdline').unlink()
        directory.rmdir()
    for target in ('/dev/kfd', '/dev/dri/renderD128', '/dev/bus/usb/001/002'):
      with self.subTest(fd=target):
        directory = self.process(2222, fds=(target,))
        self.assertFalse(self.inspect()['admitted'])
        for path in (directory / 'fd').iterdir():
          path.unlink()
        (directory / 'fd').rmdir()
        (directory / 'comm').unlink()
        (directory / 'cmdline').unlink()
        directory.rmdir()
    self.process(3333, args=('python', 'hardwared.py'), fds=('/dev/bus/usb/001/002',))
    # Own worker descendants may own the GPU, but are never exempt from daemon-role checks.
    self.assertTrue(safety.process_snapshot(self.proc, allowed_pids=(3333,))['conflicts'])

  def test_bootloader_mismatch_and_production_paths_fail_without_firmware_or_process_operations(self):
    for product in ('custom ed4e39b7-CLEAN', 'AS2462'):
      (self.usb / 'product').write_text(product)
      self.assertFalse(self.inspect()['admitted'])
    (self.usb / 'idVendor').write_text('174c')
    (self.usb / 'idProduct').write_text('2464')
    self.assertTrue(self.inspect()['usb'][0]['rom_mode'])
    with self.assertRaises(HardwareAdmissionError):
      safety.separate_paths(self.checkout, self.checkout / 'results')
    with self.assertRaises(HardwareAdmissionError):
      safety.separate_paths(self.checkout, self.workspace, protected=(str(self.checkout),))

  def test_discovered_compute_architecture_interface_and_renderer_are_all_required(self):
    USBIface = type('USBIface', (), {})
    device = SimpleNamespace(arch='gfx1200', is_usb=True, iface=USBIface(), renderer=SimpleNamespace(
      target=SimpleNamespace(device='AMD', renderer='LLVM', arch='gfx1200', interface='USB')))
    self.assertEqual(worker.admitted_device(device)['architecture'], 'gfx1200')
    for attribute, wrong in (('arch', 'gfx1100'), ('is_usb', False), ('iface', object())):
      with self.subTest(attribute=attribute):
        changed = copy.copy(device)
        setattr(changed, attribute, wrong)
        with self.assertRaises(ValueError):
          worker.admitted_device(changed)
    with self.assertRaises(RuntimeError):
      worker.NoPersistentFlash().find_spec('openpilot.system.hardware.chestnut.flash')
    self.assertIsNone(worker.NoPersistentFlash().find_spec('tinygrad.runtime.ops_amd'))

  def recording(self):
    path = self.root / 'recording.npz'
    count, size = 164, 128 * (32 + 16)
    np.savez_compressed(path, img=np.zeros((count, size), np.uint8), big_img=np.zeros((count, size), np.uint8),
                        M_img=np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)),
                        M_big_img=np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)),
                        desire=np.zeros((count, 8), np.float32), traffic_convention=np.tile(np.array([1, 0], np.float32), (count, 1)),
                        action_t=np.full((count, 2), 0.1, np.float32),
                        camera_timestamp_ns=np.arange(count, dtype=np.int64) * 50_000_000 + 1,
                        big_camera_timestamp_ns=np.arange(count, dtype=np.int64) * 50_000_000 + 1,
                        frame_id=np.arange(count, dtype=np.uint32), big_frame_id=np.arange(count, dtype=np.uint32))
    return path

  def test_plan_has_bootstrap_before_export_correct_warp_and_explicit_long_sequence(self):
    source = self.root / 'fixture.onnx'
    source.write_bytes(b'inert source identity fixture')
    recording = self.recording()
    with patch.object(runner, 'sha256', return_value=runner.SOURCE_ONNX_SHA256):
      plan = runner.build_plan(self.checkout, self.workspace, source, recording, width=16, height=16, synthetic_navigation=True)
    names = [step['name'] for step in plan['steps']]
    self.assertLess(names.index('compile-amd'), names.index('export-amd'))
    self.assertLess(names.index('warp-amd'), names.index('export-amd'))
    warp = next(step for step in plan['steps'] if step['name'] == 'warp-amd')
    self.assertIn('16,16,128,32,16,6144', warp['arguments'])
    self.assertIn('--synthetic-long-sequence', plan['steps'][-1]['arguments'])
    self.assertNotIn('--capture-recurrent-state', plan['steps'][-1]['arguments'])
    self.assertTrue(all('--capture-recurrent-state' in step['arguments'] for step in plan['steps'] if step['name'].endswith('comparison')))
    self.assertEqual(plan['memory_limit_bytes'], 4 << 30)
    command = runner.worker_command(plan, plan['steps'][0])
    self.assertIn('tools.navigation_hardware.worker', command)
    self.assertFalse(self.workspace.exists())
    with self.assertRaisesRegex(HardwareAdmissionError, 'official digest'):
      runner.build_plan(self.checkout, self.workspace, source, recording, width=16, height=16)

  def predictions(self, root, backend, *, value=0.5, frames=4, long=False):
    root.mkdir()
    (root / 'provenance.json').write_text(json.dumps({'backend': backend, 'recording_sha256': 'recording-fixture'}))
    events = []
    for index in range(frames):
      reset = index in (0, 144, 145)
      event = {'event': 'skipped' if long and index == 144 else 'frame', 'arm': 'both', 'frame': index,
               'camera_ns': index + 1, 'generation': 1 if index < 144 else 2,
               'reason': 'invalid' if long and index == 144 else 'valid', 'reset': reset,
               'reset_reason': 'transition' if reset else None, 'model_role': 'primary', 'scenario': 'explicit synthetic fixture'}
      if not long:
        path = root / f'{index}.npz'
        np.savez(path, plan=np.full((1, 33, 15), value, np.float32), action=np.array([[0.25, 0.5]], np.float32))
        event.update(prediction_file=path.name, prediction_sha256=safety.sha256(path))
      events.append(event)
    (root / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    (root / 'summary.json').write_text(json.dumps({'frames_per_arm': frames, 'arms': ['both'], 'failures': 0}))

  def test_comparison_uses_identical_prepared_file_and_temporal_identity_with_explicit_tolerances(self):
    cpu, gpu = self.root / 'cpu', self.root / 'gpu'
    self.predictions(cpu, 'CPU')
    self.predictions(gpu, 'AMD', value=0.5001)
    result = comparison.compare_results(cpu, gpu, expected_recording_sha256='recording-fixture')
    self.assertTrue(result['passed'])
    self.assertEqual(result['paired_predictions'], 4)
    with self.assertRaises(HardwareAdmissionError):
      comparison.compare_results(cpu, gpu, expected_recording_sha256='another-file')
    path = gpu / '0.npz'
    np.savez(path, plan=np.zeros((1, 33, 15), np.float32), action=np.array([[0.25, 0.5]], np.float32))
    events = [json.loads(line) for line in (gpu / 'events.jsonl').read_text().splitlines()]
    events[0]['prediction_sha256'] = safety.sha256(path)
    (gpu / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    self.assertFalse(comparison.compare_results(cpu, gpu)['passed'])
    with self.assertRaises(HardwareAdmissionError):
      comparison.compare_results(cpu, gpu, atol=1)

  def test_long_validation_requires_full_history_then_reset_and_recovery(self):
    root = self.root / 'long'
    self.predictions(root, 'AMD', frames=164, long=True)
    result = runner.validate_long_run(root)
    self.assertGreaterEqual(result['arms']['both']['maximum_consecutive_valid_primary_frames'], 132)
    self.assertFalse(result['driving_fallback_validated'])
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    events[128].update(event='skipped', reason='invalid', reset=True)
    (root / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    with self.assertRaisesRegex(HardwareAdmissionError, '132 history'):
      runner.validate_long_run(root)
    (root / 'summary.json').write_text(json.dumps({'frames_per_arm': 8, 'arms': ['both'], 'failures': 0}))
    with self.assertRaises(HardwareAdmissionError):
      runner.validate_long_run(root)

  def test_archive_excludes_weights_input_cameras_caches_and_unrelated_files(self):
    self.workspace.mkdir()
    for folder in ('artifacts', 'cache', 'logs'):
      (self.workspace / folder).mkdir()
    (self.workspace / 'plan.json').write_text(json.dumps({'schema_version': 1, 'workspace': str(self.workspace), 'normal_prediction_publishers': False}))
    (self.workspace / 'artifacts/model.sptg').write_bytes(b'weights must not be archived')
    (self.workspace / 'artifacts/model.verification.json').write_text('{}')
    (self.workspace / 'cache/cache.db').write_bytes(b'cache')
    (self.workspace / 'recording.npz').write_bytes(b'private camera fixture')
    (self.workspace / 'unrelated-secret.txt').write_text('unrelated fixture')
    (self.workspace / 'logs/compiler.log').write_text('fixture log')
    path = runner.collect_results(self.workspace)
    with tarfile.open(path) as archive:
      names = archive.getnames()
    self.assertIn('logs/compiler.log', names)
    self.assertIn('identities/artifact-reports/model.verification.json', names)
    self.assertNotIn('artifacts/model.sptg', names)
    self.assertNotIn('recording.npz', names)
    self.assertNotIn('unrelated-secret.txt', names)
    self.assertEqual(len(names), len(set(names)))

  def test_bounded_runner_starts_and_reaps_only_its_own_inert_child(self):
    self.workspace.mkdir()
    for folder in ('logs', 'telemetry'):
      (self.workspace / folder).mkdir()
    plan = {'checkout': str(self.checkout), 'workspace': str(self.workspace)}
    step = {'name': 'bounded-fixture', 'backend': 'CPU', 'timeout': 10}
    with patch.object(runner, 'worker_command', return_value=[sys.executable, '-c', 'print("inert child")']), \
         patch.object(runner, 'process_snapshot', return_value={'conflicts': [], 'inaccessible_pids': []}), \
         patch.object(runner, 'telemetry', return_value={'usb': [{'device_node': '/dev/bus/usb/001/002'}]}):
      result = runner.run_step(plan, step, '/dev/bus/usb/001/002')
    self.assertEqual(result['exit_code'], 0)
    self.assertIn('inert child', (self.workspace / 'logs/bounded-fixture.log').read_text())
    self.assertIsNone(result['abort_reason'])

  def test_timeout_targets_only_the_created_child_process_group(self):
    process = SimpleNamespace(pid=12345, poll=lambda: None, wait=lambda timeout: 0)
    with patch('os.killpg') as kill:
      runner.stop_own_child(process)
    kill.assert_called_once_with(12345, signal.SIGTERM)

  def test_cli_inspection_serializes_the_actual_fixture_preflight_without_device_execution(self):
    report = self.inspect()
    stream = io.StringIO()
    with patch.object(cli, 'preflight', return_value=report), \
         patch.object(sys, 'argv', ['hardware', 'inspect', '--checkout', str(self.checkout), '--workspace', str(self.workspace)]), \
         redirect_stdout(stream):
      cli.main()
    self.assertEqual(json.loads(stream.getvalue())['scope'], 'read_only_sysfs_proc_git')
    self.assertFalse(self.workspace.exists())

  def test_warp_verification_writes_two_seed_results_and_rejects_reload_mismatch(self):
    class Tensor:
      def __init__(self, array, device=None):
        self.array = np.asarray(array)
      def realize(self):
        return self
      def numpy(self):
        return self.array
      def __getitem__(self, index):
        return Tensor(self.array[index])
      @staticmethod
      def stack(*values):
        return Tensor(np.stack([value.array for value in values]))
    def warp(image, matrix):
      return Tensor(np.resize(image.array, (6, 128, 256)))
    def reload(**inputs):
      return Tensor.stack(*(warp(inputs['input_frame'][index], inputs['M_inv'][index]) for index in range(2)))
    artifact = self.root / 'inert-warp.sptg'
    artifact.write_bytes(b'inert fixture; never executed')
    arguments = ['--output', str(artifact), '--frame', 'fixture', '--warp-to', '512x256']
    modules = {
      'tinygrad': SimpleNamespace(Tensor=Tensor, Device=SimpleNamespace(DEFAULT='CPU')),
      'examples.openpilot.compile_warp': SimpleNamespace(make_frame_prepare=lambda *args: warp,
                                                       parse_frame=lambda value: SimpleNamespace(size=32), parse_size=lambda value: (512, 256)),
      'openpilot.sunnypilot.modeld_v2.helpers': SimpleNamespace(load_model=lambda *args, **kwargs: {'run': reload}),
    }
    with patch.dict(sys.modules, modules):
      worker.verify_warp(arguments, {'backend': 'CPU'}, self.identity)
    report_path = artifact.with_name(artifact.name + '.verification.json')
    report = json.loads(report_path.read_text())
    self.assertEqual(report['status'], 'passed_source_reload_consistency')
    self.assertEqual([run['seed'] for run in report['runs']], [42, 43])
    modules['openpilot.sunnypilot.modeld_v2.helpers'] = SimpleNamespace(load_model=lambda *args, **kwargs: {
      'run': lambda **inputs: Tensor(np.zeros((2, 6, 128, 256), np.uint8))})
    with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, 'uint8 images differ'):
      worker.verify_warp(arguments, {'backend': 'CPU'}, self.identity)
    self.assertEqual(json.loads(report_path.read_text())['status'], 'failed')


if __name__ == '__main__':
  unittest.main()
