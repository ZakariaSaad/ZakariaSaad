"""Isolated GPU-only probe/compiler/Bigv2 smoke worker. No vehicle streams."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import resource
import runpy
import statistics
import subprocess
import sys
import time

import numpy as np

from tools.navigation_hardware.safety import sha256
from tools.navigation_hardware.worker import NoPersistentFlash, admitted_device


@contextmanager
def isolated_io(*, allow_downloads=True):
  from unittest.mock import patch
  import openpilot.cereal.messaging as messaging
  import msgq
  def denied(*args, **kwargs):
    raise RuntimeError('Smoke workers cannot publish or subscribe to live vehicle data')
  class NoControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
      if fullname.startswith(('panda', 'openpilot.selfdrive.controls', 'openpilot.sunnypilot.selfdrive.controls',
                              'openpilot.system.manager', 'openpilot.selfdrive.pandad', 'openpilot.system.camerad.camerad')):
        raise RuntimeError('Vehicle controller/Panda/camera imports forbidden in GPU smoke workers')
      return None
  guards = [NoPersistentFlash(), NoControllers()]
  sys.meta_path[:0] = guards
  try:
    with ExitStack() as stack:
      for owner in (messaging, msgq):
        for name in ('PubMaster', 'SubMaster', 'pub_sock', 'sub_sock', 'PubSocket', 'SubSocket'):
          if hasattr(owner, name):
            stack.enter_context(patch.object(owner, name, denied))
      if not allow_downloads:
        import urllib.request
        def uncached(url, *args, **kwargs):
          from urllib.parse import urlsplit
          address = urlsplit(getattr(url, 'full_url', str(url)))._replace(query='', fragment='').geturl()
          raise RuntimeError('Runtime dependency is not cached; no download in smoke mode. Run prepare once: ' + address)
        stack.enter_context(patch.object(urllib.request, 'urlopen', uncached))
      yield
  finally:
    for guard in guards:
      sys.meta_path.remove(guard)


def verified_artifact(artifact, onnx, compiler):
  """Admit only a locally compiled native Bigv2 with complete real-AMD proof."""
  from openpilot.selfdrive.navigation.experimental_catalog import DEFAULT_CATALOG, _header, _model_contract, load_experimental_catalog
  from openpilot.selfdrive.navigation.export_experimental_model import _verification_report
  entry = load_experimental_catalog(DEFAULT_CATALOG, experimental=True)['entries'][0]
  report = json.loads(Path(str(artifact) + '.verification.json').read_text())
  header = _header(artifact, entry['target'])
  _verification_report(report, sha256(artifact), sha256(compiler), entry['contract'], header)
  if sha256(onnx) != entry['source_onnx']['sha256']:
    raise ValueError('Source ONNX hash mismatch')
  named = {row['name']: row for row in header['inputs'] if row['name'] is not None}
  metadata = {'output_slices': entry['contract']['output_slices']}
  for kind in ('input', 'output'):
    metadata[kind + '_specs'] = {row['name']: {'shape': row['shape'], 'dtype': row['dtype'],
      'device': named[row['name']]['device'] if kind == 'input' else 'AMD'} for row in entry['contract'][kind + 's']}
  _model_contract(header, metadata, entry['contract'])
  return {'artifact_sha256': sha256(artifact), 'source_onnx_sha256': sha256(onnx), 'compiler_sha256': sha256(compiler),
          'verification_report_sha256': sha256(Path(str(artifact) + '.verification.json')), 'contract': entry['contract']}


def validate_loaded_contract(loaded, contract):
  for kind in ('input', 'output'):
    expected = contract[kind + 's']
    actual = loaded.get(kind + '_specs', {})
    if list(actual) != [row['name'] for row in expected]:
      raise ValueError('Loaded model tensor names/order differ from Bigv2: ' + kind)
    for row in expected:
      shape, dtype, device = actual[row['name']]
      if tuple(shape) != tuple(row['shape']) or np.dtype(dtype) != np.dtype({'FLOAT': 'float32', 'UINT8': 'uint8'}.get(row['dtype'], row['dtype'])) \
         or device not in (('AMD', 'NPY') if kind == 'input' else ('AMD',)):
        raise ValueError('Loaded model shape/dtype/device differs from Bigv2: ' + row['name'])


def synthetic_inputs(specs, index, arm):
  """Prewarped artificial pixels, not cameras, GPS, route replay or model-quality data."""
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in specs.items() if not name.startswith('state_')}
  image = arrays['new_img']
  image[:] = np.arange(image.shape[-1], dtype=np.uint8) + index * 7
  arrays['traffic_convention'].flat[0] = 1
  arrays['action_t'][:] = (.15, .375)
  if arm in ('map_only', 'both'):
    arrays['nav_map'][..., 97:104, 30:230] = 255
    arrays['nav_map'][..., 100:193, 127:130] = 128
  if arm in ('maneuver_only', 'both'):
    arrays['navigation'].flat[(10, 15, 20)[index % 3]] = 1
    arrays['navigation'].flat[-1] = 1.0 - index * .1
  return arrays


def infer_sequence(loaded, device, *, arms=('neutral', 'map_only', 'maneuver_only', 'both')):
  from examples.openpilot.helpers import allocate_inputs
  from openpilot.sunnypilot.modeld_v2.model_adapters import input_view
  from tinygrad import Tensor
  specs, output_specs = loaded['input_specs'], loaded['output_specs']
  pairs = {name: 'next_' + name for name in specs if name.startswith('state_')}
  if len(pairs) != 5 or not all(name in output_specs for name in pairs.values()):
    raise ValueError('Bigv2 must expose exactly five recurrent-state pairs')
  records: list[dict] = []
  for arm in arms:
    inputs, outputs = allocate_inputs(specs), allocate_inputs(output_specs)
    for state, next_state in pairs.items():
      queue = inputs[state]
      # Distinct Tensor/Buffer view keeps all17 JIT slots while sharing state memory, exactly as the production adapter.
      outputs[next_state] = input_view(queue._buffer(), tuple(int(d) for d in queue.shape), queue.dtype, 0)
    for index in range(2):
      arrays = synthetic_inputs(specs, index, arm)
      for name, value in arrays.items():
        inputs[name].assign(Tensor(value, device=specs[name][2])).realize()
      started = time.perf_counter()
      loaded['run'](output_buffers=outputs, **inputs)
      device.synchronize()
      seconds = time.perf_counter() - started
      tensors = {name: value.numpy().copy() for name, value in outputs.items()}
      for name, value in tensors.items():
        if tuple(value.shape) != tuple(output_specs[name][0]) or np.dtype(value.dtype) != np.dtype(output_specs[name][1]) \
           or not np.isfinite(value).all():
          raise ValueError('Output/state shape, dtype or finite check failed: ' + name)
      if not np.array_equal(tensors['next_state_img_q'][:, -1], arrays['new_img']):
        raise ValueError('Image recurrent queue did not append the synthetic frame correctly')
      records.append({'arm': arm, 'frame': index, 'inference_seconds': seconds, 'all_outputs_finite': True,
                      'output_digests': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in tensors.items()}})
  if len({row['output_digests']['outputs'] for row in records}) < 2:
    raise ValueError('Predictions did not vary; cannot exclude no-op execution')
  timings = [row['inference_seconds'] * 1000 for row in records]
  return {'timing_ms': {'minimum': min(timings), 'median': statistics.median(timings), 'maximum': max(timings)},
          'predictions': records, 'recurrent_pairs': pairs, 'synthetic_input': 'built-in prewarped gradients and raw0..255 map/maneuver arrays',
          'scope': 'eight cold-start diagnostic predictions; no camera warp, independent CPU parity or learned quality validation'}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('probe', 'compile', 'infer'))
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--artifact', type=Path)
  parser.add_argument('--onnx', type=Path)
  args = parser.parse_args()
  if os.environ.get('NAV_SMOKE_WORKER') != '1' or os.environ.get('DEV') != 'USB+AMD:LLVM':
    parser.error('Launch only through the supervised off-road smoke command')
  if any(os.environ.get(key, '0') not in ('', '0') for key in ('MOCKGPU', 'MOCKUSB', 'MOCKKFD', 'ASM_CALL', 'NO_HIPCC')):
    parser.error('Emulated/no-op GPU execution is forbidden')
  owner = subprocess.check_output(['systemctl', 'show', 'navigation-chestnut-smoke.service', '--property=MainPID', '--value'],
                                  text=True, timeout=10).strip()
  if owner != str(os.getppid()):
    parser.error('GPU worker requires the registered, supervising off-road coordinator')
  result = {'action': args.action, 'passed': False, 'normal_prediction_publishing': False, 'vehicle_io': False,
            'activation_approved': False}
  resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
  try:
    with isolated_io(allow_downloads=os.environ.get('NAV_SMOKE_ALLOW_RUNTIME_DOWNLOADS') == '1'):
      # Match existing supported compiler initialization, including safe volatile firmware decompression.
      from openpilot.sunnypilot.modeld_v2.helpers import patch_fetch_fw
      patch_fetch_fw()
      import tinygrad
      from openpilot.selfdrive.navigation.experimental_catalog import TINYGRAD_REF
      runtime_root = Path(tinygrad.__file__).resolve().parent.parent
      if runtime_root != Path(__file__).resolve().parents[2] / 'tinygrad_repo':
        raise RuntimeError('Imported tinygrad is not the installed pinned runtime')
      result['runtime'] = {'revision': TINYGRAD_REF, 'import_root': str(runtime_root), 'artifact_format': 'SPTGPLN1', 'runner': 'native_tinygrad'}
      from tinygrad import Device, Tensor
      device = Device['AMD']
      result['device'] = admitted_device(device)
      result['device']['reported_total_vram_bytes'] = int(device.iface.dev_impl.vram_size)
      result['device']['free_vram_measured'] = False
      if args.action == 'probe':
        started = time.perf_counter()
        value = ((Tensor.arange(256).to('AMD') + 1) * 2).sum().numpy().item()
        if value != 65792:
          raise ValueError('GPU arithmetic probe returned an incorrect result')
        result.update(probe_value=value, elapsed_seconds=time.perf_counter() - started)
      else:
        compiler = Path(__file__).resolve().parents[2] / 'openpilot/sunnypilot/modeld_v2/compile_modeld_v2.py'
        if args.action == 'compile':
          sys.argv = [str(compiler), str(args.onnx), str(args.artifact), '--device-input', '*', '--benchmark-runs', '0', '--verification-runs', '2']
          runpy.run_path(str(compiler), run_name='__main__')
        proof = verified_artifact(args.artifact, args.onnx, compiler)
        result['artifact'] = {key: value for key, value in proof.items() if key != 'contract'}
        if args.action == 'infer':
          from openpilot.sunnypilot.modeld_v2.helpers import load_model
          def contract_guard(parts, extras):
            validate_loaded_contract(extras, proof['contract'])
            return parts, extras
          loaded = load_model(str(args.artifact), cache=False, extend=contract_guard)
          result['inference'] = infer_sequence(loaded, device)
      result['passed'] = True
  except BaseException as error:
    result['error'] = f'{type(error).__name__}: {error}'
    raise
  finally:
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
  main()
