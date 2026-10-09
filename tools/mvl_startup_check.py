#!/usr/bin/env python3
"""Inspect MVL selection or warm the ordinary model offroad, without publication.

This is a diagnostic process, not manager/modeld main. It never starts controls,
publishes predictions, clears a V3 fault, or enables a model.
"""
import argparse
import json
import os
import sys
import threading


def check_startup(params, camera_size, warmup=False):
  from openpilot.cereal import custom
  from openpilot.common.hardware.hw import Paths
  from openpilot.selfdrive.modeld.helpers import chestnut_present
  from openpilot.sunnypilot.models.helpers import get_active_bundle, _bundle_artifacts, _bundle_is_valid_locally
  from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref
  chestnut = chestnut_present()
  bundle = get_active_bundle(params, chestnut=chestnut)
  runner = bundle.runner if bundle is not None else custom.ModelManagerSP.Runner.stock
  runner_name = 'tinygrad' if runner == custom.ModelManagerSP.Runner.tinygrad else 'stock' if runner == custom.ModelManagerSP.Runner.stock else str(runner)
  report = {'chestnut_present': chestnut, 'runner': runner_name, 'selected_bundle': bundle.ref if bundle else 'hardware default',
            'tinygrad_ref': get_tinygrad_ref(), 'v3_enabled': params.get_bool('CinqueV3Enabled'),
            'v3_fault': params.get('CinqueV3Fault'), 'v3_init_journal': params.get('CinqueV3InitJournal'),
            'artifacts': [{'filename': name, 'catalog_sha256': digest, 'present': os.path.isfile(os.path.join(Paths.model_root(), name))}
                          for name, digest in _bundle_artifacts(bundle)], 'warmup': 'not requested'}
  if warmup:
    if not params.get_bool('IsOffroad'):
      raise RuntimeError('Selected-model warmup requires offroad mode')
    if report['v3_enabled'] or params.get_bool('CinqueV3UseFallback') or report['v3_init_journal']:
      raise RuntimeError('Explicitly disable V3 and resolve interrupted initialization before an ordinary model check')
    if bundle and not _bundle_is_valid_locally(bundle):
      raise RuntimeError('Selected model failed its catalog checksum check')
    os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'
    cam_w, cam_h = camera_size
    if runner == custom.ModelManagerSP.Runner.tinygrad:
      from openpilot.sunnypilot.modeld_v2.modeld import ModelState
      from openpilot.sunnypilot.modeld_v2.startup import load_models
      model = load_models(params, cam_w, cam_h, chestnut, ModelState).model
    elif runner == custom.ModelManagerSP.Runner.stock:
      from openpilot.selfdrive.modeld.modeld import ModelState
      model = ModelState(cam_w, cam_h, chestnut=chestnut)
    else:
      raise RuntimeError(f'Unsupported selected runner: {runner}')
    model.warmup()
    report['warmup'] = 'passed; no predictions published'
    report['active_device'] = 'Chestnut' if model.chestnut else 'built-in device'
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--warmup', action='store_true', help='Offroad only: initialize/reset the ordinary model; never publish')
  parser.add_argument('--camera-size', default='1344x760')
  args = parser.parse_args()
  from openpilot.common.params import Params
  def expired():
    print('Ordinary model startup check timed out; no predictions published', file=sys.stderr, flush=True)
    os._exit(1)
  timer = threading.Timer(90., expired)
  timer.daemon = True
  if args.warmup:
    timer.start()
  try:
    cam_w, cam_h = map(int, args.camera_size.split('x'))
    if cam_w <= 0 or cam_h <= 0:
      raise ValueError('Camera dimensions must be positive')
    print(json.dumps(check_startup(Params(), (cam_w, cam_h), args.warmup), indent=2))
  finally:
    timer.cancel()


if __name__ == '__main__':
  main()
