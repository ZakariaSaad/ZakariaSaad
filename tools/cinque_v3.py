#!/usr/bin/env python3
"""Install a separately compiled V3 artifact, check startup, or roll back.

This tool never downloads a catalog model, alters Honda settings, starts
vehicle controls, or enables V3 for assisted driving.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile

from openpilot.sunnypilot.modeld_v2.cinque_guard import MODEL_COMMIT, ONNX_SHA256, CinqueGuard, INIT_TIMEOUT, verify_manifest


def install_artifact(source, expected_sha256, tinygrad_ref, model_root):
  source, model_root = Path(source), Path(model_root)
  if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
    raise ValueError('A lowercase SHA-256 checksum is required')
  manifest = {'model': 'Cinque Terre V3', 'model_commit': MODEL_COMMIT, 'source_onnx_sha256': ONNX_SHA256,
              'tinygrad_ref': tinygrad_ref, 'sha256': expected_sha256, 'file_name': f'cinque-v3-{expected_sha256}.pkl'}
  # Verify the copy before publishing it; never overwrite a fallback artifact.
  model_root.mkdir(parents=True, exist_ok=True)
  temp_path = None
  try:
    with source.open('rb') as src:
      with tempfile.NamedTemporaryFile(dir=model_root, prefix='.cinque-install-', delete=False) as dst:
        temp_path = Path(dst.name)
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    with temp_path.open('rb') as f:
      if hashlib.file_digest(f, 'sha256').hexdigest() != expected_sha256:
        raise ValueError('Compiled artifact checksum mismatch')
    destination = model_root / manifest['file_name']
    if destination.exists():
      verify_manifest(manifest, model_root, tinygrad_ref)
    else:
      # link is atomic and fails if another installation already created it.
      os.link(temp_path, destination)
    verify_manifest(manifest, model_root, tinygrad_ref)
    return manifest
  finally:
    if temp_path is not None:
      temp_path.unlink(missing_ok=True)


def require_offroad(params):
  if not params.get_bool('IsOffroad'):
    raise RuntimeError('V3 installation/checks/rollback must be performed offroad')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  action = parser.add_mutually_exclusive_group(required=True)
  action.add_argument('--install', metavar='PKL', help='Local artifact; not a GitHub download')
  action.add_argument('--check', action='store_true', help='Verify artifacts, fallback, recurrence, and startup; keep V3 disabled')
  action.add_argument('--rollback', action='store_true', help='Latch V3 off and retain the verified legacy selection')
  parser.add_argument('--sha256', help='Independently recorded SHA-256 of the compiled artifact (not the ONNX hash)')
  parser.add_argument('--tinygrad-ref', help='Tinygrad revision used for the local compilation')
  parser.add_argument('--camera-size', default='1344x760')
  args = parser.parse_args()
  from openpilot.common.params import Params
  from openpilot.common.hardware.hw import Paths
  from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref
  params = Params()
  require_offroad(params)
  if args.rollback:
    CinqueGuard(params, None, None).disable('Operator rollback; explicit offroad verification required')
    print('V3 and Experimental Mode disabled. Legacy model selection and Honda settings preserved.')
  elif args.install:
    if not args.sha256 or not args.tinygrad_ref or args.tinygrad_ref != get_tinygrad_ref():
      parser.error('--install requires the compiled SHA-256 and the matching --tinygrad-ref')
    CinqueGuard(params, None, None).disable('New artifact installed; live-model validation required')
    manifest = install_artifact(args.install, args.sha256, args.tinygrad_ref, Paths.model_root())
    params.put('CinqueV3Manifest', manifest, block=True)
    print(json.dumps(manifest, indent=2))
  else:
    from openpilot.sunnypilot.modeld_v2.cinque import CinqueModelState, load_fallback
    from tinygrad import Context
    cam_w, cam_h = map(int, args.camera_size.split('x'))
    CinqueGuard(params, None, None).disable('Offroad check only; live-model validation required')
    guard = CinqueGuard(params, None, None)
    os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'
    def preflight():
      fallback = load_fallback(cam_w, cam_h)
      fallback.warmup()
      candidate = CinqueModelState(cam_w, cam_h)
      with Context(DEV=candidate.model_device):
        candidate.warmup()
      fallback.warmup()
      return candidate.model_device
    device = guard.checked_call(preflight, INIT_TIMEOUT, 'offroad check')
    print(f'Artifacts, startup, state reset/replay, and fallback passed on {device}. V3 remains disabled.')


if __name__ == '__main__':
  main()
