"""Opt-in V3 startup and recovery. Faults survive modeld/manager restarts."""
import hashlib
import re
import os
import threading
import time
from pathlib import Path
from contextlib import contextmanager

BASELINE_COMMIT = '73dca34cece29339f43835f1c9296a5809586f4b'
MODEL_COMMIT = 'c6d13eb723e0f6eee47ea7d56e47a700fba21380'
STATEFUL_COMMIT = '64f9b47b6ea7b622967acd7042148e7a92f0ccc4'
ONNX_SHA256 = '404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4'
ONNX_SIZE = 766354845
ONNX_URL = 'https://huggingface.co/commaai/openpilot_driving_models/resolve/main/' + \
           'f78ed37d-afad-4dbc-8050-40ea885eedde/12864/big_driving_supercombo.onnx'
INIT_TIMEOUT = 60.
INFERENCE_TIMEOUT = .25


@contextmanager
def open_artifact(path):
  from openpilot.common.file_chunker import open_file_chunked
  stream = Path(path).open('rb') if Path(path).is_file() else open_file_chunked(path)
  try:
    yield stream
  finally:
    # MVL's ChunkStream does not close a partially consumed last chunk.
    if hasattr(stream, 'raw') and getattr(stream.raw, '_f', None) is not None:
      stream.raw._f.close()
    stream.close()


def verify_manifest(manifest, model_root, tinygrad_ref):
  if not isinstance(manifest, dict) or manifest.get('model') != 'Cinque Terre V3':
    raise ValueError('A reviewed Cinque Terre V3 manifest is required')
  if manifest.get('source_onnx_sha256') != ONNX_SHA256 or manifest.get('model_commit') != MODEL_COMMIT:
    raise ValueError('V3 manifest must identify the official Cinque Terre V3 source')
  if not re.fullmatch(r'[0-9a-f]{64}', manifest.get('sha256', '')):
    raise ValueError('V3 manifest requires a SHA-256 checksum')
  if not tinygrad_ref or manifest.get('tinygrad_ref') != tinygrad_ref:
    raise ValueError('V3 artifact must use the pinned Tinygrad revision')
  filename = manifest.get('file_name', '')
  if not filename or Path(filename).name != filename:
    raise ValueError('V3 artifact must be a filename in the model directory')
  path = Path(model_root) / filename
  with open_artifact(path) as f:
    if hashlib.file_digest(f, 'sha256').hexdigest() != manifest['sha256']:
      raise ValueError('V3 artifact checksum mismatch')
  return path


class CinqueGuard:
  def __init__(self, params, fallback_factory, candidate_factory, terminate=None):
    self.params = params
    self.fallback_factory, self.candidate_factory = fallback_factory, candidate_factory
    self.model = self.fallback = None
    self.active = False
    self.terminate = terminate if terminate is not None else lambda: os._exit(1)
    self.params.put_bool('CinqueV3Active', False, block=True)

  def disable(self, reason):
    # Persist the latch before changing selection: a restart cannot retry V3.
    self.active = False
    self.params.put('CinqueV3Fault', str(reason)[:512], block=True)
    self.params.put_bool('CinqueV3Enabled', False, block=True)
    self.params.put_bool('ExperimentalMode', False, block=True)
    self.params.put_bool('CinqueV3Active', False, block=True)

  def checked_call(self, fn, timeout, phase):
    # A wedged GPU cannot be recovered by returning from a Python thread.
    # Persist the failure and terminate modeld; manager restarts only fallback.
    def expired():
      try:
        self.disable(f'V3 {phase} timeout ({timeout}s); modeld restart required')
      finally:
        self.terminate()
    timer = threading.Timer(timeout, expired)
    timer.daemon = True
    started = time.monotonic()
    timer.start()
    try:
      result = fn()
      if time.monotonic() - started >= timeout:
        raise TimeoutError(f'V3 {phase} exceeded {timeout}s')
      return result
    finally:
      timer.cancel()

  def run(self, *args, **kwargs):
    if self.active:
      return self.checked_call(lambda: self.model.run(*args, **kwargs), INFERENCE_TIMEOUT, 'inference')
    return self.model.run(*args, **kwargs)

  def start(self):
    try:
      self.fallback = self.fallback_factory()
      self.fallback.warmup()
    except Exception as error:
      self.disable(f'Fallback startup failed: {error}')
      raise
    self.model = self.fallback
    if self.params.get('CinqueV3Fault'):
      self.disable(self.params.get('CinqueV3Fault'))
    elif self.params.get_bool('CinqueV3Enabled'):
      self.params.put('CinqueV3Fault', 'V3 initialization interrupted; explicit offroad verification required', block=True)
      try:
        def initialize():
          candidate = self.candidate_factory()
          candidate.warmup()
          return candidate
        candidate = self.checked_call(initialize, INIT_TIMEOUT, 'initialization')
      except Exception as error:
        self.disable(f'V3 startup failed: {error}')
      else:
        self.model, self.active = candidate, True
        self.params.remove('CinqueV3Fault')
        self.params.put_bool('CinqueV3Active', True, block=True)
    return self.model

  def recover(self, error):
    if not self.active:
      raise RuntimeError('Working model failed; no verified recovery available') from error
    self.disable(f'V3 failed: {error}')
    # Reset and verify fallback history before publishing any new predictions.
    self.fallback.warmup()
    self.model = self.fallback
    return self.model
