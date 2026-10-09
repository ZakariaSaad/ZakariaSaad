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


def disable_v3(params, reason, use_fallback=False):
  # Fault is diagnostic data, not a model selection. Runtime recovery is separate.
  params.put('CinqueV3Fault', str(reason)[:512], block=True)
  params.put_bool('CinqueV3UseFallback', use_fallback, block=True)
  params.put_bool('CinqueV3Enabled', False, block=True)
  params.put_bool('ExperimentalMode', False, block=True)
  params.put_bool('CinqueV3Active', False, block=True)
  params.put_bool('CinqueV3RecoveryPending', False, block=True)
  params.remove('CinqueV3InitJournal')
  params.remove('ModelRunnerTypeCache')


def v3_start_requested(params):
  if not params.get_bool('CinqueV3Enabled'):
    return False
  if fault := params.get('CinqueV3Fault'):
    disable_v3(params, fault, use_fallback=params.get_bool('CinqueV3UseFallback') or params.get_bool('CinqueV3Active'))
    return False
  return True


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
    self.fallback_verified = False
    self.active = False
    self.terminate = terminate if terminate is not None else lambda: os._exit(1)
    self.params.put_bool('CinqueV3Active', False, block=True)

  def disable(self, reason, use_fallback=None):
    # Persist the latch before changing selection: a restart cannot retry V3.
    self.active = False
    disable_v3(self.params, reason, self.fallback_verified if use_fallback is None else use_fallback)

  def checked_call(self, fn, timeout, phase):
    # A wedged GPU cannot be recovered by returning from a Python thread.
    # Persist the failure and terminate modeld; manager restarts only fallback.
    def expired():
      try:
        self.disable(f'V3 {phase} timeout ({timeout}s); modeld restart required')
        self.params.put_bool('CinqueV3RecoveryPending', True, block=True)
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
      # Join even a cancelled timer: its callback may already have passed the
      # cancellation check. A late GPU return must not race durable latch writes.
      # Production termination exits this process; test termination returns.
      timer.join()

  def run(self, *args, **kwargs):
    if self.active:
      def infer():
        outputs = self.model.run(*args, **kwargs)
        if not isinstance(outputs, dict) or not outputs:
          raise ValueError('V3 returned no predictions')
        return outputs
      return self.checked_call(infer, INFERENCE_TIMEOUT, 'inference')
    return self.model.run(*args, **kwargs)

  def start(self):
    # Check the persistent latch before touching either experimental artifact.
    if journal := self.params.get('CinqueV3InitJournal'):
      self.disable(self.params.get('CinqueV3Fault') or f'V3 initialization interrupted ({journal}); explicit offroad verification required')
      return None
    if not v3_start_requested(self.params):
      return None
    # Journal fallback verification too: interruption must not retry V3.
    self.params.put('CinqueV3InitJournal', 'verifying fallback and initializing candidate', block=True)
    try:
      self.checked_call(self._load_fallback, INIT_TIMEOUT, 'fallback initialization')
    except Exception as error:
      self.disable(f'Fallback startup failed: {error}')
      return None
    self.model = self.fallback
    try:
      def initialize():
        candidate = self.candidate_factory()
        candidate.warmup()
        return candidate
      candidate = self.checked_call(initialize, INIT_TIMEOUT, 'initialization')
    except Exception as error:
      self.disable(f'V3 startup failed: {error}')
    else:
      if self.params.get('CinqueV3Fault') or not self.params.get_bool('CinqueV3Enabled'):
        self.disable(self.params.get('CinqueV3Fault') or 'V3 request withdrawn during initialization')
        return self.model
      self.model, self.active = candidate, True
      self.params.remove('CinqueV3InitJournal')
      self.params.put_bool('CinqueV3UseFallback', False, block=True)
      self.params.put_bool('CinqueV3Active', True, block=True)
    return self.model

  def _load_fallback(self):
    self.fallback_verified = False
    self.fallback = self.fallback_factory()
    self.fallback.warmup()
    self.fallback_verified = True

  def resume_fallback(self):
    # Used only for a live-session watchdog restart, never because a fault exists.
    try:
      self.checked_call(self._load_fallback, INIT_TIMEOUT, 'fallback restart')
    except Exception as error:
      self.disable(f'{self.params.get("CinqueV3Fault")}; fallback restart failed: {error}')
      return None
    self.model = self.fallback
    return self.model

  def recover(self, error):
    if not self.active:
      raise RuntimeError('Working model failed; no verified recovery available') from error
    self.disable(f'V3 failed: {error}')
    # Reset and verify fallback history before publishing any new predictions.
    self.fallback_verified = False
    try:
      self.checked_call(self.fallback.warmup, INIT_TIMEOUT, 'fallback recovery')
    except Exception as fallback_error:
      self.disable(f'V3 failed: {error}; fallback recovery failed: {fallback_error}')
      raise
    self.fallback_verified = True
    self.model = self.fallback
    return self.model
