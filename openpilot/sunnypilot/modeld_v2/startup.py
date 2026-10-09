"""Model startup policy, shared by the real modeld entry point and CI tests."""
import threading
from dataclasses import dataclass

from openpilot.cereal import custom
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.models.helpers import get_active_model_runner
from openpilot.sunnypilot.modeld_v2.cinque_guard import CinqueGuard, disable_v3, v3_start_requested


class ModelStartupError(RuntimeError):
  """No model can be loaded by this runner. Stop; publish no predictions."""


@dataclass
class StartupModels:
  model: object
  small_model: object | None
  guard: CinqueGuard | None


def load_models(params, cam_w, cam_h, chestnut, legacy_factory, big_timeout=60., v3_factories=None):
  params.put_bool('ChestnutLoading', chestnut, block=True)
  params.remove('ChestnutActive')
  model = small = guard = None
  try:
    if journal := params.get('CinqueV3InitJournal'):
      disable_v3(params, params.get('CinqueV3Fault') or f'V3 initialization interrupted ({journal}); explicit offroad verification required')
    requested = v3_start_requested(params)
    recovering = params.get_bool('CinqueV3UseFallback')
    if requested and not chestnut:
      disable_v3(params, 'Chestnut disconnected before V3 startup')
      requested = False
    if requested or recovering:
      try:
        if v3_factories is None:
          # Legacy boot does not import or initialize the experimental adapter.
          from openpilot.sunnypilot.modeld_v2.cinque import CinqueModelState, load_fallback
          v3_factories = (lambda: load_fallback(cam_w, cam_h), lambda: CinqueModelState(cam_w, cam_h))
        guard = CinqueGuard(params, *v3_factories)
        model = guard.start() if requested else guard.resume_fallback()
      except Exception as error:
        disable_v3(params, f'V3 adapter startup failed: {error}')
        params.put_bool('CinqueV3UseFallback', False, block=True)
      if model is not None:
        small = guard.fallback
      else:
        guard = None
        # A failed V3 prerequisite can change the appropriate managed runner.
        # A stock default must be started by stock modeld, not modeld_v2.
        if get_active_model_runner(params, force_check=True) != custom.ModelManagerSP.Runner.tinygrad:
          raise ModelStartupError('V3 unavailable; normal MVL selection requires stock modeld')

    if model is None:
      if chestnut:
        big = None
        def load_big():
          nonlocal big
          try:
            candidate = legacy_factory(cam_w=cam_w, cam_h=cam_h, chestnut=True)
            candidate.warmup()
            big = candidate
          except Exception:
            cloudlog.exception('chestnut load failed')
        loader = threading.Thread(target=load_big, daemon=True)
        loader.start()
        loader.join(big_timeout)
        model = big
        if model is None:
          params.put_bool('ChestnutModelError', True, block=True)
        else:
          params.remove('ChestnutModelError')
      try:
        small = legacy_factory(cam_w=cam_w, cam_h=cam_h, chestnut=False)
      except Exception as error:
        if model is None:
          raise ModelStartupError(f'Normal MVL small model could not initialize: {error}') from error
        # A working selected Chestnut model may still start. If it later fails,
        # absence of a small model must stop inference, not fabricate a fallback.
        cloudlog.exception('small model unavailable; no runtime Chestnut fallback')
      model = model if model is not None else small
    if model is None:
      raise ModelStartupError('No MVL model is available')
    params.put_bool('ChestnutActive', model.chestnut, block=True)
    return StartupModels(model, small, guard)
  finally:
    params.put_bool('ChestnutLoading', False, block=True)
