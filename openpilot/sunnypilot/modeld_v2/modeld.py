#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Callable
import gc
import os
os.environ['GMMU'] = '0'
from tinygrad.device import Device
import numpy as np
import threading
import time
from setproctitle import setproctitle

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.modeld.helpers import chestnut_present, wait_for_chestnut
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.hardware.usb import cable_connected
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.sunnypilot.modeld_v2.bigv2_desire_helper import DesireHelper as Bigv2DesireHelper
from openpilot.selfdrive.modeld.parse_model_outputs import Parser as Bigv2Parser
from openpilot.selfdrive.modeld.fill_model_msg import (fill_model_msg as fill_bigv2_model_msg,
                                                     fill_driving_model_data, fill_pose_msg as fill_bigv2_pose_msg)
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.selfdrive.modeld.modeld import ChestnutGpuState
from openpilot.selfdrive.navigation.activation import authorize_runtime, validate_runtime_metadata, validate_runtime_permit
from openpilot.selfdrive.navigation.model_inputs import (NAV_INPUT_NAMES, NavigationInputs, NavigationModelDisabled,
                                                        diagnostic_session_active, diagnostic_session_backend,
                                                        diagnostic_session_token, validate_model_inputs)

from openpilot.sunnypilot.modeld_v2.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState, get_curvature_from_output
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.meta_helper import load_meta_constants
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.modeld_v2.helpers import load_model
from openpilot.sunnypilot.modeld_v2.frame_fusion import fused_with_warp
from openpilot.sunnypilot.modeld_v2.model_adapters import get_model_adapter
from openpilot.sunnypilot.modeld_v2.numerics import nonfinite_details
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.sunnypilot.models.navigation_bundle import is_navigation_selected
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID
from openpilot.sunnypilot.selfdrive.controls.lib.relc import RoadEdgeLaneChangeController
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import drive_config_for_model
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.timing import prediction_delay

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld_tinygrad"
BIG_MODEL_TIMEOUT = 60
LIGHT_SPEED_MODEL_REF = 'e4acf619d4eb8b726fc405c264109310ac2951df'


def _pkl_exists(path):
  return os.path.exists(path)


def _find_driving_pkl(bundle):
  if (override := os.environ.get('COMBINED_MODEL_PKL')) and _pkl_exists(override):
    return override
  if bundle is None or not bundle.models:
    return None
  from openpilot.common.hardware.hw import Paths
  model_root = Paths.model_root()

  pkl_name = bundle.models[0].artifact.fileName
  pkl_path = os.path.join(model_root, pkl_name)
  if _pkl_exists(pkl_path):
    return pkl_path
  return None


def _navigation_input_specs(extras: dict) -> dict:
  if 'input_specs' in extras:
    return extras['input_specs']
  # Legacy artifacts describe only shapes and cannot establish the native dtype/state contract.
  metadata = extras.get('metadata', {})
  for entry in (metadata, *metadata.values()):
    if isinstance(entry, dict) and any(name in entry.get('input_shapes', {}) for name in NAV_INPUT_NAMES):
      raise ValueError("Navigation models require native input_specs metadata")
  return {}


def takes_nav(*models) -> bool:
  return any(name in model.numpy_inputs for model in models if model is not None for name in NAV_INPUT_NAMES)


def _reject_navigation_publication(*models) -> None:
  for model in models:
    if model is None:
      continue
    if getattr(model, 'diagnostic', False):
      raise NavigationModelDisabled('Diagnostic model predictions cannot be published')
    if takes_nav(model):
      validate_runtime_permit(getattr(model, '_driving_permit', None), getattr(model, 'navigation_input_specs', {}))


def bigv2_warp_matrices(rpy_calib, device_camera, main_wide_camera, use_extra_client):
  """a03 calibration/camera selection, without optional camera-offset tuning."""
  main_intrinsics = device_camera.wide_road.intrinsics if main_wide_camera else device_camera.narrow_road.intrinsics
  extra_intrinsics = device_camera.wide_road.intrinsics if use_extra_client or main_wide_camera else device_camera.narrow_road.intrinsics
  return (get_warp_matrix(rpy_calib, main_intrinsics, False).astype(np.float32),
          get_warp_matrix(rpy_calib, extra_intrinsics, True).astype(np.float32))


def publish_bigv2_model(msg, output, action, state, *frame_data):
  """Use the official publisher; omit SP's extra lane/edge time interpolation."""
  fill_bigv2_model_msg(msg, output, action, state, *frame_data)
  for line in (*msg.modelV2.laneLines, *msg.modelV2.roadEdges):
    line.t = []


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState(ModelStateBase):
  _inference_context: dict | None = None

  inputs: dict[str, np.ndarray]
  prev_desire: np.ndarray

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False, *, warp_path: str | None = None,
               diagnostic: bool = False, driving_permit=None):
    self.diagnostic = diagnostic
    self._driving_permit = driving_permit
    if driving_permit is not None:
      if diagnostic or not chestnut:
        raise NavigationModelDisabled('Driving navigation requires the separately approved Chestnut candidate')
      validate_runtime_permit(driving_permit)
      from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
      stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
      if driving_permit.warp_frame_size != stride * (y_height + uv_height):
        raise NavigationModelDisabled('Approved navigation warp does not match the actual camera copied extent')
      profile = driving_permit.metadata.get('camera_profile')
      if profile is not None and any(profile.get(key) != value for key, value in
                                    {'width': cam_w, 'height': cam_h, 'stride': stride, 'y_height': y_height,
                                     'uv_height': uv_height, 'frame_size': stride * (y_height + uv_height)}.items()):
        raise NavigationModelDisabled('Registered navigation warp geometry does not match the actual camera')
      warp_path = str(driving_permit.warp_path)
    self._diagnostic_token = diagnostic_session_token() if diagnostic else None
    self._diagnostic_backend = diagnostic_session_backend() if diagnostic else None
    self._check_diagnostic_session()
    if diagnostic and (self._diagnostic_backend not in ('CPU', 'AMD') or chestnut != (self._diagnostic_backend == 'AMD')):
      raise NavigationModelDisabled('Diagnostic Chestnut selection must match the scoped CPU/AMD backend')
    ModelStateBase.__init__(self)

    env_pkl = os.environ.get('COMBINED_MODEL_PKL')
    if driving_permit is not None or (env_pkl and os.path.exists(env_pkl)):
      model_bundle = None
    else:
      model_bundle = get_active_bundle(chestnut=chestnut)
    self.generation = model_bundle.generation if model_bundle is not None else None
    overrides = {override.key: override.value for override in model_bundle.overrides} if model_bundle else {}

    self.LAT_SMOOTH_SECONDS = float(overrides.get('lat', ".0"))
    self.LONG_SMOOTH_SECONDS = float(overrides.get('long', ".0"))
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL: float = 1.0
    self.chestnut = chestnut

    pkl_path = str(driving_permit.artifact_path) if driving_permit is not None else _find_driving_pkl(model_bundle)
    assert pkl_path is not None, f"No driving pkl found for {'chestnut' if chestnut else 'small model'} — all models must be compiled with compile_modeld.py"
    if warp_path is None:
      self._init_combined(pkl_path, cam_w, cam_h, model_bundle)
    else:
      self._init_combined(pkl_path, cam_w, cam_h, model_bundle, warp_path=warp_path)

  def _init_combined(self, pkl_path, cam_w, cam_h, bundle, *, warp_path: str | None = None):
    cloudlog.warning(f"loading combined pkl: {pkl_path}")
    fuse = (fused_with_warp(cam_w, cam_h, self.chestnut) if warp_path is None else
            fused_with_warp(cam_w, cam_h, self.chestnut, warp_path=warp_path))

    def checked_fusion(parts, extras):
      # load_model calls this before allocating PlainJit/GPU buffers or replaying any kernels.
      self._validate_navigation_inputs(_navigation_input_specs(extras))
      if getattr(self, '_driving_permit', None) is not None:
        validate_runtime_metadata(self._driving_permit, extras)
      return fuse(parts, extras)

    # A navigation permit pins the actual model and warp bytes. Its mutable
    # sibling boot cache is not part of that proof and contains pickled JITs,
    # so reconstruct this candidate from the reviewed native kernel binaries.
    # Ordinary models retain their existing boot-cache behavior.
    if getattr(self, '_driving_permit', None) is not None:
      jits = load_model(pkl_path, extend=checked_fusion, cache=False)
    else:
      jits = load_model(pkl_path, extend=checked_fusion)
    self.navigation_input_specs = _navigation_input_specs(jits)
    self._validate_navigation_inputs(self.navigation_input_specs)
    if getattr(self, '_driving_permit', None) is not None:
      validate_runtime_metadata(self._driving_permit, jits)

    metadata = jits['metadata']
    self.WARP_DEV = metadata.get('warp_dev', 'QCOM') if COMMA_HARDWARE else 'CPU'
    self.DEV = ('AMD' if self.chestnut else 'QCOM') if COMMA_HARDWARE else 'CPU'
    self.QUEUE_DEV = self.DEV
    if getattr(self, '_driving_permit', None) is not None:
      self.DEV = self.QUEUE_DEV = self.WARP_DEV = 'AMD'
    elif getattr(self, 'diagnostic', False):
      assert self._diagnostic_backend is not None
      self.DEV = self.QUEUE_DEV = self.WARP_DEV = self._diagnostic_backend
    adapter_options = {'diagnostic': True} if getattr(self, 'diagnostic', False) or getattr(self, '_driving_permit', None) is not None else {}
    if self.chestnut and bundle is not None and bundle.ref == LIGHT_SPEED_MODEL_REF:
      adapter_options['separate_recurrent_buffers'] = True
    self.adapter = get_model_adapter(jits, cam_w, cam_h, self.DEV, self.QUEUE_DEV, self.WARP_DEV, self.chestnut, **adapter_options)
    self.vision_output_slices = self.adapter.vision_output_slices
    self.policy_output_slices = self.adapter.policy_output_slices
    self._policy_slices_list = self.adapter._policy_slices_list
    self._combined_model_type = self.adapter._combined_model_type
    self._vision_input_names = self.adapter._vision_input_names
    self.numpy_inputs = self.adapter.numpy_inputs
    self._policy_keys = self.adapter._policy_keys
    self._has_on_policy = self.adapter._has_on_policy
    self._desire_key = self.adapter._desire_key
    self._road_key = self.adapter._road_key
    self._wide_key = self.adapter._wide_key
    self.frame_buf_params = self.adapter.frame_buf_params

    is_20hz = bundle.is20hz if bundle else self._combined_model_type in ('split', 'multi_policy')
    if is_20hz:
      from openpilot.sunnypilot.models.split_model_constants import SplitModelConstants
      self.constants = SplitModelConstants()
    else:
      self.constants = ModelConstants()

    self.parser = Bigv2Parser() if getattr(self, '_driving_permit', None) is not None else Parser()
    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)

  def _check_diagnostic_session(self) -> None:
    if getattr(self, 'diagnostic', False) and (
      self._diagnostic_token is None
      or self._diagnostic_token is not diagnostic_session_token()
      or not diagnostic_session_active()
    ):
      raise NavigationModelDisabled('Diagnostic inference requires the original active file-only diagnostic session')

  def _validate_navigation_inputs(self, specs: dict) -> bool:
    self._check_diagnostic_session()
    if getattr(self, 'diagnostic', False):
      return validate_model_inputs(specs, diagnostic=True)
    if getattr(self, '_driving_permit', None) is not None:
      return validate_model_inputs(specs, driving_permit=self._driving_permit)
    return validate_model_inputs(specs)

  def warmup(self) -> None:
    dummy_frames, transforms, dummy_inputs = self.adapter.get_dummy_inputs()
    self.run(dummy_frames, transforms, dummy_inputs)
    self.adapter.reset_warmup_buffers()
    self.prev_desire[:] = 0

  def reset_state(self, reason: str) -> None:
    """Start a new isolated inference sequence without retaining recurrent history."""
    if not isinstance(reason, str) or not reason.strip():
      raise ValueError('A state reset requires a diagnostic reason')
    self.adapter.reset_state()
    self.prev_desire[:] = 0
    self.state_reset_reason = reason
    self.state_reset_count = getattr(self, 'state_reset_count', 0) + 1

  @property
  def mlsim(self) -> bool:
    return bool(self.generation is not None and self.generation >= 11)

  @property
  def vision_input_names(self) -> list[str]:
    return self._vision_input_names

  @property
  def desire_key(self) -> str:
    return self._desire_key

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray],
          after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray] | None:
    if getattr(self, 'diagnostic', False) or takes_nav(self):
      # Recheck every invocation: an invalid permit stops inference before graph
      # execution. Production failure uses the ordinary model handover.
      try:
        self._validate_navigation_inputs(self.navigation_input_specs)
      except NavigationModelDisabled:
        if getattr(self, 'diagnostic', False):
          self.reset_state('diagnostic permission revoked')
        raise
    self.adapter.copy_frames(bufs)

    desire_key = self.desire_key
    inputs[desire_key][0] = 0
    (self.numpy_inputs[desire_key].flat if self.adapter.is_native else self.numpy_inputs[desire_key])[:] = \
      np.where(inputs[desire_key] - self.prev_desire > .99, inputs[desire_key], 0)
    self.prev_desire[:] = inputs[desire_key]

    for key in ('traffic_convention', 'lateral_control_params', 'action_t'):
      if key in self.numpy_inputs and key in inputs:
        self.numpy_inputs[key][:] = inputs[key]

    for name in NAV_INPUT_NAMES:
      if name in self.numpy_inputs:
        # Raw map pixels are 0..255; compatible graphs perform their own normalization.
        self.numpy_inputs[name][:] = inputs.get(name, 0)

    if self.adapter.is_native:
      for i, key in enumerate(self._vision_input_names):
        if key in transforms:
          self.numpy_inputs['tfm'][i] = transforms[key].reshape(3, 3)
    else:
      self.numpy_inputs['tfm'][:, :] = transforms[self._road_key].reshape(3, 3)
      self.numpy_inputs['big_tfm'][:, :] = transforms[self._wide_key].reshape(3, 3)

    inference_start = time.perf_counter() if self.chestnut else 0.0
    raw_outputs = self.adapter.run()
    if self.chestnut and (err := Device[self.DEV].error_state.host.view(fmt='q')[0]):
      raise RuntimeError(f"chestnut failed with {err}")

    if after_enqueue is not None:
      after_enqueue()

    if self._combined_model_type == 'supercombo':
      model_output = raw_outputs.numpy().flatten()
      if self.chestnut and not np.all(np.isfinite(model_output)):
        self._log_nonfinite_output(model_output, inference_ms=(time.perf_counter() - inference_start) * 1000)
        raise RuntimeError("model output not finite")
      if self.adapter.is_native:
        self.adapter.advance_state()
      sliced = {k: model_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_outputs(sliced)
      if getattr(self, '_driving_permit', None) is not None and os.getenv('SEND_RAW_PRED'):
        outputs['raw_pred'] = model_output.copy()
      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        (self.numpy_inputs['prev_feat'].flat if self.adapter.is_native else self.numpy_inputs['prev_feat'])[:] = \
          model_output[self.vision_output_slices['hidden_state']]
    else:
      vision_output = raw_outputs[0].numpy().flatten()
      vision_sliced = {k: vision_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_vision_outputs(vision_sliced)

      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        (self.numpy_inputs['prev_feat'].flat if self.adapter.is_native else self.numpy_inputs['prev_feat'])[:] = \
          vision_output[self.vision_output_slices['hidden_state']]

      for i, policy_slices in enumerate(self._policy_slices_list):
        policy_output = raw_outputs[i + 1].numpy().flatten()
        policy_sliced = {k: policy_output[np.newaxis, v] for k, v in policy_slices.items()}
        parsed = self.parser.parse_policy_outputs(policy_sliced)
        if 'off' in self._policy_keys[i] and self._has_on_policy:
          for key in ('plan', 'planplus', 'action'):
            if any(key in self._policy_slices_list[j] for j, k in enumerate(self._policy_keys) if 'on' in k.lower()):
              parsed.pop(key, None)

        outputs.update(parsed)

      if 'planplus' in outputs and 'plan' in outputs:
        outputs['plan'] = outputs['plan'] + outputs['planplus']

    if 'desired_curvature' in outputs and 'prev_desired_curv' in self.numpy_inputs:
      buf = self.numpy_inputs['prev_desired_curv']
      buf[0, :-1] = buf[0, 1:]
      buf[0, -1, :] = outputs['desired_curvature'][0, :] if not self.mlsim else 0

    return outputs

  def _log_nonfinite_output(self, output, *, inference_ms):
    """Never mask rejection or delay fallback with additional GPU readbacks."""
    try:
      details = nonfinite_details(output, self.vision_output_slices, self.numpy_inputs,
                                  context=getattr(self, '_inference_context', None), inference_ms=inference_ms)
      cloudlog.event('model_output_nonfinite', error=True, details=details)
    except Exception:
      # Even a malformed diagnostic or unavailable logger must preserve the
      # original exception and the ordinary big-to-small recovery path.
      pass

  def get_action_from_model(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                            lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    if 'action' not in model_output:
      plan = model_output['plan'][0]
      desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], self.constants.T_IDXS,
                                          action_t=long_action_t)

      curvature_plan = (plan + (self.PLANPLUS_CONTROL - 1.0) * model_output['planplus'][0]
                        if 'planplus' in model_output and self.PLANPLUS_CONTROL != 1.0 else plan)
      desired_curvature = get_curvature_from_output(model_output, curvature_plan, v_ego, lat_action_t, self.mlsim)
    else:
      desired_accel = model_output['action'][0, 1]
      desired_curvature = model_output['action'][0, 0] / (max(1.0, v_ego))**2

    stop = v_ego < 0.3 and desired_accel < 0.1
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)

    if self.generation is not None and self.generation >= 10: # smooth curvature for post FOF models
      if v_ego > self.MIN_LAT_CONTROL_SPEED:
        desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
      else:
        desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature), desiredAcceleration=float(desired_accel), shouldStop=bool(stop))


class NavigationModelState(ModelState):
  """The reviewed a03 native action head, without inventing a SunnyPilot generation."""
  def __init__(self, *args, **kwargs):
    if kwargs.get('driving_permit') is None:
      raise NavigationModelDisabled('NavigationModelState requires a branch-reviewed driving permit')
    super().__init__(*args, **kwargs)
    self.LAT_SMOOTH_SECONDS = 0.0
    self.LONG_SMOOTH_SECONDS = 0.3

  def get_action_from_model(self, model_output, prev_action, lat_action_t, long_action_t, v_ego):
    from openpilot.selfdrive.modeld.modeld import get_action_from_model
    if 'action' not in model_output:
      raise NavigationModelDisabled('The reviewed navigation checkpoint requires its native action head')
    return get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)


def load_navigation_model(params, cam_w, cam_h, permit):
  """Construct the admitted big model in the ordinary Chestnut loader thread."""
  validate_runtime_permit(permit, params=params)
  if not chestnut_present():
    raise NavigationModelDisabled('Navigation driving requires the reviewed Chestnut device')
  from tinygrad.helpers import DEV
  from tinygrad.runtime.ops_amd import AMDDevice, USBIface
  AMDDevice.wait_timeout_ms = 3000
  device = Device['AMD']
  # DEV is configuration, not resolved hardware identity. Automatic selection
  # leaves renderer/interface empty. The admitted SPTG programs retain their
  # verified LLVM targets independently of the default source-code renderer.
  target = DEV.target('AMD', arch=device.arch)
  if (not isinstance(device, AMDDevice) or device.arch != 'gfx1200' or device.is_usb is not True or type(device.iface) is not USBIface or
      target.device != 'AMD' or target.arch != 'gfx1200' or target.renderer not in ('', 'LLVM') or target.interface not in ('', 'USB')):
    observed = (f'device={type(device).__name__}, arch={device.arch}, is_usb={getattr(device, "is_usb", None)}, ' +
                f'interface={type(getattr(device, "iface", None)).__name__}, configured={target!r}')
    raise NavigationModelDisabled('Actual GPU does not match the reviewed AMD LLVM gfx1200 USB target: ' + observed)
  return NavigationModelState(cam_w, cam_h, chestnut=True, driving_permit=permit)


def navigation_runtime_state(params, role, reason):
  """Report only; canonical model selection is the sole runtime authority."""
  params.put('NavigationModelRuntimeState', {
    'candidate_id': MODEL_ID, 'role': role, 'reason': reason,
    'active_source': 'chestnut' if role == 'candidate' else 'qcom' if role == 'ordinary_small' else None,
  }, block=True)


def load_selected_models(params, cam_w, cam_h, chestnut, navigation_selected):
  """Upstream big-loader/retry/small-preparation path, with a native Bigv2 adapter.

  Compatibility admission belongs to the selected big loader. An admission
  failure must follow the same small-model fallback as any other load failure;
  it must never substitute a separately saved ordinary big-model selection.
  """
  model = None
  navigation_permit = None
  big_load_error = None
  if navigation_selected:
    navigation_runtime_state(params, 'loading' if chestnut else 'small_loading',
                             'Loading selected corrected Bigv2' if chestnut else 'Chestnut unavailable; preparing ordinary small model')
  try:
    if chestnut:
      big_model = None
      def load_big():
        nonlocal big_model, big_load_error, navigation_permit
        try:
          wait_for_chestnut()
          if navigation_selected:
            permit = authorize_runtime(params)
            if permit is None:
              raise NavigationModelDisabled('Corrected Bigv2 is no longer the selected big model')
            m = load_navigation_model(params, cam_w, cam_h, permit)
          else:
            permit = None
            m = ModelState(cam_w=cam_w, cam_h=cam_h, chestnut=True)
          m.warmup()
          _reject_navigation_publication(m)
          navigation_permit = permit
          big_model = m
        except Exception as error:
          big_load_error = str(error)
          cloudlog.exception("chestnut load failed")
      loader = threading.Thread(target=load_big, daemon=True)
      loader.start()
      loader.join(BIG_MODEL_TIMEOUT)
      model = big_model
      params.put_bool("ChestnutActive", model is not None, block=True)
      if model is None and not params.get_bool("ChestnutRetry"):
        # A crank can cut Chestnut power during loading. Preserve upstream's
        # one-cycle retry; its key survives that synthetic off-road transition.
        cloudlog.error("big model load failed, cycling onroad to retry")
        params.put_bool("ChestnutRetry", True, block=True)
        params.put_bool("OnroadCycleRequested", True)

    # Upstream prepares the ordinary fallback, but does not infer with it while
    # the selected big model is active. No shadow inference or custom priming.
    small_model = ModelState(cam_w=cam_w, cam_h=cam_h, chestnut=False) if model is None or chestnut else None
    if model is None:
      model = small_model
    assert model is not None
    _reject_navigation_publication(model, small_model)
    if navigation_selected:
      navigation_runtime_state(params, 'ready' if model.chestnut else 'small_loading',
                               'Corrected Bigv2 loaded; awaiting predictions' if model.chestnut else
                               'Ordinary small model prepared; awaiting predictions. ' +
                               (big_load_error or ('Big model load timed out' if chestnut else 'Chestnut unavailable')))
    return model, small_model, navigation_permit
  except BaseException as error:
    params.put_bool("ChestnutActive", False, block=True)
    if navigation_selected:
      navigation_runtime_state(params, 'failed', 'Model initialization failed: ' + str(error))
    raise
  finally:
    params.put_bool("ChestnutLoading", False, block=True)


def main(demo=False):
  cloudlog.warning("modeld init")

  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)

  params = Params()
  navigation_selected = is_navigation_selected(params)
  selected_big = navigation_selected or _find_driving_pkl(get_active_bundle(params, chestnut=True)) is not None
  CHESTNUT = selected_big and (chestnut_present() or cable_connected())
  if CHESTNUT:
    from tinygrad.runtime.ops_amd import AMDDevice
    AMDDevice.wait_timeout_ms = 3000

  params.put_bool("ChestnutLoading", CHESTNUT, block=True)
  params.remove("ChestnutActive")
  if navigation_selected:
    navigation_runtime_state(params, 'loading', 'Waiting for camera streams')
  else:
    params.remove('NavigationModelRuntimeState')

  gc.disable()

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_NARROW_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_NARROW_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_NARROW_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  cloudlog.warning("loading model")
  st = time.monotonic()

  model, small_model, navigation_permit = load_selected_models(
    params, vipc_client_main.width, vipc_client_main.height, CHESTNUT, navigation_selected)
  reported_model = None
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  config_realtime_process(7, 54)

  # messaging
  pub_socks = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"] + (["chestnutGpuState"] if CHESTNUT else [])
  pm = PubMaster(pub_socks)
  nav_inputs = NavigationInputs(params, driving_permit=navigation_permit) if takes_nav(model, small_model) else None
  sm = SubMaster(["deviceState", "carState", "narrowRoadCameraState", "extrinsicsCalibration", "driverMonitoringState",
                  "carControl", "controlsState", "lateralDelay"] +
                 (["navInstruction", "navModelStateSP", "gpsLocation", "gpsLocationExternal"] if nav_inputs is not None else []))

  publish_state = PublishState()
  chestnut_state = ChestnutGpuState(pm, model.chestnut) if CHESTNUT else None

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / model.constants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  restore_ordinary_conditioning = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  camera_offset_helper = CameraOffsetHelper()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)
  firmware_config = None if demo else drive_config_for_model(params, CP)

  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + model.LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = Bigv2DesireHelper() if navigation_permit is not None and model.chestnut else DesireHelper()
  meta_constants = load_meta_constants()
  RELC = RoadEdgeLaneChangeController()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                       extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    native_bigv2 = navigation_permit is not None and model.chestnut
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["narrowRoadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if native_bigv2:
      # Preserve the required firmware-controller prediction horizon. Without
      # that controller, use the checkpoint's current lateralDelay directly.
      prediction_horizon = prediction_delay(v_ego) if firmware_config is not None else sm["lateralDelay"].lateralDelay
      lat_delay = prediction_horizon + model.LAT_SMOOTH_SECONDS
    else:
      if sm.frame % 60 == 0 or restore_ordinary_conditioning:
        if firmware_config is None:
          model.lat_delay = get_lat_delay(params, sm["lateralDelay"].lateralDelay)
        model.PLANPLUS_CONTROL = params.get("PlanplusControl", return_default=True)
        camera_offset_helper.set_offset(params.get("CameraOffset", return_default=True))
      if firmware_config is not None:
        model.lat_delay = prediction_delay(v_ego)
      lat_delay = model.lat_delay + model.LAT_SMOOTH_SECONDS
    if (sm.updated["extrinsicsCalibration"] or restore_ordinary_conditioning) and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["extrinsicsCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      if native_bigv2:
        model_transform_main, model_transform_extra = bigv2_warp_matrices(device_from_calib_euler, dc, main_wide_camera, use_extra_client)
      else:
        main_intrinsics = dc.wide_road.intrinsics if main_wide_camera else dc.narrow_road.intrinsics
        model_transform_main = get_warp_matrix(device_from_calib_euler, main_intrinsics, False).astype(np.float32)
        model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.wide_road.intrinsics, True).astype(np.float32)
        model_transform_main, model_transform_extra = camera_offset_helper.update(model_transform_main, model_transform_extra, sm, main_wide_camera)
      live_calib_seen = True
      restore_ordinary_conditioning = False

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(model.constants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < model.constants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}

    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay

    inputs:dict[str, np.ndarray] = {
      model.desire_key: vec_desire,
      'traffic_convention': traffic_convention,
    }

    if 'lateral_control_params' in model.numpy_inputs:
      inputs['lateral_control_params'] = np.array([v_ego, lat_delay], dtype=np.float32)

    if 'action_t' in model.numpy_inputs:
      inputs['action_t'] = np.array([lat_action_t, long_action_t], dtype=np.float32)

    mt1 = time.perf_counter()
    try:
      if nav_inputs is not None and takes_nav(model):
        inputs.update(nav_inputs.read_for_model(sm, v_ego, meta_main.timestamp_sof) if navigation_permit is not None else
                      nav_inputs.read(sm, v_ego, meta_main.timestamp_sof))
      if navigation_permit is not None and takes_nav(model):
        model._inference_context = {
          'frame_id': int(meta_main.frame_id), 'camera_sof_ns': int(meta_main.timestamp_sof),
          'camera_eof_ns': int(meta_main.timestamp_eof), 'camera_extra_skew_ns': int(meta_extra.timestamp_sof - meta_main.timestamp_sof),
          'run_count': run_count, 'map_reason': getattr(nav_inputs, 'map_reason', 'unavailable'),
          'maneuver_reason': getattr(nav_inputs, 'maneuver_reason', 'unavailable'),
        }
      if native_bigv2:
        mt1 = time.perf_counter()  # Official execution timing starts after input preparation.
      send_chestnut = (chestnut_state is not None and
                       run_count % round(model.constants.MODEL_FREQ / SERVICE_LIST['chestnutGpuState'].frequency) == 0)
      model_output = model.run(bufs, transforms, inputs, chestnut_state.send if send_chestnut else None)
    except Exception as error:
      if not model.chestnut:
        if navigation_selected:
          navigation_runtime_state(params, 'failed', 'Ordinary small-model inference failed: ' + str(error))
        raise
      cloudlog.exception("big model failed, fall back to small")
      params.put_bool("ChestnutActive", False)
      assert small_model is not None
      if navigation_selected:
        # Preserve this branch's model-specific timing on the ordinary failure
        # handover; no navigation-data event initiates a model switch.
        if firmware_config is None:
          small_model.lat_delay = get_lat_delay(params, sm['lateralDelay'].lateralDelay)
        long_delay = CP.longitudinalActuatorDelay + small_model.LONG_SMOOTH_SECONDS
        navigation_runtime_state(params, 'small_loading', 'Big model failed; awaiting ordinary small-model predictions: ' + str(error))
      if native_bigv2:
        # Restore optional ordinary-model conditioning without inventing a new
        # desire transition: carry the official helper's common state forward.
        ordinary_desire = DesireHelper()
        for name, value in vars(DH).items():
          setattr(ordinary_desire, name, value)
        DH = ordinary_desire
        restore_ordinary_conditioning = live_calib_seen
      model = small_model
      if chestnut_state is not None:
        chestnut_state.big = False
      run_count = 0
      model_output = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      mdv2sp_send = messaging.new_message('modelDataV2SP')

      if not native_bigv2 and not sm['carControl'].latActive:
        prev_action.desiredCurvature = sm['controlsState'].curvature
      action = model.get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      if native_bigv2:
        publish_bigv2_model(modelv2_send, model_output, action, publish_state,
                            meta_main.frame_id, meta_extra.frame_id, frame_id,
                            frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)
      else:
        fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                       publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                       frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen, meta_constants)
      modelv2_send.modelV2.big = model.chestnut

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      if native_bigv2:
        DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      else:
        left_edge, right_edge = RELC.update_and_fill(modelv2_send.modelV2, mdv2sp_send.modelDataV2SP, v_ego)
        DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob, left_edge, right_edge)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      mdv2sp_send.valid = modelv2_send.valid
      mdv2sp_send.modelDataV2SP.laneTurnDirection = 'none' if native_bigv2 else DH.lane_turn_direction
      if native_bigv2:
        fill_driving_model_data(drivingdata_send, modelv2_send)
      else:
        drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
        drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      pose_publisher = fill_bigv2_pose_msg if native_bigv2 else fill_pose_msg
      pose_publisher(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      pm.send('modelDataV2SP', mdv2sp_send)
      if navigation_selected and reported_model is not model:
        navigation_runtime_state(params, 'candidate' if model.chestnut else 'ordinary_small',
                                 'Corrected Bigv2 publishing predictions' if model.chestnut else 'Ordinary small model publishing predictions')
        reported_model = model
    last_vipc_frame_id = meta_main.frame_id

if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    cloudlog.exception("modeld exception")
    raise
