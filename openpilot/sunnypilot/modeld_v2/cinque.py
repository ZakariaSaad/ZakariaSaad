"""V3 adapter that keeps MVL's model publication and Honda control stack."""
import os
import base64
import pickle
import re
import numpy as np
from tinygrad import Tensor, Context
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.realize import lower_and_compile

from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.cereal import log, messaging
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.sunnypilot.modeld_v2.cinque_guard import verify_manifest, open_artifact
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Meta
from openpilot.sunnypilot.modeld_v2.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.sunnypilot.modeld_v2.modeld import ModelState
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.stateful import StatefulRunner, PackedStatefulRunner, validate_predictions
from openpilot.selfdrive.modeld.compile_modeld import make_warp, NV12Frame
from openpilot.sunnypilot.models.helpers import get_active_bundle, _bundle_artifacts, _bundle_is_valid_locally
from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref


class VerifiedFallback(ModelState):
  def run(self, *args, **kwargs):
    outputs = super().run(*args, **kwargs)
    validate_predictions(outputs)
    return outputs


def load_fallback(cam_w, cam_h):
  if os.environ.get('COMBINED_MODEL_PKL'):
    raise ValueError('V3 cannot use an unverified COMBINED_MODEL_PKL fallback')
  bundle = get_active_bundle(chestnut=False)
  if bundle is None or str(bundle.runner) != 'tinygrad':
    raise ValueError('Select and download a working legacy Tinygrad fallback before enabling V3')
  artifacts = _bundle_artifacts(bundle)
  # Each chunk must be authenticated, not merely enough chunks to match the
  # number of models. The general catalog helper can omit missing checksums.
  for model in bundle.models:
    artifact = model.artifact
    if not artifact.fileName or os.path.basename(artifact.fileName) != artifact.fileName:
      raise ValueError('Fallback artifact must be a safe model filename')
    checksums = [chunk.sha256 for chunk in artifact.chunks] if artifact.chunks else [artifact.downloadUri.sha256]
    if not checksums or any(not re.fullmatch(r'[0-9a-fA-F]{64}', digest) for digest in checksums):
      raise ValueError('Fallback model artifacts must have complete verified checksums')
  if len(artifacts) < len(bundle.models) or not artifacts or not _bundle_is_valid_locally(bundle):
    raise ValueError('Fallback model artifacts must have verified checksums')
  model = VerifiedFallback(cam_w, cam_h, chestnut=False)
  if model.chestnut or 'AMD' in model.WARP_DEV.upper() or 'AMD' in model.DEV.upper():
    raise ValueError('Verified V3 fallback must run and warp on the built-in device')
  return model


class CinqueModelState(ModelState):
  def __init__(self, cam_w, cam_h, manifest=None):
    ModelStateBase.__init__(self)
    manifest = manifest if manifest is not None else Params().get('CinqueV3Manifest')
    path = verify_manifest(manifest, Paths.model_root(), get_tinygrad_ref())
    # Strict unpickling: never remap Tinygrad enums/classes across revisions.
    with open_artifact(path) as f:
      artifact = load_oob(f)
    self.packed_format = isinstance(artifact, dict) and {'input_shapes', 'state_pairs', 'run_model', 'input_devices', 'metadata'} <= artifact.keys()
    if not self.packed_format and (not isinstance(artifact, dict) or not {'input_specs', 'output_specs', 'run', 'metadata'} <= artifact.keys()):
      raise ValueError('V3 requires all input_specs/output_specs and named next-state outputs; recompile legacy single-output pickles')
    # Accept the official explicit-buffer export as well as our local export.
    if 'tinygrad_ref' in artifact and artifact['tinygrad_ref'] != get_tinygrad_ref():
      raise ValueError('V3 artifact Tinygrad revision mismatch')
    metadata = artifact['metadata']
    if 'output_slices' not in metadata:
      metadata['output_slices'] = pickle.loads(base64.b64decode(metadata['metadata']['output_slices']))
    if 'run' in artifact and getattr(artifact['run'], 'captured', None) is not None:
      artifact['run'].captured._linear = lower_and_compile(artifact['run'].captured._linear)
    stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
    self.frame_copy_size = stride * (y_height + uv_height)
    self.stateful = (PackedStatefulRunner(artifact, (cam_w, cam_h), self.frame_copy_size, Parser()) if self.packed_format else
                     StatefulRunner(artifact, Parser()))
    self.model_device = self.stateful.input_specs['new_img'][2]
    self.chestnut = 'AMD' in self.model_device
    self.generation = None
    # The separately installed artifact need not include camera warps.
    # Generate them on the model's device, using MVL's NV12 implementation.
    if self.packed_format:
      self.warp = None
    elif 'warps' in artifact:
      self.warp = artifact['warps'][(cam_w, cam_h)]
    else:
      nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
      _, _, model_h, model_w = artifact['input_specs']['new_img'][0]
      with Context(DEV=self.model_device):
        # Six YUV planes are half the full warped image's height/width.
        self.warp = TinyJit(make_warp(nv12, 2 * model_w, 2 * model_h))
    self.constants = ModelConstants()
    self.meta_constants = Meta
    self.LAT_SMOOTH_SECONDS, self.LONG_SMOOTH_SECONDS = 0., .3
    self.MIN_LAT_CONTROL_SPEED, self.PLANPLUS_CONTROL = .3, 1.
    self._vision_input_names, self._desire_key = ['img', 'big_img'], 'desire'
    self.frames = {k: np.zeros(self.frame_copy_size, dtype=np.uint8) for k in self.vision_input_names}
    self.numpy_inputs = {k: np.zeros(shape, dtype=dtype) for k, (shape, dtype, _) in self.stateful.input_specs.items()
                         if k in ('desire', 'traffic_convention', 'action_t')}
    self.frame_tensors = {k: Tensor(value, device='NPY').realize() for k, value in self.frames.items()}
    self.transforms = {k: np.eye(3, dtype=np.float32) for k in self.vision_input_names}
    self.transform_tensors = {k: Tensor(value, device='NPY').realize() for k, value in self.transforms.items()}

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    if self.packed_format:
      outputs = self.stateful.run_frames(bufs, transforms, inputs)
      if after_enqueue is not None:
        after_enqueue()
      return outputs
    for name in self.vision_input_names:
      data = bufs[name].data if hasattr(bufs[name], 'data') else bufs[name]
      np.copyto(self.frames[name], np.frombuffer(data, dtype=np.uint8, count=self.frame_copy_size))
      if not np.all(np.isfinite(transforms[name])):
        raise ValueError('Nonfinite V3 camera transform')
      np.copyto(self.transforms[name], transforms[name])
    with Context(DEV=self.model_device):
      warped = self.warp(tfm=self.transform_tensors['img'], big_tfm=self.transform_tensors['big_img'],
                         frame=self.frame_tensors['img'], big_frame=self.frame_tensors['big_img'])
    outputs = self.stateful.run(warped, inputs['desire'], inputs['traffic_convention'], inputs['action_t'])
    if after_enqueue is not None:
      after_enqueue()
    return outputs

  def warmup(self):
    # Exercise recurrence and reset/replay determinism before any live frame.
    frames = {k: np.zeros(self.frame_copy_size, dtype=np.uint8) for k in self.vision_input_names}
    transforms = dict.fromkeys(self.vision_input_names, np.eye(3, dtype=np.float32))
    inputs = {k: np.zeros_like(v) for k, v in self.numpy_inputs.items()}
    inputs['traffic_convention'][0, 0] = 1
    self.stateful.reset()
    first = {k: value.copy() for k, value in self.run(frames, transforms, inputs).items()}
    self.check_publication(first)
    self.run(frames, transforms, inputs)
    self.stateful.reset()
    repeated = self.run(frames, transforms, inputs)
    for name, value in first.items():
      np.testing.assert_allclose(repeated[name], value, rtol=1e-5, atol=1e-6, err_msg=f'V3 reset mismatch: {name}')
    self.stateful.reset()

  def check_publication(self, outputs):
    model, driving, pose = (messaging.new_message(service) for service in ('modelV2', 'drivingModelData', 'cameraOdometry'))
    action = self.get_action_from_model(outputs, log.ModelDataV2.Action(), .2, .3, 10.)
    if not np.isfinite(action.desiredCurvature) or not np.isfinite(action.desiredAcceleration):
      raise ValueError('Nonfinite V3 action')
    fill_model_msg(driving, model, outputs, action, PublishState(), 0, 0, 0, 0., 0, 0., False, Meta)
    fill_pose_msg(pose, outputs, 0, 0, 0, False)
    for msg in (model, driving, pose):
      msg.to_bytes()
