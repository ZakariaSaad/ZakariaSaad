"""Differential model conditioning/publication against exact official a03 source.

Synthetic host tensors and messages; no GPU, model compilation or vehicle I/O.
"""
import ast
import copy
from collections.abc import Callable
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import capnp
import numpy as np
import pytest
from tinygrad.device import Buffer
from tinygrad.dtype import DType, dtypes, _from_np_dtype
from tinygrad.helpers import round_up
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp

from openpilot.cereal import log, messaging
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.controls.lib import drive_helpers
from openpilot.selfdrive.modeld.constants import ModelConstants, Meta, Plan
from openpilot.selfdrive.navigation.tests.official_reference import load_reference, source
from openpilot.selfdrive.navigation.tests.test_activation import reviewed_fixture
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.tests.test_driving_navigation import fixture_data


def reference_action():
  deps = {'log': log, 'ModelConstants': ModelConstants, 'Plan': Plan,
          **{name: getattr(drive_helpers, name) for name in
             ('get_accel_from_plan', 'get_curvature_from_plan', 'should_stop', 'smooth_value')}}
  return load_reference('modeld', names=('get_action_from_model', 'LAT_SMOOTH_SECONDS',
                        'LONG_SMOOTH_SECONDS', 'MIN_LAT_CONTROL_SPEED'), namespace=deps).get_action_from_model


def native_state():
  """Real native ModelState construction/dispatch with inert artifact reader."""
  jits, metadata = fixture_data()
  with reviewed_fixture(jits['input_specs'], metadata) as (_, permit, _), patch.object(modeld, 'load_model', return_value=jits):
    return modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)


@pytest.mark.parametrize('seed', range(8))
def test_native_parser_and_raw_action_match_official_checkpoint(seed):
  state = native_state()
  reference = load_reference('parse_model_outputs', namespace={'ModelConstants': ModelConstants}).Parser()
  rng = np.random.default_rng(seed)
  raw = rng.normal(0., 2., (1, 18580)).astype(np.float32)
  sliced = {name: raw[:, bounds].copy() for name, bounds in state.vision_output_slices.items()}
  actual = state.parser.parse_outputs(copy.deepcopy(sliced))
  expected = reference.parse_outputs(copy.deepcopy(sliced))
  assert actual.keys() == expected.keys()
  for name in actual:
    np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
  assert actual['action'].shape == (1, 4), 'Official native action is not reinterpreted as an MDN head'
  previous = log.ModelDataV2.Action(desiredCurvature=.012, desiredAcceleration=-.7)
  get_action = reference_action()
  for speed in (0., .3, .30001, 1., 10., 30.):
    for lateral, longitudinal in ((.075, .575), (.245, .675), (.475, 1.25)):
      expected_action = get_action(expected, previous, lateral, longitudinal, speed)
      actual_action = state.get_action_from_model(actual, previous, lateral, longitudinal, speed)
      assert actual_action.to_dict() == expected_action.to_dict()


def test_desire_sequence_matches_official_without_optional_sp_conditions():
  official = load_reference('desire_helper', namespace={'log': log, 'CV': CV, 'DT_MDL': DT_MDL}).DesireHelper()
  actual = modeld.Bigv2DesireHelper()
  state = SimpleNamespace(vEgo=22., leftBlinker=False, rightBlinker=False, steeringPressed=False,
                          steeringTorque=0., leftBlindspot=False, rightBlindspot=False)
  sequence = [({}, True, .5), ({'leftBlinker': True}, True, .5),
              ({'steeringPressed': True, 'steeringTorque': 1., 'leftBlindspot': True}, True, .5),
              ({'leftBlindspot': False}, True, .5)]
  sequence += [({}, True, .01)] * 14
  sequence += [({'leftBlinker': False}, True, .01), ({'rightBlinker': True, 'steeringTorque': -1.}, True, .01)]
  sequence += [({}, True, .01)] * 14
  sequence += [({}, False, 0.), ({'vEgo': 2.}, True, 0.)]
  pulses, reference_pulses = [], []
  previous = np.zeros(8, np.float32)
  previous_ref = previous.copy()
  for updates, active, probability in sequence:
    for name, value in updates.items():
      setattr(state, name, value)
    official.update(state, active, probability)
    actual.update(state, active, probability)
    assert vars(actual) == vars(official)
    desire = np.eye(8, dtype=np.float32)[int(actual.desire)]
    desire_ref = np.eye(8, dtype=np.float32)[int(official.desire)]
    desire[0] = desire_ref[0] = 0
    pulses.append(np.where(desire - previous > .99, desire, 0))
    reference_pulses.append(np.where(desire_ref - previous_ref > .99, desire_ref, 0))
    previous[:], previous_ref[:] = desire, desire_ref
  np.testing.assert_array_equal(pulses, reference_pulses)


@pytest.mark.parametrize('main_wide,use_extra', ((False, False), (False, True), (True, False)))
def test_camera_transforms_match_exact_official_main_block(main_wide, use_extra):
  # Execute the calibration body from official main, with identical intrinsics,
  # calibration and stream availability. No re-created reference calculation.
  text = source('modeld')
  start = text.index('      device_from_calib_euler =')
  end = text.index('      extrinsics_calibration_seen = True', start)
  block = '\n'.join(line[6:] for line in text[start:end].splitlines())
  sm = {'deviceState': SimpleNamespace(deviceType='mici'),
        'narrowRoadCameraState': SimpleNamespace(sensor='os04c10')}
  camera = DEVICE_CAMERAS[('mici', 'os04c10')]
  for calibration in ((0., 0., 0.), (.005, -.02, .03), (-.02, .01, -.04)):
    sm['extrinsicsCalibration'] = SimpleNamespace(rpyCalib=calibration)
    context = {'sm': sm, 'np': np, 'DEVICE_CAMERAS': DEVICE_CAMERAS,
               'get_warp_matrix': modeld.get_warp_matrix, 'main_wide_camera': main_wide, 'use_extra_client': use_extra}
    exec(compile(block, 'official_a03_camera_block', 'exec'), context)
    actual = modeld.bigv2_warp_matrices(np.array(calibration, np.float32), camera, main_wide, use_extra)
    np.testing.assert_array_equal(actual[0], context['model_transform_main'])
    np.testing.assert_array_equal(actual[1], context['model_transform_extra'])


def test_native_packed_camera_bytes_and_recurrent_aliases_match_official_layout():
  state = native_state()
  adapter = state.adapter
  # Official host layout uses 128-byte alignment in input_specs insertion order.
  official_class = next(node for node in ast.parse(source('modeld')).body if isinstance(node, ast.ClassDef) and node.name == 'ModelState')
  pack = next(node for node in official_class.body if isinstance(node, ast.FunctionDef) and node.name == 'pack_inputs')
  assignments = [node for node in ast.walk(pack) if isinstance(node, ast.Assign)]
  shapes_stmt = next(node for node in assignments if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'shapes')
  size_stmt = next(node for node in assignments if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'npy_size')
  context = {'self': adapter, 'np': np, 'math': __import__('math'), 'round_up': lambda n, align: -(-n // align) * align}
  exec(compile(ast.Module(body=[shapes_stmt, size_stmt], type_ignores=[]), 'official_a03_pack_layout', 'exec'), context)
  assert len(adapter.packed_input) == context['npy_size'] + 2 * adapter.frame_copy_size
  offset = 0
  for name, (shape, dtype) in context['shapes'].items():
    array = adapter.numpy_inputs[name]
    assert array.shape == shape and array.dtype == dtype
    assert array.ctypes.data - adapter.packed_input.ctypes.data == offset
    offset += -(-array.nbytes // 128) * 128
  frames = {'img': np.arange(adapter.frame_copy_size, dtype=np.uint8),
            'big_img': 255 - np.arange(adapter.frame_copy_size, dtype=np.uint8)}
  adapter.copy_frames(frames)
  np.testing.assert_array_equal(adapter.frames[0], frames['img'])
  np.testing.assert_array_equal(adapter.frames[1], frames['big_img'])
  for name, next_name in adapter.state_pairs.items():
    incoming = adapter.input_queues[name]
    outgoing = adapter.outputs[next_name]
    assert incoming._buffer().base is outgoing._buffer().base
    np.testing.assert_array_equal(incoming.numpy(), np.zeros(incoming.shape, dtype=incoming.numpy().dtype))


@pytest.mark.parametrize('dual', (False, True))
def test_actual_camera_intake_block_matches_official_with_delayed_and_missing_frames(dual):
  def intake_block(text):
    start = text.index('    while meta_main.timestamp_sof <')
    end = text.index('    sm.update(0)', start)
    # Keep the outer loop so an absent camera frame executes its real continue.
    return compile('for _ in range(1):\n' + '\n'.join(line[2:] for line in text[start:end].splitlines()),
                   'modeld_camera_intake', 'exec')

  actual_code, official_code = intake_block(inspect.getsource(modeld.main)), intake_block(source('modeld'))
  official_meta = load_reference('modeld', names=('FrameMeta',)).FrameMeta

  class Camera:
    frame_id = timestamp_sof = timestamp_eof = 0

    def __init__(self, rows):
      self.rows = iter(rows)
      self.calls = 0

    def recv(self):
      frame, millis, present = next(self.rows)
      self.calls += 1
      self.frame_id, self.timestamp_sof = frame, 100_000_000_000 + millis * 1_000_000
      self.timestamp_eof = self.timestamp_sof + 10_000_000
      return SimpleNamespace(data=bytes([frame])) if present else None

  main = [(1, 0, True), (1, 0, True), (2, 50, True), (4, 150, True), (5, 200, False), (6, 250, True)]
  extra = [(0, -50, True), (1, 5, True), (2, 55, True), (3, 105, True), (4, 155, True), (6, 255, True)]

  def context(meta):
    return {'meta_main': meta(), 'meta_extra': meta(), 'FrameMeta': meta, 'use_extra_client': dual,
            'vipc_client_main': Camera(main), 'vipc_client_extra': Camera(extra), 'cloudlog': Mock(),
            'buf_main': None, 'buf_extra': None}

  actual, expected = context(modeld.FrameMeta), context(official_meta)
  for _ in range(5):
    exec(actual_code, actual)
    exec(official_code, expected)
    for name in ('meta_main', 'meta_extra'):
      assert vars(actual[name]) == vars(expected[name])
    for name in ('buf_main', 'buf_extra'):
      assert actual[name] == expected[name]
    for name in ('vipc_client_main', 'vipc_client_extra'):
      assert actual[name].calls == expected[name].calls


def test_native_publication_matches_official_model_pose_and_driving_data():
  state = native_state()
  raw = np.random.default_rng(19).normal(0., .2, (1, 18580)).astype(np.float32)
  parsed = state.parser.parse_outputs({key: raw[:, value].copy() for key, value in state.vision_output_slices.items()})
  reference = load_reference('fill_model_msg', namespace={'capnp': capnp, 'log': log, 'ModelConstants': ModelConstants,
                                                        'Plan': Plan, 'Meta': Meta})
  actual_state, expected_state = modeld.PublishState(), reference.PublishState()
  previous = log.ModelDataV2.Action()
  for frame in range(1, 45):
    action = reference_action()(parsed, previous, .225, .575, 17.)
    actual, expected = messaging.new_message('modelV2'), messaging.new_message('modelV2')
    args = (frame, frame, frame + 1, .025, frame * 50_000_000, .032, True)
    modeld.publish_bigv2_model(actual, parsed, action, actual_state, *args)
    reference.fill_model_msg(expected, parsed, action, expected_state, *args)
    assert actual.modelV2.to_dict() == expected.modelV2.to_dict()
    actual_driving, expected_driving = messaging.new_message('drivingModelData'), messaging.new_message('drivingModelData')
    modeld.fill_driving_model_data(actual_driving, actual)
    reference.fill_driving_model_data(expected_driving, expected)
    assert actual_driving.drivingModelData.to_dict() == expected_driving.drivingModelData.to_dict()
    actual_pose, expected_pose = messaging.new_message('cameraOdometry'), messaging.new_message('cameraOdometry')
    modeld.fill_bigv2_pose_msg(actual_pose, parsed, frame, 0, args[4], True)
    reference.fill_pose_msg(expected_pose, parsed, frame, 0, args[4], True)
    assert actual_pose.cameraOdometry.to_dict() == expected_pose.cameraOdometry.to_dict()
    previous = action


def test_real_cpu_dispatch_matches_official_run_pulses_raw_outputs_and_aliased_states():
  """Run exact official ModelState.run and native dispatch on current-host CPU.

  A small trusted synthetic Tensor graph consumes the real packed inputs and
  updates all five checkpoint-sized recurrent buffers. This tests the runtime
  ABI/state sequence, not Bigv2 weights, camera-warp kernels, AMD or performance.
  """
  jits, metadata = fixture_data()
  for field in ('input_specs', 'output_specs'):
    jits[field] = {name: (shape, dtype, 'CPU') for name, (shape, dtype, _) in jits[field].items()}
  metadata['output_specs'] = {name: {'shape': list(shape), 'dtype': np.dtype(dtype).name, 'device': device}
                              for name, (shape, dtype, device) in jits['output_specs'].items()}
  oracle = load_reference('modeld', names=('ModelState', 'input_view', 'SEND_RAW_PRED'), namespace={
    'Callable': Callable, 'VisionBuf': SimpleNamespace, 'ModelConstants': ModelConstants,
    'Tensor': Tensor, 'Buffer': Buffer, 'DType': DType, 'dtypes': dtypes,
    '_from_np_dtype': _from_np_dtype, 'UOp': UOp, 'round_up': round_up})
  expected = object.__new__(oracle.ModelState)
  expected.input_shapes = {name: (shape, np.dtype(dtype)) for name, (shape, dtype, _) in jits['input_specs'].items()}
  expected.model_device = 'CPU'
  expected.state_pairs = {name: f'next_{name}' for name in expected.input_shapes if f'next_{name}' in jits['output_specs']}
  expected.frame_copy_size = 1622016
  expected.vision_input_names = ('img', 'big_img')
  expected.prev_desire = np.zeros(8, np.float32)
  expected.chestnut = True
  expected.pack_inputs()
  expected.outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                      for name, (shape, dtype, device) in jits['output_specs'].items()}
  for name, next_name in expected.state_pairs.items():
    tensor = expected.input_queues[name]
    expected.outputs[next_name] = oracle.input_view(tensor._buffer(), tensor.shape, tensor.dtype, 0)
  expected.parser = load_reference('parse_model_outputs', namespace={'ModelConstants': ModelConstants}).Parser()

  def synthetic_warp(input_frame, M_inv):
    # Both cameras and all nine transform entries contribute. The actual warp
    # is already a separately verified artifact; this is an ABI probe.
    return (input_frame[0, :64].cast(dtypes.float32).sum() +
            input_frame[1, :64].cast(dtypes.float32).sum() * 2 + M_inv.sum()).reshape(1).realize()

  def synthetic_graph(output_buffers, **inputs):
    seed = (inputs['new_img'].sum() * .000001 + inputs['action_t'].sum() + inputs['desire'].sum() +
            inputs['navigation'].sum() * .001 + inputs['nav_map'].mean() * .001 + inputs['traffic_convention'].sum())
    summary = (inputs['state_feat_q'][0, 0, 0] + seed).contiguous().realize()
    for name, next_name in expected.state_pairs.items():
      state = inputs[name]
      output_buffers[next_name].assign((state.cast(dtypes.float32) * .5 + seed).cast(state.dtype)).realize()
    output_buffers['outputs'].assign(summary.reshape(1, 1).expand(1, 18580) * .001).realize()

  expected.run_warp, expected.run_model = synthetic_warp, synthetic_graph
  with reviewed_fixture(jits['input_specs'], metadata) as (_, permit, _), patch.object(modeld, 'load_model', return_value=jits), \
       patch.object(modeld, 'Device', {'AMD': SimpleNamespace(error_state=SimpleNamespace(host=SimpleNamespace(view=lambda **_: [0])))}):
    actual = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
    expected.output_slices = actual.vision_output_slices

    def fused_graph(host, output_buffers, **states):
      packed = host.numpy()
      base = actual.adapter.packed_input.ctypes.data
      host_inputs = {name: Tensor(np.ndarray(array.shape, dtype=array.dtype, buffer=packed,
                                             offset=array.ctypes.data - base).copy(), device='CPU').realize()
                     for name, array in actual.numpy_inputs.items()}
      frames = actual.adapter.frames
      frame_tensor = Tensor(np.ndarray(frames.shape, dtype=frames.dtype, buffer=packed,
                                       offset=frames.ctypes.data - base).copy(), device='CPU').realize()
      image = synthetic_warp(frame_tensor, host_inputs.pop('tfm'))
      synthetic_graph(output_buffers, **states, **host_inputs, new_img=image)

    actual.adapter.run_model = fused_graph
    expected.warmup()
    actual.warmup()
    np.testing.assert_array_equal(actual.adapter.packed_input, expected.packed_input)
    for name in expected.state_pairs:
      np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), expected.input_queues[name].numpy(), err_msg=name)
    frames = {name: SimpleNamespace(data=np.full(1622016, value, np.uint8)) for name, value in (('img', 17), ('big_img', 193))}
    transforms = {name: np.eye(3, dtype=np.float32) * value for name, value in (('img', 1.), ('big_img', 2.))}
    for index, desire in enumerate((0, 3, 3, 0, 4, 4, 0, 3, 0)):
      common = {'traffic_convention': np.array([1., 0.], np.float32),
                'action_t': np.array([.2 + index*.01, .575], np.float32)}
      if index not in (3, 6):
        common.update(nav_map=np.full((256, 256), index*19, np.float32),
                      navigation=np.eye(47, dtype=np.float32)[index])
      pulse = np.eye(8, dtype=np.float32)[desire]
      wanted = expected.run(frames, transforms, {'desire_pulse': pulse.copy(), **copy.deepcopy(common)})
      got = actual.run(frames, transforms, {'desire': pulse.copy(), **copy.deepcopy(common)})
      np.testing.assert_array_equal(actual.numpy_inputs['desire'], expected.npy['desire'])
      np.testing.assert_array_equal(actual.adapter.packed_input, expected.packed_input)
      np.testing.assert_array_equal(actual.adapter.outputs['outputs'].numpy(), expected.outputs['outputs'].numpy())
      for name in expected.state_pairs:
        np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), expected.input_queues[name].numpy(), err_msg=name)
      for name in wanted:
        np.testing.assert_array_equal(got[name], wanted[name], err_msg=name)
