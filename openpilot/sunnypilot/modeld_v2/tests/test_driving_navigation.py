"""Actual ModelState/adapter paths with synthetic trusted JITs, never a GPU or CAN."""
import codecs
from contextlib import ExitStack
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tinygrad.helpers import Context, DEV
from tinygrad.runtime.ops_amd import AMDDevice, USBIface

from openpilot.selfdrive.navigation import activation, model_inputs
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub, reviewed_fixture
from openpilot.selfdrive.navigation.tests.test_model_input_admission import Client, Messages
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.tests.test_navigation_inputs import FIXTURES, native_fixture


def fixture_data():
  signature = json.loads(FIXTURES.read_text())['models'][-1]
  jits = native_fixture(signature)
  metadata = {'output_specs': {name: {'shape': list(shape), 'dtype': np.dtype(dtype).name, 'device': device}
                               for name, (shape, dtype, device) in jits['output_specs'].items()},
              'output_slices': signature['output_slices']}
  return jits, metadata


def physical_amd_fixture(*, arch='gfx1200', is_usb=True, iface=None):
  """Use pinned device/interface types without invoking hardware constructors."""
  device = object.__new__(AMDDevice)
  device.device, device.arch, device.is_usb = 'AMD', arch, is_usb
  device.iface = object.__new__(USBIface) if iface is None else iface
  return device


class TestDrivingNavigation(unittest.TestCase):
  def test_startup_selects_initialized_candidate_and_keeps_small_available_without_running_it(self):
    """Exercise production initialization only; no publishers or vehicle processes."""
    class InitializationFinished(Exception):
      pass

    for mode, expected_loads in (('approved', [False]), ('off', [True, False]), ('blocked', [True, False]),
                                 ('candidate_failed', [False]), ('small_failed', [False])):
      with self.subTest(mode=mode), ExitStack() as stack:
        params = ParamsStub(NavigationModelDrivingEnabled=mode != 'off', NavigationModelCandidate='big-nav-v2', ChestnutActive=True)
        permit = SimpleNamespace(model_id='big-nav-v2')
        small = SimpleNamespace(numpy_inputs={}, chestnut=False, run=Mock(), warmup=Mock())
        big = SimpleNamespace(numpy_inputs={}, chestnut=True, warmup=Mock(), run=Mock())
        candidate = SimpleNamespace(numpy_inputs={'navigation': np.zeros(47)}, chestnut=True, warmup=Mock(), run=Mock())
        if mode == 'candidate_failed':
          candidate.warmup.side_effect = RuntimeError('candidate initialization failed')

        def construct_model(*args, big=big, small=small, mode=mode, **kwargs):
          if kwargs['chestnut']:
            return big
          if mode == 'small_failed':
            raise RuntimeError('small initialization failed')
          return small

        models = stack.enter_context(patch.object(modeld, 'ModelState', side_effect=construct_model))
        candidate_loader = stack.enter_context(patch.object(modeld, 'NavigationModelState', return_value=candidate))
        approval = stack.enter_context(patch.object(modeld, 'authorize_runtime', return_value=None if mode == 'off' else permit))
        if mode == 'blocked':
          approval.side_effect = model_inputs.NavigationModelDisabled('branch has not approved activation')
        camera = SimpleNamespace(width=1344, height=760, buffer_len=1622016, connect=lambda _: True)
        clients = Mock(return_value=camera)
        clients.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]
        device = physical_amd_fixture()
        for name, value in {'Params': Mock(return_value=params), 'setproctitle': Mock(), 'get_active_bundle': Mock(),
                            '_find_driving_pkl': Mock(return_value='selected-ordinary-big.sptg'), 'chestnut_present': Mock(return_value=True),
                            'wait_for_chestnut': Mock(), 'VisionIpcClient': clients, 'Device': {'AMD': device},
                            'validate_runtime_permit': Mock(), '_reject_navigation_publication': Mock(),
                            'config_realtime_process': Mock(side_effect=InitializationFinished), 'PubMaster': Mock()}.items():
          stack.enter_context(patch.object(modeld, name, value))
        stack.enter_context(patch.object(modeld.gc, 'disable'))
        if mode == 'small_failed':
          with self.assertRaisesRegex(RuntimeError, 'small initialization failed'):
            modeld.main()
        else:
          with self.assertRaises(InitializationFinished):
            modeld.main()
        self.assertEqual([call.kwargs['chestnut'] for call in models.call_args_list], expected_loads)
        self.assertEqual(candidate_loader.call_count, int(mode in ('approved', 'candidate_failed', 'small_failed')))
        if mode in ('approved', 'candidate_failed', 'small_failed'):
          candidate.warmup.assert_called_once()
          big.warmup.assert_not_called()
        else:
          big.warmup.assert_called_once()
        small.warmup.assert_not_called()
        small.run.assert_not_called()
        candidate.run.assert_not_called()
        if mode == 'approved':
          self.assertIs(params.get('ChestnutActive'), True)
          self.assertEqual(params.get('NavigationModelRuntimeState')['active_source'], 'chestnut')
        elif mode == 'candidate_failed':
          self.assertEqual(params.get('NavigationModelRuntimeState')['active_source'], 'qcom')
          self.assertTrue(params.get_bool('OnroadCycleRequested'), 'Preserve existing SunnyPilot initialization recovery')
        modeld.PubMaster.assert_not_called()

  def test_driving_reader_rejects_missing_unknown_or_malformed_upcoming_guidance(self):
    with reviewed_fixture() as (params, permit, _), patch.object(model_inputs, 'PC', False):
      params.values.update(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
      cases = [([], 'missing_maneuver_metadata'),
               ([SimpleNamespace(distance=10., type='unknown', modifier='left')], 'unsupported_upcoming_maneuver'),
               ([SimpleNamespace(distance=float('nan'), type='turn', modifier='left')], 'invalid_maneuver_metadata')]
      for maneuvers, reason in cases:
        sm = Messages(100.)
        sm.data['navInstruction'].allManeuvers = maneuvers
        reader = model_inputs.NavigationInputs(params, Client(99_100_000_000), clock=lambda: 100., driving_permit=permit)
        with self.subTest(reason=reason):
          self.assertEqual(reader.read(sm, 0., 100_000_000_000), {})
          self.assertEqual(reader.input_reason, reason)
          self.assertIsNone(reader._map)
      for kind in ('depart', 'arrive', 'merge'):
        sm = Messages(100.)
        sm.data['navInstruction'].allManeuvers = [SimpleNamespace(distance=10., type=kind, modifier='straight')]
        reader = model_inputs.NavigationInputs(params, Client(99_100_000_000), clock=lambda: 100., driving_permit=permit)
        inputs = reader.read(sm, 0., 100_000_000_000)
        self.assertTrue(inputs)
        self.assertFalse(inputs['navigation'].any())

  def test_actual_modelstate_approved_loading_guard_and_upstream_action_profile(self):
    jits, metadata = fixture_data()
    with reviewed_fixture(jits['input_specs'], metadata) as (_, permit, _), patch.object(modeld, 'load_model', return_value=jits) as load:
      state = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
      self.assertIs(load.call_args.kwargs['cache'], False, 'Mutable boot-cache JITs are outside the reviewed artifact proof')
      self.assertIsNone(state.generation)
      self.assertEqual((state.LAT_SMOOTH_SECONDS, state.LONG_SMOOTH_SECONDS), (0., .3))
      modeld._reject_navigation_publication(state)
      previous = modeld.log.ModelDataV2.Action(desiredCurvature=.7, desiredAcceleration=0.)
      output = {'action': np.array([[4., 1.]], np.float32)}
      low = state.get_action_from_model(output, previous, .2, .4, .2)
      high = state.get_action_from_model(output, previous, .2, .4, 10.)
      self.assertAlmostEqual(low.desiredCurvature, .7)
      self.assertAlmostEqual(high.desiredCurvature, .04)
      self.assertGreater(high.desiredAcceleration, 0.)
      self.assertLess(high.desiredAcceleration, 1.)
      state.diagnostic = True
      with self.assertRaises(model_inputs.NavigationModelDisabled):
        modeld._reject_navigation_publication(state)

  def test_ordinary_native_model_retains_existing_boot_cache_behavior(self):
    jits, _ = fixture_data()
    jits['input_specs'] = {name: spec for name, spec in jits['input_specs'].items() if name not in ('nav_map', 'navigation')}
    with patch.object(modeld, 'get_active_bundle', return_value=None), \
         patch.object(modeld, '_find_driving_pkl', return_value='ordinary-native.sptg'), \
         patch.object(modeld, 'load_model', return_value=jits) as load:
      state = modeld.ModelState(1344, 760, chestnut=False)
      self.assertNotIn('cache', load.call_args.kwargs)
      self.assertFalse(modeld.takes_nav(state))

  def test_wrong_camera_extent_rejected_before_any_artifact_loading(self):
    with patch.object(modeld.ModelState, '__init__') as initialize:
      with self.assertRaises(model_inputs.NavigationModelDisabled):
        modeld.NavigationModelState(1344, 760)
      initialize.assert_not_called()
    with reviewed_fixture() as (_, permit, _), patch.object(modeld, 'load_model') as load:
      with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'actual camera'):
        modeld.ModelState(1928, 1208, chestnut=True, driving_permit=permit)
      load.assert_not_called()

  def test_registered_camera_geometry_rejects_equal_extent_before_artifact_loading(self):
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    profile = {'width': 1344, 'height': 760, 'stride': 1408, 'y_height': 768, 'uv_height': 384, 'frame_size': 1622016}
    for width, height in ((1342, 760), (1344, 758)):
      stride, y_height, uv_height, _ = get_nv12_info(width, height)
      # NV12 alignment can hide different image geometries behind the same
      # copied byte extent. An approved warp must match the actual geometry.
      self.assertEqual(stride * (y_height + uv_height), profile['frame_size'])
      with self.subTest(width=width, height=height), reviewed_fixture(metadata={'camera_profile': profile}) as (_, permit, _), \
           patch.object(modeld.ModelState, '_init_combined') as initialize, patch.object(modeld, 'load_model') as load:
        with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'warp geometry'):
          modeld.ModelState(width, height, chestnut=True, driving_permit=permit)
        initialize.assert_not_called()
        load.assert_not_called()

  def test_equivalent_negative_output_slices_are_admitted_but_changed_actions_are_not(self):
    jits, metadata = fixture_data()
    slices = {key: slice(value['start'], value['stop'], value['step']) for key, value in metadata['output_slices'].items()}
    slices['pad'] = slice(-2, None)
    jits['metadata']['metadata']['output_slices'] = codecs.encode(pickle.dumps(slices), 'base64').decode()
    with reviewed_fixture(jits['input_specs'], metadata) as (_, permit, _):
      activation.validate_runtime_metadata(permit, jits)
      slices['action'] = slice(2063, 2067)
      jits['metadata']['metadata']['output_slices'] = codecs.encode(pickle.dumps(slices), 'base64').decode()
      with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'action slices'):
        activation.validate_runtime_metadata(permit, jits)

  def test_real_reader_neutral_navigation_and_reroute_preserve_native_recurrent_state(self):
    jits, metadata = fixture_data()
    count = [0]

    def trusted_jit(host, output_buffers, **queues):
      count[0] += 1
      for name, tensor in queues.items():
        dtype = np.uint8 if name == 'state_img_q' else np.float32
        data = np.frombuffer(tensor._buffer().as_memoryview(allow_zero_copy=True), dtype=dtype)
        data[:] += 1

    jits['run'] = trusted_jit
    now = [100.]
    client = Client(int(now[0]*1e9)-900_000_000)
    with reviewed_fixture(jits['input_specs'], metadata) as (params, permit, _), \
         patch.object(modeld, 'load_model', return_value=jits), \
         patch.object(modeld, 'Device', {'AMD': SimpleNamespace(error_state=SimpleNamespace(host=SimpleNamespace(view=lambda **kwargs: [0])))}), \
         patch.object(model_inputs, 'PC', False):
      params.values.update(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
      state = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
      reader = model_inputs.NavigationInputs(params, client, clock=lambda: now[0], driving_permit=permit)
      frames = {'img': bytes(1622016), 'big_img': bytes(1622016)}
      transforms = {'img': np.eye(3, dtype=np.float32), 'big_img': np.eye(3, dtype=np.float32)}
      observed = []
      for index in range(4):
        now[0] = 100. + index*.05
        sm = Messages(now[0])
        generation = 1 if index == 0 else 2
        sm.data['navInstruction'].routeGeneration = generation
        sm.data['navModelStateSP'].routeGeneration = generation
        sm.data['navModelStateSP'].frameId = client.frame_id = index
        client.timestamp_sof = sm.data['navModelStateSP'].locationMonoTime
        client.timestamp_eof = sm.data['navModelStateSP'].renderMonoTime
        client.frame = SimpleNamespace(data=bytes([128])*(256*256), width=256, height=256)
        if index == 2:
          params.remove('NavDestination')
        elif index == 3:
          params.put_bool('NavigationEnabled', False)
        navigation = reader.read_for_model(sm, 10., int(now[0]*1e9))
        observed.append(navigation)
        state.run(frames, transforms, {'desire': np.zeros(8, np.float32), 'action_t': np.array([.2, .4], np.float32), **navigation})
        self.assertTrue(all(np.all(tensor.numpy() == index+1) for tensor in state.adapter.input_queues.values()))
      self.assertEqual(count[0], 4)
      self.assertEqual(observed[0]['navigation'][15], 1.)
      self.assertTrue(observed[1]['nav_map'].any(), 'A route identity change does not impose a model activation delay')
      self.assertTrue(all(not item['nav_map'].any() and not item['navigation'].any() for item in observed[2:]))

  def test_actual_device_identity_gate_before_candidate_loading(self):
    specs = {'navigation': ((1, 47), 'float32', 'NPY')}
    with reviewed_fixture(specs) as (params, permit, _):
      candidate = SimpleNamespace(numpy_inputs={'navigation': np.zeros(47)}, navigation_input_specs=specs,
                                   _driving_permit=permit, diagnostic=False, warmup=Mock())
      good = physical_amd_fixture()
      for present, device in ((False, good), (True, physical_amd_fixture(arch='gfx1100')),
                              (True, physical_amd_fixture(is_usb=False)),
                              (True, physical_amd_fixture(iface=object.__new__(type('MOCKUSBIface', (USBIface,), {})))),
                              (True, physical_amd_fixture(iface=type('USBIface', (), {})())),
                              (True, SimpleNamespace(arch='gfx1200', is_usb=True, iface=good.iface))):
        with patch.object(modeld, 'chestnut_present', return_value=present), patch.object(modeld, 'Device', {'AMD': device}), \
             patch.object(modeld, 'NavigationModelState') as loader:
          with self.assertRaises(model_inputs.NavigationModelDisabled):
            modeld.load_navigation_model(params, 1344, 760, permit)
          loader.assert_not_called()
      with patch.object(modeld, 'chestnut_present', return_value=True), patch.object(modeld, 'Device', {'AMD': good}), \
           patch.object(modeld, 'NavigationModelState', return_value=candidate):
        self.assertIs(modeld.load_navigation_model(params, 1344, 760, permit), candidate)
        candidate.warmup.assert_not_called()

  def test_pinned_automatic_target_hints_are_not_the_compilers_explicit_target(self):
    with reviewed_fixture() as (params, permit, _), patch.object(modeld, 'chestnut_present', return_value=True), \
         patch.object(modeld, 'Device', {'AMD': physical_amd_fixture()}), \
         patch.object(modeld, 'NavigationModelState') as load:
      # Source renderers need not be initialized to run an admitted precompiled
      # LLVM artifact. Actual USB/architecture identity is checked separately.
      with patch.object(AMDDevice, '_select_renderer', side_effect=AssertionError('Do not initialize an unused source renderer')):
        for configured in ('', 'AMD', 'AMD:LLVM', 'USB+AMD', 'USB+AMD:LLVM', 'USB+AMD:LLVM:gfx1200'):
          with self.subTest(configured=configured), Context(DEV=configured):
            target = DEV.target('AMD', arch='gfx1200')
            if configured in ('', 'AMD'):
              self.assertEqual((target.renderer, target.interface), ('', ''))
            modeld.load_navigation_model(params, 1344, 760, permit)
        self.assertEqual(load.call_count, 6)
        for configured in ('KFD+AMD:LLVM', 'MOCKUSB+AMD:LLVM', 'USB+AMD:HIP', 'USB+AMD:LLVM:gfx1100'):
          with self.subTest(configured=configured), Context(DEV=configured):
            load.reset_mock()
            with self.assertRaisesRegex(model_inputs.NavigationModelDisabled, 'configured='):
              modeld.load_navigation_model(params, 1344, 760, permit)
            load.assert_not_called()


if __name__ == '__main__':
  unittest.main()
