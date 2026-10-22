"""Actual ModelState/adapter paths with synthetic trusted JITs, never a GPU or CAN."""
import codecs
from contextlib import ExitStack
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.selfdrive.navigation import activation, model_inputs
from openpilot.selfdrive.navigation.driving import NavigationDrivingSession
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


class TestDrivingNavigation(unittest.TestCase):
  def test_startup_admits_candidate_before_ordinary_big_allocation_and_preserves_off_or_failed_startup(self):
    """Exercise main initialization only; no publishers, controllers or GPU are started."""
    class InitializationFinished(Exception):
      pass

    for mode, expected_loads in (('approved', [False]), ('off', [True, False]), ('blocked', [True, False]),
                                 ('candidate_failed', [False, True, False]), ('small_failed', [False, True, False])):
      with self.subTest(mode=mode), ExitStack() as stack:
        params = ParamsStub(NavigationModelDrivingEnabled=mode != 'off', NavigationModelCandidate='big-nav-v2', ChestnutActive=True)
        permit = SimpleNamespace(model_id='big-nav-v2')
        small = SimpleNamespace(numpy_inputs={}, chestnut=False)
        big = SimpleNamespace(numpy_inputs={}, chestnut=True, warmup=Mock())
        candidate = SimpleNamespace(numpy_inputs={'navigation': np.zeros(47)}, warmup=Mock())
        if mode == 'candidate_failed':
          candidate.warmup.side_effect = RuntimeError('candidate initialization failed')
        small_loads = 0

        def construct_model(*args, big=big, small=small, mode=mode, **kwargs):
          nonlocal small_loads
          if kwargs['chestnut']:
            return big
          small_loads += 1
          if mode == 'small_failed' and small_loads == 1:
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
        device = SimpleNamespace(arch='gfx1200', is_usb=True, iface=type('USBIface', (), {})(),
                                 renderer=SimpleNamespace(target=SimpleNamespace(device='AMD', renderer='LLVM', arch='gfx1200', interface='USB')))
        for name, value in {'Params': Mock(return_value=params), 'setproctitle': Mock(), 'get_active_bundle': Mock(),
                            '_find_driving_pkl': Mock(return_value='selected-ordinary-big.sptg'), 'chestnut_present': Mock(return_value=True),
                            'wait_for_chestnut': Mock(), 'VisionIpcClient': clients, 'Device': {'AMD': device},
                            'validate_runtime_permit': Mock(), '_reject_navigation_publication': Mock(),
                            'config_realtime_process': Mock(side_effect=InitializationFinished), 'PubMaster': Mock()}.items():
          stack.enter_context(patch.object(modeld, name, value))
        stack.enter_context(patch.object(modeld.gc, 'disable'))
        with self.assertRaises(InitializationFinished):
          modeld.main()
        self.assertEqual([call.kwargs['chestnut'] for call in models.call_args_list], expected_loads)
        self.assertEqual(candidate_loader.call_count, int(mode in ('approved', 'candidate_failed')))
        if mode == 'approved':
          candidate.warmup.assert_called_once()
          big.warmup.assert_not_called()
          self.assertIsNone(params.get('ChestnutActive'), 'initial small-model priming is intentional fallback, not active/failed Chestnut')
        else:
          big.warmup.assert_called_once()
          self.assertIs(params.get('ChestnutActive'), True)
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

  def test_real_reader_and_native_queues_over_full_histories_and_reroute(self):
    jits, metadata = fixture_data()
    count = [0]

    def trusted_jit(host, output_buffers, **queues):
      count[0] += 1
      for name, tensor in queues.items():
        dtype = np.uint8 if name == 'state_img_q' else np.float32
        data = np.frombuffer(tensor._buffer().as_memoryview(allow_zero_copy=True), dtype=dtype).reshape(tensor.shape)
        if name == 'state_img_q':
          data[:, :-1] = data[:, 1:]
          data[:, -1] = count[0] % 250
        else:
          data[:-1] = data[1:]
          data[-1] = count[0]

    jits['run'] = trusted_jit
    now = [100.]
    client = Client(int(now[0]*1e9)-900_000_000)
    fallback = SimpleNamespace(desire_key='desire_pulse', LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.2, run=Mock(return_value={}))
    with reviewed_fixture(jits['input_specs'], metadata) as (params, permit, _), \
         patch.object(modeld, 'load_model', return_value=jits), \
         patch.object(modeld, 'Device', {'AMD': SimpleNamespace(error_state=SimpleNamespace(host=SimpleNamespace(view=lambda **kwargs: [0])))}), \
         patch.object(model_inputs, 'PC', False):
      params.values.update(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
      state = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
      reader = model_inputs.NavigationInputs(params, client, clock=lambda: now[0], driving_permit=permit)
      session = NavigationDrivingSession(permit, state, fallback, params, reader)
      frames = {'img': bytes(1622016), 'big_img': bytes(1622016)}
      transforms = {'img': np.eye(3, dtype=np.float32), 'big_img': np.eye(3, dtype=np.float32)}
      inputs = {'desire_pulse': np.zeros(8, np.float32), 'action_t': np.array([.2, .4], np.float32)}

      def tick(index, generation):
        now[0] = 100. + index*.05
        sm = Messages(now[0])
        sm.data['navInstruction'].routeGeneration = generation
        sm.data['navModelStateSP'].routeGeneration = generation
        sm.data['navModelStateSP'].frameId = client.frame_id = index
        client.timestamp_sof = sm.data['navModelStateSP'].locationMonoTime
        client.timestamp_eof = sm.data['navModelStateSP'].renderMonoTime
        client.frame = SimpleNamespace(data=bytes([128])*(256*256), width=256, height=256)
        return session.prepare(sm, 10., int(now[0]*1e9))

      for index in range(140):
        selected = tick(index, 1)
        if selected is state:
          session.run_ordinary(frames, transforms, inputs, 'desire_pulse', 10., .1, .2, .05, .025)
          state.run(frames, transforms, {'desire': np.zeros(8, np.float32), 'action_t': inputs['action_t'], **session.inputs})
        else:
          self.assertIs(selected, fallback)
          fallback.run(frames, transforms, inputs)
          session.shadow(frames, transforms, inputs, 'desire_pulse', 10., .1, .2, .05, .025)
        self.assertEqual(fallback.run.call_count, index+1)
      self.assertEqual(count[0], 140)
      self.assertEqual(session.valid_frames, 132)
      self.assertEqual(state.adapter.input_queues['state_navigation_features_q'].numpy()[0, 0, 0], 13)
      self.assertEqual(state.adapter.input_queues['state_desire_q'].numpy()[0, 0, 0], 9)
      self.assertIs(tick(140, 2), fallback)
      self.assertEqual(session.valid_frames, 0)
      self.assertTrue(all(not tensor.numpy().any() for tensor in state.adapter.input_queues.values()))
      for index in range(140, 280):
        selected = tick(index+1, 2)
        if selected is state:
          state.run(frames, transforms, {'desire': np.zeros(8, np.float32), 'action_t': inputs['action_t'], **session.inputs})
        else:
          session.shadow(frames, transforms, inputs, 'desire_pulse', 10., .1, .2, .05, .025)
      self.assertTrue((state.adapter.input_queues['state_navigation_features_q'].numpy() >= 141).all())
      params.values['NavigationModelDrivingEnabled'] = False
      self.assertIs(tick(281, 2), fallback)
      self.assertEqual(session.valid_frames, 0)
      self.assertTrue(all(not tensor.numpy().any() for tensor in state.adapter.input_queues.values()))

  def test_failed_or_unavailable_candidate_keeps_ordinary_and_never_opens_gpu_when_off(self):
    with patch.object(modeld, 'authorize_runtime', return_value=None), patch.object(modeld, 'NavigationModelState') as load:
      self.assertIsNone(modeld.load_navigation_session(object(), 1344, 760, object(), object()))
      load.assert_not_called()
    with patch.object(modeld, 'authorize_runtime', side_effect=model_inputs.NavigationModelDisabled('not reviewed')):
      params = ParamsStub(NavigationModelDrivingEnabled=True)
      self.assertIsNone(modeld.load_navigation_session(params, 1344, 760, object(), object()))
      self.assertEqual(params.values['NavigationModelRuntimeState']['role'], 'blocked')
      self.assertIsNone(params.values['NavigationModelRuntimeState']['active_source'])

  def test_handover_updates_model_specific_timing_without_false_gpu_failure(self):
    params = ParamsStub()
    CP = SimpleNamespace(longitudinalActuatorDelay=.2)
    state = SimpleNamespace(big=False)
    candidate = SimpleNamespace(LONG_SMOOTH_SECONDS=.3, chestnut=True, lat_delay=.1)
    small = SimpleNamespace(LONG_SMOOTH_SECONDS=.05, chestnut=False, lat_delay=.1)
    self.assertAlmostEqual(modeld.navigation_handover(candidate, CP, params, {}, object(), state), .5)
    self.assertTrue(params.get_bool('ChestnutActive'))
    self.assertTrue(state.big)
    self.assertAlmostEqual(modeld.navigation_handover(small, CP, params, {}, object(), state), .25)
    self.assertIsNone(params.get('ChestnutActive'), 'an intentional small fallback must not assert GPU failure')
    self.assertFalse(state.big)
    modeld.navigation_handover(small, CP, params, {}, object(), state, failed=True)
    self.assertIs(params.get('ChestnutActive'), False)

  def test_actual_device_and_firmware_identity_gate_before_candidate_loading(self):
    specs = {'navigation': ((1, 47), 'float32', 'NPY')}
    with reviewed_fixture(specs) as (params, permit, _):
      ordinary = SimpleNamespace(numpy_inputs={})
      candidate = SimpleNamespace(numpy_inputs={'navigation': np.zeros(47)}, navigation_input_specs=specs,
                                   _driving_permit=permit, diagnostic=False, warmup=Mock())
      usb_iface = type('USBIface', (), {})()
      good = SimpleNamespace(arch='gfx1200', is_usb=True, iface=usb_iface,
                             renderer=SimpleNamespace(target=SimpleNamespace(device='AMD', renderer='LLVM', arch='gfx1200', interface='USB')))
      for present, device in ((False, good), (True, SimpleNamespace(arch='gfx1100', renderer=good.renderer))):
        with patch.object(modeld, 'chestnut_present', return_value=present), patch.object(modeld, 'Device', {'AMD': device}), \
             patch.object(modeld, 'NavigationModelState') as loader:
          self.assertIsNone(modeld.load_navigation_session(params, 1344, 760, ordinary, ordinary))
          loader.assert_not_called()
      with patch.object(modeld, 'chestnut_present', return_value=True), patch.object(modeld, 'Device', {'AMD': good}), \
           patch.object(modeld, 'NavigationModelState', return_value=candidate):
        session = modeld.load_navigation_session(params, 1344, 760, ordinary, ordinary)
        assert session is not None
        self.assertIs(session.ordinary, ordinary)

  def test_continuous_small_inputs_and_failure_policy_and_runtime_identity(self):
    params = ParamsStub()
    candidate = SimpleNamespace(reset_state=Mock())
    small = SimpleNamespace(desire_key='desire_pulse', LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.2,
                             run=Mock(return_value={'action': np.ones((1, 2))}))
    reader = SimpleNamespace(reset_required=False, reason='valid', read=Mock(return_value={'navigation': np.zeros(47)}))
    permit = SimpleNamespace(model_id='big-nav-v2', minimum_valid_frames=132)
    session = NavigationDrivingSession(permit, candidate, small, params, reader)
    session.valid_frames = 132
    inputs = {'desire': np.ones(8, np.float32), 'action_t': np.array([.2, .4]), 'navigation': np.ones(47), 'nav_map': np.ones((256, 256))}
    with patch('openpilot.selfdrive.navigation.driving.validate_runtime_permit'):
      self.assertIs(session.prepare({}, 10., 100), candidate)
      session.run_ordinary({}, {}, inputs, 'desire', 10., .1, .2, .05, .025)
      copied = small.run.call_args.args[2]
      self.assertFalse({'nav_map', 'navigation', 'desire'} & copied.keys())
      np.testing.assert_allclose(copied['action_t'], [.275, .475])
      np.testing.assert_array_equal(inputs['action_t'], [.2, .4])
      with patch.object(params, 'put', wraps=params.put) as write:
        session.publish_runtime_state(candidate)
        session.publish_runtime_state(candidate)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(params.get('NavigationModelRuntimeState')['active_source'], 'chestnut')
      small.run.side_effect = RuntimeError('small fallback failed')
      with self.assertRaisesRegex(RuntimeError, 'small fallback failed'):
        session.run_ordinary({}, {}, inputs, 'desire', 10., .1, .2, .05, .025)
      self.assertTrue(session.failed)
      self.assertIs(session.prepare({}, 10., 150), small)
      session.publish_runtime_state(small)
      self.assertEqual(params.get('NavigationModelRuntimeState')['role'], 'failed')
      self.assertEqual(params.get('NavigationModelRuntimeState')['active_source'], 'qcom')
