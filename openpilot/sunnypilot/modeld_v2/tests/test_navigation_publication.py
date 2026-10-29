"""Exercise modeld's actual publication loop with trusted synthetic CPU/NPY data.

No VisionIPC, GPU, controllers or vehicle interfaces are opened by these tests.
This verifies software wiring and handovers, not inference performance or driving safety.
"""
from contextlib import ExitStack
from types import MethodType, SimpleNamespace
import unittest
import traceback

from openpilot.selfdrive.navigation.tests.official_reference import source
from unittest.mock import Mock, patch

import numpy as np

from openpilot.selfdrive.navigation.activation import validate_runtime_permit
from openpilot.selfdrive.navigation.tests.test_activation import reviewed_fixture
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.constants import Meta, ModelConstants
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.tests.test_driving_navigation import fixture_data


class EndOfFrames(Exception):
  pass


class TestNavigationPublication(unittest.TestCase):
  def test_actual_main_runs_one_selected_model_and_uses_small_only_after_failure(self):
    for transition in ('healthy', 'missing_navigation', 'route_change', 'destination_cleared', 'navigation_off',
                       'timing_change', 'eps_off', 'eps_off_timing_change', 'dropped_frames', 'selection_changed', 'legacy_flag_ignored',
                       'candidate_failure', 'candidate_nonfinite', 'small_failure', 'ordinary_selected'):
      with self.subTest(transition=transition):
        self._exercise_transition(transition)

  def _exercise_transition(self, transition):
    with ExitStack() as stack:
      presented_ids = [1, 2, 4, 5, 7, 8, 10, 11, 13, 16, 19, 22] if transition == 'dropped_frames' else [1, 2, 3]
      jits, metadata = fixture_data()
      # CPU state storage lets the normal warmup issue its real zero-fill kernels.
      # The driving graph itself remains the trusted synthetic writer below.
      for field in ('input_specs', 'output_specs'):
        jits[field] = {name: (shape, dtype, 'CPU') for name, (shape, dtype, _) in jits[field].items()}
      metadata['output_specs'] = {name: {'shape': list(shape), 'dtype': np.dtype(dtype).name, 'device': device}
                                  for name, (shape, dtype, device) in jits['output_specs'].items()}
      params, permit, _ = stack.enter_context(reviewed_fixture(jits['input_specs'], metadata))
      params.values.update(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.}, CameraOffset=.22, PlanplusControl=.2)
      stack.enter_context(patch.object(modeld, 'load_model', return_value=jits))
      error_state = SimpleNamespace(host=SimpleNamespace(view=lambda **_: [0]))
      stack.enter_context(patch.object(modeld, 'Device', {'AMD': SimpleNamespace(error_state=error_state)}))
      candidate = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
      captured_inputs, candidate_calls, small_calls, captured_transforms = [], [], [], []
      offset = modeld.CameraOffsetHelper()
      offset.update = Mock(wraps=offset.update)
      offset.set_offset = Mock(wraps=offset.set_offset)
      stack.enter_context(patch.object(modeld, 'CameraOffsetHelper', return_value=offset))
      reset_state = stack.enter_context(patch.object(candidate, 'reset_state', wraps=candidate.reset_state))

      def native_graph(host, output_buffers, **states):
        # Trusted fixture code writes the native primary output in-place, as
        # the real PlainJit does. The real adapter, parser and publisher run.
        output = np.frombuffer(output_buffers['outputs']._buffer().as_memoryview(allow_zero_copy=True), dtype=np.float32)
        output[:] = 0
        output[2062:2066] = [4., -1., 0., 0.]
        if transition == 'candidate_nonfinite' and camera.frame_id == 2:
          output[2062] = np.nan
        for name, state in states.items():
          dtype = np.uint8 if name == 'state_img_q' else np.float32
          data = np.frombuffer(state._buffer().as_memoryview(allow_zero_copy=True), dtype=dtype)
          data[:] += 1

      jits['run'].side_effect = native_graph
      real_candidate_run = candidate.run

      def candidate_run(bufs, transforms, inputs, *args):
        candidate_calls.append(camera.frame_id)
        captured_inputs.append((camera.frame_id, {key: value.copy() for key, value in inputs.items()}))
        captured_transforms.append((camera.frame_id, {key: value.copy() for key, value in transforms.items()}))
        if transition in ('candidate_failure', 'small_failure') and camera.frame_id == 2:
          raise RuntimeError('synthetic candidate inference failure')
        return real_candidate_run(bufs, transforms, inputs, *args)

      stack.enter_context(patch.object(candidate, 'run', side_effect=candidate_run))
      candidate.warmup = Mock(wraps=candidate.warmup)
      slices = candidate.vision_output_slices
      raw_small = np.zeros(18580, np.float32)
      raw_small[2062:2066] = [2., .5, 0., 0.]

      def small_run(bufs, transforms, inputs, *args):
        small_calls.append((camera.frame_id, {key: value.copy() for key, value in inputs.items()}))
        if transition == 'small_failure':
          assert params.get('NavigationModelRuntimeState')['role'] == 'small_loading'
          raise RuntimeError('synthetic ordinary small inference failure')
        return Parser().parse_outputs({key: raw_small[np.newaxis, value].copy() for key, value in slices.items()})

      small = SimpleNamespace(chestnut=False, numpy_inputs={'action_t': np.zeros(2, np.float32)},
                              constants=ModelConstants(), desire_key='desire_pulse', vision_input_names=['img', 'big_img'],
                              LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.2, MIN_LAT_CONTROL_SPEED=.3,
                              PLANPLUS_CONTROL=1., generation=None, lat_delay=.1, run=small_run)
      small.get_action_from_model = MethodType(modeld.ModelState.get_action_from_model, small)
      def read_inputs(*_):
        validate_runtime_permit(permit, params=params)
        neutral = transition in ('missing_navigation', 'destination_cleared', 'navigation_off') and camera.frame_id >= 2
        nav = np.zeros(47, np.float32)
        if not neutral:
          direction = 10 if transition == 'route_change' and camera.frame_id >= 2 else 15
          nav[direction], nav[46] = 1., 2.5
        return {'navigation': nav, 'nav_map': np.full((256, 256), 0 if neutral else 128, np.float32)}

      reader = SimpleNamespace(read_for_model=Mock(side_effect=read_inputs))
      if transition == 'ordinary_selected':
        params.put('ModelManager_ActiveBundleChestnut', {'ref': 'ordinary'})

      class Camera:
        width, height, buffer_len = 1344, 760, 1622016
        frame_id = timestamp_sof = timestamp_eof = frame_index = 0

        def connect(self, _):
          return True

        def recv(self):
          self.frame_index += 1
          if self.frame_index > len(presented_ids):
            raise EndOfFrames()
          self.frame_id = presented_ids[self.frame_index - 1]
          self.timestamp_sof = 100_000_000_000 + self.frame_id * 50_000_000
          self.timestamp_eof = self.timestamp_sof + 10_000_000
          return SimpleNamespace(data=bytes(self.buffer_len))

      camera = Camera()
      load_errors = []
      stack.enter_context(patch.object(modeld.cloudlog, 'exception',
                                       side_effect=lambda *args, **kwargs: load_errors.append((args, traceback.format_exc()))))
      numerical_events = stack.enter_context(patch.object(modeld.cloudlog, 'event'))
      clients = Mock(return_value=camera)
      clients.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]

      class Messages:
        frame = 0
        updated = {'extrinsicsCalibration': True}
        seen = {'narrowRoadCameraState': True, 'deviceState': True}
        data = {'carState': SimpleNamespace(vEgo=10., leftBlinker=False, rightBlinker=False,
                                                steeringPressed=False, steeringTorque=0., leftBlindspot=False, rightBlindspot=False),
                'driverMonitoringState': SimpleNamespace(isRHD=False),
                'narrowRoadCameraState': SimpleNamespace(frameId=0, sensor='os04c10'),
                'deviceState': SimpleNamespace(deviceType='mici'), 'extrinsicsCalibration': SimpleNamespace(rpyCalib=[0., 0., 0.], height=[1.22]),
                'lateralDelay': SimpleNamespace(lateralDelay=.1), 'carControl': SimpleNamespace(latActive=True),
                'controlsState': SimpleNamespace(curvature=0.)}

        def update(self, _):
          self.frame += 1
          self.data['narrowRoadCameraState'].frameId = camera.frame_id
          self.updated['extrinsicsCalibration'] = camera.frame_id == 1
          if transition in ('timing_change', 'eps_off_timing_change'):
            self.data['lateralDelay'].lateralDelay = .1 + camera.frame_id * .01
          if camera.frame_id == 2:
            if transition == 'selection_changed':
              # Simulate corrupted/on-road state outside the guarded UI: reject it.
              params.put('ModelManager_ActiveBundleChestnut', {'ref': 'ordinary'})
            elif transition == 'legacy_flag_ignored':
              params.put_bool('NavigationModelDrivingEnabled', False)
            elif transition == 'navigation_off':
              params.put_bool('NavigationEnabled', False)
            elif transition == 'destination_cleared':
              params.remove('NavDestination')

        def __getitem__(self, key):
          return self.data[key]

      messages = Messages()
      sent = []
      publisher = SimpleNamespace(send=lambda name, message: sent.append((name, message.to_dict())))
      DH = SimpleNamespace(desire=0, lane_change_state=0, lane_change_direction=0, lane_turn_direction=0, update=Mock())
      gpu_state = SimpleNamespace(big=False, send=Mock())
      for name, value in {'Params': Mock(return_value=params), 'setproctitle': Mock(),
                          'get_active_bundle': Mock(),
                          '_find_driving_pkl': Mock(return_value=None if transition == 'ordinary_selected' else 'ordinary-big.sptg'),
                          'chestnut_present': Mock(return_value=True), 'VisionIpcClient': clients,
                          'authorize_runtime': Mock(return_value=None if transition == 'ordinary_selected' else permit),
                          'load_navigation_model': Mock(return_value=candidate), 'ModelState': Mock(return_value=small),
                          'NavigationInputs': Mock(return_value=reader), 'wait_for_chestnut': Mock(),
                          'config_realtime_process': Mock(),
                          'PubMaster': Mock(return_value=publisher), 'SubMaster': Mock(return_value=messages),
                          'ChestnutGpuState': Mock(return_value=gpu_state), 'DesireHelper': Mock(return_value=DH),
                          'RoadEdgeLaneChangeController': Mock(return_value=SimpleNamespace(update_and_fill=lambda *_: (0., 0.))),
                          'load_meta_constants': Mock(return_value=Meta),
                          'drive_config_for_model': Mock(return_value=None if transition.startswith('eps_off') else object())}.items():
        stack.enter_context(patch.object(modeld, name, value))
      stack.enter_context(patch.object(modeld.gc, 'disable'))
      stack.enter_context(patch.object(modeld.messaging, 'log_from_bytes', return_value=SimpleNamespace(brand='honda', longitudinalActuatorDelay=.2)))
      if transition == 'small_failure':
        with self.assertRaisesRegex(RuntimeError, 'ordinary small inference failure'):
          modeld.main()
      else:
        with self.assertRaises(EndOfFrames):
          modeld.main()

      initialization_errors = [error for error in load_errors if 'chestnut load failed' in error[0][0]]
      self.assertEqual(initialization_errors, [], initialization_errors)
      published = [message['modelV2'] for name, message in sent if name == 'modelV2']
      failure = transition in ('candidate_failure', 'candidate_nonfinite', 'selection_changed', 'small_failure')
      small_failed = transition == 'small_failure'
      disabled = transition == 'ordinary_selected'
      expected_ids = [1] if small_failed else [1, 3] if failure else presented_ids
      self.assertEqual([message['frameId'] for message in published], expected_ids)
      expected_big = [False] * 3 if disabled else [True] if small_failed else [True, False] if failure else [True] * len(presented_ids)
      self.assertEqual([message['big'] for message in published], expected_big)
      frame_candidate_calls = [frame for frame in candidate_calls if frame > 0]
      expected_candidate_calls = [] if disabled else [1] if transition == 'selection_changed' else [1, 2] if failure else presented_ids
      self.assertEqual(frame_candidate_calls, expected_candidate_calls)
      self.assertEqual([frame for frame, _ in small_calls], [1, 2, 3] if disabled else [3] if failure else [])
      # Normal selected-model warmup is separate from production frame execution.
      self.assertEqual(candidate_calls.count(0), int(not disabled))
      self.assertEqual(candidate.warmup.call_count, int(not disabled))
      reset_state.assert_not_called()
      self.assertTrue(all(not ({'navigation', 'nav_map'} & inputs.keys()) for _, inputs in small_calls))
      if disabled:
        modeld.load_navigation_model.assert_not_called()
        modeld.NavigationInputs.assert_not_called()
        self.assertEqual(candidate_calls, [], 'Disabled mode must not execute Bigv2, even in shadow')
        self.assertAlmostEqual(published[0]['action']['desiredCurvature'], .02)
      else:
        self.assertAlmostEqual(published[0]['action']['desiredCurvature'], .04)
        self.assertLess(published[0]['action']['desiredAcceleration'], 0.)
        first_inputs = next(inputs for frame, inputs in captured_inputs if frame == 1)
        self.assertEqual(first_inputs['navigation'][15], 1.)
        self.assertEqual(first_inputs['navigation'][46], 2.5)
        self.assertEqual(first_inputs['nav_map'][0, 0], 128.)
        expected_lateral_horizon = (modeld.prediction_delay(10.) if not transition.startswith('eps_off') else
                                    .11 if transition == 'eps_off_timing_change' else .1) + .05 + .025
        np.testing.assert_allclose(first_inputs['action_t'], [expected_lateral_horizon, .2 + .3 + .05 + .025])
        if transition in ('timing_change', 'eps_off_timing_change'):
          for frame, inputs in captured_inputs:
            if frame > 0:
              expected_delay = .1 + frame*.01 if transition.startswith('eps_off') else modeld.prediction_delay(10.)
              np.testing.assert_allclose(inputs['action_t'][0], expected_delay + .075)
        if transition in ('missing_navigation', 'destination_cleared', 'navigation_off'):
          for frame, inputs in captured_inputs:
            if frame >= 2:
              self.assertFalse(inputs['nav_map'].any())
              self.assertFalse(inputs['navigation'].any())
          self.assertTrue(all(np.all(tensor.numpy() == 3) for tensor in candidate.adapter.input_queues.values()))
        if transition == 'route_change':
          changed_inputs = next(inputs for frame, inputs in captured_inputs if frame == 2)
          self.assertEqual(changed_inputs['navigation'][10], 1.)
          self.assertTrue(all(np.all(tensor.numpy() == 3) for tensor in candidate.adapter.input_queues.values()))
        state = params.get('NavigationModelRuntimeState')
        self.assertEqual(state['active_source'], None if small_failed else 'qcom' if failure else 'chestnut')
        self.assertEqual(state['role'], 'failed' if small_failed else 'ordinary_small' if failure else 'candidate')
      if transition == 'dropped_frames':
        # Execute the official frame-drop accounting against the actual main
        # publications. Missing frame IDs cause no invented recurrent steps.
        official = source('modeld')
        start = official.index('    vipc_dropped_frames =')
        end = official.index('    bufs =', start)
        block = '\n'.join(line[4:] for line in official[start:end].splitlines())
        context = {'meta_main': SimpleNamespace(frame_id=0), 'last_vipc_frame_id': 0, 'run_count': 0,
                   'frame_dropped_filter': modeld.FirstOrderFilter(0., 10., 1. / 20)}
        for message in published:
          context['meta_main'].frame_id = message['frameId']
          exec(compile(block, 'official_a03_frame_drop_accounting', 'exec'), context)
          self.assertAlmostEqual(message['frameDropPerc'], context['frame_drop_ratio'] * 100, places=6)
          context['last_vipc_frame_id'] = message['frameId']
        self.assertTrue(all(np.all(tensor.numpy() == len(presented_ids)) for tensor in candidate.adapter.input_queues.values()))
      if not disabled:
        camera_config = modeld.DEVICE_CAMERAS[('mici', 'os04c10')]
        expected_transforms = modeld.bigv2_warp_matrices(np.zeros(3, np.float32), camera_config, False, False)
        for frame, transforms in captured_transforms:
          if frame > 0:
            np.testing.assert_array_equal(transforms['img'], expected_transforms[0])
            np.testing.assert_array_equal(transforms['big_img'], expected_transforms[1])
        if failure:
          offset.set_offset.assert_called_once_with(.22)
          offset.update.assert_called_once()
        else:
          offset.set_offset.assert_not_called()
          offset.update.assert_not_called()
      if failure:
        if not small_failed:
          self.assertAlmostEqual(published[-1]['action']['desiredCurvature'], .02)
        np.testing.assert_allclose(small_calls[0][1]['action_t'], [modeld.prediction_delay(10.) + .1 + .05 + .025, .2 + .2 + .05 + .025])
        self.assertIs(params.get('ChestnutActive'), False)
      if transition == 'candidate_nonfinite':
        numerical_events.assert_called_once()
        event = numerical_events.call_args
        self.assertEqual(event.args[0], 'model_output_nonfinite')
        self.assertEqual(event.kwargs['details']['context']['frame_id'], 2)
        self.assertEqual(event.kwargs['details']['first_invalid_indices'], [2062])
        self.assertEqual(event.kwargs['details']['invalid_output_sections']['action']['nan'], 1)
        self.assertEqual(event.kwargs['details']['host_input_stats']['navigation']['finite'], 47)
      else:
        numerical_events.assert_not_called()
      self.assertEqual({name for name, _ in sent}, {'modelV2', 'drivingModelData', 'cameraOdometry', 'modelDataV2SP'})


if __name__ == '__main__':
  unittest.main()
