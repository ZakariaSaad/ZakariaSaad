"""Exercise modeld's actual publication loop with trusted synthetic CPU/NPY data.

No VisionIPC, GPU, controllers or vehicle interfaces are opened by these tests.
This verifies software wiring and handovers, not inference performance or driving safety.
"""
from contextlib import ExitStack
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.selfdrive.navigation.driving import NavigationDrivingSession
from openpilot.selfdrive.navigation.tests.test_activation import reviewed_fixture
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.constants import Meta, ModelConstants
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.tests.test_driving_navigation import fixture_data


class EndOfFrames(Exception):
  pass


class TestNavigationPublication(unittest.TestCase):
  def test_actual_main_publishes_native_predictions_and_hands_back_to_warm_small_model(self):
    for transition in ('navigation_off', 'missing_navigation', 'candidate_failure'):
      with self.subTest(transition=transition):
        self._exercise_transition(transition)

  def _exercise_transition(self, transition):
    with ExitStack() as stack:
      jits, metadata = fixture_data()
      params, permit, _ = stack.enter_context(reviewed_fixture(jits['input_specs'], metadata))
      params.values.update(NavigationEnabled=True, NavDestination={'latitude': 1., 'longitude': 2.})
      stack.enter_context(patch.object(modeld, 'load_model', return_value=jits))
      error_state = SimpleNamespace(host=SimpleNamespace(view=lambda **_: [0]))
      stack.enter_context(patch.object(modeld, 'Device', {'AMD': SimpleNamespace(error_state=error_state)}))
      candidate = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
      captured_inputs, candidate_calls, small_calls = [], [], []

      def native_graph(host, output_buffers, **states):
        # Trusted fixture code writes the native primary output in-place, as
        # the real PlainJit does. The real adapter, parser and publisher run.
        output = np.frombuffer(output_buffers['outputs']._buffer().as_memoryview(allow_zero_copy=True), dtype=np.float32)
        output[:] = 0
        output[2062:2066] = [4., -1., 0., 0.]

      jits['run'].side_effect = native_graph
      real_candidate_run = candidate.run

      def candidate_run(bufs, transforms, inputs, *args):
        candidate_calls.append(camera.frame_id)
        captured_inputs.append({key: value.copy() for key, value in inputs.items()})
        if transition == 'candidate_failure' and camera.frame_id == 2:
          raise RuntimeError('synthetic candidate inference failure')
        return real_candidate_run(bufs, transforms, inputs, *args)

      candidate.run = candidate_run
      slices = candidate.vision_output_slices
      raw_small = np.zeros(18580, np.float32)
      raw_small[2062:2066] = [2., .5, 0., 0.]

      def small_run(bufs, transforms, inputs, *args):
        small_calls.append((camera.frame_id, {key: value.copy() for key, value in inputs.items()}))
        return Parser().parse_outputs({key: raw_small[np.newaxis, value].copy() for key, value in slices.items()})

      small = SimpleNamespace(chestnut=False, numpy_inputs={'action_t': np.zeros(2, np.float32)},
                              constants=ModelConstants(), desire_key='desire_pulse', vision_input_names=['img', 'big_img'],
                              LAT_SMOOTH_SECONDS=.1, LONG_SMOOTH_SECONDS=.2, MIN_LAT_CONTROL_SPEED=.3,
                              PLANPLUS_CONTROL=1., generation=None, lat_delay=.1, run=small_run)
      small.get_action_from_model = MethodType(modeld.ModelState.get_action_from_model, small)
      reader = SimpleNamespace(reset_required=False, reset_reason=None, reason='valid', input_reason='valid')

      def read_inputs(*_):
        if transition == 'missing_navigation' and camera.frame_id >= 2:
          reader.input_reason = 'missing_gps'
          return {}
        nav = np.zeros(47, np.float32)
        nav[15], nav[46] = 1., 2.5
        return {'navigation': nav, 'nav_map': np.full((256, 256), 128, np.float32)}

      reader.read = read_inputs
      session = NavigationDrivingSession(permit, candidate, small, params, reader)
      # Full132-frame real-reader/native-buffer priming is covered separately.
      # Start at its transition boundary to exercise the real publishing loop.
      session.valid_frames = permit.minimum_valid_frames

      class Camera:
        width, height, buffer_len = 1344, 760, 1622016
        frame_id = timestamp_sof = timestamp_eof = 0

        def connect(self, _):
          return True

        def recv(self):
          self.frame_id += 1
          if self.frame_id > 3:
            raise EndOfFrames()
          self.timestamp_sof = 100_000_000_000 + self.frame_id * 50_000_000
          self.timestamp_eof = self.timestamp_sof + 10_000_000
          return SimpleNamespace(data=bytes(self.buffer_len))

      camera = Camera()
      clients = Mock(return_value=camera)
      clients.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]

      class Messages:
        frame = 0
        updated = {'extrinsicsCalibration': True}
        seen = {'narrowRoadCameraState': True, 'deviceState': True}
        data = {'carState': SimpleNamespace(vEgo=10.), 'driverMonitoringState': SimpleNamespace(isRHD=False),
                'narrowRoadCameraState': SimpleNamespace(frameId=0, sensor='os04c10'),
                'deviceState': SimpleNamespace(deviceType='mici'), 'extrinsicsCalibration': SimpleNamespace(rpyCalib=[0., 0., 0.], height=[1.22]),
                'lateralDelay': SimpleNamespace(lateralDelay=.1), 'carControl': SimpleNamespace(latActive=True),
                'controlsState': SimpleNamespace(curvature=0.)}

        def update(self, _):
          self.frame += 1
          self.data['narrowRoadCameraState'].frameId = camera.frame_id
          if transition == 'navigation_off' and camera.frame_id == 2:
            params.put_bool('NavigationModelDrivingEnabled', False)

        def __getitem__(self, key):
          return self.data[key]

      messages = Messages()
      sent = []
      publisher = SimpleNamespace(send=lambda name, message: sent.append((name, message.to_dict())))
      DH = SimpleNamespace(desire=0, lane_change_state=0, lane_change_direction=0, lane_turn_direction=0, update=Mock())
      gpu_state = SimpleNamespace(big=False, send=Mock())
      for name, value in {'Params': Mock(return_value=params), 'setproctitle': Mock(),
                          'get_active_bundle': Mock(), '_find_driving_pkl': Mock(return_value='ordinary-big.sptg'),
                          'chestnut_present': Mock(return_value=True), 'VisionIpcClient': clients,
                          'load_navigation_first': Mock(return_value=session), 'config_realtime_process': Mock(),
                          'PubMaster': Mock(return_value=publisher), 'SubMaster': Mock(return_value=messages),
                          'ChestnutGpuState': Mock(return_value=gpu_state), 'DesireHelper': Mock(return_value=DH),
                          'RoadEdgeLaneChangeController': Mock(return_value=SimpleNamespace(update_and_fill=lambda *_: (0., 0.))),
                          'load_meta_constants': Mock(return_value=Meta), 'drive_config_for_model': Mock(return_value=object())}.items():
        stack.enter_context(patch.object(modeld, name, value))
      stack.enter_context(patch.object(modeld.gc, 'disable'))
      stack.enter_context(patch.object(modeld.messaging, 'log_from_bytes', return_value=SimpleNamespace(brand='honda', longitudinalActuatorDelay=.2)))
      with self.assertRaises(EndOfFrames):
        modeld.main()

      published = [message['modelV2'] for name, message in sent if name == 'modelV2']
      expected_ids = [1, 3] if transition == 'candidate_failure' else [1, 2, 3]
      self.assertEqual([message['frameId'] for message in published], expected_ids)
      self.assertEqual([message['big'] for message in published], [True] + [False] * (len(expected_ids)-1))
      self.assertAlmostEqual(published[0]['action']['desiredCurvature'], .04)
      self.assertLess(published[0]['action']['desiredAcceleration'], 0.)
      self.assertAlmostEqual(published[-1]['action']['desiredCurvature'], .02)
      self.assertEqual([frame for frame, _ in small_calls], [1, 2, 3])
      self.assertTrue(all(not ({'navigation', 'nav_map'} & inputs.keys()) for _, inputs in small_calls))
      self.assertEqual(captured_inputs[0]['navigation'][15], 1.)
      self.assertEqual(captured_inputs[0]['navigation'][46], 2.5)
      self.assertEqual(captured_inputs[0]['nav_map'][0, 0], 128.)
      expected_lateral_horizon = modeld.prediction_delay(10.) + .05 + .025
      np.testing.assert_allclose(captured_inputs[0]['action_t'], [expected_lateral_horizon, .2 + .3 + .05 + .025])
      self.assertEqual(session.valid_frames, 0)
      self.assertFalse(any(tensor.numpy().any() for tensor in candidate.adapter.input_queues.values()))
      self.assertEqual(params.get('NavigationModelRuntimeState')['active_source'], 'qcom')
      self.assertEqual({name for name, _ in sent}, {'modelV2', 'drivingModelData', 'cameraOdometry', 'modelDataV2SP'})


if __name__ == '__main__':
  unittest.main()
