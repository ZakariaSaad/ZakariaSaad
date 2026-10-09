"""Real frame-loop and serialization; cameras/vehicle inputs are CI fixtures."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.cinque_guard import CinqueGuard
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Meta
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
from openpilot.sunnypilot.modeld_v2.startup import StartupModels
from openpilot.sunnypilot.modeld_v2.tests.test_stateful import artifact


class TestModeldLoop(unittest.TestCase):
  def exercise(self, fail_v3=None, fail_publication=False, guard_used=True, fail_recovery=False, legacy_gpu_failure=False,
               fail_serialization=False, late_fault=False):
    with tempfile.TemporaryDirectory() as root:
      params = Params(root)
      parsed = Parser().parse_outputs({k: np.zeros((1, sl.stop - sl.start), dtype='float32')
                                      for k, sl in artifact()['metadata']['output_slices'].items()})
      def model(chestnut):
        value = modeld.ModelState.__new__(modeld.ModelState)
        value.constants, value.meta_constants, value.generation = ModelConstants(), Meta, None
        value.chestnut, value.lat_delay = chestnut, .2
        value.LAT_SMOOTH_SECONDS, value.LONG_SMOOTH_SECONDS, value.PLANPLUS_CONTROL = 0., .3, 1.
        value._vision_input_names, value._desire_key = ['img', 'big_img'], 'desire'
        value.numpy_inputs = {'action_t': np.zeros((1, 2))}
        value.run, value.warmup = Mock(side_effect=lambda *a: {k: v.copy() for k, v in parsed.items()}), Mock()
        return value
      small, candidate = model(False), model(True)
      guard = CinqueGuard(params, lambda: small, lambda: candidate)
      if guard_used:
        params.put_bool('CinqueV3Enabled', True, block=True)
        guard.start()
        if fail_v3 is not None:
          candidate.run.side_effect = fail_v3
      selected = candidate if guard_used else small
      if fail_recovery:
        small.warmup.side_effect = RuntimeError('fallback reset failed')
      if legacy_gpu_failure:
        from openpilot.sunnypilot.modeld_v2.meta_20hz import Meta20hz
        selected = candidate
        candidate.meta_constants, candidate.LONG_SMOOTH_SECONDS = Meta20hz, 1.
        candidate.run.side_effect = RuntimeError('legacy GPU failed')
        params.put_bool('ChestnutActive', True, block=True)
      if fail_publication:
        bad = {k: v.copy() for k, v in parsed.items()}
        bad['meta'] = np.zeros((1, 1), dtype='float32')
        candidate.run.side_effect = lambda *a: bad
      published = []
      class Camera:
        width, height, buffer_len = 8, 8, 96
        frame_id, timestamp_sof, timestamp_eof = 0, 0, 0
        def connect(self, block):
          return True
        def recv(self):
          if published:
            raise KeyboardInterrupt
          self.frame_id += 1
          self.timestamp_sof = self.frame_id * 50_000_000
          self.timestamp_eof = self.timestamp_sof + 10_000_000
          if self.frame_id > 3:
            raise AssertionError('Startup loop without model publication')
          return SimpleNamespace(data=bytes(96))
      class Subscriptions:
        frame = 1
        updated, seen = {'extrinsicsCalibration': False}, {}
        def update(self, timeout):
          if late_fault and guard.active:
            params.put('CinqueV3Fault', 'operator safety stop', block=True)
        def __getitem__(self, key):
          return {'carState': car.CarState(vEgo=10.), 'carControl': car.CarControl(latActive=False),
                  'driverMonitoringState': SimpleNamespace(isRHD=False),
                  'narrowRoadCameraState': SimpleNamespace(frameId=1)}[key]
      class Publisher:
        def send(self, service, message):
          published.append((service, message if isinstance(message, bytes) else message.to_bytes()))
      vipc = Mock(side_effect=lambda *args: Camera())
      vipc.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]
      def startup_loader(*args):
        params.put_bool('ChestnutActive', selected.chestnut, block=True)
        return StartupModels(selected, small, guard if guard_used else None)
      original_new_message = modeld.messaging.new_message
      class BadSerialization:
        def __init__(self, message):
          object.__setattr__(self, 'message', message)
        def __getattr__(self, name):
          return getattr(self.message, name)
        def __setattr__(self, name, value):
          setattr(self.message, name, value)
        def to_bytes(self):
          raise ValueError('fourth message serialization failed')
      def new_message(service, *args, **kwargs):
        msg = original_new_message(service, *args, **kwargs)
        return BadSerialization(msg) if fail_serialization and service == 'modelDataV2SP' and guard.active else msg
      with patch.object(modeld, 'Params', return_value=params), patch.object(modeld, 'chestnut_present', return_value=False), \
           patch.object(modeld, 'config_realtime_process'), patch.object(modeld, 'VisionIpcClient', vipc), \
           patch.object(modeld, 'PubMaster', return_value=Publisher()), patch.object(modeld, 'SubMaster', return_value=Subscriptions()), \
           patch.object(modeld, 'get_demo_car_params', return_value=car.CarParams(brand='honda', longitudinalActuatorDelay=.2)), \
           patch.object(modeld, 'load_models', side_effect=startup_loader), \
           patch.object(modeld, 'fill_model_msg', wraps=modeld.fill_model_msg) as fill, \
           patch.object(modeld.messaging, 'new_message', side_effect=new_message):
        with self.assertRaises(RuntimeError if fail_recovery else KeyboardInterrupt):
          modeld.main(demo=True)
      if fail_recovery:
        self.assertEqual(published, [])
        self.assertIn('fallback reset failed', params.get('CinqueV3Fault'))
        return
      self.assertEqual([name for name, _ in published], ['modelV2', 'drivingModelData', 'cameraOdometry', 'modelDataV2SP'])
      recovered = guard_used and (fail_v3 is not None or fail_publication or fail_serialization or late_fault)
      with log.Event.from_bytes(published[0][1]) as message:
        self.assertEqual(message.modelV2.big, guard_used and not recovered)
        self.assertEqual(message.modelV2.action.desiredAcceleration, 0)
        self.assertEqual(message.modelV2.frameId, 2 if recovered or legacy_gpu_failure else 1)
      self.assertIs(fill.call_args.args[-1], Meta)
      active = candidate if guard_used and not recovered else small
      self.assertAlmostEqual(active.run.call_args.args[2]['action_t'][1], .575)
      if guard_used:
        self.assertEqual(candidate.run.call_count, 0 if late_fault else 1)
        self.assertEqual(params.get_bool('CinqueV3Enabled'), not recovered)
        self.assertEqual(bool(params.get('CinqueV3Fault')), recovered)
        self.assertEqual(small.warmup.call_count, 2 if recovered else 1)
      else:
        self.assertIsNone(params.get('CinqueV3Fault'))

  def test_inference_failure_discards_frame_then_publishes_clean_fallback(self):
    self.exercise(fail_v3=RuntimeError('GPU failure'))

  def test_missing_v3_predictions_are_a_fault_not_silent_missing_publications(self):
    self.exercise(fail_v3=lambda *args: None)

  def test_publication_error_discards_frame_and_resets_fallback(self):
    self.exercise(fail_publication=True)

  def test_ordinary_mvl_loop_publishes_all_services_without_v3(self):
    self.exercise(guard_used=False)

  def test_failed_recovery_stops_without_any_publication(self):
    self.exercise(fail_v3=RuntimeError('GPU failure'), fail_recovery=True)

  def test_legacy_gpu_fallback_uses_small_model_metadata_and_delays(self):
    self.exercise(guard_used=False, legacy_gpu_failure=True)

  def test_initialized_v3_publishes_all_validated_serialized_services(self):
    self.exercise()

  def test_final_message_serialization_failure_publishes_no_v3_frame(self):
    self.exercise(fail_serialization=True)

  def test_new_fault_blocks_inference_even_if_enabled_flag_was_not_cleared(self):
    self.exercise(late_fault=True)


if __name__ == '__main__':
  unittest.main()
