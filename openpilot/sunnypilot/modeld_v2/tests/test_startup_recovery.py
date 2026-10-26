"""Exercise startup recovery with registered native Params and inert model loaders.

No camera, GPU, publisher, controller or vehicle interface is opened. Real Params
are intentional: a permissive dictionary would hide an unregistered retry key.
"""
from contextlib import contextmanager, ExitStack
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID
from openpilot.sunnypilot.modeld_v2 import modeld


class StartupFinished(Exception):
  pass


class TestStartupRecovery(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.params = Params()
    self.params.put_bool('ChestnutLoading', True, block=True)
    self.params.put_bool('ChestnutActive', True, block=True)
    self.params.put_bool('ChestnutRetry', False, block=True)
    self.params.put_bool('OnroadCycleRequested', False, block=True)
    self.params.put('NavigationModelCandidate', MODEL_ID, block=True)
    self.params.put('NavigationModelManifest', '/data/navigation-models/unchanged/manifest.json', block=True)
    self.params.put('ModelManager_ActiveBundle', {'preserve': 'ordinary-small'}, block=True)
    self.params.put('ModelManager_ActiveBundleChestnut', {'preserve': 'ordinary-big'}, block=True)
    self.preserved = {key: self.params.get(key) for key in ('NavigationModelCandidate', 'NavigationModelManifest',
                                                          'ModelManager_ActiveBundle', 'ModelManager_ActiveBundleChestnut')}

  def assert_flag(self, key, expected):
    deadline = time.monotonic() + 2
    while self.params.get_bool(key) != expected and time.monotonic() < deadline:
      time.sleep(0.005)
    self.assertEqual(self.params.get_bool(key), expected, key)

  def assert_selection_preserved(self, navigation):
    self.assertEqual(self.preserved, {key: self.params.get(key) for key in self.preserved})
    self.assertEqual(self.params.get_bool('NavigationModelDrivingEnabled'), navigation)

  def runtime_state(self):
    deadline = time.monotonic() + 2
    state = self.params.get('NavigationModelRuntimeState')
    while state is None and time.monotonic() < deadline:
      time.sleep(0.005)
      state = self.params.get('NavigationModelRuntimeState')
    self.assertIsInstance(state, dict)
    return state

  @contextmanager
  def startup(self, *, navigation=True, big_error=None, small_error=False, timeout=False):
    self.params.put_bool('NavigationModelDrivingEnabled', navigation, block=True)
    permit = SimpleNamespace(model_id=MODEL_ID) if navigation else None
    small = SimpleNamespace(chestnut=False, numpy_inputs={}, warmup=Mock(), run=Mock())
    big = SimpleNamespace(chestnut=True, numpy_inputs={}, warmup=Mock(), run=Mock())
    release = threading.Event()
    entered = threading.Event()
    threads = []
    real_thread = threading.Thread

    def load_big(*_args, **_kwargs):
      entered.set()
      if timeout and not release.wait(2):
        raise RuntimeError('Test did not release the inert late loader')
      if big_error is not None:
        raise big_error
      return big

    def construct_model(*_args, **kwargs):
      if kwargs['chestnut']:
        return load_big()
      if small_error:
        raise RuntimeError('ordinary small initialization failed')
      return small

    def make_thread(*args, **kwargs):
      thread = real_thread(*args, **kwargs)
      threads.append(thread)
      return thread

    camera = SimpleNamespace(width=1344, height=760, buffer_len=1622016, connect=lambda _: True)
    clients = Mock(return_value=camera)
    clients.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]
    with ExitStack() as stack:
      models = stack.enter_context(patch.object(modeld, 'ModelState', side_effect=construct_model))
      candidate = stack.enter_context(patch.object(modeld, 'load_navigation_model', side_effect=load_big))
      for name, replacement in {'Params': Mock(return_value=self.params), 'setproctitle': Mock(),
                                'get_active_bundle': Mock(), '_find_driving_pkl': Mock(return_value='ordinary-big.sptg'),
                                'chestnut_present': Mock(return_value=True), 'wait_for_chestnut': Mock(),
                                'VisionIpcClient': clients, 'authorize_runtime': Mock(return_value=permit),
                                '_reject_navigation_publication': Mock(),
                                'config_realtime_process': Mock(side_effect=StartupFinished), 'PubMaster': Mock()}.items():
        stack.enter_context(patch.object(modeld, name, replacement))
      stack.enter_context(patch.object(modeld.gc, 'disable'))
      stack.enter_context(patch.object(modeld.threading, 'Thread', side_effect=make_thread))
      if timeout:
        stack.enter_context(patch.object(modeld, 'BIG_MODEL_TIMEOUT', 0.01))
      try:
        yield SimpleNamespace(small=small, big=big, models=models, candidate=candidate, release=release,
                              entered=entered, threads=threads)
      finally:
        release.set()
        for thread in threads:
          thread.join(2)
          self.assertFalse(thread.is_alive(), 'The inert startup loader did not finish')

  def test_failed_navigation_loader_uses_small_and_requests_only_registered_retry(self):
    error = NavigationModelDisabled('Actual GPU does not match the reviewed AMD LLVM gfx1200 USB target')
    with self.startup(big_error=error) as fixture:
      with self.assertRaises(StartupFinished):
        modeld.main()
      fixture.candidate.assert_called_once()
      self.assertEqual([call.kwargs['chestnut'] for call in fixture.models.call_args_list], [False])
      fixture.small.run.assert_not_called()
      fixture.big.warmup.assert_not_called()
      modeld.PubMaster.assert_not_called()
    self.assert_flag('ChestnutLoading', False)
    self.assert_flag('ChestnutActive', False)
    self.assert_flag('ChestnutRetry', True)
    self.assert_flag('OnroadCycleRequested', True)
    state = self.runtime_state()
    self.assertEqual(state['active_source'], 'qcom')
    self.assertIn(str(error), state['reason'])
    self.assert_selection_preserved(True)

  def test_failed_ordinary_big_loader_uses_existing_small_fallback(self):
    with self.startup(navigation=False, big_error=RuntimeError('ordinary big load failed')) as fixture:
      with self.assertRaises(StartupFinished):
        modeld.main()
      fixture.candidate.assert_not_called()
      self.assertEqual([call.kwargs['chestnut'] for call in fixture.models.call_args_list], [True, False])
      fixture.small.run.assert_not_called()
    self.assert_flag('ChestnutLoading', False)
    self.assert_flag('ChestnutActive', False)
    self.assert_flag('ChestnutRetry', True)
    self.assert_flag('OnroadCycleRequested', True)
    self.assert_selection_preserved(False)

  def test_consumed_retry_does_not_request_another_onroad_cycle(self):
    self.params.put_bool('ChestnutRetry', True, block=True)
    with self.startup(big_error=RuntimeError('candidate load failed')):
      with self.assertRaises(StartupFinished):
        modeld.main()
    self.assert_flag('ChestnutLoading', False)
    self.assert_flag('ChestnutActive', False)
    self.assert_flag('ChestnutRetry', True)
    self.assert_flag('OnroadCycleRequested', False)
    self.assert_selection_preserved(True)

  def test_small_initialization_exception_clears_loading_and_active_before_propagating(self):
    with self.startup(small_error=True):
      with self.assertRaisesRegex(RuntimeError, 'ordinary small initialization failed'):
        modeld.main()
    self.assert_flag('ChestnutLoading', False)
    self.assert_flag('ChestnutActive', False)
    self.assert_selection_preserved(True)

  def test_publication_rejection_clears_startup_flags_without_creating_publishers(self):
    def reject_selected(*models):
      if len(models) == 2:
        raise NavigationModelDisabled('Selected model publication denied')
    with self.startup(), patch.object(modeld, '_reject_navigation_publication', side_effect=reject_selected):
      with self.assertRaisesRegex(NavigationModelDisabled, 'Selected model publication denied'):
        modeld.main()
      modeld.PubMaster.assert_not_called()
    self.assert_flag('ChestnutLoading', False)
    self.assert_flag('ChestnutActive', False)
    self.assert_selection_preserved(True)

  def test_timed_out_loader_cannot_reactivate_big_after_small_is_selected(self):
    with self.startup(timeout=True) as fixture:
      with self.assertRaises(StartupFinished):
        modeld.main()
      self.assert_flag('ChestnutLoading', False)
      self.assert_flag('ChestnutActive', False)
      self.assert_flag('ChestnutRetry', True)
      self.assert_flag('OnroadCycleRequested', True)
      self.assertEqual(self.runtime_state()['active_source'], 'qcom')
      fixture.release.set()
      for thread in fixture.threads:
        thread.join(2)
      fixture.big.warmup.assert_called_once()
      fixture.small.run.assert_not_called()
    self.assert_flag('ChestnutLoading', False)
    self.assert_flag('ChestnutActive', False)
    self.assertEqual(self.runtime_state()['active_source'], 'qcom')
    self.assert_selection_preserved(True)

  def test_registered_retry_survives_onroad_cycle_and_resets_at_ignition_or_manager_start(self):
    self.params.put_bool('ChestnutRetry', True, block=True)
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)
    self.assertTrue(self.params.get_bool('ChestnutRetry'))
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_IGNITION_ON)
    self.assertFalse(self.params.get_bool('ChestnutRetry'))
    self.params.put_bool('ChestnutRetry', True, block=True)
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
    self.assertFalse(self.params.get_bool('ChestnutRetry'))
