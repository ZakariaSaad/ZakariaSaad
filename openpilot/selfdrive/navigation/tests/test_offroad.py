"""Activation reads real software state; ignition is not an off-road proxy."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openpilot.selfdrive.navigation import offroad
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled


class ParamsStub:
  def __init__(self, **values):
    self.values = {'IsOffroad': True, 'IsEngaged': False, **values}

  def get_bool(self, key):
    return bool(self.values.get(key, False))


class StateStub(dict):
  def __init__(self):
    processes = [SimpleNamespace(name=name, running=False, shouldBeRunning=False)
                 for name in ('controlsd', 'selfdrived', 'card', 'modeld', 'modeld_tinygrad')]
    super().__init__(deviceState=SimpleNamespace(started=False), managerState=SimpleNamespace(processes=processes),
                     selfdriveState=SimpleNamespace(enabled=False, active=False),
                     carControl=SimpleNamespace(latActive=False, longActive=False))
    self.seen = dict.fromkeys(self, True)
    self.valid = dict.fromkeys(self, True)
    self.logMonoTime = dict.fromkeys(self, 99_900_000_000)


class TestOffroadAdmission(unittest.TestCase):
  def setUp(self):
    self.params, self.state = ParamsStub(), StateStub()

  def reason(self):
    return offroad.offroad_reason(self.params, self.state, now=100.0)

  def test_fresh_offroad_disengaged_software_state_is_sufficient_without_ignition(self):
    self.assertEqual(self.reason(), '')
    # There is deliberately no Panda/ignition topic in this snapshot.
    self.assertNotIn('pandaStates', self.state)

  def test_params_must_agree_with_fresh_software_state(self):
    for key, value in (('IsOffroad', False), ('IsEngaged', True)):
      with self.subTest(key=key):
        self.params.values[key] = value
        self.assertTrue(self.reason())
        self.params.values[key] = not value

  def test_started_device_is_rejected_even_if_params_claim_offroad(self):
    self.state['deviceState'].started = True
    self.assertIn('deviceState', self.reason())

  def test_required_topics_must_be_seen_valid_and_fresh(self):
    for topic in ('deviceState', 'managerState'):
      for attribute, value in (('seen', False), ('valid', False), ('logMonoTime', 90_000_000_000),
                               ('logMonoTime', 101_000_000_000), ('logMonoTime', 0)):
        with self.subTest(topic=topic, attribute=attribute, value=value):
          original = getattr(self.state, attribute)[topic]
          getattr(self.state, attribute)[topic] = value
          self.assertIn(topic, self.reason())
          getattr(self.state, attribute)[topic] = original

  def test_fresh_control_messages_cannot_claim_engagement(self):
    for topic, field in (('selfdriveState', 'enabled'), ('selfdriveState', 'active'),
                         ('carControl', 'latActive'), ('carControl', 'longActive')):
      with self.subTest(topic=topic, field=field):
        setattr(self.state[topic], field, True)
        self.assertIn(topic, self.reason())
        setattr(self.state[topic], field, False)

  def test_manager_must_confirm_every_control_and_model_process_is_stopped(self):
    for process in self.state['managerState'].processes:
      for field in ('running', 'shouldBeRunning'):
        with self.subTest(process=process.name, field=field):
          setattr(process, field, True)
          self.assertIn(process.name, self.reason())
          setattr(process, field, False)

  def test_missing_or_duplicate_required_process_record_is_rejected(self):
    records = self.state['managerState'].processes
    for altered in (records[1:], records + [copy.copy(records[0])]):
      with self.subTest(count=len(altered)):
        self.state['managerState'].processes = altered
        self.assertTrue(self.reason())
    self.state['managerState'].processes = records

  def test_absent_onroad_only_control_topics_do_not_block_verified_stopped_processes(self):
    for topic in ('selfdriveState', 'carControl'):
      self.state.seen[topic] = False
      self.state.valid[topic] = False
      self.state.logMonoTime[topic] = 0
    self.assertEqual(self.reason(), '')

  def test_old_control_topic_from_last_drive_does_not_override_fresh_stopped_manager_state(self):
    self.state['selfdriveState'].enabled = True
    self.state['carControl'].latActive = True
    self.state.logMonoTime['selfdriveState'] = self.state.logMonoTime['carControl'] = 90_000_000_000
    self.assertEqual(self.reason(), '')

  def test_admission_is_read_only(self):
    original_params = copy.deepcopy(self.params.values)
    original_state = copy.deepcopy(self.state)
    self.assertEqual(self.reason(), '')
    self.assertEqual(self.params.values, original_params)
    self.assertEqual(vars(self.state['deviceState']), vars(original_state['deviceState']))
    self.assertEqual(self.state.logMonoTime, original_state.logMonoTime)

  def test_live_guard_only_subscribes_and_accepts_the_same_snapshot(self):
    self.state.update = Mock()
    messaging = SimpleNamespace(SubMaster=Mock(return_value=self.state))
    with patch.dict(sys.modules, {'openpilot.cereal': SimpleNamespace(messaging=messaging)}), \
         patch.object(offroad.time, 'monotonic', return_value=100.0):
      offroad.require_offroad(self.params)
    messaging.SubMaster.assert_called_once_with(list(offroad.SERVICES), poll='deviceState')
    self.state.update.assert_called_once_with(100)

  def test_live_guard_refuses_missing_native_messaging_with_original_error(self):
    import builtins
    original_import = builtins.__import__

    def without_native(name, *args, **kwargs):
      if name == 'openpilot.cereal':
        raise ImportError('native messaging dependency is unavailable')
      return original_import(name, *args, **kwargs)

    with patch.object(builtins, '__import__', side_effect=without_native), self.assertRaises(NavigationModelDisabled) as raised:
      offroad.require_offroad(self.params)
    self.assertIn('Cannot read fresh off-road/control state', str(raised.exception))
    self.assertIn('native messaging dependency is unavailable', str(raised.exception))


if __name__ == '__main__':
  unittest.main()
