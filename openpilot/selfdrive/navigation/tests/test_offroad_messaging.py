"""Native IPC admission with synthetic messages in isolated PC test namespaces.

No vehicle services run here. These tests exercise the installed cereal schemas
and real SubMaster subscriptions used by Settings and the SSH activation CLI.
"""
from contextlib import contextmanager
import threading

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import REQUIRED_PROCESSES, require_offroad


class TestOffroadNativeMessaging(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.params = Params()
    self.params.put_bool('IsOffroad', True, block=True)
    self.params.put_bool('IsEngaged', False, block=True)

  @contextmanager
  def publish_state(self, *, started=False, enabled=False, lat_active=False, controls=False):
    services = ['managerState', 'deviceState']
    if controls:
      # Publish controls before the polled deviceState so one native update
      # receives every already-queued topic before evaluating admission.
      services = ['selfdriveState', 'carControl', *services]
    publisher = messaging.PubMaster(services)
    stop, ready = threading.Event(), threading.Event()
    errors = []

    def publish():
      try:
        while not stop.is_set():
          for service in services:
            message = messaging.new_message(service, valid=True)
            if service == 'deviceState':
              message.deviceState.started = started
            elif service == 'managerState':
              message.managerState.processes = [{'name': name, 'running': False, 'shouldBeRunning': False}
                                                for name in sorted(REQUIRED_PROCESSES)]
            elif service == 'selfdriveState':
              message.selfdriveState.enabled = enabled
              message.selfdriveState.active = enabled
            else:
              message.carControl.latActive = lat_active
              message.carControl.longActive = False
            publisher.send(service, message)
          ready.set()
          stop.wait(0.02)
      except Exception as error:
        errors.append(error)
        ready.set()

    thread = threading.Thread(target=publish, name='offroad-native-test-publisher', daemon=True)
    thread.start()
    try:
      self.assertTrue(ready.wait(2), 'Synthetic native IPC publisher did not start')
      self.assertFalse(errors, f'Synthetic native IPC publisher failed: {errors}')
      yield
    finally:
      stop.set()
      thread.join(2)
      self.assertFalse(thread.is_alive(), 'Synthetic native IPC publisher did not stop')
      self.assertFalse(errors, f'Synthetic native IPC publisher failed: {errors}')
      publisher.sock.clear()

  def test_real_subscriptions_accept_fresh_offroad_and_stopped_control_processes(self):
    # On-road-only control publishers are normally absent while off road.
    with self.publish_state():
      require_offroad(self.params)
    self.assertTrue(self.params.get_bool('IsOffroad'))
    self.assertFalse(self.params.get_bool('IsEngaged'))

  def test_real_subscriptions_reject_started_or_active_control_messages(self):
    cases = (({'started': True}, 'deviceState.started'),
             ({'enabled': True}, 'selfdriveState'),
             ({'lat_active': True}, 'carControl'))
    for state, reason in cases:
      with self.subTest(state=state), self.publish_state(controls=True, **state), \
           self.assertRaisesRegex(NavigationModelDisabled, reason):
        require_offroad(self.params)
    self.assertTrue(self.params.get_bool('IsOffroad'))
    self.assertFalse(self.params.get_bool('IsEngaged'))
