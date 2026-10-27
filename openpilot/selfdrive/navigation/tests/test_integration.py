"""Backport boundaries: persistent settings, real daemon lifecycle and the installed update channel."""
import os
import time
from unittest.mock import patch

import numpy as np
from msgq.visionipc import VisionIpcClient
from opendbc.car.structs import car

from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.navigation.renderer import SIZE
from openpilot.system.manager.process import ensure_running
from openpilot.system.manager.process_config import managed_processes
from openpilot.system.updated.updated import Updater

BRANCHES = ('mvl-honda-eps-selector21-chestnut-v28', 'mvl-sp-steer-eps-nav')


class TestNavigationIntegration(OpenpilotTestCase):
  def test_existing_comma_api_compatibility(self):
    from openpilot.common.api.comma_connect import API_HOST, CommaConnectApi
    from openpilot.selfdrive.navigation.navd import RouteEngine
    from openpilot.selfdrive.navigation.tiles import Api as TilesApi
    params = Params()
    params.put('DongleId', '0123456789abcdef', block=True)
    with patch.dict(os.environ), patch.object(CommaConnectApi, 'get_key_pair', return_value=(None, None, None)):
      # Use the real existing API class, not a fake with the old wrapper's missing attributes.
      os.environ.pop('MAPBOX_TOKEN', None)
      engine = RouteEngine(None, None, params)
      self.assertEqual(engine.api.dongle_id, '0123456789abcdef')
      self.assertEqual(engine.mapbox_host, API_HOST.rstrip('/') + '/v1/nav')
      self.assertEqual(TilesApi('0123456789abcdef').dongle_id, '0123456789abcdef')

  def test_parameter_lifetime(self):
    params = Params()
    self.assertFalse(params.get_bool('NavigationEnabled'))
    self.assertFalse(params.get('NavigationEnabled', return_default=True))
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
      params.put_bool('NavigationEnabled', True, block=True)
      params.put('NavDestination', {'latitude': 32.8, 'longitude': -117.2}, block=True)
      params.clear_all(flag)
      self.assertTrue(Params().get_bool('NavigationEnabled'))
      self.assertIsNone(params.get('NavDestination'))

  def test_native_daemon_lifecycle(self):
    params = Params()
    cp = car.CarParams.new_message()
    procs = {name: managed_processes[name] for name in ('navd', 'mapsd')}
    self.addCleanup(lambda: [p.stop() for p in procs.values()])
    self.assertEqual(ensure_running(procs.values(), True, params, cp), [])
    params.put_bool('NavigationEnabled', True, block=True)
    self.assertEqual(ensure_running(procs.values(), False, params, cp), [])

    sm = messaging.SubMaster(['navInstruction', 'navRoute'])
    client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
    self.assertEqual(len(ensure_running(procs.values(), True, params, cp)), 2)
    deadline = time.monotonic() + 10
    frame = None
    while time.monotonic() < deadline:
      if client.is_connected() or client.connect(False):
        frame = client.recv(timeout_ms=100)
        if frame is not None:
          break
      time.sleep(0.01)
    self.assertIsNotNone(frame, 'mapsd failed to publish through the pinned native VisionIPC runtime')
    self.assertEqual((client.width, client.height), (SIZE, SIZE))
    self.assertEqual(client.timestamp_sof, 0)  # no GPS or destination must never become a usable map
    self.assertFalse(np.frombuffer(frame.data, np.uint8).any())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not sm.seen['navInstruction']:
      sm.update(100)
    self.assertTrue(sm.seen['navInstruction'])
    self.assertFalse(sm.valid['navInstruction'])
    self.assertTrue(all(p.proc.is_alive() for p in procs.values()))

    # Disabling the feature stops both real child processes; re-enabling/offroad also works.
    params.put_bool('NavigationEnabled', False, block=True)
    self.assertEqual(ensure_running(procs.values(), True, params, cp), [])
    self.assertTrue(all(p.stop() == 0 for p in procs.values()))
    params.put_bool('NavigationEnabled', True, block=True)
    sm = messaging.SubMaster(['navInstruction'])
    restarted_client = VisionIpcClient('mapsd', VisionStreamType.VISION_STREAM_MAP, True)
    ensure_running(procs.values(), True, params, cp)
    deadline = time.monotonic() + 5
    restarted_frame = None
    # Both daemons must finish startup before a clean-shutdown assertion. navd
    # can publish before mapsd installs its normal SIGINT handler; stopping it
    # during Python's after-fork bootstrap tests a race rather than lifecycle.
    while time.monotonic() < deadline and (not sm.seen['navInstruction'] or restarted_frame is None):
      sm.update(50)
      if restarted_frame is None and (restarted_client.is_connected() or restarted_client.connect(False)):
        restarted_frame = restarted_client.recv(timeout_ms=50)
    self.assertTrue(sm.seen['navInstruction'])
    self.assertIsNotNone(restarted_frame, 'restarted mapsd did not become ready')
    self.assertEqual(ensure_running(procs.values(), False, params, cp), [])
    self.assertTrue(all(p.stop() == 0 for p in procs.values()))

  def test_installed_branch_tracking_and_commit_detection(self):
    params = Params()
    for branch, device in ((branch, device) for branch in BRANCHES for device in ('mici', 'tizi', 'tici')):
      with self.subTest(branch=branch, device=device), patch('openpilot.system.updated.updated.HARDWARE.get_device_type', return_value=device), \
           patch.object(Updater, 'get_branch', return_value=branch):
        updater = Updater()
        self.assertEqual(updater.target_branch, branch)
        params.put('UpdaterTargetBranch', branch, block=True)
        self.assertEqual(Updater().target_branch, branch)
        updater.branches[branch] = '1' * 40
        with patch('openpilot.system.updated.updated.os.path.isdir', return_value=True), \
             patch.object(Updater, 'get_commit_hash', return_value='0' * 40):
          self.assertTrue(updater.update_available)
        with patch('openpilot.system.updated.updated.os.path.isdir', return_value=True), \
             patch.object(Updater, 'get_commit_hash', return_value='1' * 40):
          self.assertFalse(updater.update_available)
        params.remove('UpdaterTargetBranch')
