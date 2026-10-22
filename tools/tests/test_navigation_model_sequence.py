"""Long-fixture provenance and real diagnostic transition execution, without a model."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openpilot.selfdrive.navigation.temporal import NavigationTransitions, context_reason
from tools.navigation_model_diagnostic import prepare_contexts
from tools.navigation_model_sequence import prepare_sequence


class TestLongNavigationSequence(unittest.TestCase):
  def test_repeated_camera_fixture_requires_opt_in_and_never_claims_recorded_navigation(self):
    with tempfile.TemporaryDirectory() as folder:
      source, output = Path(folder) / 'source.npz', Path(folder) / 'long.npz'
      times = 10_000_000_000 + np.arange(8, dtype=np.int64) * 50_000_000
      np.savez(source, img=np.arange(8*12, dtype=np.uint8).reshape(8, 12), big_img=np.zeros((8, 12), np.uint8),
               camera_timestamp_ns=times, big_camera_timestamp_ns=times + 40_000,
               gps_source_ns=times, gps_valid=np.ones(8, bool), nav_valid=np.zeros(8, bool))
      with self.assertRaisesRegex(ValueError, 'explicit acknowledgment'):
        prepare_sequence(source, output)
      provenance = prepare_sequence(source, output, allow_repeated_cameras=True)
      self.assertFalse(provenance['driving_quality_validation'])
      self.assertFalse(provenance['navigation_synthesized_by_this_tool'])
      with np.load(output, allow_pickle=False) as archive:
        self.assertEqual(archive['img'].shape, (164, 12))
        self.assertTrue(archive['synthetic_repeated_camera'].all())
        self.assertFalse(archive['nav_valid'].any())
        self.assertFalse(archive['gps_valid'].any())
        self.assertTrue((np.diff(archive['camera_timestamp_ns']) == 50_000_000).all())
        self.assertTrue((archive['big_camera_timestamp_ns'] - archive['camera_timestamp_ns'] == 40_000).all())
      frames = [{'navigation': {}} for _ in range(164)]
      contexts, scenarios = prepare_contexts(output, frames, synthetic_long_sequence=True)
      self.assertTrue(all(context_reason(context) == 'valid' for context in contexts[:144]))
      defects = {144: 'map_route_mismatch', 148: 'missing_gps', 152: 'permission_revoked',
                 156: 'source_time_out_of_range', 160: 'map_frame_mismatch'}
      state = NavigationTransitions()
      resets = []
      for index, context in enumerate(contexts):
        reason = context_reason(context)
        decision = state.update(context.identity, reason == 'valid', context.camera_ns, reason)
        if decision.reset:
          resets.append(index)
        if index in defects:
          self.assertEqual(reason, defects[index])
          self.assertFalse(decision.infer)
          self.assertEqual(scenarios[index], {'source_time_out_of_range': 'stale_map',
                                            'map_route_mismatch': 'reroute_old_map'}.get(reason, reason))
      self.assertEqual(resets[:2], [0, 144])
      self.assertTrue(all(index+1 in resets for index in defects))
      self.assertTrue(all(context_reason(contexts[index+1]) == 'valid' for index in defects))
      with self.assertRaisesRegex(ValueError, 'new output'):
        prepare_sequence(source, output, allow_repeated_cameras=True)
      with self.assertRaisesRegex(ValueError, 'explicit acknowledgment'):
        prepare_sequence(output, Path(folder) / 'copied.npz')
      copied = prepare_sequence(output, Path(folder) / 'copied.npz', allow_repeated_cameras=True)
      self.assertIn('synthetic repeated', copied['camera_sequence'])

  def test_short_long_suite_is_rejected(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / 'short.npz'
      np.savez(path, camera_timestamp_ns=np.array([10_000_000_000]), big_camera_timestamp_ns=np.array([10_000_000_000]))
      with self.assertRaisesRegex(ValueError, '164'):
        prepare_contexts(path, [{'navigation': {}}], synthetic_long_sequence=True)
