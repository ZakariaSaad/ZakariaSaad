import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.navigation_model_offline import compare_runs, load_recording


class TestOfflineReplay(unittest.TestCase):
  def test_independent_recurrent_arms_and_identical_camera_timing(self):
    calls = []

    class Model:
      def __init__(self):
        self.state = 0

      def warmup(self):
        self.state = 0

      def run(self, bufs, transforms, inputs):
        calls.append((bufs, transforms, inputs['action_t'].copy()))
        self.state += float(np.sum(inputs.get('navigation', 0)))
        return {'action': np.array([[self.state, -self.state]])}

    frames = [
      {'bufs': object(), 'transforms': object(), 'inputs': {'action_t': np.array([0.2, 0.3])}, 'navigation': {'navigation': np.array([1.0, 2.0])}}
      for _ in range(2)
    ]
    out = compare_runs(Model, frames)
    self.assertEqual(out['frames'], 2)
    self.assertEqual(out['max_abs_difference']['action'], 6.0)
    for before, after in zip(calls[:2], calls[2:], strict=True):
      self.assertIs(before[0], after[0])
      self.assertIs(before[1], after[1])
      np.testing.assert_array_equal(before[2], after[2])

  def test_recording_defaults_to_missing_navigation(self):
    with tempfile.TemporaryDirectory() as root:
      p = Path(root) / 'recording.npz'
      np.savez(
        p,
        img=np.zeros((1, 64), np.uint8),
        big_img=np.zeros((1, 64), np.uint8),
        M_img=np.eye(3)[None],
        M_big_img=np.eye(3)[None],
        desire=np.zeros((1, 8)),
        traffic_convention=np.array([[1.0, 0.0]]),
        action_t=np.array([[0.2, 0.3]]),
      )
      frames = load_recording(p, 'desire')
      self.assertEqual(frames[0]['navigation'], {})
      self.assertEqual(len(frames[0]['bufs']['img'].data), 64)

  def test_object_pickles_rejected(self):
    with tempfile.TemporaryDirectory() as root:
      p = Path(root) / 'recording.npz'
      np.savez(p, img=np.array([object()], dtype=object))
      with self.assertRaises(ValueError):
        load_recording(p, 'desire')

  def test_nonfinite_model_output_rejected(self):
    class Model:
      def warmup(self):
        pass

      def run(self, *args):
        return {'action': np.array([np.nan])}

    with self.assertRaises(ValueError):
      compare_runs(Model, [{'bufs': {}, 'transforms': {}, 'inputs': {}, 'navigation': {}}])

  def test_input_mutations_do_not_cross_replay_arms(self):
    class Model:
      def warmup(self):
        pass

      def run(self, bufs, transforms, inputs):
        before = inputs['action_t'].copy()
        inputs['action_t'][:] = 100
        inputs.get('navigation', np.zeros(47))[:] = 100
        return {'action': before}

    original = np.array([0.2, 0.3])
    navigation = np.zeros(47)
    out = compare_runs(Model, [{'bufs': {}, 'transforms': {}, 'inputs': {'action_t': original}, 'navigation': {'navigation': navigation}}])
    self.assertEqual(out['max_abs_difference']['action'], 0)
    np.testing.assert_array_equal(original, [0.2, 0.3])
    np.testing.assert_array_equal(navigation, np.zeros(47))

  def test_navigation_validity_requires_explicit_booleans(self):
    arrays = {
      'img': np.zeros((1, 64), np.uint8), 'big_img': np.zeros((1, 64), np.uint8),
      'M_img': np.eye(3)[None], 'M_big_img': np.eye(3)[None], 'desire': np.zeros((1, 8)),
      'traffic_convention': np.array([[1.0, 0.0]]), 'action_t': np.array([[0.2, 0.3]]),
      'nav_map': np.zeros((1, 256, 256), np.uint8), 'navigation': np.zeros((1, 47), np.float32),
    }
    with tempfile.TemporaryDirectory() as root:
      p = Path(root) / 'recording.npz'
      for invalid in (np.array([np.nan]), np.array(['False']), np.array([1]), np.array([[True]])):
        with self.subTest(validity=invalid):
          np.savez(p, allow_pickle=False, **arrays, nav_valid=invalid)
          with self.assertRaisesRegex(ValueError, 'nav_valid must be a boolean'):
            load_recording(p, 'desire')
      np.savez(p, allow_pickle=False, **arrays, nav_valid=np.array([True]))
      self.assertEqual(set(load_recording(p, 'desire')[0]['navigation']), {'nav_map', 'navigation'})
