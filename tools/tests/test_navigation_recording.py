"""Recorded-camera preparation tests with source provenance kept explicit."""
import unittest

import numpy as np

from tools.navigation_recording import latest_context, match_camera_frames, padded_nv12
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class TestNavigationRecording(unittest.TestCase):
  def test_native_stride_and_plane_padding_preserves_visible_pixels(self):
    width, height = 1344, 760
    frame = np.arange(width * height * 3 // 2, dtype=np.uint8)
    padded = padded_nv12(frame, width, height)
    stride, y_height, uv_height, _ = get_nv12_info(width, height)
    self.assertEqual(padded.size, stride * (y_height + uv_height))
    planes = padded.reshape(y_height + uv_height, stride)
    original = frame.reshape(height * 3 // 2, width)
    np.testing.assert_array_equal(planes[:height, :width], original[:height])
    np.testing.assert_array_equal(planes[y_height:y_height + height // 2, :width], original[height:])
    self.assertFalse(planes[:, width:].any())
    self.assertFalse(planes[height:y_height].any())

  def test_bad_decoded_frame_cannot_be_silently_reshaped(self):
    for frame in (np.zeros(10, np.uint8), np.zeros(12, np.float32)):
      with self.assertRaises(ValueError):
        padded_nv12(frame, 4, 2)

  def test_pairing_uses_capture_timestamps_instead_of_matching_frame_ids(self):
    narrow = [{'frame_id': 10, 'timestamp_sof': 100}, {'frame_id': 11, 'timestamp_sof': 200}]
    wide = [{'frame_id': 10, 'timestamp_sof': 1}, {'frame_id': 12, 'timestamp_sof': 103}, {'frame_id': 13, 'timestamp_sof': 204}]
    pairs = match_camera_frames(narrow, wide, tolerance_ns=10)
    self.assertEqual([big['frame_id'] for _, big in pairs], [12, 13])

  def test_camera_skew_missing_and_repeated_frames_rejected(self):
    with self.assertRaises(ValueError):
      match_camera_frames([{'timestamp_sof': 100}], [{'timestamp_sof': 150}], tolerance_ns=10)
    with self.assertRaises(ValueError):
      match_camera_frames([{'timestamp_sof': 100}, {'timestamp_sof': 101}], [{'timestamp_sof': 100}], tolerance_ns=10)

  def test_context_is_causal_and_fresh(self):
    events = [(10, 'old'), (20, 'current'), (30, 'future')]
    self.assertEqual(latest_context(events, 25, max_age_ns=10), (20, 'current'))
    for timestamp, timeout in ((5, 10), (25, 5)):
      with self.assertRaises(ValueError):
        latest_context(events, timestamp, max_age_ns=timeout)


if __name__ == '__main__':
  unittest.main()
