"""Edge-case maneuver encoding must match the pinned official Bigv2 helper."""
import unittest
from types import SimpleNamespace

import numpy as np

from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.navigation.helpers import navigation_input
from openpilot.selfdrive.navigation.tests.official_reference import load_reference


def maneuver(kind, modifier, distance=100):
  return SimpleNamespace(type=kind, modifier=modifier, distance=distance)


class TestNavigationInputHardening(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.official = load_reference('helpers')

  def assert_official_encoding(self, maneuvers, traveled):
    result = navigation_input(maneuvers, traveled)
    expected = self.official.navigation_input(maneuvers, traveled)
    np.testing.assert_array_equal(result, expected)
    self.assertEqual(result.dtype, expected.dtype)
    self.assertEqual(result.shape, expected.shape)
    return result

  @parameterized.expand([('unknown', 'unknown'), ('turn', 'typo'), ('fork', 'typo')])
  def test_unknown_future_instruction_uses_official_default(self, kind, modifier):
    self.assert_official_encoding([maneuver(kind, modifier), maneuver('turn', 'left', 200)], 0)

  @parameterized.expand([float('nan'), float('inf'), -1])
  def test_nonfinite_and_negative_traveled_match_official(self, value):
    self.assert_official_encoding([maneuver('turn', 'left')], value)

  def test_nonfinite_maneuver_distance_matches_official(self):
    self.assert_official_encoding([maneuver('turn', 'left', float('nan'))], 0)

  def test_intentional_merge_omission_and_known_values_preserved(self):
    result = self.assert_official_encoding([maneuver('merge', 'right'), maneuver('turn', 'left', 200)], 0)
    self.assertEqual(result[15], 1)
    self.assertEqual(result[-1], 2)
    self.assertEqual(result.dtype, np.float32)
