"""Malformed/unknown navigation metadata must not manufacture a model command."""
import unittest
from types import SimpleNamespace

import numpy as np

from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.navigation.helpers import navigation_input


def maneuver(kind, modifier, distance=100):
  return SimpleNamespace(type=kind, modifier=modifier, distance=distance)


class TestNavigationInputHardening(unittest.TestCase):
  @parameterized.expand([('unknown', 'unknown'), ('turn', 'typo'), ('fork', 'typo')])
  def test_unknown_future_instruction_blocks_vector(self, kind, modifier):
    self.assertFalse(navigation_input([maneuver(kind, modifier), maneuver('turn', 'left', 200)], 0).any())

  @parameterized.expand([float('nan'), float('inf'), -1])
  def test_invalid_traveled_fails_closed(self, value):
    self.assertFalse(navigation_input([maneuver('turn', 'left')], value).any())

  def test_invalid_maneuver_distance_fails_closed(self):
    self.assertFalse(navigation_input([maneuver('turn', 'left', float('nan'))], 0).any())

  def test_intentional_merge_omission_and_known_values_preserved(self):
    result = navigation_input([maneuver('merge', 'right'), maneuver('turn', 'left', 200)], 0)
    self.assertEqual(result[15], 1)
    self.assertEqual(result[-1], 2)
    self.assertEqual(result.dtype, np.float32)
