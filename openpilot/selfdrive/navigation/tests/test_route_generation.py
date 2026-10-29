"""Optional diagnostic identity utilities; production follows official receipt semantics.

Producer parity comparisons are in test_official_route_production.py.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


from openpilot.common.parameterized import parameterized
from openpilot.selfdrive.navigation.generation import UINT64_MAX, destination_key, gps_source, next_generation


class TestDestinationIdentity(unittest.TestCase):
  def test_identity_ignores_display_metadata_and_numeric_spelling(self):
    reference = destination_key({'latitude': 32, 'longitude': -117})
    self.assertEqual(len(reference), 64)
    self.assertEqual(reference, destination_key({'latitude': 32., 'longitude': -117., 'place_name': 'New label'}))
    self.assertEqual(destination_key({'latitude': -0., 'longitude': 0}), destination_key({'latitude': 0, 'longitude': 0.}))
    self.assertNotEqual(reference, destination_key({'latitude': 32.0000001, 'longitude': -117}))

  @parameterized.expand([None, {}, [], {'latitude': True, 'longitude': 0}, {'latitude': 0, 'longitude': float('nan')},
                         {'latitude': 0, 'longitude': float('inf')}, {'latitude': 91, 'longitude': 0},
                         {'latitude': 0, 'longitude': 181}, {'latitude': '0', 'longitude': 0},
                         {'latitude': 10**500, 'longitude': 0}])
  def test_invalid_destination_has_no_identity(self, destination):
    self.assertEqual(destination_key(destination), '')

  def test_route_ids_advance_and_separate_normal_restarts(self):
    first = next_generation(clock=lambda: 100)
    second = next_generation(first, clock=lambda: 100)
    restarted = next_generation(clock=lambda: 102)
    self.assertEqual((first, second, restarted), (100, 101, 102))
    self.assertEqual(next_generation(clock=lambda: 0), 1)
    with self.assertRaises(OverflowError):
      next_generation(UINT64_MAX, clock=lambda: 1)

  def test_physical_gps_time_checked_independently_of_delivery(self):
    sm = MagicMock()
    sm.valid = {'gpsLocation': True, 'gpsLocationExternal': False}
    sm.recv_time = dict.fromkeys(sm.valid, 100.)
    sm.__getitem__.return_value = SimpleNamespace(hasFix=True, latitude=0., longitude=0.)
    for source_ns, expected in ((99_000_000_000, 'gpsLocation'), (98_000_000_000, None),
                                (100_300_000_000, None), (110_000_000_000, None), (0, None)):
      sm.logMonoTime = dict.fromkeys(sm.valid, source_ns)
      with self.subTest(source_ns=source_ns):
        self.assertEqual(gps_source(sm, now=100.), expected)
    sm.logMonoTime = dict.fromkeys(sm.valid, 1)
    self.assertEqual(gps_source(sm, now=100., check_source_time=False), 'gpsLocation')  # explicit legacy viewer policy
