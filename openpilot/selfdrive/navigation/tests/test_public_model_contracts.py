"""CPU-only compatibility checks against verified public model graph signatures.

The fixtures contain signatures and provenance only, never weights or pickle data.
These tests exercise admission, parser contracts and recurrent-state pair layouts;
they do not run neural-network inference or validate driving behavior.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.navigation import model_inputs
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser

MODELS = json.loads((Path(__file__).parent / 'fixtures' / 'nav_models.json').read_text())['models']
DTYPES = {'UINT8': 'uint8', 'FLOAT': 'float32'}


def input_specs(model):
  return {entry['name']: (tuple(entry['shape']), DTYPES[entry['dtype']], 'CPU') for entry in model['inputs']}


def params(enabled):
  return SimpleNamespace(get_bool=lambda key: key == 'NavigationModelInputsEnabled' and enabled)


class TestPublicNavigationModelContracts(unittest.TestCase):
  def test_public_models_require_explicit_offline_admission(self):
    # Every gate is independent: parameter alone, environment alone, or hardware
    # must not admit a navigation-conditioned model to live driving.
    cases = [(False, '1', True), (True, '', True), (True, '1', False)]
    for model in MODELS:
      for pc, offline, enabled in cases:
        with self.subTest(model=model['tag'], pc=pc, offline=offline, enabled=enabled), \
             patch.object(model_inputs, 'PC', pc), patch.dict(os.environ, {'NAV_MODEL_OFFLINE': offline}):
          with self.assertRaises(model_inputs.NavigationModelDisabled):
            model_inputs.validate_model_inputs(input_specs(model), params(enabled))

  def test_verified_public_signatures_are_supported_offline(self):
    with patch.object(model_inputs, 'PC', True), patch.dict(os.environ, {'NAV_MODEL_OFFLINE': '1'}):
      for model in MODELS:
        with self.subTest(model=model['tag']):
          self.assertTrue(model_inputs.validate_model_inputs(input_specs(model), params(True)))

  def test_unknown_map_shapes_and_dtypes_fail_closed(self):
    with patch.object(model_inputs, 'PC', True), patch.dict(os.environ, {'NAV_MODEL_OFFLINE': '1'}):
      for model in MODELS:
        for key, shape, dtype in (
          ('nav_map', (1, 3, 256, 256), 'uint8'),
          ('nav_map', (1, 1, 128, 256), 'uint8'),
          ('nav_map', (1, 1, 256, 256), 'uint16'),
          ('navigation', (1, 150), 'float32'),
          ('navigation', (47,), 'float64'),
        ):
          specs = input_specs(model)
          specs[key] = (shape, dtype, 'CPU')
          with self.subTest(model=model['tag'], key=key, shape=shape, dtype=dtype), self.assertRaises(ValueError):
            model_inputs.validate_model_inputs(specs, params(True))

  def test_existing_non_navigation_signatures_do_not_need_experimental_gate(self):
    for model in MODELS:
      specs = {key: value for key, value in input_specs(model).items()
               if key not in ('nav_map', 'navigation') and not key.startswith(('state_map', 'state_nav'))}
      with self.subTest(model=model['tag']), patch.object(model_inputs, 'PC', False), \
           patch.dict(os.environ, {'NAV_MODEL_OFFLINE': ''}):
        self.assertFalse(model_inputs.validate_model_inputs(specs, params(False)))

  def test_native_recurrent_inputs_have_exact_next_output_pairs(self):
    # The generations use different navigation-history protocols. Matching only
    # immediate map/vector shapes would omit mandatory persistent model state.
    for model in MODELS:
      outputs = {entry['name']: entry for entry in model['outputs']}
      with self.subTest(model=model['tag']):
        for entry in model['inputs']:
          if not entry['name'].startswith('state_'):
            continue
          next_state = outputs['next_' + entry['name']]
          self.assertEqual(next_state['shape'], entry['shape'])
          self.assertEqual(next_state['dtype'], entry['dtype'])

  def test_current_parser_accepts_real_flat_output_contracts(self):
    for model in MODELS:
      size = next(entry['shape'][1] for entry in model['outputs'] if entry['name'] == 'outputs')
      raw = np.zeros((1, size), dtype=np.float32)
      slices = model['output_slices']
      # Distinct synthetic values verify the existing action interface receives
      # the metadata's action mean slots, not a neighboring feature/history slice.
      raw[0, slices['action']['start']:slices['action']['start'] + 2] = (0.25, -0.75)
      heads = {name: raw[:, bounds['start']:bounds['stop']].copy() for name, bounds in slices.items()}
      parsed = Parser().parse_outputs(heads)
      with self.subTest(model=model['tag']):
        self.assertEqual(parsed['plan'].shape, (1, 33, 15))
        self.assertEqual(parsed['lead'].shape, (1, 3, 6, 4))
        self.assertEqual(parsed['lane_lines'].shape, (1, 4, 33, 2))
        np.testing.assert_array_equal(parsed['action'], [[0.25, -0.75]])
        for value in parsed.values():
          self.assertTrue(np.isfinite(value).all())
        # All flat output bytes are accounted for by the statically decoded
        # metadata. This includes Bigv2's extra learned map/navigation features.
        ranges = sorted((bounds['start'], bounds['stop']) for bounds in slices.values())
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], size)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(ranges[:-1], ranges[1:], strict=True)))
