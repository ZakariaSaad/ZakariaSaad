"""Isolated inference resets use real CPU/NPY tensors, never cameras or publishers."""
import json
from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
from tinygrad import Tensor

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2.model_adapters import BaseModelAdapter, NativeTinygradAdapter
from openpilot.sunnypilot.modeld_v2 import modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.modeld import ModelState
from openpilot.sunnypilot.modeld_v2.tests.test_navigation_inputs import FIXTURES, native_fixture


class TestDiagnosticStateReset(OpenpilotTestCase):
  @staticmethod
  def fill_tensor(tensor, value):
    array = tensor.numpy()
    array[:] = value
    tensor._buffer().as_memoryview(allow_zero_copy=True)[:] = array.tobytes()

  def test_all_native_history_and_aliases_reset_in_place(self):
    for signature, device in product(json.loads(FIXTURES.read_text())['models'], ('NPY', 'CPU')):
      with self.subTest(model=signature['tag'], device=device):
        fixture = native_fixture(signature)
        for key in ('input_specs', 'output_specs'):
          fixture[key] = {name: (shape, dtype, device) for name, (shape, dtype, _) in fixture[key].items()}
        adapter = NativeTinygradAdapter(fixture, 1344, 760, device, device, device)
        queues = dict(adapter.input_queues)
        buffers = {name: tensor._buffer() for name, tensor in queues.items()}
        outputs = dict(adapter.outputs)
        for tensor in queues.values():
          self.fill_tensor(tensor, 7)
        for tensor in outputs.values():
          self.fill_tensor(tensor, 9)
        adapter.packed_input[:] = 11
        adapter.full_frames['img'] = object()
        adapter._blob_cache['old-frame'] = object()

        adapter.reset_state()

        for name, tensor in queues.items():
          self.assertIs(adapter.input_queues[name], tensor)
          self.assertIs(tensor._buffer(), buffers[name])
          self.assertFalse(tensor.numpy().any(), name)
        for name, tensor in outputs.items():
          self.assertIs(adapter.outputs[name], tensor)
          self.assertFalse(tensor.numpy().any(), name)
        self.assertFalse(adapter.packed_input.any())
        self.assertFalse(adapter.full_frames)
        self.assertFalse(adapter._blob_cache)
        # Output aliases must still point to the same captured recurrent state.
        for name, next_name in adapter.state_pairs.items():
          self.fill_tensor(adapter.input_queues[name], 3)
          self.assertTrue((adapter.outputs[next_name].numpy() == 3).all(), name)
        adapter.jits['run'].assert_not_called()

  def test_legacy_image_feature_and_desire_queues_also_reset(self):
    adapter = BaseModelAdapter({}, 1344, 760, 'NPY', 'NPY', 'NPY')
    adapter.input_queues = {name: Tensor(np.ones((2, 3), np.float32), device='NPY').realize()
                            for name in ('img_q', 'big_img_q', 'feat_q', 'desire_q', 'packed_npy_inputs')}
    adapter.numpy_inputs = {'prev_feat': np.ones((2, 3)), 'desire': np.ones(8)}
    adapter.frame_slots = {'img': np.ones(12, np.uint8), 'big_img': np.ones(12, np.uint8)}
    adapter.reset_state()
    self.assertTrue(all(not tensor.numpy().any() for tensor in adapter.input_queues.values()))
    self.assertTrue(all(not array.any() for array in adapter.numpy_inputs.values()))
    self.assertTrue(all(not array.any() for array in adapter.frame_slots.values()))

  def test_explicit_model_reset_clears_desire_and_records_reason(self):
    state = object.__new__(ModelState)
    state.adapter = SimpleNamespace(reset_state=Mock())
    state.prev_desire = np.ones(8, np.float32)
    state.reset_state('route generation changed')
    state.adapter.reset_state.assert_called_once_with()
    self.assertFalse(state.prev_desire.any())
    self.assertEqual(state.state_reset_reason, 'route generation changed')
    self.assertEqual(state.state_reset_count, 1)
    state.reset_state('GPS invalid')
    self.assertEqual(state.state_reset_count, 2)
    invalid_reasons: tuple[Any, ...] = ('', ' ', None)
    for invalid in invalid_reasons:
      with self.subTest(reason=invalid), self.assertRaises(ValueError):
        state.reset_state(invalid)
    self.assertEqual(state.state_reset_count, 2)


class TestDiagnosticAdmission(OpenpilotTestCase):
  def test_constructor_rejects_without_capability_before_loading(self):
    with patch.object(modeld_module, 'diagnostic_session_token', return_value=None), \
         patch.object(modeld_module, 'diagnostic_session_backend', return_value=None), \
         patch.object(modeld_module, 'get_active_bundle') as bundle, \
         patch.object(modeld_module.ModelState, '_init_combined') as load:
      with self.assertRaisesRegex(modeld_module.NavigationModelDisabled, 'original active'):
        ModelState(1344, 760, chestnut=True, diagnostic=True)
    bundle.assert_not_called()
    load.assert_not_called()

  def test_constructor_binds_original_session(self):
    token = object()
    with patch.object(modeld_module, 'diagnostic_session_token', return_value=token), \
         patch.object(modeld_module, 'diagnostic_session_active', return_value=True), \
         patch.object(modeld_module, 'diagnostic_session_backend', return_value='AMD'), \
         patch.object(modeld_module, 'get_active_bundle', return_value=None), \
         patch.object(modeld_module, '_find_driving_pkl', return_value='trusted-synthetic-model'), \
         patch.object(modeld_module.ModelState, '_init_combined') as load:
      state = ModelState(1344, 760, chestnut=True, diagnostic=True)
    self.assertTrue(state.diagnostic)
    self.assertIs(state._diagnostic_token, token)
    load.assert_called_once_with('trusted-synthetic-model', 1344, 760, None)

  def test_backend_must_match_chestnut_before_loading(self):
    for backend, chestnut in (('CPU', True), ('AMD', False), ('QCOM', False), (None, False)):
      with self.subTest(backend=backend, chestnut=chestnut), \
           patch.object(modeld_module, 'diagnostic_session_token', return_value=object()), \
           patch.object(modeld_module, 'diagnostic_session_active', return_value=True), \
           patch.object(modeld_module, 'diagnostic_session_backend', return_value=backend), \
           patch.object(modeld_module.ModelState, '_init_combined') as load:
        with self.assertRaisesRegex(modeld_module.NavigationModelDisabled, 'Chestnut selection'):
          ModelState(1344, 760, chestnut=chestnut, diagnostic=True)
        load.assert_not_called()

  def test_replaced_closed_or_revoked_session_resets_before_inference(self):
    for changed_token, active in ((object(), True), (None, False), (None, True)):
      for has_navigation in (False, True):
        with self.subTest(active=active, navigation=has_navigation):
          state = object.__new__(ModelState)
          state.diagnostic = True
          state._diagnostic_token = object()
          state.numpy_inputs = {'navigation': np.ones(47, np.float32)} if has_navigation else {}
          state.navigation_input_specs = {}
          state.prev_desire = np.ones(8, np.float32)
          state.adapter = SimpleNamespace(reset_state=Mock(), copy_frames=Mock(), run=Mock())
          with patch.object(modeld_module, 'diagnostic_session_token', return_value=changed_token), \
               patch.object(modeld_module, 'diagnostic_session_active', return_value=active):
            with self.assertRaisesRegex(modeld_module.NavigationModelDisabled, 'original active'):
              state.run({}, {}, {})
          state.adapter.reset_state.assert_called_once_with()
          state.adapter.copy_frames.assert_not_called()
          state.adapter.run.assert_not_called()
          self.assertFalse(state.prev_desire.any())
          self.assertEqual(state.state_reset_reason, 'diagnostic permission revoked')

  def test_revoked_permission_with_original_token_resets(self):
    state = object.__new__(ModelState)
    state.diagnostic = True
    state._diagnostic_token = object()
    state.numpy_inputs = {}
    state.navigation_input_specs = {}
    state.prev_desire = np.ones(8, np.float32)
    state.adapter = SimpleNamespace(reset_state=Mock(), copy_frames=Mock(), run=Mock())
    with patch.object(modeld_module, 'diagnostic_session_token', return_value=state._diagnostic_token), \
         patch.object(modeld_module, 'diagnostic_session_active', return_value=False):
      with self.assertRaises(modeld_module.NavigationModelDisabled):
        state.run({}, {}, {})
    state.adapter.reset_state.assert_called_once_with()
    state.adapter.copy_frames.assert_not_called()

  def test_diagnostic_admission_is_explicit_and_never_published(self):
    state = object.__new__(ModelState)
    state.diagnostic = True
    state._diagnostic_token = object()
    state.numpy_inputs = {}
    with patch.object(modeld_module, 'diagnostic_session_token', return_value=state._diagnostic_token), \
         patch.object(modeld_module, 'diagnostic_session_active', return_value=True), \
         patch.object(modeld_module, 'validate_model_inputs', return_value=True) as validate:
      self.assertTrue(state._validate_navigation_inputs({'navigation': 'typed-spec'}))
      validate.assert_called_once_with({'navigation': 'typed-spec'}, diagnostic=True)
    with self.assertRaises(modeld_module.NavigationModelDisabled):
      modeld_module._reject_navigation_publication(state, None)
