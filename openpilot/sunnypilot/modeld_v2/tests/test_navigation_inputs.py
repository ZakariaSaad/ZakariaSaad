"""Navigation-model plumbing with trusted synthetic kernels and CPU/NPY buffers only."""
import ast
import codecs
import inspect
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2 import frame_fusion, modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.frame_fusion import pack_layout
from openpilot.sunnypilot.modeld_v2.model_adapters import NativeTinygradAdapter
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser

FIXTURES = Path(__file__).resolve().parents[3] / 'selfdrive/navigation/tests/fixtures/nav_models.json'
DTYPES = {'UINT8': np.uint8, 'FLOAT': np.float32}


def signature_specs(items):
  return {item['name']: (tuple(item['shape']), np.dtype(DTYPES[item['dtype']]), 'NPY') for item in items}


def native_fixture(signature):
  # This pickle is generated from the checked-in JSON slice bounds, never downloaded model data.
  slices = {name: slice(value['start'], value['stop'], value['step']) for name, value in signature['output_slices'].items()}
  return {
    'input_specs': signature_specs(signature['inputs']),
    'output_specs': signature_specs(signature['outputs']),
    'metadata': {'output_shapes': {item['name']: tuple(item['shape']) for item in signature['outputs']},
                 'metadata': {'output_slices': codecs.encode(pickle.dumps(slices), 'base64').decode()}},
    'run': Mock(),
  }


class TestNavigationModelInputs(OpenpilotTestCase):
  def test_native_uint8_and_float32_packing(self):
    signatures = json.loads(FIXTURES.read_text())['models']
    source = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
    for signature in signatures:
      with self.subTest(model=signature['tag']):
        adapter = NativeTinygradAdapter(native_fixture(signature), 1344, 760, 'NPY', 'NPY', 'NPY')
        adapter.numpy_inputs['nav_map'][:] = source
        vector = np.zeros(47, np.float32)
        vector[15], vector[-1] = 1, 2.5
        adapter.numpy_inputs['navigation'][:] = vector
        np.testing.assert_array_equal(adapter.numpy_inputs['nav_map'].reshape(256, 256), source)
        np.testing.assert_array_equal(adapter.numpy_inputs['navigation'].reshape(47), vector)
        expected = DTYPES[next(item['dtype'] for item in signature['inputs'] if item['name'] == 'nav_map')]
        self.assertEqual(adapter.numpy_inputs['nav_map'].dtype, np.dtype(expected))
        self.assertEqual(len(adapter.state_pairs), 5)
        for name, next_name in adapter.state_pairs.items():
          self.assertEqual(adapter.input_queues[name].dtype, adapter.outputs[next_name].dtype)
          self.assertEqual(adapter.input_queues[name].shape, adapter.outputs[next_name].shape)
        self.assertFalse(adapter.frames.any())
        adapter.jits['run'].assert_not_called()

  def test_fused_upload_uses_the_same_typed_offsets(self):
    for map_dtype in (np.dtype('uint8'), np.dtype('float32')):
      with self.subTest(dtype=map_dtype):
        dtype_name = 'dtypes.uint8' if map_dtype == np.uint8 else 'dtypes.float32'
        model = {'inputs': [
          {'slot': 0, 'name': 'new_img', 'dtype': 'dtypes.uint8', 'numel': 48, 'device': 'NPY'},
          {'slot': 1, 'name': 'nav_map', 'dtype': dtype_name, 'numel': 65536, 'device': 'NPY'},
          {'slot': 2, 'name': 'navigation', 'dtype': 'dtypes.float32', 'numel': 47, 'device': 'NPY'},
          {'slot': 3, 'name': None, 'dtype': 'dtypes.float32', 'numel': 10, 'device': 'NPY'}],
          'buffers': [], 'calls': [{'kind': 'kernel', 'args': [{'input': 1}, {'input': 2}]}]}
        warp = {'inputs': [{'slot': 0, 'name': 'input_frame', 'numel': 2000, 'dtype': 'dtypes.uint8'},
                           {'slot': 1, 'name': 'tfm', 'numel': 18, 'dtype': 'dtypes.float32'}],
                'returns': {'tensors': [{'buffer': 0, 'dtype': 'dtypes.uint8', 'offset': 0, 'nbytes': 48, 'shape': [1, 12, 2, 2]}]},
                'buffers': [{'nbytes': 48, 'device': 'NPY', 'dtype': 'dtypes.uint8'}], 'calls': []}
        specs = {'new_img': ((1, 12, 2, 2), np.uint8, 'NPY'), 'nav_map': ((1, 1, 256, 256), map_dtype, 'NPY'),
                 'navigation': ((1, 47), np.float32, 'NPY')}
        extras = {'input_specs': specs, 'metadata': {'output_shapes': {'outputs': (1, 10)}}}
        with patch.object(frame_fusion, 'read_parts', return_value=({'run': (warp, b'warp')}, {})):
          fused = frame_fusion.frame_part('synthetic-warp', {'run': (model, b'model')}, extras)['run'][0]
        offsets, total = pack_layout({'nav_map': ((1, 1, 256, 256), map_dtype), 'navigation': ((1, 47), np.float32)}, 1000)
        self.assertEqual(fused['inputs'][0]['numel'], total)
        image, vector = fused['calls'][-1]['args']
        self.assertEqual((image['offset'], image['nbytes'], image['dtype']), (offsets['nav_map'], 65536 * map_dtype.itemsize, dtype_name))
        self.assertEqual((vector['offset'], vector['nbytes']), (offsets['navigation'], 188))

  def test_explicit_scratch_warp_and_default_asset_selection(self):
    scratch_path = '/tmp/offline-only-cpu-warp.pkl'
    overridden = frame_fusion.fused_with_warp(1344, 760, False, warp_path=scratch_path)
    self.assertIs(overridden.func, frame_fusion.frame_part)
    self.assertEqual(overridden.args, (scratch_path,))
    for chestnut in (False, True):
      default = frame_fusion.fused_with_warp(1344, 760, chestnut)
      prefix = 'big_' if chestnut else ''
      self.assertEqual(default.args, (f'{frame_fusion.BASEDIR}/openpilot/sunnypilot/modeld_v2/models/{prefix}driving_warp_1344x760_tinygrad.pkl',))

  def test_modelstate_forwards_only_explicit_warp_override(self):
    with patch.object(modeld_module, 'get_active_bundle', return_value=None), \
         patch.object(modeld_module, '_find_driving_pkl', return_value='synthetic-driving-model'), \
         patch.object(modeld_module.ModelState, '_init_combined') as init:
      modeld_module.ModelState(1344, 760)
      init.assert_called_once_with('synthetic-driving-model', 1344, 760, None)
      init.reset_mock()
      modeld_module.ModelState(1344, 760, warp_path='/tmp/offline-cpu-warp.pkl')
      init.assert_called_once_with('synthetic-driving-model', 1344, 760, None, warp_path='/tmp/offline-cpu-warp.pkl')

  def test_existing_float32_layout_is_unchanged(self):
    offsets, total = pack_layout({'desire': ((1, 8), np.float32), 'traffic_convention': ((1, 2), np.float32),
                                  'action_t': ((1, 2), np.float32)}, 1000)
    self.assertEqual(offsets, {'tfm': 0, 'desire': 128, 'traffic_convention': 256, 'action_t': 384, 'frames': 512})
    self.assertEqual(total, 2512)

  def test_invalidated_navigation_inputs_are_zeroed(self):
    state = object.__new__(modeld_module.ModelState)
    state.numpy_inputs = {'desire': np.zeros(8, np.float32), 'tfm': np.zeros((2, 3, 3), np.float32),
                          'nav_map': np.zeros((1, 1, 256, 256), np.uint8), 'navigation': np.zeros((1, 47), np.float32)}
    state.navigation_input_specs = {'nav_map': ((1, 1, 256, 256), np.uint8, 'NPY'), 'navigation': ((1, 47), np.float32, 'NPY')}
    state.adapter = SimpleNamespace(is_native=True, copy_frames=Mock(), run=Mock(return_value=SimpleNamespace(numpy=lambda: np.zeros(1, np.float32))))
    state._desire_key = 'desire'
    state.prev_desire = np.zeros(8, np.float32)
    state._vision_input_names = ['img', 'big_img']
    state._combined_model_type = 'supercombo'
    state.vision_output_slices = {}
    state.chestnut = False
    state.parser = Parser()
    vector = np.zeros(47, np.float32)
    vector[15] = 1
    with patch.object(modeld_module, 'validate_model_inputs', return_value=True) as permit:
      state.run({}, {}, {'desire': np.zeros(8, np.float32), 'nav_map': np.full((256, 256), 255, np.uint8), 'navigation': vector})
      self.assertTrue(state.numpy_inputs['nav_map'].all())
      self.assertEqual(state.numpy_inputs['navigation'][0, 15], 1)
      state.run({}, {}, {'desire': np.zeros(8, np.float32)})
      self.assertFalse(state.numpy_inputs['nav_map'].any())
      self.assertFalse(state.numpy_inputs['navigation'].any())
      self.assertEqual(permit.call_count, 2)
    copies, runs = state.adapter.copy_frames.call_count, state.adapter.run.call_count
    with patch.object(modeld_module, 'validate_model_inputs', side_effect=PermissionError('offline permit withdrawn')):
      with self.assertRaises(PermissionError):
        state.run({}, {}, {'desire': np.zeros(8, np.float32)})
    self.assertEqual(state.adapter.copy_frames.call_count, copies)
    self.assertEqual(state.adapter.run.call_count, runs)

  def test_preflight_rejects_before_warp_or_gpu_loading(self):
    state = object.__new__(modeld_module.ModelState)
    state.chestnut = True
    extras = {'input_specs': {'nav_map': ((1, 1, 256, 256), np.uint8, 'AMD')}}
    fuse = Mock()
    allocated = Mock()

    def load_before_gpu(path, *, extend):
      extend({}, extras)
      allocated()
      raise AssertionError('GPU loading must not be reached')

    with patch.object(modeld_module, 'load_model', side_effect=load_before_gpu), \
         patch.object(modeld_module, 'fused_with_warp', return_value=fuse), \
         patch.object(modeld_module, 'validate_model_inputs', side_effect=PermissionError('disabled')):
      with self.assertRaises(PermissionError):
        state._init_combined('not-a-real-model', 1344, 760, None)
    fuse.assert_not_called()
    allocated.assert_not_called()

  def test_production_publication_rejects_active_or_fallback_nav_model(self):
    standard = SimpleNamespace(numpy_inputs={'desire': np.zeros(8)})
    navigation = SimpleNamespace(numpy_inputs={'nav_map': np.zeros((1, 1, 256, 256), np.uint8)})
    modeld_module._reject_navigation_publication(standard, None)
    for models in ((navigation, standard), (standard, navigation)):
      with self.subTest(nav_model=models[0] is navigation):
        with self.assertRaisesRegex(modeld_module.NavigationModelDisabled, 'branch-reviewed runtime permit'):
          modeld_module._reject_navigation_publication(*models)
    # This safety boundary must remain ahead of any production publisher, including demo mode.
    tree = ast.parse(inspect.getsource(modeld_module.main))
    main = tree.body[0]
    self.assertIsInstance(main, ast.FunctionDef)
    assert isinstance(main, ast.FunctionDef)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    # Check the unconditional main-startup guard, separately from the additional
    # post-warmup check inside the asynchronous big-model loader.
    # Startup's try/finally now clears ChestnutLoading on every exit. Its direct
    # body is unconditional on successful startup; nested loader functions and
    # conditional branches must not satisfy this publication guard assertion.
    startup = [*main.body, *(statement for node in main.body if isinstance(node, ast.Try) for statement in node.body)]
    guards = [node.lineno for node in startup if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == '_reject_navigation_publication']
    publishers = [node.lineno for node in calls if node.func.id == 'PubMaster']
    self.assertEqual(len(guards), 1)
    self.assertTrue(publishers)
    self.assertLess(guards[0], min(publishers))

  def test_legacy_nav_contract_rejected_non_navigation_unchanged(self):
    self.assertEqual(modeld_module._navigation_input_specs({'metadata': {'model': {'input_shapes': {'desire': (1, 8)}}}}), {})
    with self.assertRaisesRegex(ValueError, 'native input_specs'):
      modeld_module._navigation_input_specs({'metadata': {'model': {'input_shapes': {'nav_map': (1, 1, 256, 256)}}}})
    self.assertFalse(modeld_module.takes_nav(SimpleNamespace(numpy_inputs={'desire': np.zeros(8)}), None))
    self.assertTrue(modeld_module.takes_nav(SimpleNamespace(numpy_inputs={'desire': np.zeros(8)}),
                                          SimpleNamespace(numpy_inputs={'navigation': np.zeros(47)})))
