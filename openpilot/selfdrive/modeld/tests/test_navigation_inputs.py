"""Native stock model input packing and admission without loading a model pickle or GPU."""
import ast
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import dtypes

from openpilot.selfdrive.modeld import modeld
from openpilot.selfdrive.navigation import model_inputs


class FakeParams:
  def __init__(self, enabled=True):
    self.enabled = enabled

  def get_bool(self, key):
    return self.enabled if key == 'NavigationModelInputsEnabled' else False


def input_specs(nav_dtype=None):
  specs: dict = {
    'new_img': ((2, 6, 128, 256), np.uint8, 'CPU'),
    'desire': ((8,), np.float32, 'CPU'),
    'traffic_convention': ((1, 2), np.float32, 'CPU'),
    'action_t': ((1, 2), np.float32, 'CPU'),
  }
  if nav_dtype is not None:
    specs |= {
      'nav_map': ((1, 1, 256, 256), nav_dtype, 'CPU'),
      'navigation': (((47,) if nav_dtype == np.uint8 else (1, 47)), np.float32, 'CPU'),
    }
  return specs


def packed_model(nav_dtype=None):
  model = object.__new__(modeld.ModelState)
  model._input_specs = input_specs(nav_dtype)
  model._navigation_input_names = tuple(name for name in model_inputs.NAV_INPUT_NAMES if name in model._input_specs)
  model.input_shapes = {name: (shape, np.dtype(dtype)) for name, (shape, dtype, _) in model._input_specs.items()}
  model.state_pairs = {}
  model.model_device = 'CPU'
  model.frame_copy_size = 64
  model.vision_input_names = ('img', 'big_img')
  model.prev_desire = np.zeros(8, np.float32)
  model.chestnut = False
  model.pack_inputs()
  model.run_warp = Mock(return_value=Tensor(np.zeros((2, 6, 128, 256), np.uint8)).realize())
  model.run_model = Mock()
  model.outputs = {'outputs': Tensor(np.array([[1., 2.]], np.float32)).realize()}
  model.output_slices = {'test': slice(0, 2)}
  model.parser = modeld.Parser(ignore_missing=True)
  return model


def frame_inputs():
  bufs = {name: SimpleNamespace(data=bytes(range(64))) for name in ('img', 'big_img')}
  transforms = {name: np.eye(3, dtype=np.float32) for name in bufs}
  inputs = {'desire_pulse': np.ones(8, np.float32), 'traffic_convention': np.array([1., 0.], np.float32),
            'action_t': np.array([0.2, 0.3], np.float32)}
  return bufs, transforms, inputs


class TestStockNavigationInputs(unittest.TestCase):
  def test_production_publication_rejects_active_or_fallback_navigation_models(self):
    ordinary = object.__new__(modeld.ModelState)
    ordinary._navigation_input_names = ()
    navigation = object.__new__(modeld.ModelState)
    navigation._navigation_input_names = ('nav_map', 'navigation')
    with patch.dict(os.environ, {'NAV_MODEL_OFFLINE': '1'}):
      modeld._require_non_navigation_publication(ordinary, None)
      modeld._require_non_navigation_publication(ordinary, ordinary)
      for active, fallback in ((navigation, None), (navigation, ordinary), (ordinary, navigation)):
        with self.subTest(active=active, fallback=fallback), self.assertRaises(model_inputs.NavigationModelDisabled):
          modeld._require_non_navigation_publication(active, fallback)

  def test_publication_guard_is_unconditional_and_precedes_pubmaster(self):
    # This boundary must apply to demo mode too and run before native publisher creation.
    main = ast.parse(inspect.getsource(modeld.main)).body[0]
    guards = [node for node in main.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
              isinstance(node.value.func, ast.Name) and node.value.func.id == '_require_non_navigation_publication']
    self.assertEqual(len(guards), 1)
    publications = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                    node.func.id == 'PubMaster']
    self.assertTrue(publications)
    self.assertLess(guards[0].lineno, min(node.lineno for node in publications))

  def test_recurrent_navigation_queues_survive_the_host_frame_upload(self):
    model = packed_model(np.uint8)
    states = {'state_map_q': ((32, 1, 256, 256), np.uint8, 'CPU'), 'state_nav_q': ((32, 47), np.float32, 'CPU')}
    model._input_specs |= states
    model.input_shapes |= {name: (shape, np.dtype(dtype)) for name, (shape, dtype, _) in states.items()}
    model.state_pairs = {name: 'next_' + name for name in states}
    model.pack_inputs()
    self.assertTrue(all(name not in model.npy for name in states))
    model.input_queues['state_map_q'].assign(23).realize()
    model.input_queues['state_nav_q'].assign(0.75).realize()
    with patch.dict(os.environ, {'NAV_MODEL_OFFLINE': '1'}), \
         patch.object(modeld, 'validate_model_inputs', side_effect=lambda specs: model_inputs.validate_model_inputs(specs, FakeParams())):
      model.run(*frame_inputs())
    self.assertTrue(np.all(model.input_queues['state_map_q'].numpy() == 23))
    self.assertTrue(np.all(model.input_queues['state_nav_q'].numpy() == 0.75))

  def test_uint8_and_float_map_packing_and_clear_on_missing_data(self):
    for dtype in (np.uint8, np.float32):
      with self.subTest(dtype=dtype), patch.dict(os.environ, {'NAV_MODEL_OFFLINE': '1'}), \
           patch.object(modeld, 'validate_model_inputs', side_effect=lambda specs: model_inputs.validate_model_inputs(specs, FakeParams())):
        model = packed_model(dtype)
        self.assertEqual(model.npy['nav_map'].dtype, dtype)
        self.assertEqual(model.input_queues['nav_map'].dtype, dtypes.uint8 if dtype == np.uint8 else dtypes.float32)
        for name, host_input in model.npy.items():
          self.assertEqual((host_input.ctypes.data - model.packed_input.ctypes.data) % 128, 0, name)
          self.assertEqual(model.input_queues.get(name, model.warp_inputs['M_inv']).dtype,
                           dtypes.uint8 if host_input.dtype == np.uint8 else dtypes.float32)
        bufs, transforms, inputs = frame_inputs()
        inputs['nav_map'] = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        inputs['navigation'] = np.zeros(47, np.float32)
        inputs['navigation'][15] = 1
        inputs['navigation'][-1] = 1.75
        after = Mock()
        result = model.run(bufs, transforms, inputs, after)
        np.testing.assert_array_equal(model.npy['nav_map'][0, 0], inputs['nav_map'])
        np.testing.assert_array_equal(model.input_queues['nav_map'].numpy()[0, 0], inputs['nav_map'])
        np.testing.assert_array_equal(model.npy['navigation'].reshape(-1), inputs['navigation'])
        np.testing.assert_array_equal(model.frames, np.tile(np.arange(64, dtype=np.uint8), (2, 1)))
        np.testing.assert_array_equal(result['test'], [[1., 2.]])
        after.assert_called_once()
        self.assertEqual(model.run_model.call_count, 1)
        del inputs['nav_map'], inputs['navigation']
        model.run(bufs, transforms, inputs)
        self.assertFalse(model.npy['nav_map'].any())
        self.assertFalse(model.npy['navigation'].any())
        self.assertFalse(model.input_queues['nav_map'].numpy().any())

  def test_gate_revocation_precedes_any_model_or_warp_execution(self):
    params = FakeParams()
    with patch.dict(os.environ, {'NAV_MODEL_OFFLINE': '1'}), \
         patch.object(modeld, 'validate_model_inputs', side_effect=lambda specs: model_inputs.validate_model_inputs(specs, params)):
      model = packed_model(np.uint8)
      model.run(*frame_inputs())
      old_bytes = model.packed_input.copy()
      params.enabled = False
      with self.assertRaises((ValueError, RuntimeError)):
        model.run(*frame_inputs())
      self.assertEqual(model.run_model.call_count, 1)
      self.assertEqual(model.run_warp.call_count, 1)
      np.testing.assert_array_equal(model.packed_input, old_bytes)

  def test_non_navigation_models_ignore_extra_navigation_and_never_read_gate_per_frame(self):
    model = packed_model()
    bufs, transforms, inputs = frame_inputs()
    with patch.object(modeld, 'validate_model_inputs', side_effect=AssertionError('ordinary models need no per-frame navigation gate')):
      before = model.run(bufs, transforms, inputs)
      inputs['nav_map'] = np.full((256, 256), 255, np.uint8)
      inputs['navigation'] = np.ones(47, np.float32)
      after = model.run(bufs, transforms, inputs)
    np.testing.assert_array_equal(before['test'], after['test'])
    self.assertNotIn('nav_map', model.npy)
    self.assertNotIn('navigation', model.npy)
    np.testing.assert_array_equal(model.npy['traffic_convention'], [[1., 0.]])
    np.testing.assert_allclose(model.npy['action_t'], [[0.2, 0.3]])

  def test_navigation_gate_precedes_model_initialization_and_warmup(self):
    specs = input_specs(np.uint8)
    with patch.object(modeld.ModelStateBase, '__init__', return_value=None), \
         patch.object(modeld, 'load_oob', return_value={'input_specs': specs}), \
         patch.object(modeld, 'validate_model_inputs', side_effect=RuntimeError('offline gate closed')) as validate, \
         patch.object(modeld.ModelState, 'pack_inputs') as pack, patch.object(modeld, 'lower_and_compile') as compile_model:
      with self.assertRaisesRegex(RuntimeError, 'offline gate closed'):
        modeld.ModelState(1928, 1208, False)
      validate.assert_called_once_with(specs)
      pack.assert_not_called()
      compile_model.assert_not_called()
