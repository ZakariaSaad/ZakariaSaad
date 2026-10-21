"""Independent checks of the offline gate and unchanged non-navigation model admission."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from openpilot.selfdrive.navigation import model_inputs


class ParamsStub:
  def __init__(self, **values):
    self.values = values
    self.reads = []

  def get_bool(self, key):
    self.reads.append(key)
    return self.values.get(key, False)


class TestExperimentalBoundaries(unittest.TestCase):
  def setUp(self):
    self.env = patch.dict(os.environ)
    self.env.start()
    self.addCleanup(self.env.stop)
    os.environ.pop('NAV_MODEL_OFFLINE', None)

  @staticmethod
  def nav_specs(dtype=np.uint8):
    return {'nav_map': ((1, 1, 256, 256), dtype, 'CPU'), 'navigation': ((47,), np.float32, 'CPU')}

  def test_agnos_rejects_every_authorization_combination(self):
    for explicit_env in (None, '', '0', '1', 'true'):
      for model_toggle in (False, True):
        for display_toggle in (False, True):
          with self.subTest(env=explicit_env, model=model_toggle, display=display_toggle), patch.object(model_inputs, 'PC', False):
            if explicit_env is None:
              os.environ.pop('NAV_MODEL_OFFLINE', None)
            else:
              os.environ['NAV_MODEL_OFFLINE'] = explicit_env
            params = ParamsStub(NavigationModelInputsEnabled=model_toggle, NavigationEnabled=display_toggle)
            self.assertFalse(model_inputs.offline_navigation_enabled(params))
            with self.assertRaises(model_inputs.NavigationModelDisabled):
              model_inputs.validate_model_inputs(self.nav_specs(), params)

  def test_native_parameter_default_is_disabled(self):
    from openpilot.common.params import Params
    from openpilot.common.prefix import OpenpilotPrefix
    with OpenpilotPrefix():
      params = Params()
      self.assertFalse(params.get_bool('NavigationModelInputsEnabled'))
      self.assertFalse(params.get_default_value('NavigationModelInputsEnabled'))
      params.put_bool('NavigationEnabled', True, block=True)
      self.assertFalse(params.get_bool('NavigationModelInputsEnabled'))
      self.assertFalse(model_inputs.offline_navigation_enabled(params))

  def test_pc_requires_explicit_environment_and_model_toggle(self):
    with patch.object(model_inputs, 'PC', True):
      for explicit_env in (None, '', '0', '1', 'true', 'yes'):
        for model_toggle in (False, True):
          with self.subTest(env=explicit_env, model=model_toggle):
            if explicit_env is None:
              os.environ.pop('NAV_MODEL_OFFLINE', None)
            else:
              os.environ['NAV_MODEL_OFFLINE'] = explicit_env
            params = ParamsStub(NavigationModelInputsEnabled=model_toggle)
            expected = explicit_env == '1' and model_toggle
            self.assertEqual(model_inputs.offline_navigation_enabled(params), expected)

  def test_navigation_display_toggle_does_not_authorize_model(self):
    os.environ['NAV_MODEL_OFFLINE'] = '1'
    with patch.object(model_inputs, 'PC', True):
      params = ParamsStub(NavigationEnabled=True, NavigationModelInputsEnabled=False)
      self.assertFalse(model_inputs.offline_navigation_enabled(params))
      with self.assertRaises(model_inputs.NavigationModelDisabled):
        model_inputs.validate_model_inputs(self.nav_specs(), params)

  def test_non_navigation_models_do_not_require_navigation_authorization(self):
    specs = {'action_t': ((1, 2), np.float32, 'CPU'), 'traffic_convention': ((1, 2), np.float32, 'CPU')}
    for pc in (False, True):
      with self.subTest(pc=pc), patch.object(model_inputs, 'PC', pc):
        params = ParamsStub()
        self.assertFalse(model_inputs.validate_model_inputs(specs, params))
        self.assertEqual(params.reads, [])

  def test_both_published_map_dtypes_admitted_only_in_offline_mode(self):
    os.environ['NAV_MODEL_OFFLINE'] = '1'
    with patch.object(model_inputs, 'PC', True):
      params = ParamsStub(NavigationModelInputsEnabled=True)
      for dtype in (np.uint8, np.float32):
        with self.subTest(dtype=dtype):
          self.assertTrue(model_inputs.validate_model_inputs(self.nav_specs(dtype), params))

  def test_unsupported_nav_tensor_contract_fails_closed(self):
    os.environ['NAV_MODEL_OFFLINE'] = '1'
    with patch.object(model_inputs, 'PC', True):
      params = ParamsStub(NavigationModelInputsEnabled=True)
      invalid = [
        {'nav_map': ((1, 1, 128, 128), np.uint8, 'CPU')},
        {'nav_map': ((1, 1, 256, 256), np.float64, 'CPU')},
        {'navigation': ((46,), np.float32, 'CPU')},
        {'navigation': ((47,), np.float64, 'CPU')},
      ]
      for specs in invalid:
        with self.subTest(specs=specs), self.assertRaises(ValueError):
          model_inputs.validate_model_inputs(specs, params)

  def test_revocation_invalidates_a_previously_admitted_model(self):
    os.environ['NAV_MODEL_OFFLINE'] = '1'
    with patch.object(model_inputs, 'PC', True):
      params = ParamsStub(NavigationModelInputsEnabled=True)
      self.assertTrue(model_inputs.validate_model_inputs(self.nav_specs(), params))
      params.values['NavigationModelInputsEnabled'] = False
      with self.assertRaises(model_inputs.NavigationModelDisabled):
        model_inputs.validate_model_inputs(self.nav_specs(), params)
      params.values['NavigationModelInputsEnabled'] = True
      os.environ.pop('NAV_MODEL_OFFLINE')
      with self.assertRaises(model_inputs.NavigationModelDisabled):
        model_inputs.validate_model_inputs(self.nav_specs(), params)

  def test_production_entrypoints_reject_before_constructing_publishers(self):
    # Execute the actual main guard statement followed by the actual PubMaster
    # assignment, with only publisher construction replaced. This does not load
    # a model, start a process, or open any messaging socket.
    from openpilot.common.basedir import BASEDIR
    cases = (
      ('openpilot/selfdrive/modeld/modeld.py', '_require_non_navigation_publication'),
      ('openpilot/sunnypilot/modeld_v2/modeld.py', '_reject_navigation_publication'),
    )
    for path, guard_name in cases:
      with self.subTest(path=path):
        module = ast.parse((Path(BASEDIR) / path).read_text())
        main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        guards = [(index, node) for index, node in enumerate(main.body)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                  isinstance(node.value.func, ast.Name) and node.value.func.id == guard_name]
        pubs = [(index, node) for index, node in enumerate(main.body)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Name) and node.value.func.id == 'PubMaster']
        self.assertEqual(len(guards), 1, 'all main/demo paths must pass an unconditional publication guard')
        self.assertEqual(len(pubs), 1)
        self.assertLess(guards[0][0], pubs[0][0])
        definitions = [copy.deepcopy(node) for node in module.body if isinstance(node, ast.FunctionDef) and
                       node.name in ('takes_nav', guard_name)]
        for definition in definitions:
          definition.returns = None
          for argument in (*definition.args.args, *definition.args.kwonlyargs):
            argument.annotation = None
          if definition.args.vararg is not None:
            definition.args.vararg.annotation = None
        probe = ast.FunctionDef(name='probe', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='model'), ast.arg(arg='small_model')],
                                                               kwonlyargs=[], kw_defaults=[], defaults=[]),
                                body=[copy.deepcopy(guards[0][1]), copy.deepcopy(pubs[0][1])], decorator_list=[], type_params=[])
        executable = ast.fix_missing_locations(ast.Module(body=[*definitions, probe], type_ignores=[]))
        publisher = MagicMock()
        namespace: dict[str, Any] = {'NAV_INPUT_NAMES': model_inputs.NAV_INPUT_NAMES,
                     'NavigationModelDisabled': model_inputs.NavigationModelDisabled,
                     'PubMaster': publisher, 'pub_socks': ['modelV2']}
        exec(compile(executable, path, 'exec'), namespace)
        ordinary = SimpleNamespace(numpy_inputs={}, _navigation_input_names=())
        nav_map = SimpleNamespace(numpy_inputs={'nav_map': None}, _navigation_input_names=('nav_map',))
        maneuvers = SimpleNamespace(numpy_inputs={'navigation': None}, _navigation_input_names=('navigation',))
        for active, fallback in ((nav_map, None), (ordinary, nav_map), (maneuvers, ordinary)):
          for pc in (False, True):
            with self.subTest(active=active, fallback=fallback, pc=pc), patch.object(model_inputs, 'PC', pc):
              os.environ['NAV_MODEL_OFFLINE'] = '1'
              with self.assertRaises(model_inputs.NavigationModelDisabled):
                namespace['probe'](active, fallback)
              publisher.assert_not_called()
        namespace['probe'](ordinary, None)
        publisher.assert_called_once_with(['modelV2'])
