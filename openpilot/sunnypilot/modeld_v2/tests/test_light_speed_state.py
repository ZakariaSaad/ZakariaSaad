"""Light Speed-only state ownership and actual CPU PlainJit buffer rebinding.

The reference ModelState is an unchanged excerpt of comma.ai's MIT-licensed
b20a8e4017525febac7d8810a964e7b505f9861d (full source blob
42861548f1b3b188f523ccbf4aa9e4c6557a994b). Only its external artifact/warp
functions are replaced with a trusted tiny synthetic CPU graph. No model
weights, AMD hardware, messaging or vehicle processes are used.
"""
import ast
import base64
from collections.abc import Callable
from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up

from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.helpers import PlainJit, _Jit, cache_is_fresh, load_model, write_bundle
from openpilot.sunnypilot.modeld_v2.model_adapters import NativeTinygradAdapter, input_view
from openpilot.sunnypilot.modeld_v2.tests.test_plain_artifact import compile_jit
from openpilot.sunnypilot.modeld_v2.tests.test_driving_navigation import fixture_data
from openpilot.selfdrive.navigation.tests.test_activation import reviewed_fixture


LIGHT_SPEED = 'e4acf619d4eb8b726fc405c264109310ac2951df'
REFERENCE = Path(__file__).with_name('fixtures') / 'light_speed_b20_modelstate.txt'
STATE_SPECS = {'state_a': ((1, 4), np.float32, 'CPU'), 'state_b': ((1, 4), np.float16, 'CPU')}
SLICES = {'action': slice(0, 4)}


def native_jits(run=None):
  inputs = {'new_img': ((1,), np.float32, 'CPU'), 'desire': ((1, 8), np.float32, 'CPU'),
            'traffic_convention': ((1, 2), np.float32, 'CPU'), 'action_t': ((1, 2), np.float32, 'CPU'), **STATE_SPECS}
  outputs = {'outputs': ((1, 4), np.float32, 'CPU'), **{f'next_{name}': spec for name, spec in STATE_SPECS.items()}}
  return {'input_specs': inputs, 'output_specs': outputs, 'run': run,
          'metadata': {'output_shapes': {name: spec[0] for name, spec in outputs.items()},
                       'metadata': {'output_slices': base64.b64encode(pickle.dumps(SLICES)).decode()}}}


def make_model(ref=LIGHT_SPEED, *, chestnut=True, jits=None):
  bundle = None if ref is None else SimpleNamespace(ref=ref, generation=12, overrides=[], is20hz=True)
  with patch.object(modeld, 'get_active_bundle', return_value=bundle), \
       patch.object(modeld, '_find_driving_pkl', return_value='synthetic-local.sptg'), \
       patch.object(modeld, 'load_model', return_value=jits or native_jits()):
    state = modeld.ModelState(1344, 760, chestnut=chestnut)
  state.parser = SimpleNamespace(parse_outputs=lambda values: values)
  return state


def raw_synthetic(host, state_a, state_b, output_buffers):
  # Writes next states before reading original state into the primary output.
  # Separate buffers preserve the original inputs; aliasing changes the result.
  drive = host[:1].to('CPU').cast(dtypes.float32) + 1
  output_buffers['next_state_a'].assign(state_a + drive).realize()
  output_buffers['next_state_b'].assign((state_b.float() * .5 + state_a * 2 + drive).cast(dtypes.float16)).realize()
  output_buffers['outputs'].assign(state_a * 3 + state_b.float()).realize()


@pytest.fixture(scope='module')
def plain_part():
  adapter = NativeTinygradAdapter(native_jits(), 1344, 760, 'CPU', 'CPU', 'CPU', separate_recurrent_buffers=True)
  arguments = {'host': adapter.host, **adapter.input_queues, 'output_buffers': adapter.outputs}
  # Compile only this tiny trusted test graph, not a driving model/artifact.
  return compile_jit(raw_synthetic, arguments, 'output_buffers')


def official_state(jits, graph):
  source = REFERENCE.read_text()
  assert hashlib.sha256(source.encode()).hexdigest() == '347654ee8a68a0c1a032027a685c2b4f23b802e14ea9831e281553be960e2edd'
  namespace = {'np': np, 'Tensor': Tensor, 'math': math, 'round_up': round_up,
               'input_view': input_view, 'dtypes': dtypes, 'VisionBuf': object,
               'Callable': Callable, 'SEND_RAW_PRED': False,
               'ModelConstants': SimpleNamespace(DESIRE_LEN=8)}
  exec(compile(source, 'official_b20_ModelState', 'exec'), namespace)
  expected = object.__new__(namespace['ModelState'])
  expected.model_device = 'CPU'
  expected.input_shapes = {name: (shape, np.dtype(dtype)) for name, (shape, dtype, _) in jits['input_specs'].items()}
  expected.state_pairs = {name: f'next_{name}' for name in STATE_SPECS}
  expected.frame_copy_size = 1622016
  expected.vision_input_names = ('img', 'big_img')
  expected.prev_desire = np.zeros(8, np.float32)
  expected.chestnut = True
  expected.output_slices = SLICES
  expected.pack_inputs()
  # Execute the upstream allocation expression, not a replacement alias policy.
  init = next(node for node in ast.parse(source).body[0].body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
  allocation = next(node for node in init.body if isinstance(node, ast.Assign) and
                    isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == 'outputs')
  exec(compile(ast.Module(body=[allocation], type_ignores=[]), 'official_b20_outputs', 'exec'),
       {**namespace, 'self': expected, 'jits': jits})
  expected.parser = SimpleNamespace(parse_outputs=lambda values: values)
  expected.run_warp = lambda **_: Tensor.zeros(1, device='CPU').realize()
  # The official runner uploads/warps separately, while the native adapter has
  # an already fused host argument. Both pass identical recurrence and host bytes.
  expected_host = Tensor(expected.packed_input, device='NPY')
  expected.run_model = lambda output_buffers, **inputs: graph(host=expected_host, output_buffers=output_buffers,
                                                             **{name: inputs[name] for name in STATE_SPECS})
  return expected, namespace


def frame(adapter):
  bufs, transforms, inputs = adapter.get_dummy_inputs()
  wrapped = {name: SimpleNamespace(data=value) for name, value in bufs.items()}
  inputs['action_t'][:] = [.175, .575]
  return wrapped, transforms, inputs


def pair(plain_part, runner_factory=None):
  factory = runner_factory or (lambda: PlainJit(*plain_part))
  actual = make_model(jits=native_jits(factory()))
  expected, namespace = official_state(native_jits(), factory())
  errors = [0]
  devices = {'CPU': SimpleNamespace(error_state=SimpleNamespace(host=SimpleNamespace(view=lambda **_: errors)))}
  namespace['Device'] = devices
  return actual, expected, devices, errors


def run_both(actual, expected):
  bufs, transforms, inputs = frame(actual.adapter)
  wanted_inputs = {'desire_pulse': inputs['desire'].reshape(-1).copy(),
                   'traffic_convention': inputs['traffic_convention'], 'action_t': inputs['action_t']}
  wanted = expected.run(bufs, transforms, wanted_inputs)
  got = actual.run(bufs, transforms, inputs)
  np.testing.assert_array_equal(got['action'], wanted['action'])
  for name, next_name in actual.adapter.state_pairs.items():
    np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), expected.input_queues[name].numpy())
    np.testing.assert_array_equal(actual.adapter.outputs[next_name].numpy(), expected.outputs[next_name].numpy())


def pointers(state):
  return {name: (tensor._buffer().base, state.outputs[next_name]._buffer().base)
          for name, next_name in state.state_pairs.items() for tensor in [state.input_queues[name]]}


@pytest.mark.parametrize('cached', [False, True])
def test_plainjit_state_sequence_matches_official_and_dynamically_rebinds(plain_part, tmp_path, cached):
  factory = None
  if cached:
    path = str(tmp_path / 'trusted-small-synthetic.sptg')
    write_bundle(path, {'run': plain_part})
    assert isinstance(load_model(path)['run'], PlainJit)
    assert cache_is_fresh(path)
    def factory():
      return load_model(path)['run']
  actual, expected, devices, _ = pair(plain_part, factory)
  original = pointers(actual.adapter)
  assert all(a is not b for a, b in original.values())
  with patch.object(modeld, 'Device', devices):
    for index in range(8):
      run_both(actual, expected)
      current = pointers(actual.adapter)
      for name, (a, b) in original.items():
        assert current[name] == ((b, a) if index % 2 == 0 else (a, b))
      assert type(actual.adapter.run_model) is (_Jit if cached else PlainJit)
      np.testing.assert_array_equal(actual.adapter.input_queues['state_a'].numpy(), np.full((1, 4), index + 1, np.float32))


def test_initial_warmup_and_explicit_reset_preserve_separate_zero_history(plain_part):
  actual, expected, devices, _ = pair(plain_part)
  initial = pointers(actual.adapter)
  for name, next_name in actual.adapter.state_pairs.items():
    np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), 0)
    np.testing.assert_array_equal(actual.adapter.outputs[next_name].numpy(), 0)
  with patch.object(modeld, 'Device', devices):
    actual.warmup()
    expected.warmup()
    for name, next_name in actual.adapter.state_pairs.items():
      assert pointers(actual.adapter)[name] == initial[name][::-1]
      np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), expected.input_queues[name].numpy())
      np.testing.assert_array_equal(actual.adapter.outputs[next_name].numpy(), expected.outputs[next_name].numpy())
    run_both(actual, expected)
    run_both(actual, expected)
    before_reset = pointers(actual.adapter)
    actual.reset_state('isolated test reset')
    assert pointers(actual.adapter) == before_reset
    for tensor in (*actual.adapter.input_queues.values(), *actual.adapter.outputs.values()):
      np.testing.assert_array_equal(tensor.numpy(), 0)
    assert not actual.prev_desire.any()
    expected, namespace = official_state(native_jits(), PlainJit(*plain_part))
    namespace['Device'] = devices
    run_both(actual, expected)
    # Warmup input reset is upstream's reset: no requirement to clear dirty
    # scratch outputs after repeated warmups is invented here.


@pytest.mark.parametrize('failure', ['graph', 'device', 'callback', 'readback', 'nonfinite', 'parser'])
def test_failure_swap_boundary_matches_official(plain_part, failure):
  actual, expected, devices, errors = pair(plain_part)
  with patch.object(modeld, 'Device', devices), ExitStack() as stack:
    run_both(actual, expected)
    before = pointers(actual.adapter)
    wanted_before = pointers(expected)
    incoming = {name: tensor.numpy().copy() for name, tensor in actual.adapter.input_queues.items()}
    if failure == 'device':
      errors[0] = 7
    elif failure == 'readback':
      for owner in (actual.adapter, expected):
        stack.enter_context(patch.object(owner.outputs['outputs'], 'numpy', side_effect=RuntimeError('synthetic readback failure')))
    elif failure == 'parser':
      for owner in (actual, expected):
        owner.parser.parse_outputs = lambda _: (_ for _ in ()).throw(RuntimeError('synthetic parser failure'))
    elif failure == 'graph':
      for owner in (actual.adapter, expected):
        original = owner.run_model
        def failed_graph(*args, original=original, **kwargs):
          original(*args, **kwargs)
          raise RuntimeError('synthetic graph failure')
        owner.run_model = failed_graph
    def callback(owner):
      if failure == 'callback':
        raise RuntimeError('synthetic callback failure')
      if failure == 'nonfinite':
        owner.outputs['outputs'].assign(float('nan')).realize()
    bufs, transforms, inputs = frame(actual.adapter)
    wanted_inputs = {'desire_pulse': inputs['desire'].reshape(-1).copy(),
                     'traffic_convention': inputs['traffic_convention'], 'action_t': inputs['action_t']}
    with pytest.raises(RuntimeError):
      expected.run(bufs, transforms, wanted_inputs, lambda: callback(expected))
    with pytest.raises(RuntimeError):
      actual.run(bufs, transforms, inputs, lambda: callback(actual.adapter))
    for name in actual.adapter.state_pairs:
      assert pointers(actual.adapter)[name] == (before[name][::-1] if failure == 'parser' else before[name])
      assert pointers(expected)[name] == (wanted_before[name][::-1] if failure == 'parser' else wanted_before[name])
      np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), expected.input_queues[name].numpy())
      if failure != 'parser':
        np.testing.assert_array_equal(actual.adapter.input_queues[name].numpy(), incoming[name])


@pytest.mark.parametrize('ref,chestnut,separate', [(LIGHT_SPEED, True, True), (LIGHT_SPEED, False, False),
                                                (LIGHT_SPEED + '-other', True, False), ('mountain-dew', True, False),
                                                ('other-ordinary', True, False), (None, True, False)])
def test_exact_selected_big_model_ref_is_only_opt_in(ref, chestnut, separate):
  state = make_model(ref, chestnut=chestnut)
  before = pointers(state.adapter)
  assert all((a is not b) == separate for a, b in before.values())
  state.adapter.advance_state()
  for name, pair_before in before.items():
    assert pointers(state.adapter)[name] == (pair_before[::-1] if separate else pair_before)


def test_corrected_bigv2_all_state_aliases_remain_unchanged():
  jits, metadata = fixture_data()
  with reviewed_fixture(jits['input_specs'], metadata) as (_, permit, _), patch.object(modeld, 'load_model', return_value=jits):
    state = modeld.NavigationModelState(1344, 760, chestnut=True, driving_permit=permit)
  before = pointers(state.adapter)
  assert len(before) == 5
  assert all(a is b for a, b in before.values())
  state.adapter.advance_state()
  assert pointers(state.adapter) == before
