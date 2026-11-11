"""Fixed Fidx input plumbing through real packed buffers and a synthetic graph.

The catalog Fidx model declares float32 (1, 1), verified from the compiled
artifact at upstream 3471915de772c645d60a563cc3c335b17da1fe0b. Ordinary scalar
shapes below match that artifact; images, outputs and recurrent state are tiny
test fixtures. CPU/NPY storage and mocked graph dispatch do not exercise AMD
hardware, driving model weights, cameras, publishers, or vehicle interfaces.
"""
import ast
import base64
import inspect
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.frame_fusion import pack_layout
from openpilot.sunnypilot.modeld_v2.model_adapters import NativeTinygradAdapter


def make_model(*, fidx):
  inputs = {'new_img': ((1,), np.uint8, 'CPU'), 'desire': ((8,), np.float32, 'CPU'),
            'traffic_convention': ((1, 2), np.float32, 'CPU'), 'action_t': ((1, 2), np.float32, 'CPU')}
  if fidx:
    inputs['fidx'] = ((1, 1), np.float32, 'CPU')
  inputs['state_history'] = ((1, 4), np.float32, 'CPU')
  outputs = {'outputs': ((1, 4), np.float32, 'CPU'), 'next_state_history': inputs['state_history']}
  # This literal trusted slice metadata is constructed locally, not downloaded.
  slices = {'action': slice(0, 4)}
  captures = []

  def graph(*, host, output_buffers, **states):
    # Inspect the actual packed tensor handed to graph dispatch, not the input
    # dictionary or a replacement ModelState/adapter.run implementation.
    packed = host.numpy().copy()
    values = {name: np.ndarray(shape, dtype=dtype, buffer=packed, offset=offsets[name]).copy()
              for name, (shape, dtype) in specs.items()}
    captures.append(SimpleNamespace(packed=packed, values=values,
                                    history=states['state_history'].numpy().copy()))

  jits = {'input_specs': inputs, 'output_specs': outputs, 'run': graph,
          'metadata': {'output_shapes': {name: spec[0] for name, spec in outputs.items()},
                       'metadata': {'output_slices': base64.b64encode(pickle.dumps(slices)).decode()}}}
  adapter = NativeTinygradAdapter(jits, 1344, 760, 'CPU', 'CPU', 'CPU')
  specs = {name: (shape, dtype) for name, (shape, dtype, _) in inputs.items()
           if name not in adapter.state_pairs and name != 'new_img'}
  offsets, _ = pack_layout(specs, adapter.warp_frame_size)
  state = object.__new__(modeld.ModelState)
  state.adapter = adapter
  state.numpy_inputs = adapter.numpy_inputs
  state._desire_key = adapter._desire_key
  state._vision_input_names = adapter._vision_input_names
  state.prev_desire = np.zeros(8, np.float32)
  state.chestnut = False
  state._combined_model_type = 'supercombo'
  state.vision_output_slices = slices
  state.parser = SimpleNamespace(parse_outputs=lambda values: values)
  return state, captures


def frame_inputs(state):
  # Execute main's actual per-frame input-building block with inert upstream
  # values. This covers its producer without starting cameras or messaging, and
  # feeds the result through the actual ModelState and native adapter below.
  main = ast.parse(inspect.getsource(modeld.main)).body[0]
  loop = next(node for node in main.body if isinstance(node, ast.While) and
              any(isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.target.id == 'inputs'
                  for child in node.body))
  start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.AnnAssign) and
               isinstance(node.target, ast.Name) and node.target.id == 'inputs')
  end = next(i for i, node in enumerate(loop.body[start:], start) if isinstance(node, ast.Assign) and
             any(isinstance(target, ast.Name) and target.id == 'mt1' for target in node.targets))
  namespace = {'np': np, 'model': state, 'vec_desire': np.zeros(8, np.float32),
               'traffic_convention': np.array([0., 1.], np.float32), 'v_ego': 21., 'lat_delay': .15,
               'lat_action_t': .225, 'long_action_t': .575}
  exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]), 'modeld_main_frame_inputs', 'exec'), namespace)
  frames, transforms, _ = state.adapter.get_dummy_inputs()
  return frames, transforms, namespace['inputs']


def assert_default(capture):
  value = capture.values['fidx']
  assert value.dtype == np.dtype(np.float32)
  assert value.shape == (1, 1)
  np.testing.assert_array_equal(value, [[3.0]])


def test_normal_inference_copies_fixed_default_to_graph_buffer():
  state, captures = make_model(fidx=True)
  frames, transforms, inputs = frame_inputs(state)
  assert inputs['fidx'].shape == (1, 1)
  assert inputs['fidx'].dtype == np.dtype(np.float32)
  for stale_value in (0., -17.):
    state.numpy_inputs['fidx'][:] = stale_value
    state.run(frames, transforms, inputs)
    assert_default(captures[-1])
    np.testing.assert_array_equal(captures[-1].values['traffic_convention'], [[0., 1.]])
    np.testing.assert_array_equal(captures[-1].values['action_t'], np.array([[.225, .575]], np.float32))


def test_warmup_uses_default_and_first_real_frame_restores_after_clear():
  state, captures = make_model(fidx=True)
  state.adapter.input_queues['state_history'].assign(7).realize()
  state.warmup()
  assert len(captures) == 1
  assert_default(captures[0])
  assert not state.adapter.packed_input.any()
  assert not state.adapter.input_queues['state_history'].numpy().any()
  assert not state.prev_desire.any()
  state.run(*frame_inputs(state))
  assert_default(captures[-1])
  assert not captures[-1].history.any()


@pytest.mark.parametrize('reset', ['warmup_buffers', 'state'])
def test_reset_preserves_buffer_and_next_inference_rewrites_default(reset):
  state, captures = make_model(fidx=True)
  state.run(*frame_inputs(state))
  packed = state.adapter.packed_input
  state.adapter.input_queues['state_history'].assign(7).realize()
  if reset == 'warmup_buffers':
    state.adapter.reset_warmup_buffers()
  else:
    state.reset_state('synthetic sequence change')
  assert state.adapter.packed_input is packed
  assert not packed.any()
  assert not state.adapter.input_queues['state_history'].numpy().any()
  state.run(*frame_inputs(state))
  assert_default(captures[-1])
  assert not captures[-1].history.any()


def test_switch_and_reload_use_current_model_input_capabilities():
  with_fidx, fidx_captures = make_model(fidx=True)
  without_fidx, ordinary_captures = make_model(fidx=False)
  reloaded, reload_captures = make_model(fidx=True)
  for state, captures, expected_fidx in ((with_fidx, fidx_captures, True), (without_fidx, ordinary_captures, False),
                                         (with_fidx, fidx_captures, True), (reloaded, reload_captures, True),
                                         (without_fidx, ordinary_captures, False)):
    frames, transforms, inputs = frame_inputs(state)
    assert ('fidx' in inputs) == expected_fidx
    state.run(frames, transforms, inputs)
    if expected_fidx:
      assert_default(captures[-1])
    else:
      assert 'fidx' not in captures[-1].values
  for name in ('desire', 'traffic_convention', 'action_t'):
    np.testing.assert_array_equal(fidx_captures[-1].values[name], ordinary_captures[-1].values[name])


def test_model_without_fidx_ignores_extra_value_and_keeps_identical_inputs():
  state, captures = make_model(fidx=False)
  frames, transforms, inputs = frame_inputs(state)
  assert set(inputs) == {'desire', 'traffic_convention', 'action_t'}
  state.run(frames, transforms, inputs)
  state.run(frames, transforms, inputs | {'fidx': np.full((1, 1), 3., np.float32)})
  np.testing.assert_array_equal(captures[0].packed, captures[1].packed)
  state.warmup()
  assert 'fidx' not in captures[-1].values
  assert not state.adapter.packed_input.any()


def test_missing_required_input_fails_before_graph_dispatch():
  state, captures = make_model(fidx=True)
  frames, transforms, inputs = frame_inputs(state)
  inputs.pop('fidx', None)
  with pytest.raises(KeyError, match='fidx'):
    state.run(frames, transforms, inputs)
  assert not captures
