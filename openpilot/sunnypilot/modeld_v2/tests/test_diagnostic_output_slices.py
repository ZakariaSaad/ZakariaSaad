"""Diagnostic metadata is locally serialized from primitives; no public pickle executes."""
import base64
import codecs
import json
from pathlib import Path
import pickle
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.sunnypilot.modeld_v2 import model_adapters

FIXTURES = Path(__file__).resolve().parents[3] / 'selfdrive/navigation/tests/fixtures/nav_models.json'
SIDE_EFFECTS = []


def encode(value):
  return codecs.encode(pickle.dumps(value, protocol=4), 'base64').decode()


def side_effect():
  SIDE_EFFECTS.append('must never execute')
  return {'outputs': slice(0, 1)}


class MaliciousReducer:
  def __reduce__(self):
    return side_effect, ()


def real_fixture():
  signature = json.loads(FIXTURES.read_text())['models'][1]
  slices = {name: slice(value['start'], value['stop'], value['step']) for name, value in signature['output_slices'].items()}
  dtypes = {'UINT8': 'uint8', 'FLOAT': 'float32'}

  def specs(items):
    return {item['name']: (tuple(item['shape']), dtypes[item['dtype']], 'NPY') for item in items}

  return {'input_specs': specs(signature['inputs']), 'output_specs': specs(signature['outputs']),
          'metadata': {'output_shapes': {item['name']: tuple(item['shape']) for item in signature['outputs']},
                       'metadata': {'output_slices': encode(slices)}}, 'run': Mock()}, slices


def test_real_bigv2_fixture_loads_without_unrestricted_pickle():
  fixture, slices = real_fixture()
  with patch.object(model_adapters.pickle, 'loads', side_effect=AssertionError('unrestricted pickle must not execute')):
    adapter = model_adapters.NativeTinygradAdapter(fixture, 1344, 760, 'NPY', 'NPY', 'NPY', diagnostic=True)
  assert adapter.vision_output_slices == slices
  assert adapter.outputs['outputs'].shape == (1, 18580)
  assert adapter.numpy_inputs['nav_map'].dtype == np.float32
  fixture['run'].assert_not_called()


def test_production_default_keeps_existing_local_slice_loading_path():
  fixture, slices = real_fixture()
  with patch.object(model_adapters, 'diagnostic_output_slices', side_effect=AssertionError('diagnostic path selected')):
    adapter = model_adapters.NativeTinygradAdapter(fixture, 1344, 760, 'NPY', 'NPY', 'NPY')
  assert adapter.vision_output_slices == slices


def test_malicious_reducer_is_rejected_before_tensor_allocation_without_side_effect():
  fixture, _ = real_fixture()
  fixture['metadata']['metadata']['output_slices'] = encode(MaliciousReducer())
  with patch.object(model_adapters, 'Tensor') as tensor:
    with pytest.raises(ValueError, match='output_slices metadata'):
      model_adapters.NativeTinygradAdapter(fixture, 1344, 760, 'NPY', 'NPY', 'NPY', diagnostic=True)
    tensor.assert_not_called()
  assert SIDE_EFFECTS == []


def test_cached_pickle_extension_cannot_bypass_constructor_restriction():
  payload = base64.b64encode(b'\x80\x04\x82\xf6)R.').decode()
  with patch.dict('pickle._extension_cache', {246: side_effect}):
    with pytest.raises(ValueError, match='output_slices metadata'):
      model_adapters.diagnostic_output_slices(payload, 10)
  assert SIDE_EFFECTS == []


@pytest.mark.parametrize('mapping', [
  [], {}, {'bad': 1}, {1: slice(0, 2)}, {'': slice(0, 2)}, {'bad': slice(True, 2)},
  {'bad': slice(0, 2.5)}, {'bad': slice(0, 20)}, {'bad': slice(-11, None)},
  {'bad': slice(8, 2)}, {'bad': slice(1, 1)}, {'bad': slice(None, None, 0)}, {'bad': slice(None, None, -1)},
])
def test_malformed_mapping_and_out_of_range_slices_are_rejected(mapping):
  with pytest.raises(ValueError):
    model_adapters.diagnostic_output_slices(encode(mapping), 10)


def test_literal_open_and_negative_bounds_remain_supported():
  selections = {'all': slice(None, None), 'pad': slice(-2, None), 'sparse': slice(0, 10, 2)}
  assert model_adapters.diagnostic_output_slices(encode(selections), 10) == selections


@pytest.mark.parametrize('payload', ['!!!', '', 'A' * 131073, encode({'all': slice(0, 10)}) + '!!!!'])
def test_malformed_base64_and_oversized_encoding_are_rejected(payload):
  with pytest.raises(ValueError):
    model_adapters.diagnostic_output_slices(payload, 10)


def test_trailing_pickle_payload_is_rejected():
  payload = base64.b64encode(pickle.dumps({'all': slice(0, 10)}) + b'trailing').decode()
  with pytest.raises(ValueError):
    model_adapters.diagnostic_output_slices(payload, 10)


def test_factory_forwards_diagnostic_only_to_native_adapter():
  with patch.object(model_adapters, 'NativeTinygradAdapter') as native, patch.object(model_adapters, 'LegacyModelAdapter') as legacy:
    model_adapters.get_model_adapter({'input_specs': {}}, 1, 2, 'NPY', 'NPY', 'NPY', diagnostic=True)
    assert native.call_args.kwargs == {'chestnut': False, 'diagnostic': True}
    model_adapters.get_model_adapter({}, 1, 2, 'NPY', 'NPY', 'NPY', diagnostic=True)
    assert legacy.call_args.kwargs == {'chestnut': False}
