"""File-only diagnostic orchestration with mocked inference and real admission rules."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.selfdrive.navigation.model_inputs import NavigationModelDisabled, diagnostic_session
from openpilot.selfdrive.navigation.temporal import NavigationContext, context_reason
from tools import navigation_model_diagnostic as diagnostic


def context(index=0, generation=1, **overrides):
  timestamp = 10_000_000_000 + index * 50_000_000
  key = f'route-{generation}'
  base = NavigationContext(timestamp, timestamp, timestamp, timestamp, timestamp, generation, key, generation, key,
                           generation, key, index, index, True, True, True)
  return replace(base, **overrides)


def frames(count):
  return [
    {'bufs': {'img': SimpleNamespace(data=b'camera'), 'big_img': SimpleNamespace(data=b'wide-camera')},
     'transforms': {'img': np.eye(3, dtype=np.float32), 'big_img': np.eye(3, dtype=np.float32)},
     'inputs': {'desire': np.zeros(8, np.float32), 'action_t': np.array([index + 0.2, 0.3], np.float32)},
     'navigation': {'nav_map': np.full((256, 256), 7, np.uint8), 'navigation': np.full(47, 0.5, np.float32)}}
    for index in range(count)
  ]


class FakeModel:
  def __init__(self, *, mutate=False, fail=False, nonfinite=False):
    self.calls, self.resets = [], []
    self.state = 0
    self.mutate, self.fail, self.nonfinite = mutate, fail, nonfinite
    self.warmups = 0

  def warmup(self):
    self.warmups += 1

  def reset_state(self, reason):
    self.resets.append(reason)
    self.state = 0

  def run(self, bufs, transforms, inputs):
    self.calls.append({name: value.copy() for name, value in inputs.items()})
    self.state += 1
    if self.fail:
      raise RuntimeError('primary graph failed')
    influence = float(np.asarray(inputs.get('nav_map', [0])).reshape(-1)[0])
    influence += float(np.asarray(inputs.get('navigation', [0])).reshape(-1)[0])
    result = {'action': np.array([[self.state, influence]], np.float32)}
    if self.nonfinite:
      result['action'][0, 0] = np.nan
    if self.mutate:
      for array in inputs.values():
        array[:] = 99
    return result


class TestDiagnosticRuns(unittest.TestCase):
  def test_amd_cli_requires_compiler_verified_manifest_before_loading(self):
    arguments = ['diagnostic', '--recording', '/unused.npz', '--output', '/unused-output',
                 '--backend', 'AMD', '--experimental-catalog', '/unused.json']
    with patch.object(diagnostic.sys, 'argv', arguments), patch.dict(diagnostic.os.environ, {'NAV_MODEL_DIAGNOSTIC': '1'}), \
         patch.object(diagnostic, 'verify_runtime'), \
         patch.object(diagnostic, 'load_experimental_model', return_value=SimpleNamespace(compiler_verification_validated=False)), \
         redirect_stderr(io.StringIO()) as error_output, self.assertRaises(SystemExit) as error:
      diagnostic.main()
    self.assertEqual(error.exception.code, 2)
    self.assertIn('compiler-verified', error_output.getvalue())

  def run_case(self, factory, sample_frames, contexts, *, fallback=None, arms=diagnostic.ARMS):
    root = tempfile.TemporaryDirectory()
    self.addCleanup(root.cleanup)
    output = Path(root.name)
    with patch.object(diagnostic, 'measurements', return_value={'mock_memory_bytes': 0}), redirect_stdout(io.StringIO()):
      summary = diagnostic.run_diagnostic(factory, sample_frames, contexts, ['test-scene'] * len(contexts), output,
                                          arms=arms, fallback_factory=fallback)
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    return summary, events, output

  def test_four_arms_have_independent_states_and_immutable_auxiliary_inputs(self):
    models = []

    def factory():
      model = FakeModel(mutate=True)
      models.append(model)
      return model

    sample = frames(2)
    original = [{key: {name: value.copy() for name, value in frame[key].items()} for key in ('inputs', 'navigation', 'transforms')}
                for frame in sample]
    summary, events, output = self.run_case(factory, sample, [context(0), context(1)])
    self.assertEqual(len({id(model) for model in models}), 4)
    self.assertEqual(summary['completed_predictions'], 8)
    self.assertEqual(summary['failures'], 0)
    expected_navigation = (set(), {'nav_map'}, {'navigation'}, {'nav_map', 'navigation'})
    for arm, model, expected in zip(diagnostic.ARMS, models, expected_navigation, strict=True):
      self.assertEqual(model.warmups, 1)
      self.assertEqual(model.resets, ['initial_state'])
      self.assertEqual(len(model.calls), 2)
      for call in model.calls:
        self.assertEqual(set(call) & {'nav_map', 'navigation'}, expected)
      with np.load(output / arm / '000001.npz', allow_pickle=False) as data:
        self.assertEqual(set(data.files), {'action'})
        self.assertEqual(data['action'][0, 0], 2, 'recurrent state must start independently for every arm')
    for before, after in zip(original, sample, strict=True):
      for key in before:
        for name in before[key]:
          np.testing.assert_array_equal(before[key][name], after[key][name])
    self.assertEqual(len(events), 8)
    self.assertTrue(all(item['paired_primary_frames'] == 2 for item in summary['comparisons'].values()))

  def test_generation_and_validity_changes_reset_and_skip(self):
    model = FakeModel()
    contexts = [context(0), context(1), context(2, generation=2), context(3, generation=2, gps_valid=False), context(4, generation=2)]
    summary, events, _ = self.run_case(lambda: model, frames(5), contexts, arms=('neutral',))
    self.assertEqual(model.resets, ['initial_state', 'route_changed', 'navigation_lost', 'navigation_reacquired'])
    self.assertEqual([event['event'] for event in events], ['frame', 'frame', 'frame', 'skipped', 'frame'])
    self.assertEqual(events[3]['reason'], 'missing_gps')
    self.assertEqual(len(model.calls), 4)
    self.assertEqual(summary['skipped'], 1)

  def test_invalid_context_runs_only_explicit_non_navigation_fallback(self):
    primary, fallback = FakeModel(), FakeModel()
    summary, events, _ = self.run_case(lambda: primary, frames(1), [context(gps_valid=False)],
                                       fallback=lambda: fallback, arms=('both',))
    self.assertFalse(primary.calls)
    self.assertEqual(len(fallback.calls), 1)
    self.assertFalse(set(fallback.calls[0]) & {'nav_map', 'navigation'})
    self.assertEqual(events[-1]['model_role'], 'non_navigation_fallback')
    self.assertEqual(summary['comparisons']['both']['paired_primary_frames'], 0)

  def test_primary_failure_resets_and_runs_fresh_non_navigation_fallback(self):
    primary, fallback = FakeModel(fail=True), FakeModel()
    summary, events, _ = self.run_case(lambda: primary, frames(1), [context()],
                                       fallback=lambda: fallback, arms=('both',))
    self.assertIn('inference failure', primary.resets)
    self.assertEqual(fallback.warmups, 1)
    self.assertEqual(len(fallback.calls), 1)
    self.assertFalse(set(fallback.calls[0]) & {'nav_map', 'navigation'})
    self.assertTrue(any(event['event'] == 'failure' and 'primary graph failed' in event['reason'] for event in events))
    self.assertTrue(any(event['event'] == 'frame' and event['model_role'] == 'non_navigation_fallback' for event in events))
    self.assertEqual(summary['failures'], 1)
    self.assertEqual(summary['comparisons']['both']['paired_primary_frames'], 0)

  def test_failed_primary_is_quarantined_for_remaining_frames(self):
    primary, fallback = FakeModel(fail=True), FakeModel()
    summary, _, _ = self.run_case(lambda: primary, frames(2), [context(0), context(1)],
                                  fallback=lambda: fallback, arms=('both',))
    self.assertEqual(len(primary.calls), 1)
    self.assertEqual(len(fallback.calls), 2)
    self.assertEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 2)

  def test_fallback_initialization_failure_is_logged(self):
    factory = Mock(side_effect=RuntimeError('fallback load failed'))
    summary, events, _ = self.run_case(FakeModel, frames(1), [context(gps_valid=False)], fallback=factory, arms=('neutral',))
    factory.assert_called_once_with()
    self.assertGreaterEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 0)
    self.assertTrue(any(event['event'] == 'failure' and 'fallback load failed' in event['reason'] for event in events))

  def test_fallback_warmup_failure_is_logged_without_a_prediction(self):
    fallback = FakeModel()
    with patch.object(fallback, 'warmup', side_effect=RuntimeError('fallback warmup failed')):
      summary, events, output = self.run_case(FakeModel, frames(1), [context(gps_valid=False)],
                                             fallback=lambda: fallback, arms=('neutral',))
    self.assertGreaterEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 0)
    self.assertFalse(fallback.calls)
    self.assertEqual(list(output.rglob('*.npz')), [])
    self.assertTrue(any(event['event'] == 'failure' and 'fallback warmup failed' in event['reason'] for event in events))

  def test_failed_fallback_warmup_is_not_reused_on_the_next_frame(self):
    failed, fresh = FakeModel(), FakeModel()
    factory = Mock(side_effect=[failed, fresh])
    with patch.object(failed, 'warmup', side_effect=RuntimeError('fallback warmup failed')):
      summary, _, _ = self.run_case(FakeModel, frames(2), [context(0, gps_valid=False), context(1, gps_valid=False)],
                                    fallback=factory, arms=('neutral',))
    self.assertEqual(factory.call_count, 2)
    self.assertFalse(failed.calls, 'a model whose warmup failed cannot be considered initialized')
    self.assertEqual(fresh.warmups, 1)
    self.assertEqual(len(fresh.calls), 1)
    self.assertEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 1)

  def test_primary_initialization_failure_is_visible_in_summary(self):
    summary, events, output = self.run_case(Mock(side_effect=RuntimeError('primary load failed')), frames(1), [context()], arms=('neutral',))
    self.assertGreaterEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 0)
    self.assertEqual(list(output.rglob('*.npz')), [])
    self.assertTrue(any(event['event'] == 'failure' and 'primary load failed' in event['reason'] for event in events))

  def test_primary_warmup_failure_is_visible_without_running_frames(self):
    primary = FakeModel()
    with patch.object(primary, 'warmup', side_effect=RuntimeError('primary warmup failed')):
      summary, events, _ = self.run_case(lambda: primary, frames(1), [context()], arms=('neutral',))
    self.assertFalse(primary.calls)
    self.assertGreaterEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 0)
    self.assertTrue(any(event['event'] == 'failure' and 'primary warmup failed' in event['reason'] for event in events))

  def test_nonfinite_predictions_are_reset_and_never_written(self):
    model = FakeModel(nonfinite=True)
    summary, events, output = self.run_case(lambda: model, frames(1), [context()], arms=('neutral',))
    self.assertEqual(summary['failures'], 1)
    self.assertEqual(summary['completed_predictions'], 0)
    self.assertIn('inference failure', model.resets)
    self.assertEqual(list(output.rglob('*.npz')), [])
    self.assertIn('Non-finite', events[0]['reason'])


class TestDiagnosticContexts(unittest.TestCase):
  def prepare(self, narrow, wide, *, synthetic=False, **metadata):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / 'input.npz'
      np.savez(path, camera_timestamp_ns=narrow, big_camera_timestamp_ns=wide, **metadata)
      sample = frames(len(narrow))
      return diagnostic.prepare_contexts(path, sample, synthetic_suite=synthetic), sample

  @staticmethod
  def prepared_metadata():
    return {
      'nav_valid': np.array([True]), 'gps_valid': np.array([True]),
      'route_generation': np.array([1]), 'instruction_generation': np.array([1]), 'map_generation': np.array([1]),
      'destination_key': np.array(['route-1']), 'instruction_destination_key': np.array(['route-1']),
      'map_destination_key': np.array(['route-1']), 'frame_id': np.array([17]), 'metadata_frame_id': np.array([17]),
      **{name: np.array([10_000_000_000]) for name in ('gps_source_ns', 'instruction_source_ns', 'map_source_ns', 'map_render_ns')},
    }

  def test_prepared_identity_and_frame_metadata_are_independent(self):
    timestamps = np.array([10_000_000_000], np.int64)
    metadata = self.prepared_metadata()
    (contexts, _), _ = self.prepare(timestamps, timestamps, **metadata)
    self.assertEqual(context_reason(contexts[0]), 'valid')
    self.assertEqual(contexts[0].frame_id, 17)
    defects = [
      ('instruction_generation', np.array([2]), 'instruction_route_mismatch'),
      ('instruction_destination_key', np.array(['another-route']), 'instruction_route_mismatch'),
      ('map_generation', np.array([2]), 'map_route_mismatch'),
      ('map_destination_key', np.array(['another-route']), 'map_route_mismatch'),
      ('metadata_frame_id', np.array([18]), 'map_frame_mismatch'),
    ]
    for name, value, expected in defects:
      with self.subTest(field=name):
        (contexts, _), _ = self.prepare(timestamps, timestamps, **(metadata | {name: value}))
        self.assertEqual(context_reason(contexts[0]), expected)

  def test_valid_prepared_navigation_requires_recorded_provenance(self):
    timestamps = np.array([10_000_000_000], np.int64)
    metadata = self.prepared_metadata()
    for name in metadata.keys() - {'nav_valid'}:
      with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'independent provenance'):
        self.prepare(timestamps, timestamps, **{key: value for key, value in metadata.items() if key != name})

  def test_prepared_scalar_shapes_and_types_are_not_coerced(self):
    timestamps = np.array([10_000_000_000], np.int64)
    metadata = self.prepared_metadata()
    defects = [
      ('nav_valid', np.array([[True]])), ('gps_valid', np.array([1])),
      ('instruction_generation', np.array([1.5])), ('instruction_destination_key', np.array([b'route-1'])),
      ('map_render_ns', np.array([[10_000_000_000]])), ('metadata_frame_id', np.array([], np.int64)),
      ('frame_id', np.array([1 << 32], np.uint64)), ('route_generation', np.array([-1])),
    ]
    for name, value in defects:
      with self.subTest(field=name), self.assertRaises(ValueError):
        self.prepare(timestamps, timestamps, **(metadata | {name: value}))

  def test_signed_and_unsigned_cameras_accept_small_skew_in_either_direction(self):
    for dtype in (np.int64, np.uint64):
      for delta in (-1_000_000, 1_000_000):
        with self.subTest(dtype=dtype, skew=delta):
          narrow = np.array([10_000_000_000, 10_050_000_000], dtype=dtype)
          wide = np.array([10_000_000_000 + delta, 10_050_000_000 + delta], dtype=dtype)
          (contexts, _), _ = self.prepare(narrow, wide)
          self.assertEqual(len(contexts), 2)

  def test_camera_order_skew_dtype_zero_and_integer_overflow_rejected(self):
    valid = np.array([10_000_000_000, 10_050_000_000], np.int64)
    cases = [
      (valid[::-1], valid[::-1]),
      (valid, valid[::-1]),
      (valid.astype(np.uint64)[::-1], valid.astype(np.uint64)[::-1]),
      (np.array([valid[0], valid[0]]), valid),
      (valid, valid + 10_000_001),
      (valid.astype(np.float64), valid),
      (np.array([0, valid[1]]), valid),
      (np.array([2**63, 2**63 + 1], np.uint64), np.array([2**63, 2**63 + 1], np.uint64)),
    ]
    for narrow, wide in cases:
      with self.subTest(narrow=narrow, wide=wide), self.assertRaises(ValueError):
        self.prepare(narrow, wide)

  def test_simulated_navigation_is_explicit_and_preserves_camera_pixels(self):
    timestamps = np.arange(9, dtype=np.int64) * 50_000_000 + 10_000_000_000
    (contexts, scenarios), sample = self.prepare(timestamps, timestamps, synthetic=True)
    self.assertEqual(scenarios[5:], ['reroute_old_map', 'missing_gps', 'stale_map', 'route_reacquired'])
    self.assertEqual([context_reason(item) for item in contexts[5:]],
                     ['map_route_mismatch', 'missing_gps', 'source_time_out_of_range', 'valid'])
    self.assertTrue(all(frame['bufs']['img'].data == b'camera' for frame in sample))
    self.assertTrue(all(frame['bufs']['big_img'].data == b'wide-camera' for frame in sample))


class TestDiagnosticBoundary(unittest.TestCase):
  def test_file_only_scope_denies_all_messaging_and_vision_aliases(self):
    import openpilot.cereal.messaging as messaging
    import openpilot.cereal.visionipc as cereal_vision
    import msgq
    import msgq.visionipc as native_vision
    from openpilot.sunnypilot.modeld_v2 import modeld

    symbols: list[tuple[ModuleType, str]] = [(messaging, name) for name in ('PubMaster', 'pub_sock', 'SubMaster', 'sub_sock')]
    symbols += [(owner, name) for owner in (messaging, msgq) for name in ('PubSocket', 'SubSocket')]
    symbols += [(msgq, name) for name in ('pub_sock', 'sub_sock')]
    symbols += [(modeld, name) for name in ('PubMaster', 'SubMaster', 'VisionIpcClient')]
    symbols += [(native_vision, 'VisionIpcClient')]
    if hasattr(cereal_vision, 'VisionIpcClient'):
      symbols += [(cereal_vision, 'VisionIpcClient')]
    original = {(id(owner), name): getattr(owner, name) for owner, name in symbols}
    with diagnostic.no_vehicle_io():
      for owner, name in symbols:
        with self.subTest(owner=owner.__name__, name=name):
          blocked = getattr(owner, name)
          self.assertIsNot(blocked, original[id(owner), name], 'do not call an unpatched live socket constructor')
          with self.assertRaises(NavigationModelDisabled):
            blocked('diagnostic-must-never-connect')
    for owner, name in symbols:
      self.assertIs(getattr(owner, name), original[id(owner), name])

  def test_diagnostic_models_require_session_and_cannot_publish_inside_it(self):
    from openpilot.common.params import Params
    from openpilot.common.prefix import OpenpilotPrefix
    from openpilot.sunnypilot.modeld_v2 import modeld

    with patch.dict('os.environ', {'NAV_MODEL_DIAGNOSTIC': '1'}), OpenpilotPrefix():
      Params().put_bool('NavigationModelInputsEnabled', True, block=True)
      with self.assertRaises(NavigationModelDisabled):
        modeld.ModelState(1344, 760, diagnostic=True)
      with diagnostic_session('CPU'), diagnostic.no_vehicle_io(), \
           patch.object(modeld, 'get_active_bundle', return_value=None), \
           patch.object(modeld, '_find_driving_pkl', return_value='trusted-synthetic-model'), \
           patch.object(modeld.ModelState, '_init_combined'):
        model = modeld.ModelState(1344, 760, diagnostic=True)
        model.numpy_inputs = {}
        with self.assertRaises(NavigationModelDisabled):
          modeld._reject_navigation_publication(model, None)
      with self.assertRaises(NavigationModelDisabled):
        model._check_diagnostic_session()

  def test_cli_requires_explicit_opt_in_before_runtime_or_artifact_loading(self):
    argv = ['navigation_model_diagnostic.py', '--recording', '/nonexistent.npz', '--output', '/nonexistent', '--backend', 'CPU']
    with patch('sys.argv', argv), patch.dict('os.environ', {'NAV_MODEL_DIAGNOSTIC': '0'}), \
         patch.object(diagnostic, 'verify_runtime') as verify, redirect_stdout(io.StringIO()):
      with self.assertRaises(SystemExit) as error:
        diagnostic.main()
      self.assertEqual(error.exception.code, 2)
      verify.assert_not_called()
