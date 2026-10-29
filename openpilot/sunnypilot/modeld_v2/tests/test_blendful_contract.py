"""Additional branch contracts around the unchanged upstream Blendful helper."""
import hashlib
from pathlib import Path

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2 import blendful
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.tests.test_blendful import drive, frame


def test_algorithm_is_exact_final_merged_upstream_source():
  # PR #2073, 97135965effb3630ed55993903449edb73468d50. No subsequent core
  # changes through inspected master 6f7a2c94c10dffcf9a83b120ab1e6985e4496868.
  assert hashlib.sha256(Path(blendful.__file__).read_bytes()).hexdigest() == \
    '652c52e164933a0fe585ef121e642f60e3e0a616554dcc9366b2231efefe5920'


def test_disabled_does_not_inspect_model_arrays_or_add_correction():
  state = Blendful()
  for speed in (0., 4.99, 5., 15., 30.):
    for curvature in (-.02, 0., .02):
      assert state.update({}, speed, True, curvature, enabled=False) == 0.
      assert state.curvature == state.trust == 0.
  drive(state, 5., center=.5)
  assert state.curvature > 0.
  assert state.update({}, 20., True, .01, enabled=False) == 0.
  assert state.curvature == state.trust == state.smooth == 0.


def test_inactive_gate_fades_correction_with_upstream_release_rate():
  # The runtime supplies active=False during lane changes or non-none desires.
  # The helper fades the previous correction rather than inventing a hard cut.
  state = Blendful()
  drive(state, 5., v=10., center=.5)
  previous = state.curvature
  assert previous > 0.
  changes = []
  for _ in range(100):
    current = state.update(frame(center=.5), 10., False, 0.)
    changes.append(current)
    assert state.trust == state.line_trust == 0.
    assert not state.usual_offset.initialized and not state.lane_width.initialized
    assert not state.going_around
    assert abs(current - previous) <= .5 / 10.**2 * DT_MDL + 1e-12
    previous = current
  assert changes[0] > 0.
  assert abs(changes[-1]) < 1e-6


def test_both_missing_lines_release_without_reinterpreting_model_output():
  # Absent lines are represented by zero probabilities in the parsed contract,
  # not by deleting required output tensor names or fabricating new predictions.
  state = Blendful()
  drive(state, 5., center=.5)
  before = state.trust
  for _ in range(400):
    output = frame(center=.5, probs=(0., 0.))
    original = {name: array.copy() for name, array in output.items()}
    state.update(output, 20., True, 0.)
    assert np.isfinite([state.curvature, state.trust]).all()
    for name in output:
      np.testing.assert_array_equal(output[name], original[name])
  assert state.trust < before
  assert state.trust < 1e-6
  assert abs(state.curvature) < 1e-6


def test_turning_off_then_on_matches_a_fresh_upstream_helper():
  state, fresh = Blendful(), Blendful()
  drive(state, 5., center=.5)
  state.update({}, 20., True, 0., enabled=False)
  for index in range(120):
    output = frame(center=.3, plan=.8 if index >= 80 else 0.)
    assert state.update(output, 20., True, 0.) == fresh.update(output, 20., True, 0.)
    for name in ('curvature', 'trust', 'line_trust', 'dk', 'l_prob', 'r_prob', 'model_curvature', 'smooth', 'going_around'):
      assert getattr(state, name) == getattr(fresh, name)
    for name in ('lane_width', 'line_gone', 'line_probs_slow', 'line_probs', 'usual_offset'):
      np.testing.assert_array_equal(getattr(state, name).x, getattr(fresh, name).x)
