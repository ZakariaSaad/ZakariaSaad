"""Navigation sample association and explicit recurrent-state boundaries.

These are diagnostic transport rules, not a claim about trained fallback behavior.
An invalid sample is skipped; it is never silently promoted to a valid zero route.
"""
from dataclasses import dataclass
import math

SOURCE_TIMEOUT = 2.0  # A physical GPS sample can be 1 Hz.
MAP_RECEIPT_TIMEOUT = 0.5
SOURCE_FUTURE_TOLERANCE = 0.2
MAX_CAMERA_STEP_NS = 75_000_000  # 20Hz checkpoints: a missing sample requires a cold start.


@dataclass(frozen=True)
class NavigationContext:
  camera_ns: int
  gps_ns: int
  instruction_ns: int
  map_source_ns: int
  map_render_ns: int
  generation: int
  destination_key: str
  instruction_generation: int
  instruction_destination_key: str
  map_generation: int
  map_destination_key: str
  frame_id: int
  metadata_frame_id: int
  gps_valid: bool
  instruction_valid: bool
  map_valid: bool
  map_age: float = 0.0
  instruction_age: float = 0.0
  metadata_age: float = 0.0

  @property
  def identity(self):
    return self.generation, self.destination_key


def context_reason(context: NavigationContext) -> str:
  if not context.gps_valid:
    return 'missing_gps'
  if not context.instruction_valid or not context.map_valid:
    return 'invalid_producer'
  if context.generation <= 0 or not context.destination_key:
    return 'missing_route_identity'
  if (context.instruction_generation, context.instruction_destination_key) != context.identity:
    return 'instruction_route_mismatch'
  if (context.map_generation, context.map_destination_key) != context.identity:
    return 'map_route_mismatch'
  if context.frame_id != context.metadata_frame_id:
    return 'map_frame_mismatch'
  if context.camera_ns <= 0 or min(context.gps_ns, context.instruction_ns, context.map_source_ns, context.map_render_ns) <= 0:
    return 'missing_source_timestamp'
  if context.map_source_ns > context.map_render_ns:
    return 'map_source_after_render'
  for source in (context.gps_ns, context.instruction_ns, context.map_source_ns):
    age = (context.camera_ns - source) / 1e9
    if not -SOURCE_FUTURE_TOLERANCE <= age < SOURCE_TIMEOUT:
      return 'source_time_out_of_range'
  render_age = (context.camera_ns - context.map_render_ns) / 1e9
  if not -SOURCE_FUTURE_TOLERANCE <= render_age < MAP_RECEIPT_TIMEOUT:
    return 'stale_or_future_render'
  for age, maximum in ((context.map_age, MAP_RECEIPT_TIMEOUT), (context.metadata_age, MAP_RECEIPT_TIMEOUT),
                       (context.instruction_age, SOURCE_TIMEOUT)):
    if not math.isfinite(age) or not 0 <= age < maximum:
      return 'stale_delivery'
  return 'valid'


@dataclass(frozen=True)
class TransitionDecision:
  infer: bool
  reset: bool
  reason: str


class NavigationTransitions:
  """Clear all model state at route/validity/clock boundaries, then skip bad data.

  The reset deliberately includes camera history as well as navigation history.
  This avoids retaining hidden route context, at the cost of a cold start. The
  cold-start quality and subsequent convergence still require learned evaluation.
  """
  def __init__(self):
    self.identity = None
    self.valid = None
    self.camera_ns = 0

  def update(self, identity, valid: bool, camera_ns: int, reason='valid') -> TransitionDecision:
    clock_bad = camera_ns <= 0 or (self.camera_ns > 0 and camera_ns <= self.camera_ns)
    frame_gap = self.camera_ns > 0 and camera_ns - self.camera_ns > MAX_CAMERA_STEP_NS
    reset = self.valid is None or identity != self.identity or valid != self.valid or clock_bad or frame_gap
    why = ('camera_clock_discontinuity' if clock_bad else 'camera_frame_gap' if frame_gap else 'initial_state' if self.valid is None else
           'route_changed' if identity != self.identity else 'navigation_reacquired' if valid and not self.valid else
           'navigation_lost' if not valid and self.valid else reason)
    self.identity, self.valid, self.camera_ns = identity, valid and not clock_bad, camera_ns
    return TransitionDecision(valid and not clock_bad, reset, why)
