"""SunnyPilot lane changes with Bigv2's lane-change-only desire conditioning.

Use the ordinary helper's saved settings, timers and vehicle blind spots. Native
Bigv2 intentionally ignores road edges and the optional ordinary lane-turn
conditioning; its desire encoding and low-speed behavior remain unchanged.
"""
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper as OrdinaryDesireHelper, TurnDirection


class DesireHelper(OrdinaryDesireHelper):
  def update(self, carstate, lateral_active, lane_change_prob, left_edge_detected=False, right_edge_detected=False):
    # Only omit model-derived road edges. Actual carState blind spots still
    # reach the shared lane-change controller, including its optional delay.
    super().update(carstate, lateral_active, lane_change_prob)

    # Keep native Bigv2's existing conditioning, including when speed drops
    # during an ongoing lane change. Do not inject ordinary lane-turn desires.
    self.lane_turn_direction = TurnDirection.none
    self.desire = log.Desire.none
    if self.lane_change_state == log.LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == log.LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == log.LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
