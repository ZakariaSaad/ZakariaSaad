from types import SimpleNamespace

import pytest

from openpilot.cereal import custom
from openpilot.common.params import Params
from opendbc.car.structs import car
from opendbc.car.honda.values import CAR, HondaFlags
from openpilot.nrdr.features.longitudinal.driver_controls import BoschADriverControls, imported_mode, initialize_experimental
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP, ET


@pytest.fixture
def state(tmp_path, monkeypatch):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path))
  params = Params()
  cp = car.CarParams.new_message(brand="honda", carFingerprint=CAR.HONDA_INSIGHT, openpilotLongitudinalControl=True)
  return SimpleNamespace(CP=cp, params=params, events_sp=EventsSP(), experimental_mode=False, personality=1,
                         sm={"longitudinalPlanSP": custom.LongitudinalPlanSP.new_message()})


def car_state(*, pressed=None, available=True):
  cs = car.CarState.new_message()
  cs.cruiseState.available = available
  if pressed is not None:
    cs.buttonEvents = [{"type": "gapAdjustCruise", "pressed": pressed}]
  return cs


def test_short_press_cycles_all_four_profiles(state):
  controls = BoschADriverControls(state.CP, state.params)
  seen = []
  for index in range(4):
    assert not controls.update(state, car_state(pressed=True), now=1. + index)
    assert controls.update(state, car_state(pressed=False), now=1.1 + index)
    seen.append(state.personality)
  assert seen == [0, 3, 2, 1]


def test_hold_toggles_once_and_release_does_not_cycle_profile(state):
  controls = BoschADriverControls(state.CP, state.params)
  assert not controls.update(state, car_state(pressed=True), now=1.)
  for tick in range(1, 80):
    assert not controls.update(state, car_state(), now=1. + tick * .01)
  assert controls.cruise.experimental_mode_switched
  assert controls.cruise._experimental_mode is True
  assert state.events_sp.names == [custom.OnroadEventSP.EventName.experimentalModeSwitched]
  assert not controls.update(state, car_state(pressed=False), now=2.)
  assert state.personality == 1
  assert not controls.cruise.experimental_mode_switched


@pytest.mark.parametrize("reservation", ["speed_limit", "cruise_unavailable"])
def test_consumed_press_does_not_become_mode_toggle_or_profile_change(state, reservation):
  controls = BoschADriverControls(state.CP, state.params)
  if reservation == "speed_limit":
    state.sm["longitudinalPlanSP"].speedLimit.assist.state = "preActive"
  controls.update(state, car_state(pressed=True, available=reservation != "cruise_unavailable"), now=1.)
  state.sm["longitudinalPlanSP"].speedLimit.assist.state = "inactive"
  for tick in range(1, 120):
    assert not controls.update(state, car_state(), now=1. + tick * .01)
  assert not controls.update(state, car_state(pressed=False), now=3.)
  assert not state.events_sp.names
  assert state.personality == 1
  controls.update(state, car_state(pressed=True), now=4.)
  assert controls.update(state, car_state(pressed=False), now=4.1)
  assert state.personality == 0


def test_mode_alert_uses_latched_value_not_stale_param(state):
  state.params.put_bool("ExperimentalMode", False, block=True)
  state.events_sp.add(custom.OnroadEventSP.EventName.experimentalModeSwitched)
  alerts = state.events_sp.create_alerts([ET.WARNING], [], callback_args_by_event={
    custom.OnroadEventSP.EventName.experimentalModeSwitched: [True],
  })
  assert alerts[0].alert_text_1 == "Experimental Mode On"
  assert state.params.get_bool("ExperimentalMode") is False


def test_stock_transition_clears_saved_experimental_choice(state):
  state.params.put_bool("ExperimentalMode", True, block=True)
  state.CP.openpilotLongitudinalControl = False
  initialize_experimental(state.CP, state.params)
  assert state.params.get("ExperimentalMode") is False


def test_default_stock_and_mvl_modes_do_not_select_imported_controls(state):
  assert imported_mode(state.CP)
  state.CP.flags |= HondaFlags.MVL_LONGITUDINAL.value
  assert not imported_mode(state.CP)
  state.CP.flags = 0
  state.CP.openpilotLongitudinalControl = False
  assert not imported_mode(state.CP)


def test_other_vehicle_saved_experimental_choice_is_preserved(state):
  state.params.put_bool("ExperimentalMode", True, block=True)
  state.CP.brand = "toyota"
  state.CP.openpilotLongitudinalControl = False
  initialize_experimental(state.CP, state.params)
  assert state.params.get_bool("ExperimentalMode")
