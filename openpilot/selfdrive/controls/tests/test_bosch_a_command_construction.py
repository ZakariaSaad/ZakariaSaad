"""Offline Insight command-construction regressions using the real controllers.

These tests construct CAN bytes; they never initialize a vehicle, open messaging
sockets, or send CAN. Only the asynchronous parameter writer is replaced. Params,
LongControl, Honda CarController, CANPacker and the DBC decoder remain real.

The positive stopping fixture is conditioned on the observed internal target in
route B segment 31, around 4310798824968/4310818087937 ns. It establishes the
existing controller mechanism, not a reconstruction of all hidden route state or
proof that an ECU applied the request. The Bosch-A flag does not change this
actuator implementation, so planner guards alone cannot promise negative CAN
acceleration immediately after entering stopping with positive controller state.
"""

from types import SimpleNamespace

import pytest

from opendbc.can.dbc import DBC
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.honda import carcontroller
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC as HONDA_DBC
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.interfaces import _initialize_honda
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState


ACCEL_LIMITS = (-3.5, 2.0)
OBSERVED_PRIOR_ACCEL = 0.0910818204


@pytest.fixture
def insight_factory(tmp_path, monkeypatch):
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path / "params"))
  # Avoid background writes/threads, without replacing controller calculations.
  monkeypatch.setattr(carcontroller, "HondaParamWriter",
                      lambda: SimpleNamespace(put_many=lambda values: None))

  def create(*, feature_enabled=False, stock_long=False, speed=.1154, acceleration=.6837):
    fingerprint = gen_empty_fingerprint()
    cp = CarInterface.get_params(CAR.HONDA_INSIGHT, fingerprint, [], True, False, False)
    cp_sp = CarInterface.get_params_sp(cp, CAR.HONDA_INSIGHT, fingerprint, [], True, False, False)
    _initialize_honda(cp, cp_sp, {
      "BoschALongitudinalEnabled": int(feature_enabled),
      "HondaEnforceStockLongitudinal": int(stock_long),
    })
    assert bool(cp_sp.flags & HondaFlagsSP.BOSCH_A_LONG) == (feature_enabled and not stock_long)
    ci = CarInterface(cp, cp_sp)
    state, _ = ci.update([])
    state.vEgo = speed
    state.aEgo = acceleration
    state.standstill = False
    state.gasPressed = False
    state.brakePressed = False
    state.cruiseState.enabled = True
    state.cruiseState.standstill = False
    state.cruiseState.speed = 30.
    state.vCruise = 108.
    ci.CC.gasalpha = 0.
    ci.CC.gasfactor = ci.CC.gasfactor_before_gasmax = 1.
    ci.CC.windfactor = ci.CC.windfactor_before_brake = ci.CC.windfactor_before_gasmax = 1.
    ci.CC.bosch_last_gas = 1000.  # Isolate stopping shaping from the unrelated gas rise limiter.
    return SimpleNamespace(ci=ci, state=state, control=LongControl(cp, cp_sp), tick=0)

  return create


def decode_acc(frames):
  """Decode actual packed intents and verify their Honda checksums."""
  dbc = DBC(HONDA_DBC[CAR.HONDA_INSIGHT][Bus.pt])
  selected = {dbc.name_to_msg[name].address: dbc.name_to_msg[name]
              for name in ("ACC_CONTROL", "ACC_CONTROL_ON")}
  result = {}
  for address, data, bus in frames:
    if address not in selected:
      continue
    message = selected[address]
    assert len(data) == message.size
    values = {}
    for signal in message.sigs.values():
      raw = signal.get_raw_value(data)
      if signal.is_signed:
        raw -= ((raw >> (signal.size - 1)) & 1) * (1 << signal.size)
      if signal.calc_checksum is not None:
        assert raw == signal.calc_checksum(address, signal, bytearray(data))
      values[signal.name] = raw * signal.factor + signal.offset
    assert message.name not in result
    result[message.name] = (bus, values, bytes(data))
  return result


def command_tick(harness, *, target, should_stop, enabled=True, active=True):
  """Reproduce controlsd's state-before-update assignment, not the full process.

  controlsd stores actuators.longControlState before LongControl.update, while
  controlsState reports its state afterward. No mocked MPC or planner is used;
  target/should_stop are explicit inputs to the real longitudinal controller.
  """
  cc = structs.CarControl.new_message(enabled=enabled, longActive=active)
  cc.actuators.longControlState = harness.control.long_control_state
  cc.actuators.accel = float(harness.control.update(
    active, harness.state, target, should_stop, ACCEL_LIMITS,
  ))
  output, frames = harness.ci.apply(
    cc.as_reader(), structs.CarControlSP(), int(harness.tick * DT_CTRL * 1e9),
  )
  harness.tick += 1
  return cc, output, decode_acc(frames)


@pytest.mark.parametrize("feature_enabled", [False, True])
def test_positive_prior_stopping_state_can_outlive_negative_longcontrol_request(insight_factory, feature_enabled):
  h = insight_factory(feature_enabled=feature_enabled)
  h.control.long_control_state = LongCtrlState.stopping
  h.control.last_output_accel = -.01
  h.ci.CC.last_applied_brake = OBSERVED_PRIOR_ACCEL

  observed = []
  for tick in range(3):
    cc, output, messages = command_tick(h, target=.090114, should_stop=True)
    assert cc.actuators.accel == pytest.approx(-.02 - .01 * tick, abs=1e-7)
    assert cc.actuators.longControlState == LongCtrlState.stopping
    assert h.control.long_control_state == LongCtrlState.stopping
    if tick == 1:
      assert not messages  # LongControl runs at 100 Hz; ACC intents at 50 Hz.
      continue
    internal = OBSERVED_PRIOR_ACCEL - .002 * (len(observed) + 1)
    assert h.ci.CC.last_applied_brake == pytest.approx(internal)
    assert output.accel == pytest.approx(internal, abs=1e-7)
    bus, acc, _ = messages["ACC_CONTROL"]
    assert bus == h.ci.CC.CAN.pt
    assert acc["ACCEL_COMMAND"] == pytest.approx(.09)
    assert 0 < acc["GAS_COMMAND"] <= 1600
    assert acc["BRAKE_REQUEST"] == 0
    assert acc["STANDSTILL"] == 1
    assert not h.state.standstill and h.state.vEgo > 0
    observed.append(internal)
  assert observed == pytest.approx([.0890818204, .0870818204])


@pytest.mark.parametrize("feature_enabled", [False, True])
def test_restart_has_one_cycle_old_actuator_state_and_releases_protocol_stop(insight_factory, feature_enabled):
  h = insight_factory(feature_enabled=feature_enabled, speed=.02, acceleration=0.)
  h.control.long_control_state = LongCtrlState.stopping
  h.control.last_output_accel = -.2
  h.ci.CC.last_applied_brake = -.2

  cc, _, messages = command_tick(h, target=.3, should_stop=False)
  assert h.control.long_control_state == LongCtrlState.pid
  assert cc.actuators.longControlState == LongCtrlState.stopping
  assert cc.actuators.accel > 0
  assert messages["ACC_CONTROL"][1]["ACCEL_COMMAND"] > 0
  assert messages["ACC_CONTROL"][1]["STANDSTILL"] == 1

  cc, _, messages = command_tick(h, target=.3, should_stop=False)
  assert cc.actuators.longControlState == LongCtrlState.pid
  assert not messages
  _, _, messages = command_tick(h, target=.3, should_stop=False)
  assert messages["ACC_CONTROL"][1]["STANDSTILL"] == 0
  assert messages["ACC_CONTROL"][1]["STANDSTILL_RELEASE"] == 1


@pytest.mark.parametrize("feature_enabled", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_inactive_longitudinal_masks_positive_internal_state_at_command_construction(insight_factory, feature_enabled, enabled):
  h = insight_factory(feature_enabled=feature_enabled)
  h.control.long_control_state = LongCtrlState.stopping
  h.control.last_output_accel = -.02
  h.ci.CC.last_applied_brake = OBSERVED_PRIOR_ACCEL
  cc, output, messages = command_tick(h, target=.1, should_stop=True, enabled=enabled, active=False)
  assert cc.actuators.longControlState == LongCtrlState.stopping
  assert h.control.long_control_state == LongCtrlState.off
  assert cc.actuators.accel == 0
  # Internal software output can still be positive; the wire-intent gate owns authority.
  assert output.accel > 0
  acc = messages["ACC_CONTROL"][1]
  assert acc["ACCEL_COMMAND"] == 0
  assert acc["GAS_COMMAND"] == -30000
  assert acc["BRAKE_REQUEST"] == acc["STANDSTILL"] == 0
  assert acc["CONTROL_ON"] == (5 if enabled else 0)
  assert messages["ACC_CONTROL_ON"][1]["CONTROL_ON"] == int(enabled)


@pytest.mark.parametrize("feature_enabled", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_stock_selection_never_constructs_openpilot_acc_commands(insight_factory, feature_enabled, active):
  h = insight_factory(feature_enabled=feature_enabled, stock_long=True)
  assert not h.ci.CP.openpilotLongitudinalControl
  for _ in range(4):
    _, _, messages = command_tick(h, target=1., should_stop=False, active=active)
    assert not messages


@pytest.mark.parametrize("feature_enabled", [False, True])
def test_gas_compensation_and_acceleration_fields_have_different_units(insight_factory, feature_enabled):
  h = insight_factory(feature_enabled=feature_enabled, speed=10., acceleration=-.01)
  h.control.long_control_state = LongCtrlState.pid
  h.ci.CC.gasalpha = .2
  cc, output, messages = command_tick(h, target=-.01, should_stop=False)
  acc = messages["ACC_CONTROL"][1]
  assert cc.actuators.accel == pytest.approx(-.01)
  assert output.accel == pytest.approx(-.01)
  assert acc["ACCEL_COMMAND"] == pytest.approx(-.01)  # m/s² request
  assert 0 < acc["GAS_COMMAND"] <= 1600  # OEM raw gas units, not m/s²
  assert acc["BRAKE_REQUEST"] == 0
  # These output fields carry lateral telemetry in this target, not longitudinal feedback.
  assert output.gas == pytest.approx(h.ci.CC.lat_model.gain_now)
  assert output.brake == pytest.approx(h.ci.CC.lat_model.applied_correction)


def test_new_radar_flag_preserves_existing_actuator_command_bytes(insight_factory):
  off = insight_factory(feature_enabled=False, speed=10., acceleration=0.)
  on = insight_factory(feature_enabled=True, speed=10., acceleration=0.)
  for h in (off, on):
    h.control.long_control_state = LongCtrlState.pid
    h.ci.CC.last_applied_brake = OBSERVED_PRIOR_ACCEL
  for target, stop, active in [(1., False, True), (-.5, True, True), (-.5, True, True),
                                (.3, False, True), (.3, False, True), (0., False, False)]:
    left = command_tick(off, target=target, should_stop=stop, active=active)
    right = command_tick(on, target=target, should_stop=stop, active=active)
    assert left[0].actuators.accel == right[0].actuators.accel
    assert left[1].accel == right[1].accel
    assert left[2] == right[2]
