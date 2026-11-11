"""Native Params/startup and saved-choice regressions for the selective port."""
from types import SimpleNamespace
import time

import pytest

from opendbc.car.honda.values import CAR, HondaFlags
from openpilot.common.params import Params
from openpilot.nrdr.ui.honda_longitudinal import (
  donor_bosch_a_mode, live_gas_available, live_gas_status, personality_value,
  save_experimental_choice, save_live_gas_choice, save_radar_choice, save_stock_choice,
)


@pytest.fixture
def params(tmp_path):
  return Params(str(tmp_path / "params"))


def cp(*, mvl=False, long=True, brand="honda", fingerprint=CAR.HONDA_INSIGHT):
  return SimpleNamespace(brand=brand, carFingerprint=fingerprint, openpilotLongitudinalControl=long,
                         flags=int(HondaFlags.MVL_LONGITUDINAL) if mvl else 0, radarUnavailable=False)


def state(params, *, parked=True, CP=None):
  return SimpleNamespace(params=params, CP=CP or cp(), CP_SP=SimpleNamespace(),
                         is_offroad=lambda: parked, ignition=True, engaged=not parked)


def test_native_defaults_match_donor(params):
  assert params.get("HondaBoschARadar", return_default=True) is True
  assert params.get("HondaLiveLearningGas", return_default=True) is None
  assert params.get("ExperimentalMode", return_default=True) is False
  assert params.get("HondaStopAccel", return_default=True) == -2.0
  assert params.get("HondaStoppingDecelRateLong", return_default=True) == 0.3
  assert params.get("HondaVEgoStopping", return_default=True) == 0.5
  assert params.get("HondaVEgoStarting", return_default=True) == 0.5
  assert params.get("NrdrRoenAccelerationLimits", return_default=True) is False
  assert params.get("NrdrCruiseMismatchCorrection", return_default=True) == 100.0
  assert params.get("NrdrStandstillGapExtra", return_default=True) == 0.0
  for personality in ("Aggressive", "Standard", "Relaxed", "Econ"):
    assert params.get(f"LongPidTuneScale{personality}", return_default=True) == 100


@pytest.mark.parametrize("stored", [None, False, True])
def test_startup_snapshot_preserves_choice_without_writing_before_identification(params, stored):
  from openpilot.sunnypilot.selfdrive.car.interfaces import initialize_params
  if stored is not None:
    params.put_bool("HondaLiveLearningGas", stored, block=True)
  values = {key: value for entry in initialize_params(params) for key, value in entry.items()}
  assert values["HondaLiveLearningGas"] is (True if stored is None else stored)
  assert values["HondaBoschARadar"] is True
  assert params.get("HondaLiveLearningGas") is stored


@pytest.mark.parametrize("brand,stored,expected", [
  ("honda", None, True), ("honda", False, False), ("honda", True, True), ("toyota", None, None),
])
def test_identified_honda_initializes_missing_choice_only(params, monkeypatch, brand, stored, expected):
  from openpilot.sunnypilot.selfdrive.car import interfaces
  if stored is not None:
    params.put_bool("HondaLiveLearningGas", stored, block=True)
  for name in ("prepare_drive_config", "configure_firmware_transport", "_enforce_torque_lateral_control",
               "_initialize_neural_network_lateral_control", "_initialize_intelligent_cruise_button_management",
               "_initialize_torque_lateral_control", "_cleanup_unsupported_params"):
    monkeypatch.setattr(interfaces, name, lambda *args, **kwargs: None)
  monkeypatch.setattr(interfaces.STATSLOGSP, "raw", lambda *args, **kwargs: None)
  CI = SimpleNamespace(CP=SimpleNamespace(brand=brand, to_dict=dict), CP_SP=SimpleNamespace())
  interfaces.setup_interfaces(CI, params)
  assert params.get("HondaLiveLearningGas") is expected


def test_parked_ready_stock_cache_can_save_next_drive_without_changing_authority(params):
  current = state(params, CP=cp(long=False))
  current.params.put_bool("HondaLiveLearningGas", True, block=True)
  assert current.ignition and current.is_offroad()
  assert save_live_gas_choice(current, False)
  assert params.get_bool("HondaLiveLearningGas") is False
  deadline = time.monotonic() + 1.0
  while not params.get_bool("OnroadCycleRequested") and time.monotonic() < deadline:
    time.sleep(0.01)
  assert params.get_bool("OnroadCycleRequested") is True
  assert current.CP.openpilotLongitudinalControl is False
  assert current.CP.flags == 0
  assert live_gas_status(current.CP, params, is_offroad=True) == "MVL Off next drive"


def test_saved_off_does_not_clear_active_mvl_flag(params):
  current = state(params, CP=cp(mvl=True))
  assert save_live_gas_choice(current, False)
  assert current.CP.flags == HondaFlags.MVL_LONGITUDINAL
  assert live_gas_status(current.CP, params, is_offroad=False) == "MVL active\nOff next drive"


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_live_choice_rejects_nonboolean(params, value):
  assert not save_live_gas_choice(state(params), value)
  assert params.get("HondaLiveLearningGas") is None


def test_onroad_choice_rejected_without_side_effect(params):
  assert not save_live_gas_choice(state(params, parked=False), False)
  assert params.get("HondaLiveLearningGas") is None
  assert params.get("OnroadCycleRequested") is None


@pytest.mark.parametrize("bundle", [{}, {"platform": CAR.HONDA_INSIGHT},
                                    {"platform": CAR.HONDA_CIVIC, "brand": "honda"}, []])
def test_mismatched_or_incomplete_selection_rejected(params, bundle):
  params.put("CarPlatformBundle", bundle, block=True)
  assert not live_gas_available(cp(), SimpleNamespace(), params)


def test_matching_selection_and_cached_stock_identity_accepted(params):
  params.put("CarPlatformBundle", {"platform": CAR.HONDA_INSIGHT, "brand": "honda"}, block=True)
  assert live_gas_available(cp(long=False), SimpleNamespace(), params)
  assert not live_gas_available(None, SimpleNamespace(), params)
  assert not live_gas_available(cp(), None, params)


def test_malformed_physical_selection_is_not_missing(params):
  from pathlib import Path
  Path(params.get_param_path("CarPlatformBundle")).write_text("invalid JSON")
  assert not live_gas_available(cp(), SimpleNamespace(), params)


def test_radar_preference_does_not_mutate_live_cp_or_request_restart(params):
  current = state(params, parked=False)
  assert save_radar_choice(current, False)
  assert params.get_bool("HondaBoschARadar") is False
  assert current.CP.radarUnavailable is False
  assert params.get("OnroadCycleRequested") is None


def test_comma4_stock_choice_uses_existing_bosch_handover_contract(params):
  current = state(params)
  current.CP.alphaLongitudinalAvailable = True
  params.put_bool("HondaEnforceStockLongitudinal", True, block=True)
  assert save_stock_choice(current, False)
  assert not params.get_bool("HondaEnforceStockLongitudinal")
  assert current.CP.openpilotLongitudinalControl
  assert params.get("OnroadCycleRequested") is None


def test_comma4_stock_choice_rejects_onroad_edits(params):
  params.put_bool("HondaEnforceStockLongitudinal", True, block=True)
  assert not save_stock_choice(state(params, parked=False), False)
  assert params.get_bool("HondaEnforceStockLongitudinal")


@pytest.mark.parametrize("current,active", [(cp(), True), (cp(mvl=True), False), (cp(long=False), False),
                                          (cp(fingerprint=CAR.HONDA_CIVIC), False),
                                          (cp(brand="toyota", fingerprint="TOYOTA_RAV4"), False)])
def test_experimental_and_econ_only_in_imported_mode(params, current, active):
  assert donor_bosch_a_mode(current) is active
  assert save_experimental_choice(state(params, CP=current), True) is active
  assert params.get_bool("ExperimentalMode") is active
  assert personality_value(current, 3) == (3 if active else 2)


@pytest.mark.parametrize("value", [None, True, -1, 4, "3"])
def test_malformed_personality_uses_donor_standard(value):
  assert personality_value(cp(), value) == 1
