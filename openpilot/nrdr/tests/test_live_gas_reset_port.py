"""Donor reset lifecycle against isolated files and simulated Panda telemetry."""
import json
from types import SimpleNamespace as NS

import pytest

from openpilot.nrdr.car import live_gas_reset as reset

HONDA = NS(brand="honda", carFingerprint="HONDA_INSIGHT")
KEEP = {"HondaLiveLearningGas": False, "HondaBrakeFactorParams": 1.3,
        "CalibrationParams": b"preserved", "HondaSpeedFactorParams": 1.2,
        "LongPidTuneScaleStandard": 125}


class Parameters(dict):
  def put(self, key, value, *, block):
    assert block
    self[key] = value

  put_bool = put
  get_raw = dict.get

  def get_bool(self, key):
    return bool(self.get(key))

  def remove(self, key):
    self.pop(key, None)


class Streams(dict):
  fresh = True

  def all_checks(self, services):
    assert set(services) == {"deviceState", "pandaStates"}
    return self.fresh


@pytest.fixture
def parked(tmp_path):
  params = Parameters(IsOffroad=True, **dict.fromkeys(reset.GAS_KEYS, 1.2), **KEEP)
  meta = tmp_path / "honda_learner_meta.json"
  meta.write_text(json.dumps({"car_fingerprint": "HONDA_INSIGHT"}))
  bosch = tmp_path / "star7_honda_gas_learner.json"
  bosch.write_text('{"gas_factor": 1.1, "wind_factor": 0.9}')
  panda = NS(pandaType="redPanda", safetyModel="noOutput", ignitionLine=False,
             ignitionCan=False, controlsAllowed=False)
  sm = Streams(deviceState=NS(started=False), pandaStates=[panda], carParams=HONDA)
  return params, sm, NS(proc=None), meta, bosch


def consume(parked):
  params, sm, card, meta, bosch = parked
  reset.consume_reset(params, sm, card, read_cp=lambda _: HONDA, meta_path=meta, bosch_a_path=bosch)


def test_reset_clears_only_donor_gas_wind_state_once(parked):
  params, _, _, meta, bosch = parked
  reset.request_reset(params, HONDA)
  assert meta.exists() and bosch.exists()
  consume(parked)
  assert not meta.exists() and not bosch.exists()
  assert not set(reset.GAS_KEYS) & params.keys()
  assert {key: params[key] for key in KEEP} == KEEP
  assert params[reset.STATUS] == "Reset complete."
  assert params[reset.REQUEST] is False
  assert params["ParamsVersion"] == 1
  params["HondaGasFactorParams"] = 1.25
  consume(parked)
  assert params["HondaGasFactorParams"] == 1.25
  assert params["ParamsVersion"] == 1


@pytest.mark.parametrize("reason", ["started", "road", "line", "can", "controls", "output", "missing",
                                    "unknown", "stale", "card", "selection", "live_cp", "other_metadata", "unbound"])
def test_rejected_request_is_not_deferred(parked, reason):
  params, sm, card, meta, bosch = parked
  reset.request_reset(params, HONDA)
  panda = sm["pandaStates"][0]
  if reason == "started":
    sm["deviceState"].started = True
  elif reason == "road":
    params["IsOffroad"] = False
  elif reason == "line":
    panda.ignitionLine = True
  elif reason == "can":
    panda.ignitionCan = True
  elif reason == "controls":
    panda.controlsAllowed = True
  elif reason == "output":
    panda.safetyModel = "hondaBosch"
  elif reason == "missing":
    sm["pandaStates"] = []
  elif reason == "unknown":
    panda.pandaType = "unknown"
  elif reason == "stale":
    sm.fresh = False
  elif reason == "card":
    card.proc = object()
  elif reason == "selection":
    params["CarPlatformBundle"] = {"platform": "HONDA_CLARITY", "brand": "honda"}
  elif reason == "live_cp":
    sm["carParams"] = NS(carFingerprint="HONDA_CLARITY", brand="honda")
  elif reason == "other_metadata":
    meta.write_text('{"car_fingerprint":"HONDA_CLARITY"}')
  elif reason == "unbound":
    del params[reset.STATUS]
  consume(parked)
  assert meta.exists() and bosch.exists()
  assert all(params[key] == 1.2 for key in reset.GAS_KEYS)
  assert params[reset.REQUEST] is False
  assert params[reset.STATUS].startswith("Not reset")


def test_no_request_no_writes(parked):
  params, _, _, meta, bosch = parked
  before = dict(params)
  consume(parked)
  assert dict(params) == before and meta.exists() and bosch.exists()


@pytest.mark.parametrize("current", [None, NS(brand="toyota", carFingerprint="TOYOTA_CAMRY"),
                                     NS(brand="honda", carFingerprint="")])
def test_unknown_identity_rejected(parked, current):
  with pytest.raises(ValueError):
    reset.request_reset(parked[0], current)
  assert reset.REQUEST not in parked[0]


def test_partial_storage_failure_not_reported_success(parked, monkeypatch):
  params = parked[0]
  reset.request_reset(params, HONDA)
  monkeypatch.setattr(params, "remove", lambda key: None)
  consume(parked)
  assert params[reset.REQUEST] is False
  assert params[reset.STATUS].startswith("Reset incomplete")


def test_ui_reset_unavailable_in_ready_even_when_forced_offroad(parked, monkeypatch):
  from openpilot.nrdr.ui.settings import live_gas_reset as ui_reset
  params = parked[0]
  ui = NS(params=params, CP=HONDA, ignition=True, is_offroad=lambda: True)
  monkeypatch.setattr(ui_reset, "ui_state", ui)
  assert not ui_reset.available()
  ui_reset.submit()
  assert reset.REQUEST not in params
  ui.ignition = False
  assert ui_reset.available()
  ui_reset.submit()
  assert params[reset.REQUEST] is True


def test_target_raw_adapter_does_not_hide_malformed_remaining_nodes(tmp_path):
  node = tmp_path / "HondaGasFactorParams"
  params = NS(get_param_path=lambda key: str(node))
  node.write_bytes(b"not a number")
  assert reset._raw_value(params, "HondaGasFactorParams") == b"not a number"
  node.unlink()
  assert reset._raw_value(params, "HondaGasFactorParams") is None


def test_native_reset_keys_clear_at_startup_and_preserve_saved_choice(tmp_path):
  from openpilot.common.params import Params, ParamKeyFlag
  params = Params(str(tmp_path / "params"))
  assert params.get(reset.REQUEST, return_default=True) is False
  params.put_bool("HondaLiveLearningGas", False, block=True)
  params.put_bool(reset.REQUEST, True, block=True)
  params.put(reset.STATUS, "Requested: HONDA_INSIGHT", block=True)
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  assert params.get(reset.REQUEST) is None
  assert params.get(reset.STATUS) is None
  assert params.get("HondaLiveLearningGas") is False
