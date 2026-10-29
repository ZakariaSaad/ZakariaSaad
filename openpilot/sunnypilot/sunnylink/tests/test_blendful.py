"""Blendful's native parameter, active-runner eligibility, and SunnyLink wire contract."""
import base64
import gzip
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.cereal import custom
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.sunnypilot.models import helpers
from openpilot.sunnypilot.models.navigation_bundle import navigation_bundle
from openpilot.sunnypilot.sunnylink.capabilities import CAPABILITY_FIELDS, CAR_CAPABILITY_FIELDS, generate_capabilities


@pytest.fixture
def device(tmp_path, monkeypatch):
  # Exercise the real typed/persistent parameter store and actual RPC methods.
  # No SunnyLink connection, GPU, manager, or vehicle process is launched.
  monkeypatch.setenv("PARAMS_ROOT", str(tmp_path / "params"))
  params = Params()
  from openpilot.sunnypilot.sunnylink.athena import sunnylinkd
  monkeypatch.setattr(sunnylinkd, "params", params)
  return SimpleNamespace(params=params, rpc=sunnylinkd)


def schema_item(schema):
  panel = next(p for p in schema["panels"] if p["id"] == "models")
  section = next(s for s in panel["sections"] if s["id"] == "model_behavior")
  return next(item for item in section["items"] if item["key"] == "Blendful")


def wire(value, compressed=False):
  raw = str(value).lower().encode()
  return base64.b64encode(gzip.compress(raw) if compressed else raw).decode()


def test_native_parameter_defaults_off_and_survives_lifecycle_clears(device):
  params = device.params
  assert params.get_type("Blendful") == ParamKeyType.BOOL
  assert params.get_default_value("Blendful") is False
  assert params.get_bool("Blendful") is False
  for flag in (ParamKeyFlag.PERSISTENT, ParamKeyFlag.BACKUP):
    assert b"Blendful" in params.all_keys(flag)
  for value in (True, False):
    params.put_bool("Blendful", value, block=True)
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
      assert Params().get_bool("Blendful") is value


@pytest.mark.parametrize("runner,eligible", [(None, False), ("stock", False), ("tinygrad", True)])
def test_capability_follows_manager_runner_cache_not_highlighted_slot(device, runner, eligible):
  params = device.params
  # Even a saved native navigation bundle does not override the manager's
  # actual runner decision; neither does a Chestnut status/UI selection flag.
  params.put("ModelManager_ActiveBundleChestnut", navigation_bundle(), block=True)
  params.put_bool("ChestnutActive", not eligible, block=True)
  params.put_bool("Blendful", True, block=True)
  if runner is not None:
    params.put("ModelRunnerTypeCache", int(getattr(custom.ModelManagerSP.Runner, runner)), block=True)
  assert generate_capabilities(params)["custom_model_active"] is eligible
  assert params.get_bool("Blendful")  # unsupported Default preserves the saved preference


@pytest.mark.parametrize("chestnut,big_custom,small_custom,eligible", [
  (True, True, False, True),
  (True, False, True, False),
  (False, True, False, False),
  (False, False, True, True),
])
def test_manager_active_source_drives_capability(device, chestnut, big_custom, small_custom, eligible):
  params = device.params
  if big_custom:
    params.put("ModelManager_ActiveBundleChestnut", navigation_bundle(), block=True)
  if small_custom:
    params.put("ModelManager_ActiveBundle", {
      "ref": "ordinary-small", "runner": "tinygrad", "minimumSelectorVersion": helpers.REQUIRED_JSON_VERSION,
    }, block=True)
  with patch.object(helpers, "chestnut_present", return_value=chestnut):
    runner = helpers.get_active_model_runner(params, force_check=True)
  assert (runner == custom.ModelManagerSP.Runner.tinygrad) is eligible
  assert generate_capabilities(params)["custom_model_active"] is eligible


def test_actual_metadata_serves_model_toggle_and_capability_reason(device):
  device.params.put("ModelRunnerTypeCache", int(custom.ModelManagerSP.Runner.tinygrad), block=True)
  schema = json.loads(gzip.decompress(base64.b64decode(device.rpc.getParamsMetadata())))
  item = schema_item(schema)
  assert item["widget"] == "toggle"
  assert item["title"] == "Blendful (Alpha)"
  assert item["enablement"] == [{"type": "capability", "field": "custom_model_active", "equals": True}]
  assert "Default model" in item["description"]
  assert schema["capabilities"]["custom_model_active"] is True
  assert schema["capability_labels"]["custom_model_active"] == "Non-Default model selected"
  assert set(schema["car_capability_fields"]) == set(CAR_CAPABILITY_FIELDS)
  assert set(CAR_CAPABILITY_FIELDS) <= set(CAPABILITY_FIELDS)
  assert not {"custom_model_active", "protocol_version"} & set(CAR_CAPABILITY_FIELDS)
  # Existing custom model selection and Insight EPS capability contract stays served.
  assert "honda_eps_status" in schema["capabilities"]
  panel = next(p for p in schema["panels"] if p["id"] == "models")
  assert any(s["id"] == "model_selection" for s in panel["sections"])


@pytest.mark.parametrize("compressed", [False, True])
def test_remote_toggle_round_trip_controls_runtime_parameter(device, compressed):
  from openpilot.sunnypilot.modeld_v2.blendful import Blendful
  from openpilot.sunnypilot.modeld_v2.tests.test_blendful import frame

  def read_remote():
    response = device.rpc.getParams(["Blendful"], compression=compressed)
    entry = json.loads(response["params"])[0]
    assert entry["key"] == "Blendful" and entry["type"] == int(ParamKeyType.BOOL)
    assert entry["is_compressed"] is compressed
    raw = base64.b64decode(response["Blendful"])
    return (gzip.decompress(raw) if compressed else raw).decode().lower() in ("true", "1")

  assert not read_remote()
  blendful = Blendful()
  output = frame(center=0.5)
  assert blendful.update(output, 20.0, True, 0.0, enabled=Params().get_bool("Blendful")) == 0.0
  device.rpc.saveParams({"Blendful": wire(True, compressed)}, compression=compressed)
  assert read_remote() and Params().get_bool("Blendful")
  for _ in range(100):
    correction = blendful.update(output, 20.0, True, 0.0, enabled=Params().get_bool("Blendful"))
  assert correction > 0.0
  device.rpc.saveParams({"Blendful": wire(False, compressed)}, compression=compressed)
  assert not read_remote() and not Params().get_bool("Blendful")
  assert blendful.update(output, 20.0, True, 0.0, enabled=Params().get_bool("Blendful")) == 0.0
  assert blendful.curvature == blendful.trust == 0.0
