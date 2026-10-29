"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.selfdrive.car.sync_sunnylink_params import CAR_LIST_JSON_OUT

ONROAD_BRIGHTNESS_MIGRATION_VERSION: str = "1.0"
ONROAD_BRIGHTNESS_TIMER_MIGRATION_VERSION: str = "1.0"

# index → seconds mapping for OnroadScreenOffTimer (SSoT)
ONROAD_BRIGHTNESS_TIMER_VALUES = {0: 3, 1: 5, 2: 7, 3: 10, 4: 15, 5: 30, **{i: (i - 5) * 60 for i in range(6, 16)}}
VALID_TIMER_VALUES = set(ONROAD_BRIGHTNESS_TIMER_VALUES.values())


def _resolve_brand(_params) -> str:
  bundle = _params.get("CarPlatformBundle")
  if isinstance(bundle, dict) and bundle.get("brand"):
    return str(bundle["brand"])

  # Auto-fingerprinted cars have no bundle, fall back to the last known CarParams.
  CP_bytes = _params.get("CarParamsPersistent")
  if CP_bytes is None:
    return ""

  # Never raises: callers rely on "" to mean "brand unknown, skip the migration".
  try:
    from openpilot.cereal import messaging  # lazy: avoids heavy import at module level
    from opendbc.car.structs import car
    return str(messaging.log_from_bytes(CP_bytes, car.CarParams).brand)
  except Exception as e:
    cloudlog.exception(f"params_migration: failed to resolve brand from CarParamsPersistent: {e}")
    return ""


def _migrate_car_platform_bundle(_params):
  bundle = _params.get("CarPlatformBundle")
  if bundle is None:
    return

  old_platform = bundle.get("platform")
  if not old_platform:
    return

  from opendbc.car.fingerprints import MIGRATION  # lazy: avoids heavy import at module level
  if old_platform not in MIGRATION:
    return

  new_platform = str(MIGRATION[old_platform])

  with open(CAR_LIST_JSON_OUT) as f:
    car_list = json.load(f)

  candidates = [(k, v) for k, v in car_list.items() if v.get("platform") == new_platform]
  if candidates:
    old_model = bundle.get("model")
    key, data = next(((k, v) for k, v in candidates if v.get("model") == old_model), candidates[0])
    bundle = {**data, "name": key}
  else:
    bundle["platform"] = new_platform

  _params.put("CarPlatformBundle", bundle, block=True)
  cloudlog.info(f"params_migration: CarPlatformBundle migrated {old_platform!r} -> {new_platform!r}")


def _migrate_tesla_mads_screen_button(_params):
  # TeslaMadsScreenButton defaults to Off for fresh installs, but the screen button was previously always
  # active on Teslas with a vehicle bus. Seed existing Tesla installs with 3-finger to preserve that.
  try:
    if _params.get("TeslaMadsScreenButton") is not None:
      return

    if _resolve_brand(_params) != "tesla":
      return

    from opendbc.sunnypilot.car.tesla.values import MadsScreenButtonType  # lazy: avoids heavy import at module level
    _params.put("TeslaMadsScreenButton", MadsScreenButtonType.THREE_FINGER, block=True)
    cloudlog.info("params_migration: seeded TeslaMadsScreenButton with 3-finger to preserve existing behavior")
  except Exception as e:
    cloudlog.exception(f"Error migrating TeslaMadsScreenButton: {e}")


def _migrate_model_bundle_slots(_params):
  # Pre-split, a chestnut user's big-model selection lived in the single
  # ActiveBundle. Seed once; an intentionally empty modern slot means Default,
  # not permission to restore a legacy model again on the next boot.
  try:
    if _params.get_bool("ModelManager_BundleSlotsMigrated"):
      return
    active_json = _params.get("ModelManager_ActiveJson")
    modern = any(_params.get(key) is not None for key in (
      "ModelManager_ActiveBundleChestnut", "ModelManager_ModelsCache_Chestnut", "ModelManager_SelectionRequest",
      "ModelManager_SelectionState", "NavigationModelCandidate")) or isinstance(active_json, dict) and 'chestnut' in active_json
    if not modern:
      source = 'ModelManager_ActiveBundleUSBGPU'
      bundle = _params.get(source)
      if bundle is None:
        source = 'ModelManager_ActiveBundle'
        bundle = _params.get(source)
      if bundle is not None:
        from openpilot.selfdrive.navigation.candidate_identity import is_retired_bundle
        if is_retired_bundle(bundle):
          cloudlog.warning('params_migration: original Bigv2 is retired; legacy bytes retained without selecting it')
        else:
          _params.put("ModelManager_ActiveBundleChestnut", bundle, block=True)
          if _params.get("ModelManager_ActiveBundleChestnut") != bundle:
            raise RuntimeError('Could not persist legacy model-slot migration')
          cloudlog.info(f"params_migration: seeded ModelManager_ActiveBundleChestnut from {source}")
    _params.put_bool("ModelManager_BundleSlotsMigrated", True, block=True)
    if not _params.get_bool("ModelManager_BundleSlotsMigrated"):
      raise RuntimeError('Could not persist model-slot migration marker')
  except Exception as e:
    cloudlog.exception(f"Error migrating model bundle slots: {e}")


def run_migration(_params):
  # migrate OnroadScreenOffBrightness
  if _params.get("OnroadScreenOffBrightnessMigrated") != ONROAD_BRIGHTNESS_MIGRATION_VERSION:
    try:
      val = _params.get("OnroadScreenOffBrightness", return_default=True)
      if val >= 2:  # old: 5%, new: Screen Off
        new_val = val + 1
        _params.put("OnroadScreenOffBrightness", new_val, block=True)
        log_str = f"Successfully migrated OnroadScreenOffBrightness from {val} to {new_val}."
      else:
        log_str = "Migration not required for OnroadScreenOffBrightness."

      _params.put("OnroadScreenOffBrightnessMigrated", ONROAD_BRIGHTNESS_MIGRATION_VERSION, block=True)
      cloudlog.info(log_str + f" Setting OnroadScreenOffBrightnessMigrated to {ONROAD_BRIGHTNESS_MIGRATION_VERSION}")
    except Exception as e:
      cloudlog.exception(f"Error migrating OnroadScreenOffBrightness: {e}")

  # migrate OnroadScreenOffTimer
  if _params.get("OnroadScreenOffTimerMigrated") != ONROAD_BRIGHTNESS_TIMER_MIGRATION_VERSION:
    try:
      val = _params.get("OnroadScreenOffTimer", return_default=True)
      if val not in VALID_TIMER_VALUES:
        _params.put("OnroadScreenOffTimer", 15, block=True)
        log_str = f"Successfully migrated OnroadScreenOffTimer from {val} to 15 (default)."
      else:
        log_str = "Migration not required for OnroadScreenOffTimer."

      _params.put("OnroadScreenOffTimerMigrated", ONROAD_BRIGHTNESS_TIMER_MIGRATION_VERSION, block=True)
      cloudlog.info(log_str + f" Setting OnroadScreenOffTimerMigrated to {ONROAD_BRIGHTNESS_TIMER_MIGRATION_VERSION}")
    except Exception as e:
      cloudlog.exception(f"Error migrating OnroadScreenOffTimer: {e}")

  _migrate_car_platform_bundle(_params)

  # seed TeslaMadsScreenButton for existing Tesla installs
  _migrate_tesla_mads_screen_button(_params)

  # seed the chestnut model slot from the pre-split single slot
  _migrate_model_bundle_slots(_params)

  # Convert a completed corrected-native selection into the ordinary Chestnut
  # slot before any model process starts. Legacy flags are migration input only.
  try:
    from openpilot.sunnypilot.models.selection import _recover_or_migrate
    _recover_or_migrate(_params)
  except Exception as error:
    cloudlog.exception(f"Error migrating native model selection: {error}")
