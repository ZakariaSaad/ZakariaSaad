"""SunnyLink settings for the existing drive-latched Insight EPS controller.

This module never reads back firmware, flashes hardware, rewrites calibration or
changes an active drive snapshot. It only admits owner settings off road.
"""
import base64
import gzip

from openpilot.cereal import custom, messaging
from opendbc.car.structs import car
from openpilot.selfdrive.navigation.offroad import require_offroad
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.selection import (
  ATTESTATION_KEY, DRIVE_KEY, FIRMWARE_BUILD, FIRMWARE_SHA256, FirmwareDriveConfig, _validate_config,
  firmware_attestation, firmware_confirmed, get_drive_config, is_supported_vehicle, valid_attestation_vin,
)

ENABLED = "HondaEpsFirmwareEnabled"
STATUS = "HondaEpsFirmwareStatus"
CONFIRM = "HondaEpsFirmwareConfirmInstalled"
SETTINGS = frozenset((ENABLED, "HondaEpsFirmwareGeometry", "HondaEpsFirmwareResponse"))
WRITE_KEYS = SETTINGS | frozenset((CONFIRM,))
PROTECTED = frozenset(("HondaEpsFirmwareVerifiedSha256", ATTESTATION_KEY, DRIVE_KEY, STATUS))


def _car_params(params):
  # Persistent records describe the last identified vehicle off road; current
  # records describe the drive whose immutable snapshot supplies effective state.
  suffix = "Persistent" if params.get_bool("IsOffroad") else ""
  cp_bytes, sp_bytes = params.get("CarParams" + suffix), params.get("CarParamsSP" + suffix)
  if cp_bytes is None or sp_bytes is None:
    raise ValueError("Vehicle identification is unavailable. Let SunnyPilot identify the vehicle before enabling this controller.")
  try:
    return messaging.log_from_bytes(cp_bytes, car.CarParams), messaging.log_from_bytes(sp_bytes, custom.CarParamsSP)
  except Exception as error:
    raise ValueError("The saved vehicle configuration cannot be read; controller selection is blocked.") from error


def _eligibility(params, updates=None):
  updates = updates or {}
  blockers = []
  CP = CP_SP = None
  try:
    CP, CP_SP = _car_params(params)
    supported = is_supported_vehicle(CP, CP_SP)
    if not supported:
      blockers.append("Requires the identified Honda Insight Bosch TXM-A040 modified EPS and compatible sp-steer 4096 torque transport. " +
                      "Let the next drive identify the supported configuration.")
  except ValueError as error:
    supported = False
    blockers.append(str(error))
  if not firmware_confirmed(params, CP, CP_SP):
    blockers.append("The existing installed-firmware confirmation is missing or does not match the reviewed TXM-A040 image. " +
                    "Use the explicit owner-confirmation action only if " + FIRMWARE_BUILD + " is already installed. No firmware is flashed.")
  config = FirmwareDriveConfig(
    geometry=updates.get("HondaEpsFirmwareGeometry", params.get("HondaEpsFirmwareGeometry", return_default=True)),
    response=updates.get("HondaEpsFirmwareResponse", params.get("HondaEpsFirmwareResponse", return_default=True)),
  )
  try:
    _validate_config(config)
  except ValueError as error:
    blockers.append(str(error))
  return supported, blockers, CP, CP_SP, config


def status(params) -> dict:
  supported, blockers, CP, CP_SP, config = _eligibility(params)
  requested, offroad = params.get_bool(ENABLED), params.get_bool("IsOffroad")
  latched = None
  if params.get(DRIVE_KEY) is not None and CP is not None and CP_SP is not None:
    try:
      latched = get_drive_config(params, CP, CP_SP) is not None
    except ValueError as error:
      blockers.append(str(error))
  effective = latched if not offroad else False
  if offroad:
    phase = "Off road; requested setting applies on the next drive cycle."
    if latched is not None:
      phase += " Last drive selected " + ("Insight EPS." if latched else "the ordinary controller.")
  elif latched is None:
    phase = "Current drive selection is unavailable; requested settings do not replace an active controller."
  else:
    phase = "Current drive: " + ("Insight EPS controller." if latched else "ordinary controller.")
    if requested != latched:
      phase += " Requested change is pending the next drive cycle."
    phase += " Controller settings can be changed only while off road and disengaged."
  description = "Requested: " + ("ON. " if requested else "OFF. ") + phase
  if blockers:
    description += " Enabling blocked: " + " ".join(blockers)
  ordinary = [label for key, label in (("EnforceTorqueControl", "Enforce Torque Control"),
                                      ("NeuralNetworkLateralControl", "Neural Network Lateral Control")) if params.get_bool(key)]
  if ordinary:
    description += " Stored ordinary options (" + ", ".join(ordinary) + ") are preserved; Insight EPS takes priority when admitted."
  if firmware_confirmed(params, CP, CP_SP):
    description += " Installed build: " + FIRMWARE_BUILD + "; owner installation attestation, not ECU firmware readback."
  description += f" Calibration: geometry={config.geometry}, response={config.response}."
  return {"supported": supported, "eligible": not blockers, "requested": requested, "effective": effective is True,
          "latched": latched, "pending": not offroad and latched is not None and requested != latched,
          "reason": " ".join(blockers), "description": description}


def annotate_schema(schema, params):
  """Use the existing frontend description field; no hosted widget is needed."""
  description = status(params)["description"]
  for panel in schema.get("panels", []):
    for section in panel.get("sections", []):
      for item in section.get("items", []):
        if item.get("key") == ENABLED:
          item["description"] = description


def decode_update(key: str, encoded: str, compression=False):
  if key not in WRITE_KEYS:
    raise ValueError("Unsupported Insight EPS setting")
  raw = base64.b64decode(encoded, validate=True)
  if compression:
    raw = gzip.decompress(raw)
  value = raw.decode("utf-8")
  if key in (ENABLED, CONFIRM):
    if value.lower() not in ("true", "false", "1", "0"):
      raise ValueError("Insight EPS enable must be true or false")
    return value.lower() in ("true", "1")
  return value


def apply_settings(params, updates: dict) -> dict:
  """Apply a complete settings batch off road; never touch firmware or latches."""
  if not updates or not set(updates).issubset(SETTINGS):
    raise ValueError("Unsupported Insight EPS settings batch")
  if ENABLED in updates and type(updates[ENABLED]) is not bool:
    raise ValueError("Insight EPS enable must be true or false")
  for key, allowed in (("HondaEpsFirmwareGeometry", ("mvl", "james")), ("HondaEpsFirmwareResponse", ("source", "route32"))):
    if key in updates and updates[key] not in allowed:
      raise ValueError("Unsupported " + key)
  require_offroad(params)
  if updates.get(ENABLED, params.get_bool(ENABLED)):
    _, blockers, *_ = _eligibility(params, updates)
    if blockers:
      raise ValueError("Cannot enable Insight EPS controller: " + " ".join(blockers))
  previous = {key: params.get(key) for key in updates}
  require_offroad(params)
  try:
    # Complete calibration values before making the enable request visible.
    for key in sorted(updates, key=lambda name: name == ENABLED):
      params.put(key, updates[key], block=True)
      if params.get(key) != updates[key]:
        raise OSError("Could not persist " + key)
  except Exception as error:
    failures = []
    for key, value in previous.items():
      try:
        params.remove(key) if value is None else params.put(key, value, block=True)
        if params.get(key) != value:
          failures.append(key)
      except Exception:
        failures.append(key)
    detail = "; could not restore " + ", ".join(failures) if failures else ""
    raise ValueError(str(error) + detail) from error
  return status(params)


def apply_enabled(params, enabled: bool) -> dict:
  return apply_settings(params, {ENABLED: enabled})


def confirm_installed_firmware(params, confirmed: bool) -> dict:
  """Explicit UI recovery only: never called by migration, boot, or backup restore."""
  if confirmed is not True:
    raise ValueError("An explicit owner confirmation of " + FIRMWARE_BUILD + " is required")
  require_offroad(params)
  CP, CP_SP = _car_params(params)
  if not is_supported_vehicle(CP, CP_SP):
    raise ValueError("Firmware attestation requires the identified Honda Insight TXM-A040 modified EPS and supported transport")
  # Preserve valid legacy proof first; new statements need a real vehicle
  # identity so moving this comma to another same-platform Insight cannot reuse it.
  if firmware_confirmed(params, CP, CP_SP):
    return status(params)
  if not valid_attestation_vin(str(CP.carVin)):
    raise ValueError("A valid identified vehicle VIN is required for a new owner attestation; let SunnyPilot identify this Insight first")
  receipt = firmware_attestation(CP, CP_SP)
  old_receipt, old_hash = params.get(ATTESTATION_KEY), params.get("HondaEpsFirmwareVerifiedSha256")
  require_offroad(params)
  latest_CP, latest_SP = _car_params(params)
  if firmware_attestation(latest_CP, latest_SP) != receipt:
    raise ValueError("Vehicle identification changed during owner confirmation")
  try:
    params.put(ATTESTATION_KEY, receipt, block=True)
    if params.get(ATTESTATION_KEY) != receipt:
      raise OSError("Could not persist owner firmware attestation")
    require_offroad(params)
    params.put("HondaEpsFirmwareVerifiedSha256", FIRMWARE_SHA256, block=True)
    if not firmware_confirmed(params, CP, CP_SP):
      raise OSError("Could not persist installed-firmware confirmation")
  except Exception:
    # The record cannot leave a newly confirmed hash with mismatched identity.
    for key, value in (("HondaEpsFirmwareVerifiedSha256", old_hash), (ATTESTATION_KEY, old_receipt)):
      params.remove(key) if value is None else params.put(key, value, block=True)
    raise
  return status(params)
