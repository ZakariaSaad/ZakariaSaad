"""Drive-latched admission shared by card, controls and both model runners.

The diagnostic identifier proves an EPS family, never its flashed contents. The
owner's installed-image confirmation is deliberately separate from file checking.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from opendbc.car.honda.values import HondaFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.transport import uses_supported_transport

FIRMWARE_SHA256 = "7ba3743248d30758f8423bcb2a1c65f0d2d2d3563a830b8268bf448659b28a38"
FIRMWARE_BUILD = "39990-TXM,A040-PTM-20261016b"
ATTESTATION_KEY = "HondaEpsFirmwareAttestation"
PROFILE = "insight_txm_a040_20261016b"
DRIVE_KEY = "HondaEpsFirmwareDriveConfig"


@dataclass(frozen=True)
class FirmwareDriveConfig:
  geometry: str = "mvl"
  response: str = "source"
  profile: str = PROFILE
  firmware_sha256: str = FIRMWARE_SHA256


def _eps_versions(CP) -> tuple[bytes, ...]:
  return tuple(bytes(fw.fwVersion).rstrip(b"\x00 ") for fw in CP.carFw if str(fw.ecu) == "eps")


def is_supported_vehicle(CP, CP_SP) -> bool:
  """Check the actual sp-steer torque transport, never deprecated CarParams maps."""
  if CP is None or CP_SP is None:
    return False
  flags = int(CP.flags)
  return (str(CP.brand) == "honda" and str(CP.carFingerprint) == "HONDA_INSIGHT" and
          bool(flags & HondaFlags.BOSCH) and not flags & (HondaFlags.NIDEC | HondaFlags.BOSCH_RADARLESS | HondaFlags.BOSCH_CANFD) and
          bool(int(CP_SP.flags) & HondaFlagsSP.EPS_MODIFIED) and
          str(CP.steerControlType) == "torque" and CP.lateralTuning.which() == "torque" and
          uses_supported_transport(CP, CP_SP) and
          bool(_eps_versions(CP)) and all(version == b"39990-TXM,A040" for version in _eps_versions(CP)))


def _vehicle_identity(CP, CP_SP) -> str:
  # Deliberately bind only controller-relevant fields: card subsequently updates
  # passive/MADS/safety fields without changing this lateral contract.
  identity = {
    "brand": str(CP.brand), "fingerprint": str(CP.carFingerprint),
    "platform_flags": int(CP.flags), "eps_modified": bool(int(CP_SP.flags) & HondaFlagsSP.EPS_MODIFIED),
    "eps": sorted(version.hex() for version in _eps_versions(CP)),
    "control_type": str(CP.steerControlType), "tuning": CP.lateralTuning.which(),
    "transport": "linear_4096" if uses_supported_transport(CP, CP_SP) else "unsupported",
  }
  return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def valid_attestation_vin(vin) -> bool:
  import re
  return isinstance(vin, str) and re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin) is not None and len(set(vin)) > 1


def firmware_attestation(CP, CP_SP) -> dict:
  """Owner installation statement bound to this identified vehicle; no ECU readback."""
  return {"version": 1, "kind": "owner_installation_attestation", "build": FIRMWARE_BUILD,
          "firmware_sha256": FIRMWARE_SHA256, "vehicle_identity": _vehicle_identity(CP, CP_SP),
          "vin": str(CP.carVin)}


def firmware_confirmed(params, CP, CP_SP) -> bool:
  if params.get("HondaEpsFirmwareVerifiedSha256") != FIRMWARE_SHA256:
    return False
  receipt = params.get(ATTESTATION_KEY)
  if receipt is None:
    # Preserve the valid pre-migration owner confirmation without inventing a
    # receipt. Malformed newer JSON is not a legacy confirmation.
    return not Path(params.get_param_path(ATTESTATION_KEY)).exists()
  if CP is None or CP_SP is None or not isinstance(receipt, dict):
    return False
  expected = firmware_attestation(CP, CP_SP)
  if set(receipt) != set(expected) or type(receipt.get("version")) is not int:
    return False
  return receipt == expected and valid_attestation_vin(receipt["vin"])


def _validate_config(config: FirmwareDriveConfig) -> None:
  if (config.profile != PROFILE or config.firmware_sha256 != FIRMWARE_SHA256 or config.geometry not in ("mvl", "james") or
      config.response not in ("source", "route32")):
    raise ValueError("Invalid Honda EPS firmware drive calibration/geometry")
  from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.core import validate_calibration
  validate_calibration()


def _read_snapshot(params):
  snapshot = params.get(DRIVE_KEY)
  # Params' typed JSON decoder returns None for malformed JSON and JSON null.
  # Distinguish that from an unpublished key; corruption cannot select fallback.
  if snapshot is None and Path(params.get_param_path(DRIVE_KEY)).exists():
    raise ValueError("Corrupted Honda EPS firmware drive snapshot")
  return snapshot


def _is_insight(CP) -> bool:
  return str(CP.brand) == "honda" and str(CP.carFingerprint) == "HONDA_INSIGHT"


def get_drive_config(params, CP, CP_SP) -> FirmwareDriveConfig | None:
  """Never reread owner settings during a drive. Invalid admitted data fails closed."""
  snapshot = _read_snapshot(params)
  if snapshot is None:
    if _is_insight(CP):
      raise ValueError("Missing Insight drive snapshot; restart the onroad cycle")
    return None
  if (not isinstance(snapshot, dict) or type(snapshot.get("version")) is not int or snapshot["version"] != 1 or
      type(snapshot.get("enabled")) is not bool):
    raise ValueError("Invalid Honda EPS firmware drive snapshot")
  if not snapshot["enabled"]:
    return None
  if not is_supported_vehicle(CP, CP_SP) or snapshot.get("vehicle_identity") != _vehicle_identity(CP, CP_SP):
    raise ValueError("Honda EPS firmware snapshot does not match CarParams")
  raw_config = snapshot.get("config")
  if not isinstance(raw_config, dict) or set(raw_config) != {"geometry", "response", "profile", "firmware_sha256"}:
    raise ValueError("Incomplete Honda EPS firmware drive configuration")
  try:
    config = FirmwareDriveConfig(**raw_config)
  except (KeyError, TypeError) as exc:
    raise ValueError("Invalid Honda EPS firmware drive configuration") from exc
  _validate_config(config)
  return config


def prepare_drive_config(params, CP, CP_SP) -> FirmwareDriveConfig | None:
  """Publish once, synchronously before CarParams; reuse on same-drive restarts.

  Manager clears the snapshot on every onroad transition. Even a disabled
  snapshot is latched, so changing settings cannot select a new controller while
  controlsd or a model process restarts during the drive.
  """
  if _read_snapshot(params) is not None:
    return get_drive_config(params, CP, CP_SP)
  if _is_insight(CP) and params.get("CarParams") is not None:
    raise ValueError("Insight drive snapshot was lost after CarParams publication")
  config = None
  if (params.get_bool("HondaEpsFirmwareEnabled") and is_supported_vehicle(CP, CP_SP) and
      firmware_confirmed(params, CP, CP_SP)):
    config = FirmwareDriveConfig(geometry=params.get("HondaEpsFirmwareGeometry", return_default=True),
                                 response=params.get("HondaEpsFirmwareResponse", return_default=True))
    _validate_config(config)
  snapshot = {"version": 1, "enabled": config is not None}
  if config is not None:
    snapshot.update(vehicle_identity=_vehicle_identity(CP, CP_SP), config=asdict(config))
  params.put(DRIVE_KEY, snapshot, block=True)
  return config


def drive_config_for_model(params, CP) -> FirmwareDriveConfig | None:
  snapshot = _read_snapshot(params)
  if snapshot is None:
    if _is_insight(CP):
      raise ValueError("Missing Insight model drive snapshot; restart the onroad cycle")
    return None
  if (not isinstance(snapshot, dict) or type(snapshot.get("version")) is not int or snapshot["version"] != 1 or
      type(snapshot.get("enabled")) is not bool):
    raise ValueError("Invalid Honda EPS firmware model snapshot")
  if not snapshot["enabled"]:
    return None
  # The synchronous snapshot write in setup_interfaces precedes both blocking
  # CarParams publications. Only admitted firmware mode needs the SP record.
  from openpilot.cereal import custom, messaging
  CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
  return get_drive_config(params, CP, CP_SP)
