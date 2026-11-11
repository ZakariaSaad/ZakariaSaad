"""NRDR saved Honda choices adapted to the target's existing settings pages.

Selection identity and Live Learning Gas admission mirror nrdr-development
622a1479. Saved choices never mutate the running CarParams/controller.
"""
from pathlib import Path

from opendbc.car.honda.values import CAR
from openpilot.nrdr.features.longitudinal.bosch_a.gate import vehicle_supported
from openpilot.nrdr.features.longitudinal.policy import mvl_longitudinal_enabled, longitudinal_personality


MVL_DESCRIPTION = (
  "Selects full MVL longitudinal control. MVL owns gas, braking and following distances. " +
  "Turn off to use NRDR's shared Bosch-A longitudinal implementation with Alpha Long. " +
  "Speed-limit assistance, curve-speed control and custom cruise-button increments keep their own settings. " +
  "Change while parked. Applies next drive; no device reboot."
)
RADAR_DESCRIPTION = (
  "EXPERIMENTAL - Decodes the factory radar on supported Honda Bosch-A vehicles with stock longitudinal or openpilot Alpha Long. " +
  "The reverse-engineered object format has not been road-validated on every supported model, so disable this setting if radar tracks " +
  "look incorrect. With Alpha Long, these tracks feed openpilot's lead tracking and longitudinal commands; factory Honda AEB/CMBS " +
  "remains unavailable."
)
EXPERIMENTAL_DESCRIPTION = (
  "Use the driving model for end-to-end longitudinal planning, including stopping for red lights and stop signs. " +
  "Off uses ACC planning. The set speed remains an upper bound. This is experimental; mistakes should be expected."
)


def confirmed_vehicle_identity(CP, fingerprint=None, brand=None, *, selection_present=None):
  if CP is None:
    return None
  detected_fingerprint = str(getattr(CP, "carFingerprint", "") or "")
  detected_brand = str(getattr(CP, "brand", "") or "").lower()
  selected_fingerprint = str(fingerprint or "")
  selected_brand = str(brand or "").lower()
  if selection_present is None:
    selection_present = fingerprint is not None or brand is not None
  if not detected_fingerprint or not detected_brand:
    return None
  if selection_present:
    if not selected_fingerprint or not selected_brand:
      return None
    if selected_fingerprint != detected_fingerprint or selected_brand != detected_brand:
      return None
  return detected_fingerprint, detected_brand


def get_selected_car_identity(params):
  bundle = params.get("CarPlatformBundle")
  present = bundle is not None
  if bundle is None:
    try:
      present = Path(params.get_param_path("CarPlatformBundle")).is_file()
    except (AttributeError, OSError, TypeError, ValueError):
      present = False
  if isinstance(bundle, dict):
    return present, str(bundle.get("platform") or ""), str(bundle.get("brand") or "")
  return present, "", ""


def live_gas_available(CP, CP_SP, params):
  present, fingerprint, brand = get_selected_car_identity(params)
  identity = confirmed_vehicle_identity(CP, fingerprint, brand, selection_present=present)
  return bool(identity is not None and identity[1] == "honda" and identity[0] in CAR and CP_SP is not None)


def save_live_gas_choice(state, value):
  if (type(value) is not bool or not state.is_offroad()
      or not live_gas_available(state.CP, state.CP_SP, state.params)):
    return False
  state.params.put_bool("HondaLiveLearningGas", value, block=True)
  state.params.put_bool("OnroadCycleRequested", True)
  return True


def live_gas_status(CP, params, *, is_offroad):
  selected = params.get_bool("HondaLiveLearningGas")
  if is_offroad:
    return "MVL next drive" if selected else "MVL Off next drive"
  if mvl_longitudinal_enabled(CP):
    return "MVL active" if selected else "MVL active\nOff next drive"
  return "MVL next drive" if selected else "MVL inactive"


def save_radar_choice(state, value):
  # This is the donor's saved parser preference, not a running authority change.
  if type(value) is not bool or getattr(state.CP, "brand", None) != "honda":
    return False
  state.params.put_bool("HondaBoschARadar", value, block=True)
  return True


def save_stock_choice(state, value):
  """Comma 4 counterpart of the target's existing Honda factory-ACC control."""
  if (type(value) is not bool or not state.is_offroad()
      or not live_gas_available(state.CP, state.CP_SP, state.params)):
    return False
  state.params.put_bool("HondaEnforceStockLongitudinal", value, block=True)
  # Match the existing Honda page: the Bosch controller owns its radar
  # hand-back, while Nidec uses the normal cycle request.
  if not getattr(state.CP, "alphaLongitudinalAvailable", False):
    state.params.put_bool("OnroadCycleRequested", True)
  return True


def donor_bosch_a_mode(CP):
  return vehicle_supported(CP) and not mvl_longitudinal_enabled(CP)


def personality_value(CP, value):
  return longitudinal_personality(value, donor_bosch_a_mode(CP))


def save_experimental_choice(state, value):
  if type(value) is not bool or not donor_bosch_a_mode(state.CP):
    return False
  state.params.put_bool("ExperimentalMode", value, block=True)
  return True
