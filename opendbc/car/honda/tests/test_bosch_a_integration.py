"""Target-branch startup configuration and receive-only Bosch-A regressions.

These fixtures exercise the real Honda parameter and startup functions. They do
not initialize a vehicle interface, disable an ECU, or send CAN messages.
"""
from dataclasses import asdict

import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint
from opendbc.car.honda.bosch_a_radar import BOSCH_A_DBC_NAME
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.values import CAR, DBC, HONDA_BOSCH_A, CarControllerParams, HondaFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP, HondaSafetyFlagsSP
from opendbc.sunnypilot.car.interfaces import _initialize_honda


def parameters(candidate=CAR.HONDA_INSIGHT, *, enabled=False, alpha_long=True, stock_long=False,
               passive=False, dashcam_only=False, not_car=False):
  """Build fresh target CP/CP_SP and apply the startup-only opt-in.

  ``enabled=None`` omits the parameter, exercising its default-OFF behavior.
  ``alpha_long`` is the existing Honda selection, independent of the new opt-in.
  """
  fingerprint = gen_empty_fingerprint()
  # Some existing Honda parameter branches assign this class-level actuator
  # table. A multi-platform receive-only fixture must not leak it to later
  # controller tests; no controller is constructed by this helper.
  gas_lookup = CarControllerParams.BOSCH_GAS_LOOKUP_V
  try:
    cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long, False, False)
    cp_sp = CarInterface.get_params_sp(cp, candidate, fingerprint, [], alpha_long, False, False)
  finally:
    CarControllerParams.BOSCH_GAS_LOOKUP_V = gas_lookup
  cp.passive = passive
  cp.dashcamOnly = dashcam_only
  cp.notCar = not_car
  settings = {"HondaEnforceStockLongitudinal": "1" if stock_long else "0"}
  if enabled is not None:
    settings["BoschALongitudinalEnabled"] = "1" if enabled else "0"
  _initialize_honda(cp, cp_sp, settings)
  return cp, cp_sp


@pytest.mark.parametrize("candidate", sorted(HONDA_BOSCH_A))
@pytest.mark.parametrize("enabled", (None, False))
def test_missing_or_disabled_opt_in_keeps_bosch_radar_unavailable(candidate, enabled):
  cp, cp_sp = parameters(candidate, enabled=enabled)
  assert cp.openpilotLongitudinalControl
  assert cp.radarUnavailable
  assert not cp_sp.flags & HondaFlagsSP.BOSCH_A_LONG
  radar = RadarInterface(cp, cp_sp)
  assert radar.bosch_a is None
  assert radar.rcp is None


@pytest.mark.parametrize("candidate", sorted(HONDA_BOSCH_A))
def test_opt_in_changes_only_receive_availability_delay_and_non_safety_flag(candidate):
  off, off_sp = parameters(candidate, enabled=False)
  on, on_sp = parameters(candidate, enabled=True)
  assert on.openpilotLongitudinalControl
  assert not on.radarUnavailable
  assert on.radarDelay == pytest.approx(.1)
  assert DBC[candidate][Bus.radar] == BOSCH_A_DBC_NAME
  expected = off.to_dict()
  expected.update(radarUnavailable=False, radarDelay=on.radarDelay)
  assert on.to_dict() == expected
  expected_sp = asdict(off_sp)
  expected_sp["flags"] |= HondaFlagsSP.BOSCH_A_LONG
  assert asdict(on_sp) == expected_sp
  assert [config.to_dict() for config in on.safetyConfigs] == [config.to_dict() for config in off.safetyConfigs]
  assert on_sp.safetyParam == off_sp.safetyParam
  assert RadarInterface(on, on_sp).bosch_a is not None


@pytest.mark.parametrize("candidate", sorted(set(CAR) - HONDA_BOSCH_A))
def test_unsupported_hardware_is_unchanged_by_opt_in(candidate):
  off, off_sp = parameters(candidate, enabled=False)
  on, on_sp = parameters(candidate, enabled=True)
  assert on.to_dict() == off.to_dict()
  assert asdict(on_sp) == asdict(off_sp)
  assert not on_sp.flags & HondaFlagsSP.BOSCH_A_LONG
  assert RadarInterface(on, on_sp).bosch_a is None


@pytest.mark.parametrize("candidate", (CAR.HONDA_INSIGHT, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G))
def test_stock_selection_takes_priority_over_opt_in(candidate):
  stock, stock_sp = parameters(candidate, enabled=False, stock_long=True)
  opted_in, opted_in_sp = parameters(candidate, enabled=True, stock_long=True)
  assert opted_in.to_dict() == stock.to_dict()
  assert asdict(opted_in_sp) == asdict(stock_sp)
  assert not opted_in.openpilotLongitudinalControl
  assert opted_in.radarUnavailable
  assert opted_in_sp.flags & HondaFlagsSP.STOCK_LONGITUDINAL
  assert opted_in_sp.safetyParam & HondaSafetyFlagsSP.STOCK_LONGITUDINAL
  assert not opted_in_sp.flags & HondaFlagsSP.BOSCH_A_LONG
  assert RadarInterface(opted_in, opted_in_sp).bosch_a is None


@pytest.mark.parametrize("candidate", sorted(HONDA_BOSCH_A))
def test_opt_in_does_not_select_openpilot_longitudinal(candidate):
  off, off_sp = parameters(candidate, enabled=False, alpha_long=False)
  on, on_sp = parameters(candidate, enabled=True, alpha_long=False)
  assert not on.openpilotLongitudinalControl
  assert on.to_dict() == off.to_dict()
  assert asdict(on_sp) == asdict(off_sp)
  assert RadarInterface(on, on_sp).bosch_a is None


@pytest.mark.parametrize("mode", ("passive", "dashcam_only", "not_car"))
def test_non_controlling_configuration_cannot_enable_bosch_a(mode):
  off, off_sp = parameters(enabled=False, **{mode: True})
  on, on_sp = parameters(enabled=True, **{mode: True})
  assert on.to_dict() == off.to_dict()
  assert asdict(on_sp) == asdict(off_sp)
  assert on.radarUnavailable
  assert not on_sp.flags & HondaFlagsSP.BOSCH_A_LONG
  assert RadarInterface(on, on_sp).bosch_a is None


def test_platform_eligibility_excludes_canfd_radarless_and_alternate_radar():
  for candidate in CAR:
    flags = candidate.config.flags
    expected = bool(flags & HondaFlags.BOSCH and
                    not flags & (HondaFlags.BOSCH_CANFD | HondaFlags.BOSCH_RADARLESS | HondaFlags.BOSCH_ALT_RADAR))
    assert (candidate in HONDA_BOSCH_A) == expected
  assert CAR.HONDA_INSIGHT in HONDA_BOSCH_A


def test_nidec_receive_path_retains_points_and_radar_faults():
  cp, cp_sp = parameters(CAR.HONDA_CIVIC, enabled=True)
  radar = RadarInterface(cp, cp_sp)
  assert radar.bosch_a is None and radar.rcp is not None
  packer = CANPacker(DBC[cp.carFingerprint][Bus.radar])

  def messages(radar_state):
    frames = [packer.make_can_msg(0x400, 1, {"RADAR_STATE": radar_state})]
    for address in [*range(0x430, 0x43A), *range(0x440, 0x446)]:
      values = {"LONG_DIST": 255}
      if address == 0x430:
        values.update(LONG_DIST=30, LAT_DIST=1, REL_SPEED=-2, NEW_TRACK=1)
      frames.append(packer.make_can_msg(address, 1, values))
    return frames

  result = radar.update([(1_000_000_000, messages(0x79))])
  assert not any(result.errors.to_dict().values())
  assert len(result.points) == 1
  point = result.points[0]
  assert (point.dRel, point.yRel, point.vRel) == pytest.approx((30, -1, -2))
  faulted = radar.update([(1_050_000_000, messages(0x69))])
  assert faulted.errors.radarFault and faulted.errors.wrongConfig
