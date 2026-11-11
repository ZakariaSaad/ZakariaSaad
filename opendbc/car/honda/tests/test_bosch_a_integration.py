"""NRDR startup behavior through the existing target-branch parameter boundary."""
from dataclasses import asdict

import pytest

from opendbc.car import Bus, gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.values import CAR, DBC, HONDA_BOSCH_A, CarControllerParams, HondaFlags, HondaSafetyFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.interfaces import _initialize_honda


def parameters(candidate=CAR.HONDA_INSIGHT, *, enabled=True, alpha_long=True, stock_long=False, mvl=False,
               passive=False, dashcam_only=False, not_car=False):
  """Exercise actual get_params and the target's equivalent startup snapshot.

  enabled refers to the donor HondaBoschARadar setting, not an extra port switch.
  None omits that key, matching donor's missing-interface-config fallback.
  """
  fingerprint = gen_empty_fingerprint()
  table = CarControllerParams.BOSCH_GAS_LOOKUP_V
  try:
    cp = CarInterface.get_params(candidate, fingerprint, [], alpha_long, False, False)
    sp = CarInterface.get_params_sp(cp, candidate, fingerprint, [], alpha_long, False, False)
  finally:
    CarControllerParams.BOSCH_GAS_LOOKUP_V = table
  cp.passive, cp.dashcamOnly, cp.notCar = passive, dashcam_only, not_car
  settings = {"HondaEnforceStockLongitudinal": int(stock_long), "HondaLiveLearningGas": int(mvl)}
  if enabled is not None:
    settings["HondaBoschARadar"] = int(enabled)
  _initialize_honda(cp, sp, settings)
  return cp, sp


@pytest.mark.parametrize("candidate", sorted(HONDA_BOSCH_A))
@pytest.mark.parametrize("radar", (False, True))
@pytest.mark.parametrize("alpha", (False, True))
def test_donor_radar_selection_is_independent_of_longitudinal(candidate, radar, alpha):
  cp, sp = parameters(candidate, enabled=radar, alpha_long=alpha)
  assert cp.openpilotLongitudinalControl == alpha
  assert cp.radarUnavailable == (not radar)
  assert DBC[candidate][Bus.radar] == "honda_bosch_a_radar"
  assert (RadarInterface(cp, sp).bosch_a is not None) == radar
  assert bool(cp.safetyConfigs[-1].safetyParam & HondaSafetyFlags.BOSCH_LONG) == alpha
  assert not sp.flags & 32  # No previous project-specific BOSCH_A_LONG flag.


@pytest.mark.parametrize("candidate", sorted(set(CAR) - HONDA_BOSCH_A))
def test_radar_setting_does_not_change_other_platforms(candidate):
  off, off_sp = parameters(candidate, enabled=False)
  on, on_sp = parameters(candidate, enabled=True)
  assert on.to_dict() == off.to_dict()
  assert asdict(on_sp) == asdict(off_sp)


@pytest.mark.parametrize("candidate", sorted(HONDA_BOSCH_A))
def test_missing_interface_selection_keeps_radar_unavailable(candidate):
  cp, sp = parameters(candidate, enabled=None)
  assert cp.radarUnavailable
  assert RadarInterface(cp, sp).bosch_a is None


@pytest.mark.parametrize("mvl", (False, True))
@pytest.mark.parametrize("alpha", (False, True))
@pytest.mark.parametrize("stock", (False, True))
def test_donor_mvl_startup_flag_and_force_stock_authority(mvl, alpha, stock):
  cp, sp = parameters(mvl=mvl, alpha_long=alpha, stock_long=stock)
  assert bool(cp.flags & HondaFlags.MVL_LONGITUDINAL) == (mvl and alpha and not stock)
  assert cp.openpilotLongitudinalControl == (alpha and not stock)
  assert not cp.radarUnavailable  # Force-stock does not disable radar decoding.
  if stock:
    assert cp.pcmCruise
    assert sp.flags & HondaFlagsSP.STOCK_LONGITUDINAL
    assert not cp.safetyConfigs[-1].safetyParam & HondaSafetyFlags.BOSCH_LONG


def test_startup_snapshot_replaces_old_mvl_flag():
  cp, sp = parameters(mvl=True)
  _initialize_honda(cp, sp, {"HondaBoschARadar": 1, "HondaLiveLearningGas": 0})
  assert not cp.flags & HondaFlags.MVL_LONGITUDINAL
  assert not cp.radarUnavailable
