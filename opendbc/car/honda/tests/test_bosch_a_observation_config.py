"""Observation configuration must not confer longitudinal or radar authority.

Uses production parameter construction and the normal stock RadarInterface;
does not initialize hardware, open messaging sockets, or transmit CAN.
"""
from dataclasses import asdict
from itertools import product
from types import SimpleNamespace

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda import carcontroller, interface as honda_interface
from opendbc.car.honda.bosch_a_observation import observation_config_supported, observation_enabled
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.values import CAR, CarControllerParams, HondaSafetyFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.interfaces import _initialize_honda


OBS = HondaFlagsSP.BOSCH_A_RADAR_OBSERVATION


def parameters(candidate=CAR.HONDA_INSIGHT, *, observer=None, alpha=False, stock=True,
               integration=False, icbm=False, mode=None):
  fingerprint = gen_empty_fingerprint()
  saved_gas_lookup = CarControllerParams.BOSCH_GAS_LOOKUP_V
  try:
    cp = CarInterface.get_params(candidate, fingerprint, [], alpha, False, False)
    sp = CarInterface.get_params_sp(cp, candidate, fingerprint, [], alpha, False, False)
  finally:
    CarControllerParams.BOSCH_GAS_LOOKUP_V = saved_gas_lookup
  if mode is not None:
    setattr(cp, mode, True)
  # Main setup can set this later as well; the final helper is rechecked there.
  sp.pcmCruiseSpeed = not icbm
  settings = {'HondaEnforceStockLongitudinal': int(stock), 'BoschALongitudinalEnabled': int(integration)}
  if observer is not None:
    settings['BoschARadarObservationEnabled'] = int(observer)
  _initialize_honda(cp, sp, settings)
  return cp, sp


@pytest.mark.parametrize('observer', [None, False, True])
@pytest.mark.parametrize('alpha,stock,integration,icbm', list(product((False, True), repeat=4)))
def test_observer_requires_an_explicit_unambiguous_stock_selection(observer, alpha, stock, integration, icbm):
  cp, sp = parameters(observer=observer, alpha=alpha, stock=stock, integration=integration, icbm=icbm)
  expected = bool(observer and not alpha and stock and not integration and not icbm)
  assert bool(sp.flags & OBS) == expected
  assert observation_enabled(cp, sp) == expected


def test_selected_observer_changes_only_its_software_bit():
  off, off_sp = parameters(observer=False)
  on, on_sp = parameters(observer=True)
  assert on.to_dict() == off.to_dict()
  expected = asdict(off_sp)
  expected['flags'] |= OBS
  assert asdict(on_sp) == expected
  assert not on.openpilotLongitudinalControl
  assert on.pcmCruise and on_sp.pcmCruiseSpeed and on.radarUnavailable
  assert not on.safetyConfigs[0].safetyParam & HondaSafetyFlags.BOSCH_LONG
  assert not on_sp.flags & HondaFlagsSP.BOSCH_A_LONG


@pytest.mark.parametrize('candidate', sorted(set(CAR) - {CAR.HONDA_INSIGHT}))
def test_other_honda_fingerprints_do_not_select_observation(candidate):
  off, off_sp = parameters(candidate, observer=False)
  on, on_sp = parameters(candidate, observer=True)
  assert not observation_enabled(on, on_sp)
  assert on.to_dict() == off.to_dict()
  assert asdict(on_sp) == asdict(off_sp)


@pytest.mark.parametrize('mode', ['passive', 'dashcamOnly', 'notCar'])
def test_noncontrolling_modes_are_rejected_without_parameter_repair(mode):
  off, off_sp = parameters(observer=False, mode=mode)
  on, on_sp = parameters(observer=True, mode=mode)
  assert not observation_enabled(on, on_sp)
  assert on.to_dict() == off.to_dict()
  assert asdict(on_sp) == asdict(off_sp)


@pytest.mark.parametrize('conflict', ['op_long', 'pcm_cruise', 'icbm', 'stock_flag', 'integration_flag',
                                      'radar_available', 'gas_interceptor', 'safety_long', 'safety_model',
                                      'safety_empty', 'passive', 'dashcamOnly', 'notCar', 'brand', 'fingerprint'])
def test_late_configuration_conflict_invalidates_a_previously_selected_observer(conflict):
  cp, sp = parameters(observer=True)
  assert observation_enabled(cp, sp)
  if conflict == 'op_long':
    cp.openpilotLongitudinalControl = True
  elif conflict == 'pcm_cruise':
    cp.pcmCruise = False
  elif conflict == 'icbm':
    sp.pcmCruiseSpeed = False
  elif conflict == 'stock_flag':
    sp.flags = int(sp.flags) & ~int(HondaFlagsSP.STOCK_LONGITUDINAL)
  elif conflict == 'integration_flag':
    sp.flags |= HondaFlagsSP.BOSCH_A_LONG
  elif conflict == 'radar_available':
    cp.radarUnavailable = False
  elif conflict == 'gas_interceptor':
    sp.enableGasInterceptor = True
  elif conflict == 'safety_long':
    cp.safetyConfigs[0].safetyParam |= HondaSafetyFlags.BOSCH_LONG.value
  elif conflict == 'safety_model':
    cp.safetyConfigs[0].safetyModel = structs.CarParams.SafetyModel.noOutput
  elif conflict == 'safety_empty':
    cp.safetyConfigs = []
  elif conflict in ('passive', 'dashcamOnly', 'notCar'):
    setattr(cp, conflict, True)
  elif conflict == 'brand':
    cp.brand = 'toyota'
  else:
    cp.carFingerprint = CAR.HONDA_CIVIC_BOSCH
  before_cp, before_sp = cp.to_dict(), asdict(sp)
  assert not observation_config_supported(cp, sp)
  assert not observation_enabled(cp, sp)
  # Eligibility is pure: final main startup owns revoking the stored bit.
  assert cp.to_dict() == before_cp and asdict(sp) == before_sp


def test_reinitializing_honda_revokes_only_observation_bit():
  cp, sp = parameters(observer=True)
  sp.flags |= 64  # Reserved proposal bit must remain caller-owned.
  before_cp = cp.to_dict()
  before_sp = asdict(sp)
  _initialize_honda(cp, sp, {'HondaEnforceStockLongitudinal': 1})
  expected = before_sp | {'flags': before_sp['flags'] & ~int(OBS)}
  assert cp.to_dict() == before_cp
  assert asdict(sp) == expected
  assert not observation_enabled(cp, sp)


def test_other_brands_reusing_bit_are_not_modified():
  cp, sp = parameters(observer=True)
  cp.brand = 'toyota'
  before = cp.to_dict(), asdict(sp)
  _initialize_honda(cp, sp, {'BoschARadarObservationEnabled': 1})
  assert (cp.to_dict(), asdict(sp)) == before
  assert not observation_enabled(cp, sp)


def test_normal_radar_interface_and_empty_track_cadence_stay_identical():
  off, off_sp = parameters(observer=False)
  on, on_sp = parameters(observer=True)
  radars = [RadarInterface(off, off_sp), RadarInterface(on, on_sp)]
  assert all(ri.bosch_a is None and ri.rcp is None and ri.radar_off_can for ri in radars)
  outputs = [[], []]
  for _ in range(15):
    for values, radar in zip(outputs, radars, strict=True):
      result = radar.update([])
      values.append(None if result is None else result.to_dict())
  assert outputs[0] == outputs[1]
  assert any(result is not None for result in outputs[0])
  assert all(not result.get('points', []) for result in outputs[0] if result is not None)


def test_missing_configuration_never_selects_observation():
  assert not observation_config_supported(None, None)
  assert not observation_enabled(None, None)


def test_stock_initialization_never_calls_diagnostic_or_can_io(monkeypatch):
  calls = []

  def forbidden(*args, **kwargs):
    calls.append((args, kwargs))
    raise AssertionError('Stock observation must never call diagnostic or CAN I/O')

  for name in ('disable_ecu', 'clear_all_dtcs', 'clear_ecu_dtcs'):
    monkeypatch.setattr(honda_interface, name, forbidden)
  for observer in (False, True):
    cp, sp = parameters(observer=observer)
    CarInterface.init(cp, sp, forbidden, forbidden)
  assert not calls


def test_stock_controller_all_frames_match_and_never_construct_alpha_long_commands(tmp_path, monkeypatch):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  monkeypatch.setattr(carcontroller, 'HondaParamWriter', lambda: SimpleNamespace(put_many=lambda values: None))
  controllers = []
  for observer in (False, True):
    cp, sp = parameters(observer=observer)
    ci = CarInterface(cp, sp)
    state, _ = ci.update([])
    state.cruiseState.enabled = state.cruiseState.available = True
    state.cruiseState.speed = 30.
    state.vCruise = 108.
    controllers.append((ci, state))

  # Last two situations deliberately violate normal producer authority. Stock
  # selection must still never enter the alpha-long CAN construction branch.
  situations = [(False, False, False, False, False, False),
                (True, False, False, False, False, False),
                (True, False, False, True, False, False),
                (False, False, True, False, False, False),
                (True, False, False, False, True, False),
                (False, False, True, False, False, True),
                (True, True, False, False, False, False),
                (False, True, True, False, False, False)]
  addresses = set()
  for tick in range(64):
    enabled, active, cancel, resume, gas, brake = situations[tick // 8]
    cc = structs.CarControl.new_message(enabled=enabled, longActive=active, latActive=enabled)
    cc.cruiseControl.cancel, cc.cruiseControl.resume = cancel, resume
    cc.actuators.accel = 2. if tick % 2 else -3.5
    cc.actuators.longControlState = 'stopping' if tick % 3 else 'pid'
    cc.actuators.torque = .12
    outputs = []
    for ci, state in controllers:
      state.vEgo = (0., .2, 5.)[tick % 3]
      state.aEgo = 0.
      state.standstill = state.vEgo == 0.
      state.gasPressed, state.brakePressed = gas, brake
      result, frames = ci.apply(cc.as_reader(), structs.CarControlSP(), tick * 10_000_000)
      outputs.append((result.to_dict(), frames))
    assert outputs[0] == outputs[1]
    frame_ids = {address for address, _, _ in outputs[0][1]}
    assert not frame_ids & {0x1DF, 0x1EF, 0x18DAB0F1}  # ACC_CONTROL, ACC_CONTROL_ON, radar diagnostic
    addresses.update(frame_ids)
  assert {0xE5, 0x296} <= addresses  # Supplemental and stock cancel/resume are exercised.
