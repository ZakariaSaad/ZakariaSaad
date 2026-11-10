"""Final card startup gates with real target parameters and native serialization.

Socket creation, fingerprint I/O and unrelated setup are replaced with local
fixtures. Car.__init__, the final passive decision, RadarInterface construction,
Params writes and CP/CP_SP schemas are real. No messaging or CAN is transmitted.
"""
from types import SimpleNamespace

import pytest

from opendbc.car import structs
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.honda.tests.test_bosch_a_integration import parameters
from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.selfdrive.car import card as card_module
from openpilot.sunnypilot.selfdrive.controls.lib.bosch_a import gate


@pytest.mark.parametrize('field', ('passive', 'dashcamOnly', 'notCar'))
def test_final_vehicle_flags_disable_optional_gate_and_restore_receive_configuration(field):
  cp, cp_sp = parameters(enabled=True)
  assert gate.active_for(cp, cp_sp)
  # Preserve unrelated/current-future bits when clearing this one Honda flag.
  cp_sp.flags |= 1 << 20
  original_flags = int(cp_sp.flags)
  safety = cp.safetyConfigs[0].to_dict(), cp_sp.safetyParam
  setattr(cp, field, True)
  assert not gate.vehicle_supported(cp, cp_sp)
  assert not gate.active_for(cp, cp_sp)
  gate.finalize_startup(cp, cp_sp)
  assert cp_sp.flags == original_flags & ~int(HondaFlagsSP.BOSCH_A_LONG)
  assert cp.radarUnavailable
  assert (cp.safetyConfigs[0].to_dict(), cp_sp.safetyParam) == safety


def test_other_brands_reusing_numeric_flag_are_not_changed():
  cp, cp_sp = parameters(enabled=True)
  cp.brand = 'toyota'
  cp.passive = True
  before = cp.to_dict(), int(cp_sp.flags)
  assert not gate.startup_selected(cp, cp_sp)
  assert not gate.active_for(cp, cp_sp)
  gate.finalize_startup(cp, cp_sp)
  assert (cp.to_dict(), int(cp_sp.flags)) == before


def test_unsupported_honda_does_not_change_radar_availability():
  cp, cp_sp = parameters(CAR.HONDA_CIVIC, enabled=False)
  cp_sp.flags |= HondaFlagsSP.BOSCH_A_LONG
  before = cp.to_dict()
  assert not gate.startup_selected(cp, cp_sp)
  gate.finalize_startup(cp, cp_sp)
  assert not cp_sp.flags & HondaFlagsSP.BOSCH_A_LONG
  assert cp.to_dict() == before


def construct_car(monkeypatch, tmp_path, *, enabled=True, master_enabled=True, has_controller=True,
                  stock_long=False, late_dashcam=False, late_not_car=False, injected=False):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  params = Params()
  params.put_bool('OpenpilotEnabledToggle', master_enabled, block=True)
  params.put_bool('AlphaLongitudinalEnabled', True, block=True)
  params.put_bool('BoschALongitudinalEnabled', enabled, block=True)
  params.put_bool('HondaEnforceStockLongitudinal', stock_long, block=True)
  cp, cp_sp = parameters(enabled=enabled, stock_long=stock_long)
  cp.dashcamOnly = late_dashcam
  cp.notCar = late_not_car
  ci = SimpleNamespace(CP=cp, CP_SP=cp_sp, CC=SimpleNamespace() if has_controller else None,
                       CS=SimpleNamespace())
  calls = []

  def no_send(*_args, **_kwargs):
    raise AssertionError('This startup test must not transmit messages or CAN')

  monkeypatch.setattr(card_module.messaging, 'sub_sock', lambda *_args, **_kwargs: object())
  monkeypatch.setattr(card_module.messaging, 'SubMaster', lambda *_args, **_kwargs: SimpleNamespace())
  monkeypatch.setattr(card_module.messaging, 'PubMaster', lambda *_args, **_kwargs:
    SimpleNamespace(sock={'sendcan': SimpleNamespace(send=no_send)}, send=no_send))
  monkeypatch.setattr(card_module.messaging, 'recv_one_retry', lambda *_args: SimpleNamespace(can=[object()]))
  monkeypatch.setattr(card_module, 'Ratekeeper', lambda *_args, **_kwargs: SimpleNamespace(remaining=0.))
  # This hook covers unrelated lateral/model setup, not the final controller gate.
  monkeypatch.setattr(card_module.sunnypilot_interfaces, 'setup_interfaces', lambda *_args: calls.append('setup'))

  def get_car(*args):
    settings = {key: value for item in args[-2] for key, value in item.items()}
    assert bool(settings['BoschALongitudinalEnabled']) is enabled
    calls.append('get_car')
    return ci

  def create_radar(final_cp, final_cp_sp):
    calls.append(('radar', bool(final_cp.passive), bool(final_cp.radarUnavailable), int(final_cp_sp.flags)))
    return RadarInterface(final_cp, final_cp_sp)

  monkeypatch.setattr(card_module, 'get_car', get_car)
  monkeypatch.setitem(card_module.interfaces, cp.carFingerprint, SimpleNamespace(RadarInterface=create_radar))
  supplied_radar = object()
  result = card_module.Car(ci, supplied_radar) if injected else card_module.Car()
  with car.CarParams.from_bytes(params.get('CarParams')) as stored:
    saved_cp = stored.to_dict()
  with custom.CarParamsSP.from_bytes(params.get('CarParamsSP')) as stored:
    saved_flags = int(stored.flags)
  return result, calls, saved_cp, saved_flags, params, supplied_radar


@pytest.mark.parametrize('master_enabled,has_controller', ((False, True), (True, False), (False, False)))
def test_opted_in_radar_is_constructed_after_final_passive_latch(monkeypatch, tmp_path, master_enabled, has_controller):
  result, calls, saved, flags, params, _ = construct_car(
    monkeypatch, tmp_path, master_enabled=master_enabled, has_controller=has_controller)
  radar_call, = [call for call in calls if isinstance(call, tuple)]
  assert radar_call[1:3] == (True, True)
  assert not radar_call[3] & HondaFlagsSP.BOSCH_A_LONG
  assert result.CP.passive and result.CP.radarUnavailable
  assert result.RI.bosch_a is None and result.RI.rcp is None
  assert saved['passive'] and saved['radarUnavailable']
  assert not flags & HondaFlagsSP.BOSCH_A_LONG
  assert result.CP.safetyConfigs[0].safetyModel == structs.CarParams.SafetyModel.noOutput
  assert params.get_bool('BoschALongitudinalEnabled')  # Do not rewrite the owner's saved preference.


def test_active_opt_in_constructs_decoder_and_publishes_matching_configuration(monkeypatch, tmp_path):
  result, calls, saved, flags, _, _ = construct_car(monkeypatch, tmp_path)
  radar_call, = [call for call in calls if isinstance(call, tuple)]
  assert radar_call[1:3] == (False, False)
  assert radar_call[3] & HondaFlagsSP.BOSCH_A_LONG
  assert result.RI.bosch_a is not None
  assert not saved['passive'] and not saved['radarUnavailable']
  assert flags & HondaFlagsSP.BOSCH_A_LONG
  assert result.CP.safetyConfigs[0].safetyModel != structs.CarParams.SafetyModel.noOutput


@pytest.mark.parametrize('enabled,stock_long', ((False, False), (True, True)))
def test_unflagged_paths_keep_existing_radar_constructor_order(monkeypatch, tmp_path, enabled, stock_long):
  result, calls, saved, flags, _, _ = construct_car(
    monkeypatch, tmp_path, enabled=enabled, stock_long=stock_long, master_enabled=False)
  radar_call, = [call for call in calls if isinstance(call, tuple)]
  # Baseline unflagged construction precedes card's passive assignment.
  assert radar_call[1:3] == (False, True)
  assert not radar_call[3] & HondaFlagsSP.BOSCH_A_LONG
  assert result.CP.passive and saved['passive'] and saved['radarUnavailable']
  assert not flags & HondaFlagsSP.BOSCH_A_LONG
  if stock_long:
    assert flags & HondaFlagsSP.STOCK_LONGITUDINAL
    assert not saved['openpilotLongitudinalControl']


@pytest.mark.parametrize('mode', ('late_dashcam', 'late_not_car'))
def test_late_ineligible_configuration_is_revoked_before_decoder_construction(monkeypatch, tmp_path, mode):
  result, calls, saved, flags, _, _ = construct_car(monkeypatch, tmp_path, **{mode: True})
  radar_call, = [call for call in calls if isinstance(call, tuple)]
  assert radar_call[2] and not radar_call[3] & HondaFlagsSP.BOSCH_A_LONG
  assert result.RI.bosch_a is None and saved['radarUnavailable']
  assert not flags & HondaFlagsSP.BOSCH_A_LONG


def test_injected_radar_identity_is_retained_but_published_optional_gate_is_disabled(monkeypatch, tmp_path):
  result, calls, saved, flags, _, supplied_radar = construct_car(
    monkeypatch, tmp_path, master_enabled=False, injected=True)
  assert result.RI is supplied_radar
  assert not calls  # No normal fingerprint/setup/radar construction on injection.
  assert saved['passive'] and saved['radarUnavailable']
  assert not flags & HondaFlagsSP.BOSCH_A_LONG
  assert not gate.active_for(result.CP, result.CP_SP)
