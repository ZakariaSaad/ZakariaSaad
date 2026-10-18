import pytest
import uuid

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.params import Params
from openpilot.selfdrive.car.helpers import convert_to_capnp


@pytest.fixture
def params(tmp_path, monkeypatch):
  path = str(tmp_path / 'params')
  monkeypatch.setenv('PARAMS_ROOT', path)
  monkeypatch.setenv('LOG_ROOT', str(tmp_path / 'logs'))
  monkeypatch.setenv('OPENPILOT_PREFIX', 'honda_eps_' + uuid.uuid4().hex[:12])
  return Params(path)


@pytest.fixture
def car_factory():
  def make_car(candidate=CAR.HONDA_INSIGHT, versions=(b'39990-TXM,A040',), native_sp=True):
    fingerprint = gen_empty_fingerprint()
    firmware = [structs.CarParams.CarFw(ecu=structs.CarParams.Ecu.eps, address=0x18da30f1,
                                      fwVersion=version, brand='honda') for version in versions]
    CP = CarInterface.get_params(candidate, fingerprint, firmware, False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, firmware, False, False, False)
    CP.carFw = [{'ecu': 'eps', 'address': fw.address, 'fwVersion': fw.fwVersion, 'brand': 'honda'} for fw in firmware]
    return CP, convert_to_capnp(CP_SP) if native_sp else CP_SP
  return make_car
