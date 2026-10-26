"""Legacy entrypoints delegate to the shared model transaction service."""
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID, RETIRED_MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.selection import select_candidate, set_driving_enabled
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub


def test_select_and_enable_share_identical_service_semantics():
  params = ParamsStub()
  with patch('openpilot.sunnypilot.models.selection.select_navigation_now') as service:
    select_candidate(params, MODEL_ID, offroad=True)
    set_driving_enabled(params, True, candidate_id=MODEL_ID, offroad=True)
  assert service.call_count == 2
  for call in service.call_args_list:
    assert call.args == (params,) and call.kwargs == {'enabled': True, 'offroad': True}


def test_disable_uses_same_offroad_request_path():
  params = ParamsStub()
  with patch('openpilot.sunnypilot.models.selection.select_navigation_now') as service:
    set_driving_enabled(params, False, offroad=True)
  service.assert_called_once_with(params, enabled=False, offroad=True)


@pytest.mark.parametrize('candidate', [RETIRED_MODEL_ID, 'unknown-navigation-model'])
def test_retired_or_unknown_identity_never_enters_service(candidate):
  with patch('openpilot.sunnypilot.models.selection.select_navigation_now') as service:
    with pytest.raises(NavigationModelDisabled, match='retired|unknown'):
      select_candidate(ParamsStub(), candidate, offroad=True)
  service.assert_not_called()


def test_shared_service_refusal_is_preserved():
  params = ParamsStub(NavigationModelDrivingEnabled=False)
  with patch('openpilot.sunnypilot.models.selection.select_navigation_now', side_effect=NavigationModelDisabled('deviceState.started is true')):
    with pytest.raises(NavigationModelDisabled, match='deviceState.started'):
      set_driving_enabled(params, True, offroad=False)
  assert not params.get_bool('NavigationModelDrivingEnabled')
