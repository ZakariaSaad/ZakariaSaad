"""Optional CLI selections use the ordinary model transaction service."""
from unittest.mock import patch

import pytest

from openpilot.selfdrive.navigation.candidate_identity import MODEL_ID, RETIRED_MODEL_ID
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.selection import select_candidate
from openpilot.selfdrive.navigation.tests.test_activation import ParamsStub
from openpilot.sunnypilot.models.navigation_bundle import NAV_REF


def test_select_uses_the_shared_model_slot_service():
  params = ParamsStub()
  with patch('openpilot.sunnypilot.models.selection.select_model_now') as service:
    select_candidate(params, MODEL_ID, offroad=True)
  service.assert_called_once_with(params, source='chestnut', ref=NAV_REF, offroad=True)


@pytest.mark.parametrize('candidate', [RETIRED_MODEL_ID, 'unknown-navigation-model'])
def test_retired_or_unknown_identity_never_enters_service(candidate):
  with patch('openpilot.sunnypilot.models.selection.select_model_now') as service:
    with pytest.raises(NavigationModelDisabled, match='retired|unknown'):
      select_candidate(ParamsStub(), candidate, offroad=True)
  service.assert_not_called()


def test_shared_service_refusal_is_preserved():
  params = ParamsStub(NavigationModelDrivingEnabled=False)
  with patch('openpilot.sunnypilot.models.selection.select_model_now', side_effect=NavigationModelDisabled('deviceState.started is true')):
    with pytest.raises(NavigationModelDisabled, match='deviceState.started'):
      select_candidate(params, MODEL_ID, offroad=False)
  assert not params.get_bool('NavigationModelDrivingEnabled')
