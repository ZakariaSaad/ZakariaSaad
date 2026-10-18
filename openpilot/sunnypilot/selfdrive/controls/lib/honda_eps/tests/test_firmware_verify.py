import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.firmware_verify import RWD_SIZE, verify_rwd


def test_verifier_fails_closed_for_other_images_and_truncation():
  for raw in (b"", b"Z\r\n", bytes(RWD_SIZE - 1), bytes(RWD_SIZE), bytes(RWD_SIZE + 1)):
    with pytest.raises(ValueError):
      verify_rwd(raw)
