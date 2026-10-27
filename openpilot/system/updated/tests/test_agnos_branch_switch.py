"""Branch switching stages the requested OS before handing over software.

All partition operations are mocked; these tests never install an OS.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpilot.system.updated import updated


class TestAgnosBranchSwitch(unittest.TestCase):
  def test_unchanged_os_needs_no_partition_work(self):
    with mock.patch.object(updated.HARDWARE, "get_os_version", return_value="19.9"), \
         mock.patch.object(updated, "run", return_value="19.9"), \
         mock.patch.object(updated, "set_consistent_flag") as consistent, \
         mock.patch("openpilot.common.hardware.comma.agnos.flash_agnos_update") as flash:
      updated.handle_agnos_update()
      consistent.assert_not_called()
      flash.assert_not_called()

  def test_upgrade_and_rollback_both_invalidate_before_staging(self):
    for current, target in (("19.7", "19.9"), ("19.9", "19.7")):
      with self.subTest(current=current, target=target), \
           mock.patch.object(updated.HARDWARE, "get_os_version", return_value=current), \
           mock.patch.object(updated, "run", return_value=target), \
           mock.patch.object(updated, "get_agnos_manifest_path", return_value="/target/agnos.json"), \
           mock.patch.object(updated, "set_consistent_flag") as consistent, \
           mock.patch("openpilot.common.hardware.comma.agnos.get_target_slot_number", return_value=1), \
           mock.patch("openpilot.common.hardware.comma.agnos.flash_agnos_update") as flash:
        order = mock.Mock()
        order.attach_mock(consistent, "consistent")
        order.attach_mock(flash, "flash")
        updated.handle_agnos_update()
        self.assertEqual(order.mock_calls, [
          mock.call.consistent(False),
          mock.call.flash("/target/agnos.json", 1, updated.cloudlog),
        ])

  def test_failed_staging_cannot_mark_update_ready(self):
    with mock.patch.object(updated.HARDWARE, "get_os_version", return_value="19.9"), \
         mock.patch.object(updated, "run", return_value="19.7"), \
         mock.patch.object(updated, "get_agnos_manifest_path", return_value="/target/agnos.json"), \
         mock.patch.object(updated, "set_consistent_flag") as consistent, \
         mock.patch("openpilot.common.hardware.comma.agnos.get_target_slot_number", return_value=1), \
         mock.patch("openpilot.common.hardware.comma.agnos.flash_agnos_update", side_effect=OSError("interrupted")):
      with self.assertRaisesRegex(OSError, "interrupted"):
        updated.handle_agnos_update()
      consistent.assert_called_once_with(False)

  def test_manifest_lookup_supports_both_project_layouts(self):
    for rel_path in ("openpilot/common/hardware/comma/agnos.json", "system/hardware/tici/agnos.json"):
      with self.subTest(path=rel_path), tempfile.TemporaryDirectory() as root:
        manifest = Path(root, rel_path)
        manifest.parent.mkdir(parents=True)
        manifest.write_text("[]")
        self.assertEqual(updated.get_agnos_manifest_path(root), str(manifest))


if __name__ == "__main__":
  unittest.main()
