import struct
import unittest
from unittest.mock import Mock, patch

from openpilot.cereal import log
from openpilot.common.hardware.usb import CHESTNUT_FW_VERSION, CHESTNUT_USB_PRODUCT
from openpilot.system.hardware.chestnut.flash import FIRMWARE_PATH, image_product, validate_image
from openpilot.system.hardware.chestnut.monitoring import read_chestnut_state
from openpilot.system.hardware.hardwared import Chestnut


class TestChestnutIntegration(unittest.TestCase):
  def test_bundled_firmware_matches_runtime(self):
    image = FIRMWARE_PATH.read_bytes()
    validate_image(image)
    self.assertEqual(image_product(image), CHESTNUT_USB_PRODUCT)
    self.assertEqual(CHESTNUT_FW_VERSION, "8516b768")

  @patch("openpilot.system.hardware.hardwared.threading.Thread")
  @patch("openpilot.system.hardware.hardwared.time.monotonic", return_value=100.)
  def test_existing_firmware_updates_only_offroad(self, _clock, thread):
    # Mock the worker itself: this test never opens USB or invokes a flasher.
    device = {"vendorId": 0xADD1, "productId": 1, "product": "custom ed4e39b7-CLEAN"}
    updater = Chestnut()
    updater.update(False, [device])
    thread.assert_not_called()
    self.assertTrue(updater.mismatch)
    updater.update(True, [device])
    thread.assert_called_once_with(target=updater.flash, daemon=True)
    thread.return_value.start.assert_called_once()
    updater.update(True, [device])
    thread.assert_called_once()

  @patch("openpilot.system.hardware.hardwared.threading.Thread")
  def test_matching_firmware_needs_no_update(self, thread):
    updater = Chestnut()
    updater.update(True, [{"vendorId": 0x3801, "productId": 1, "product": CHESTNUT_USB_PRODUCT}])
    self.assertFalse(updater.mismatch)
    thread.assert_not_called()

  def test_usb_monitor_merges_gpu_state(self):
    handle = Mock()
    handle.controlRead.side_effect = [struct.pack("<Hh?", 12000, 1500, False), b"\x78"]
    gpu = log.ChestnutState.new_message()
    gpu.tempC = 42.
    message = read_chestnut_state(handle, gpu)
    self.assertTrue(message.valid)
    self.assertEqual(message.chestnutState.supplyVoltage, 12000)
    self.assertEqual(message.chestnutState.supplyCurrent, 1500)
    self.assertEqual(message.chestnutState.pcieLtssm, 0x78)
    self.assertEqual(message.chestnutState.tempC, 42.)

  def test_short_usb_response_is_invalid(self):
    handle = Mock()
    handle.controlRead.return_value = b"\x00"
    self.assertFalse(read_chestnut_state(handle).valid)
