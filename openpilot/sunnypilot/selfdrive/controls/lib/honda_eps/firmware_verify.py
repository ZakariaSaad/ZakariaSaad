"""Read-only verification of the owner's corrected Insight PTM RWD.

This verifies an uploaded file, never a vehicle's installed image. Both the old
and corrected firmware report 39990-TXM,A040; identification cannot prove revision.
No firmware bytes are written or flashed, and no controller parameter is changed.
RWD decoder/checksum reference: JamesL787/openpilot bfcbbf24 eps_tools/check_rwd.py.
Table provenance: f3a6a124bdad3151f875f4f29c8d7396c7617702.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.geometry import INSIGHT_POSITION_X, INSIGHT_POSITION_Y

RWD_SHA256 = "7ba3743248d30758f8423bcb2a1c65f0d2d2d3563a830b8268bf448659b28a38"
PAYLOAD_SHA256 = "089a8a7251da04e4bbe693f764c4fe7a02ad6a630afde0fa700925801d9f9728"
RWD_SIZE = 442469
FLASH_START = 0x4000
PAYLOAD_SIZE = 0x6C000
_DECODE = bytes.fromhex(
  "24e4a4ead999596483430319ae6e2ec328e8a8eedd9d5d688747071db27232c72cecacf2e1a1616c8b4b0b21b67636cb30f0b0f6e5a565708f4f0f25ba7a3acf" +
  "34f4b4fae9a9697493531329be7e3ed338f8b8feedad6d789757172dc28242d73cfcbc02f1b1717c9b5b1b31c68646db4000c006f5b575809f5f1f35ca8a4adf" +
  "4404c40af9b97984a3632339ce8e4ee34808c80efdbd7d88a767273dd29252e74c0ccc1201c1818cab6b2b41d69656eb5010d01605c58590af6f2f45da9a5aef" +
  "5414d41a09c98994b3733349de9e5ef35818d81e0dcd8d98b777374de2a262f75c1cdc2211d1919cbb7b3b51e6a666fb6020e02615d595a0bf7f3f55aa6a2aff"
)


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def verify_rwd(raw: bytes) -> dict:
  """Accept only the exact revised image, then verify its checksums and tables."""
  _require(len(raw) == RWD_SIZE, "unexpected RWD size")
  rwd_digest = hashlib.sha256(raw).hexdigest()
  _require(rwd_digest == RWD_SHA256, "RWD SHA-256 does not match the corrected TXM PTM revision")
  _require(raw[:3] == b"Z\r\n", "expected Honda 0x5A RWD")
  _require(struct.unpack_from("<I", raw, len(raw) - 4)[0] == sum(raw[:-4]) & 0xFFFFFFFF, "RWD checksum mismatch")
  headers = []
  index = 3
  for _ in range(6):
    count = raw[index]
    index += 1
    values = []
    for _ in range(count):
      size = raw[index]
      index += 1
      values.append(raw[index:index + size])
      index += size
    headers.append(values)
  _require(headers[5] == [b"\x01\x02\x03"], "unexpected RWD substitution key")
  start, length = struct.unpack_from(">II", raw, index)
  index += 8
  _require((start, length) == (FLASH_START, PAYLOAD_SIZE), "unexpected flash block")
  _require(index + length == len(raw) - 4, "unexpected firmware block length")
  payload = raw[index:index + length].translate(_DECODE)
  payload_digest = hashlib.sha256(payload).hexdigest()
  _require(payload_digest == PAYLOAD_SHA256, "decoded firmware SHA-256 mismatch")
  words = struct.unpack(f">{len(payload) // 2}H", payload)
  _require(words[0x6BF80 // 2] == sum(words[:0x6BF80 // 2]) & 0xFFFF, "firmware sum checksum mismatch")
  _require(words[0x6BFFE // 2] == -sum(words[:0x6BFFE // 2]) & 0xFFFF, "firmware negative-sum checksum mismatch")
  expected_tables = {
    "command_row_0_axis": (0x11B3C, (0, 111, 222, 333, 443, 665, 887, 1108, 1663)),
    "command_row_0_values": (0x11BBA, (0, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 30000)),
    "p_row_0_axis": (0x11E8A, (0, 222, 443, 665, 887, 1108, 1330, 1552, 1774)),
    "p_row_0_values": (0x11F08, (117, 148, 184, 220, 245, 257, 263, 265, 265)),
    "position_a_axis": (0x11374, INSIGHT_POSITION_X),
    "position_a_divisors": (0x11338, INSIGHT_POSITION_Y),
  }
  tables = {}
  for name, (address, expected) in expected_tables.items():
    actual = struct.unpack_from(f">{len(expected)}H", payload, address - start)
    _require(actual == expected, f"firmware table mismatch: {name}")
    tables[name] = {"flash_address": hex(address), "values": actual}
  return {
    "profile": "insight_txm_a040_ptm_20261016b", "rwd_sha256": rwd_digest, "payload_sha256": payload_digest,
    "flash_start": hex(start), "payload_length": length, "file_checksum": "0x03853b3d",
    "firmware_checksums": {"0x6bf80": "0x8805", "0x6bffe": "0x981b"}, "tables": tables,
    "installed_firmware_confirmed": False,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("rwd", type=Path)
  args = parser.parse_args()
  try:
    # Bound the read and reject extra data without loading arbitrary large files.
    with args.rwd.open("rb") as file:
      report = verify_rwd(file.read(RWD_SIZE + 1))
  except (OSError, ValueError, IndexError, struct.error) as error:
    parser.exit(1, f"Verification failed: {error}\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
