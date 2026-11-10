"""Bosch-A F0 STATUS per radarTracks point, from card to radard.

car.capnp's RadarPoint is upstream-owned, so card publishes the decoder's point_statuses in
carStateSP.radarPointStatus (custom.capnp), stamped with the logMonoTime of the radarTracks message they describe.
Ported from the Biscuit-Labs starpilotCarState carrier; nrdr's companion message is carStateSP.
radard uses them only when their referenced timestamp exactly matches the radarTracks message it holds.
Absent, invalid or unpaired statuses read as empty, so the STATUS tell is off (no bound).
"""
from __future__ import annotations

def write(companion, radar_tracks_mono_time: int, statuses: dict[int, int]) -> None:
  companion.radarTracksMonoTime = int(radar_tracks_mono_time)
  rows = companion.init('radarPointStatus', len(statuses))
  for row, (track_id, status) in zip(rows, sorted(statuses.items()), strict=True):
    row.trackId = int(track_id)
    row.objectStatus = int(status)


def read(sm) -> dict[int, int]:
  try:
    if not sm.valid.get('carStateSP', False):
      return {}
    companion = sm['carStateSP']
    stamp = int(companion.radarTracksMonoTime)
    tracks_stamp = int(sm.logMonoTime['radarTracks'])
    # card records the exact publication timestamp, including when it repeats
    # the status at 100 Hz. A nearby sweep can reuse the same wire track ID.
    if stamp <= 0 or stamp != tracks_stamp:
      return {}
    return {int(row.trackId): int(row.objectStatus) for row in companion.radarPointStatus}
  except (AttributeError, KeyError, TypeError, ValueError):
    return {}
