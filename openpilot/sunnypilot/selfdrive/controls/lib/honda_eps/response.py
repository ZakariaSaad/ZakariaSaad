"""Explicit firmware-controller response presets, latched with drive selection.

``source`` preserves the pinned James/NRDR numerical defaults. ``route32``
reproduces the owner's recorded settings in f133facb1b9b7420|00000032--212d34d24d:
100 ms output filtering in all speed bands and 100 ms override fade-down.
Neither preset changes calibration, geometry, gains, command scaling or safety.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FirmwareResponseProfile:
  name: str
  output_lpf_tau: tuple[float, float, float]
  override_fade_down_s: float


SOURCE_RESPONSE = FirmwareResponseProfile("source", (.07, .05, .01), 0.)
ROUTE32_RESPONSE = FirmwareResponseProfile("route32", (.1, .1, .1), .1)


def resolve_response_profile(name: str) -> FirmwareResponseProfile:
  if type(name) is not str:
    raise ValueError("Honda EPS response profile must be a named preset")
  for profile in (SOURCE_RESPONSE, ROUTE32_RESPONSE):
    if name == profile.name:
      return profile
  raise ValueError(f"unsupported Honda EPS response profile: {name}")
