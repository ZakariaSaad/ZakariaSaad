"""Numerical source comparison, not recorded-route replay or road validation.

Run from the project root with --james-root and --nrdr-root checkouts. James root
must contain common/pid.py, selfdrive/controls/lib/honda_eps_{firmware_ff,rack_map}.py,
and opendbc_repo/opendbc/car/honda/steer_ratio.py. Pin James to bfcbbf24bc857f9d
(includes calibration f3a6a124) and NRDR to PR #22's 5777cf9dc76b7b98.
Only project imports are rebound when executing the original source. Algorithms
and constants execute unchanged. James's actual PID and rack-map classes are used.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps import core, geometry

GAIN_BP = (0.0, 11.175, 11.176, 22.352)
KP = (0.018, 0.024, 0.048, 0.060)
KI = (0.006, 0.008, 0.016, 0.020)


def _load(path, name, injected=None, select=None):
  tree = ast.parse(path.read_text(), filename=str(path))
  if select is not None:
    tree.body = [node for node in tree.body if select(node)]
  else:
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
                 (node.module or "").startswith(("openpilot.", "opendbc.")))]
  module = types.ModuleType(name)
  sys.modules[name] = module  # dataclasses resolves this module while defining tables
  module.__dict__.update(injected or {})
  exec(compile(tree, str(path), "exec"), module.__dict__)
  return module


def load_sources(james_root, nrdr_root):
  rack_file = james_root / "selfdrive/controls/lib/honda_eps_rack_map.py"
  ff_file = james_root / "selfdrive/controls/lib/honda_eps_firmware_ff.py"
  pid_file = james_root / "common/pid.py"
  filter_file = james_root / "common/filter_simple.py"
  vgr_file = james_root / "opendbc_repo/opendbc/car/honda/steer_ratio.py"
  nrdr_file = nrdr_root / "openpilot/nrdr/features/lateral/vfn_eps_core.py"
  if not nrdr_file.exists():
    nrdr_file = nrdr_root / "nrdr/features/lateral/vfn_eps_core.py"
  nrdr_project = nrdr_file.parents[3]
  nrdr_pid_file = nrdr_project / "common/pid.py"
  nrdr_filter_file = nrdr_project / "common/filter_simple.py"
  nrdr_integral_file = nrdr_file.with_name("integral_pid.py")

  def vgr_selected(node):
    if isinstance(node, ast.FunctionDef):
      return node.name == "_build_vgr_position_inverse"
    if isinstance(node, ast.Assign):
      names = [name.id for target in node.targets for name in ast.walk(target) if isinstance(name, ast.Name)]
      return any(name.startswith(("_CLARITY_POSITION_", "_INSIGHT_TXM_A040_POSITION_", "CLARITY_TRW_A020_VGR_",
                                 "INSIGHT_TXM_A040_VGR_")) for name in names)
    return isinstance(node, ast.Import) and any(alias.name == "math" for alias in node.names)

  vgr = _load(vgr_file, "honda_eps_comparison_james_vgr", select=vgr_selected)
  rack = _load(rack_file, "honda_eps_comparison_james_rack")
  pid = _load(pid_file, "honda_eps_comparison_james_pid")
  source_filter = _load(filter_file, "honda_eps_comparison_james_filter")
  dependencies = {"FirstOrderFilter": source_filter.FirstOrderFilter, "PIDController": pid.PIDController,
                  "CLARITY_TRW_A020_VGR_ANGLE_BP": vgr.CLARITY_TRW_A020_VGR_ANGLE_BP,
                  "CLARITY_TRW_A020_VGR_LINEAR_BP": vgr.CLARITY_TRW_A020_VGR_LINEAR_BP}
  dependencies.update({name: getattr(rack, name) for name in
                       ("CIVIC_TBA_C020_RACK", "CIVIC_TEG_A010_RACK", "CLARITY_TRW_A020_RACK", "INSIGHT_TXM_A040_RACK", "RackMapTable")})
  james = _load(ff_file, "honda_eps_comparison_james_core", dependencies)
  nrdr_pid = _load(nrdr_pid_file, "honda_eps_comparison_nrdr_pid")
  nrdr_filter = _load(nrdr_filter_file, "honda_eps_comparison_nrdr_filter")
  nrdr_integral = _load(nrdr_integral_file, "honda_eps_comparison_nrdr_integral", {"PIDController": nrdr_pid.PIDController})
  nrdr = _load(nrdr_file, "honda_eps_comparison_nrdr_core", {
    "FirstOrderFilter": nrdr_filter.FirstOrderFilter, "PIDController": nrdr_integral.IntegralScaledPIDController,
    "NRDR_CLARITY_VGR_LINEAR_BP": vgr.CLARITY_TRW_A020_VGR_LINEAR_BP,
    "NRDR_CLARITY_VGR_ANGLE_BP": vgr.CLARITY_TRW_A020_VGR_ANGLE_BP,
  })
  sources = {"james/" + str(file.relative_to(james_root)): hashlib.sha256(file.read_bytes()).hexdigest()
             for file in (rack_file, ff_file, pid_file, filter_file, vgr_file)}
  sources.update({"nrdr/" + str(file.relative_to(nrdr_root)): hashlib.sha256(file.read_bytes()).hexdigest()
                  for file in (nrdr_file, nrdr_pid_file, nrdr_filter_file, nrdr_integral_file)})
  return james, nrdr, rack, vgr, sources


def _make_core(module, james=False):
  if james:
    ff = module.HondaEpsFirmwareFeedforward(.01, cal=module.INSIGHT_TXM_A040)
    return module.HondaEpsLateralCore(GAIN_BP, KP, GAIN_BP, KI, .01, ff=ff)
  cal = module.INSIGHT_TXM_A040_CANONICAL if module is core else module.INSIGHT_TXM_A040
  ff = module.HondaEpsFirmwareFeedforward(.01, cal=cal, load=module.INSIGHT_TXM_EPS_LOAD)
  return module.HondaEpsLateralCore(GAIN_BP, KP, GAIN_BP, KI, .01, ff=ff,
                                  p_scale=core.INSIGHT_P_SCALE, i_scale=core.INSIGHT_I_SCALE)


def run_comparison(james_root, nrdr_root):
  james, nrdr, rack, vgr, sources = load_sources(james_root, nrdr_root)
  results, golden = [], []
  for speed, amplitude in ((3., 90.), (8., 70.), (12., 30.), (22., 10.), (30., 7.)):
    source, port, nrdr_port = _make_core(james, True), _make_core(core), _make_core(nrdr)
    ours, reference, nrdr_commands = [], [], []
    for frame in range(2000):
      time = frame * .01
      desired = amplitude * math.sin(time * .6) + 1.5 * math.sin(time * 5.)
      measured = amplitude * math.sin((time - .08) * .6)
      if 13 <= time < 14:  # substantial residual with FF already joined: saturation recovery
        measured -= 80.
      pressed = 8 <= time < 8.5
      limited = pressed or 8.5 <= time < 10.
      args = (desired, .2, measured, speed, .025 * math.sin(time * .2), pressed, limited)
      src = source.update(*args)
      dst = port.update(*args)
      nrdr_output = nrdr_port.update(*args)
      reference.append(-3840. * src)
      ours.append(-4096. * (dst * (3840. / 4096.)))
      nrdr_commands.append(-4096. * nrdr_output)
      if speed == 12. and frame in (0, 50, 100, 500, 799, 825, 900, 999, 1400, 1999):
        golden.append({"frame": frame, "input": args, "canonical_output": src})
    ours_array = np.asarray(ours)
    reference_array = np.asarray(reference)
    nrdr_array = np.asarray(nrdr_commands)
    maximum_error = float(np.max(np.abs(ours_array - reference_array)))
    if maximum_error > 1e-8:
      raise AssertionError(f"physical command parity failed at {speed} m/s: {maximum_error}")
    results.append({
      "speed_m_s": speed, "frames": len(ours_array), "port_vs_james_max_abs_e4": maximum_error,
      "port_vs_james_rms_e4": float(np.sqrt(np.mean((ours_array - reference_array) ** 2))),
      # MVL transport uses int interpolation. This checks pre-rate-limit identity mapping only.
      "port_vs_james_integer_e4_mismatches": int(np.count_nonzero(ours_array.astype(int) != reference_array.astype(int))),
      "nrdr_vs_james_max_abs_e4": float(np.max(np.abs(nrdr_array - reference_array))),
      "nrdr_vs_james_rms_e4": float(np.sqrt(np.mean((nrdr_array - reference_array) ** 2))),
      "max_abs_source_e4": float(np.max(np.abs(reference_array))),
    })
  src_rack = rack.HondaEpsRackMap(2.7, (vgr.INSIGHT_TXM_A040_VGR_LINEAR_BP, vgr.INSIGHT_TXM_A040_VGR_ANGLE_BP), rack.INSIGHT_TXM_A040_RACK)
  port_rack = geometry.InsightRackMap(2.7)
  geometry_error = max(abs(src_rack.angle_from_curvature(curvature, speed, roll) -
                           port_rack.angle_from_curvature(curvature, speed, roll))
                       for speed in (0, 3, 8, 12, 22, 30, 35) for roll in (-.04, 0, .04)
                       for curvature in (-.02, -.005, -.001, 0, .001, .005, .02))
  if geometry_error > 1e-10:
    raise AssertionError(f"source rack geometry mismatch: {geometry_error}")
  return {"kind": "deterministic injected-input source comparison; no route replay or road validation",
          "james_commit": "bfcbbf24bc857f9dbf93af52b7bb72852f4540bb", "james_calibration_commit": "f3a6a124bdad3151f875f4f29c8d7396c7617702",
          "source_file_sha256": sources, "canonical_to_mvl_scale": 3840. / 4096., "results": results,
          "geometry_max_abs_angle_deg": geometry_error, "james_golden_trace": golden,
          "limits": "pre-carcontroller E4 identity mapping; excludes CAN delivery, rate limits, Panda and vehicle dynamics"}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--james-root", type=Path, required=True)
  parser.add_argument("--nrdr-root", type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(run_comparison(args.james_root, args.nrdr_root), indent=2))


if __name__ == "__main__":
  main()
