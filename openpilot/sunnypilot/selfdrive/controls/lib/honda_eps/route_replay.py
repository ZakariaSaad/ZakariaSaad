"""Replay recorded wheel-angle inputs through the original James/NRDR and port cores.

Logs are decoded with the James schema in a child process to avoid mixing two
different versions of Cap'n Proto types. No cameras, network calls, parameter
writes, vehicle control, or firmware operations are performed. Reports contain
aggregate metrics and log hashes, not GPS, VIN, signed URLs, or raw CAN payloads.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SETTING_KEYS = (
  "HondaEpsController", "HondaEpsFirmwareVgr", "HondaEpsAngleRateLimit",
  "HondaTorqueOutputLowPassFilter", "HondaTorqueOutputLpfTauLowSpeed", "HondaTorqueOutputLpfTauStandard", "HondaTorqueOutputLpfTauHighway",
  "HondaOverrideFadeDownSecs", "HondaOverrideFadeUpSecs", "HondaOverrideTorqueScale",
  "NrdrDriverOverrideThreshold",
  "NrdrLearnAngleOffset", "NrdrLearnSteerRatio", "NrdrLearnStiffness", "UseAutoSteerDelay", "SteerDelay",
)


class DeliveryLimitHistory:
  """Reproduce controlsd's compute-after-update, consume-next-tick ordering."""
  def __init__(self):
    self.next_limited = False

  def observe(self, requested, delivered_snapshot, active):
    consumed = self.next_limited
    self.next_limited = bool(active and abs(requested - delivered_snapshot) > .01)
    return consumed


def _decode(path, schema, warnings):
  import capnp
  import zstandard
  with zstandard.ZstdDecompressor().stream_reader(path.open("rb")) as reader:
    raw = reader.read()
  iterator = schema.Event.read_multiple_bytes(raw)
  while True:
    try:
      yield next(iterator)
    except StopIteration:
      return
    except capnp.KjException:
      warnings.append({"segment": int(path.parent.name), "file": path.name, "issue": "incomplete Cap'n Proto tail; complete preceding messages retained"})
      return


def extract(route_dir, schema_root):
  import capnp
  schema = capnp.load(str(schema_root / "cereal/log.capnp"))
  metadata, segments, qlogs, warnings = {}, [], [], []
  rlogs = sorted(route_dir.glob("*/rlog.zst"), key=lambda path: int(path.parent.name))
  for path in rlogs:
    controls, states, parameters, debug, outputs = [], [], [], [], []
    for event in _decode(path, schema, warnings):
      try:
        kind = event.which()
      except Exception:
        continue
      time = event.logMonoTime / 1e9
      if kind == "initData":
        init = event.initData
        metadata["git_commit"] = init.gitCommit
        metadata["git_branch"] = init.gitBranch
        metadata["dirty"] = init.dirty
        entries = init.params.to_dict().get("entries", [])
        metadata["recorded_settings"] = {
          entry["key"]: entry.get("value", b"").decode("utf-8", errors="replace")
          for entry in entries if entry["key"] in SETTING_KEYS
        }
      elif kind == "carParams":
        cp = event.carParams
        metadata["car"] = {
          "fingerprint": cp.carFingerprint, "flags": cp.flags,
          "eps_identifiers": [fw.fwVersion.decode("ascii", errors="replace").strip("\0") for fw in cp.carFw if str(fw.ecu) == "eps"],
          "pid": cp.lateralTuning.pid.to_dict(), "torque_map": cp.lateralParams.to_dict(),
          "model": {key: getattr(cp, key) for key in
                    ("mass", "rotationalInertia", "wheelbase", "centerToFront", "steerRatioRear", "tireStiffnessFront", "tireStiffnessRear", "steerRatio")},
        }
      elif kind == "carState":
        cs = event.carState
        states.append((time, cs.vEgo, cs.steeringAngleDeg, cs.steeringTorque, cs.steeringPressed))
      elif kind == "liveParameters":
        lp = event.liveParameters
        parameters.append((time, lp.angleOffsetDeg, lp.roll, lp.stiffnessFactor, lp.steerRatio))
      elif kind == "carOutput":
        co = event.carOutput.actuatorsOutput
        outputs.append((time, co.torque, co.torqueOutputCan))
      elif kind == "starpilotLateralState":
        eps = event.starpilotLateralState
        debug.append((time, eps.epsFfFeedforward, eps.epsFfR5, eps.epsFfLoad, eps.epsFfDesiredRate,
                      eps.epsFfWeight, eps.epsCalibration, eps.epsCommandDelay, eps.epsModelDelay))
      elif kind == "controlsState":
        cs = event.controlsState
        if cs.lateralControlState.which() != "pidState":
          continue
        pid = cs.lateralControlState.pidState
        controls.append((time, pid.active, pid.steeringAngleDesiredDeg, pid.steeringAngleDeg,
                         pid.i, pid.output, pid.saturated, cs.desiredCurvature))
    segments.append({"segment": int(path.parent.name), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "controls": controls, "states": states, "parameters": parameters, "debug": debug, "outputs": outputs})
  for path in sorted(route_dir.glob("*/qlog.zst"), key=lambda item: int(item.parent.name)):
    count = Counter()
    speeds, active, raw_presses, saturated = [], 0, 0, 0
    speed_bands = Counter()
    latest_speed = 0.
    start, end = None, None
    for event in _decode(path, schema, warnings):
      time = event.logMonoTime / 1e9
      try:
        kind = event.which()
      except Exception:
        continue
      if kind in ("carState", "controlsState"):
        start = time if start is None else min(start, time)
        end = time if end is None else max(end, time)
        count[kind] += 1
      if kind == "carState":
        latest_speed = event.carState.vEgo
        speeds.append(latest_speed)
        raw_presses += bool(event.carState.steeringPressed)
      elif kind == "controlsState" and event.controlsState.lateralControlState.which() == "pidState":
        pid = event.controlsState.lateralControlState.pidState
        active += bool(pid.active)
        saturated += bool(pid.active and pid.saturated)
        if pid.active:
          band = "low_below_25mph" if latest_speed < 11.176 else "medium_25_to_50mph" if latest_speed < 22.352 else "highway_above_50mph"
          speed_bands[band] += 1
    qlogs.append({"segment": int(path.parent.name), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "duration_seconds": end - start if end is not None else 0, "message_counts": dict(count),
                  "active_controls_samples": active, "raw_steering_pressed_car_state_samples": raw_presses,
                  "active_saturation_samples": saturated, "active_controls_speed_band_samples": dict(speed_bands),
                  "speed_min_m_s": min(speeds) if speeds else None,
                  "speed_max_m_s": max(speeds) if speeds else None})
  return {"metadata": metadata, "segments": segments, "qlogs": qlogs, "decode_warnings": warnings}


def _preceding(samples, time, index):
  while index + 1 < len(samples) and samples[index + 1][0] <= time:
    index += 1
  return samples[index], index


def _stats(values):
  import numpy as np
  values = np.asarray(values, dtype=float)
  if len(values) == 0:
    return {"samples": 0}
  return {"samples": len(values), "rms": float(np.sqrt(np.mean(values ** 2))),
          "mean": float(np.mean(values)), "max_abs": float(np.max(np.abs(values))),
          "p95_abs": float(np.percentile(np.abs(values), 95))}


def replay(data, james_root, nrdr_root):
  import numpy as np
  from types import SimpleNamespace
  from opendbc.car.vehicle_model import VehicleModel
  from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps import core, geometry
  from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.compare_sources import GAIN_BP, KI, KP, load_sources
  from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.driver_override import eps_modified_steering_pressed
  from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.response import ROUTE32_RESPONSE, SOURCE_RESPONSE

  james, nrdr, _, _, hashes = load_sources(james_root, nrdr_root)
  settings = data["metadata"]["recorded_settings"]
  recorded_tau = (float(settings["HondaTorqueOutputLpfTauLowSpeed"]), float(settings["HondaTorqueOutputLpfTauStandard"]),
                  float(settings["HondaTorqueOutputLpfTauHighway"]))
  recorded_lpf = settings.get("HondaTorqueOutputLowPassFilter", "1") == "1"
  if (recorded_tau != ROUTE32_RESPONSE.output_lpf_tau or not recorded_lpf or
      float(settings["HondaOverrideFadeDownSecs"]) != ROUTE32_RESPONSE.override_fade_down_s):
    raise ValueError("recorded settings do not match the explicit route32 response preset")
  gains = data["metadata"]["car"]["pid"]
  names = ("port_source_response", "port_route32_response", "james_recorded_response", "nrdr_source_response")
  segment_reports, combined, bands, golden = [], {name: [] for name in names}, {}, []
  reconstructed, actual_can_difference, geometry_difference = [], [], []
  total_overrides, total_saturated, total_active = 0, 0, 0
  integer_request_mismatches = 0

  for segment in data["segments"]:
    if not all(segment[key] for key in ("controls", "states", "parameters", "debug", "outputs")):
      raise ValueError(f"segment {segment['segment']} lacks required recorded signals")
    source_ff = james.HondaEpsFirmwareFeedforward(.01, cal=james.INSIGHT_TXM_A040)
    controllers = {
      "james_recorded_response": james.HondaEpsLateralCore(gains["kpBP"], gains["kpV"], gains["kiBP"], gains["kiV"], .01, ff=source_ff),
    }
    for name in names:
      if name == "james_recorded_response":
        continue
      module = nrdr if name.startswith("nrdr") else core
      cal = nrdr.INSIGHT_TXM_A040 if module is nrdr else core.INSIGHT_TXM_A040_CANONICAL
      ff = module.HondaEpsFirmwareFeedforward(.01, cal=cal, load=module.INSIGHT_TXM_EPS_LOAD)
      controllers[name] = module.HondaEpsLateralCore(GAIN_BP, KP, GAIN_BP, KI, .01, ff=ff,
                                                  p_scale=core.INSIGHT_P_SCALE, i_scale=core.INSIGHT_I_SCALE)
    for name in ("james_recorded_response", "port_route32_response"):
      controllers[name].output_lpf_enabled = recorded_lpf
      controllers[name].output_lpf_tau = recorded_tau
    controllers["port_route32_response"].output_lpf_tau = ROUTE32_RESPONSE.output_lpf_tau
    controllers["port_source_response"].output_lpf_tau = SOURCE_RESPONSE.output_lpf_tau
    for controller in controllers.values():
      if hasattr(controller, "override_fade_up_s"):
        controller.override_fade_up_s = float(settings.get("HondaOverrideFadeUpSecs", "1.5"))
    index = dict.fromkeys(("states", "parameters", "debug", "outputs"), 0)
    press_state = dict.fromkeys(names, (0., False))
    deltas = {name: [] for name in names}
    reconstruction, reconstructed_i, reconstructed_ff = [], [], []
    first_time = segment["controls"][0][0]
    previous_time = first_time
    delivery_limits = DeliveryLimitHistory()
    initialized = False
    vm = VehicleModel(SimpleNamespace(**data["metadata"]["car"]["model"]))
    active_count, override_count, saturated_count = 0, 0, 0
    integer_mismatches = 0
    dt_values, angle_differences = [], []
    for number, control in enumerate(segment["controls"]):
      time, active, desired, measured, logged_i, logged_output, saturated, _ = control
      signals = {}
      for key in index:
        signals[key], index[key] = _preceding(segment[key], time, index[key])
      state, parameters, output = signals["states"], signals["parameters"], signals["outputs"]
      # The EPS debug message is published immediately after controlsState.
      dbg_index = index["debug"]
      if dbg_index + 1 < len(segment["debug"]) and abs(segment["debug"][dbg_index + 1][0] - time) < abs(signals["debug"][0] - time):
        dbg_index += 1
      debug = segment["debug"][dbg_index]
      _, speed, _, torque, raw_pressed = state
      _, offset, roll, stiffness, ratio = parameters
      angle_no_offset = desired - offset
      # controlsd computes the delivery proxy AFTER updating the controller;
      # the next tick consumes that previous tick's request/snapshot pair.
      limited = delivery_limits.observe(logged_output, output[1], active)
      if not active:
        for name, controller in controllers.items():
          controller.reset()
          press_state[name] = (0., False)
        previous_time = time
        continue
      if time - first_time < 5.:
        previous_time = time
        continue
      if not initialized:
        # The first 5 s are excluded. Seed all observable states once at that
        # boundary, avoiding lasting I drift from unavailable pre-segment history.
        for name, controller in controllers.items():
          domain = 4096. if name.startswith("nrdr") else 3840.
          controller.pid.i = logged_i * 3840. / domain
          controller.output = logged_output * 3840. / domain
          controller.output_lpf.x = controller.output
          controller.ff.prev_angle = angle_no_offset
          controller.ff.output = debug[1] * 3840. / domain
          controller.ff.r5, controller.ff.load, controller.ff.rate = debug[2:5]
          speed_gate = float(np.interp(speed, [2., 4.], [0., 1.]))
          crawl = float(np.interp(speed, [5., 8.], [1., 0.]))
          crawl_gate = 1. - crawl * (1. - float(np.interp(abs(angle_no_offset), [5., 20.], [0., 1.])))
          controller.ff_ramp = min(1., debug[5] / (speed_gate * crawl_gate)) if speed_gate * crawl_gate > 0 else 0.
        initialized = True
        previous_time = time
        continue
      dt_values.append(time - previous_time)
      values = {}
      for name, controller in controllers.items():
        duration, pressed = press_state[name]
        duration, pressed = eps_modified_steering_pressed(raw_pressed, torque, controller.output, duration, pressed, .01)
        press_state[name] = (duration, pressed)
        value = controller.update(angle_no_offset, offset, measured, speed, roll, pressed, limited)
        values[name] = -value * (4096. if name.startswith("nrdr") else 3840.)
      if time - first_time >= 5.:
        reference = values["james_recorded_response"]
        integer_mismatches += int(values["port_route32_response"]) != int(reference)
        band = "low_below_25mph" if speed < 11.176 else "medium_25_to_50mph" if speed < 22.352 else "highway_above_50mph"
        bands.setdefault(band, {name: [] for name in names})
        for name in names:
          error = values[name] - reference
          deltas[name].append(error)
          combined[name].append(error)
          bands[band][name].append(error)
        error = reference + logged_output * 3840.
        reconstruction.append(error)
        reconstructed.append(error)
        reconstructed_i.append((controllers["james_recorded_response"].pid.i - logged_i) * -3840.)
        reconstructed_ff.append((controllers["james_recorded_response"].ff.output - debug[1]) * -3840.)
        actual_can_difference.append(output[2] + logged_output * 3840.)
        # Compare both local geometries at the same source desired angle/path.
        vm.update_params(max(stiffness, .1), max(ratio, .1))
        curvature = geometry.curvature_from_angle(vm, speed, roll, angle_no_offset, "james")
        mvl_angle = geometry.desired_angle_no_offset(vm, speed, roll, curvature, "mvl")
        geometry_difference.append(mvl_angle - angle_no_offset)
        angle_differences.append(mvl_angle - angle_no_offset)
        active_count += 1
        override_count += press_state["james_recorded_response"][1]
        saturated_count += saturated
        if len(golden) < 10 and number % 500 == 0:
          golden.append({"segment": segment["segment"], "frame": number, "speed_m_s": speed,
                         "logged_core_e4": -logged_output * 3840., "source_reconstructed_e4": reference,
                         "port_route32_e4": values["port_route32_response"]})
      previous_time = time
    total_active += active_count
    integer_request_mismatches += integer_mismatches
    total_overrides += override_count
    total_saturated += saturated_count
    segment_reports.append({"segment": segment["segment"], "rlog_sha256": segment["sha256"],
                            "recorded_controls_frames": len(segment["controls"]), "compared_active_frames_after_warmup": active_count,
                            "duration_seconds": segment["controls"][-1][0] - first_time,
                            "control_dt_seconds": _stats(dt_values), "driver_override_frames": override_count,
                            "logged_saturation_frames": saturated_count, "vs_james_recorded_response_e4": {name: _stats(deltas[name]) for name in names},
                            "port_route32_vs_reconstructed_james_truncated_e4_mismatches": integer_mismatches,
                            "source_reconstruction_vs_logged_core_e4": _stats(reconstruction),
                            "reconstructed_integral_minus_logged_integral_e4": _stats(reconstructed_i),
                            "reconstructed_raw_ff_minus_logged_raw_ff_e4": _stats(reconstructed_ff),
                            "mvl_minus_james_geometry_angle_deg": _stats(angle_differences)})
  return {
    "route": "f133facb1b9b7420|00000032--212d34d24d", "kind": "recorded wheel-angle input core replay",
    "metadata": data["metadata"], "reference_source_file_sha256": hashes,
    "observed_eps_calibrations": sorted({sample[6] for segment in data["segments"] for sample in segment["debug"]}),
    "observed_eps_command_delay_range_s": [min(sample[7] for segment in data["segments"] for sample in segment["debug"]),
                                            max(sample[7] for segment in data["segments"] for sample in segment["debug"])],
    "observed_eps_logged_model_delay_range_s": [min(sample[8] for segment in data["segments"] for sample in segment["debug"]),
                                                 max(sample[8] for segment in data["segments"] for sample in segment["debug"])],
    "full_rate_rlog_segments": [segment["segment"] for segment in segment_reports],
    "qlog_segments": [segment["segment"] for segment in data["qlogs"]], "qlog_summary": data["qlogs"],
    "decode_warnings": data["decode_warnings"],
    "compared_active_frames_after_warmup": total_active, "driver_override_frames": total_overrides, "logged_saturation_frames": total_saturated,
    "port_route32_vs_reconstructed_james_truncated_e4_mismatches": integer_request_mismatches,
    "per_segment": segment_reports, "vs_james_recorded_response_e4": {name: _stats(combined[name]) for name in names},
    "by_speed_band_vs_james_e4": {band: {name: _stats(errors) for name, errors in values.items()} for band, values in bands.items()},
    "source_reconstruction_vs_logged_core_e4": _stats(reconstructed),
    "logged_delivered_can_vs_logged_core_request_e4": _stats(actual_can_difference),
    "mvl_minus_james_geometry_angle_deg": _stats(geometry_difference), "sample_comparisons": golden,
    "limits": [
      f"Replays {len(data['segments'])} full-rate rlogs; {len(data['qlogs'])} qlogs are summarized without upsampling or treating them as full-rate.",
      "Uses logged desired wheel angles: geometry, command delay and model prediction are held outside the core comparison.",
      "Excludes the first five seconds; seeds observable state once at that boundary, then never re-injects logged state during validation.",
      "Latest preceding carState/liveParameters/carOutput approximate SubMaster snapshots; limited-state inference uses the recorded delivery difference.",
      "Recorded float32 desired angles and gain tables limit exact reconstruction of original double-precision runtime state.",
      "Port route32 response matches recorded LPF settings explicitly; source response retains original source default filters.",
      "Geometry comparison uses the same path with recorded learned ratio/stiffness; it is not a replacement for the selected live mode.",
      "Logged delivered CAN includes original carcontroller override fade/rate limits and is reported separately from core requests.",
      "Recorded raw steeringPressed uses 2400 counts, matched by admitted firmware mode; disabled MVL fallback keeps its original 1200.",
      "No model inference, complete controlsd process replay, closed-loop dynamics, vehicle validation or firmware-revision proof.",
    ],
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--route-dir", required=True, type=Path)
  parser.add_argument("--james-root", required=True, type=Path)
  parser.add_argument("--nrdr-root", type=Path)
  parser.add_argument("--schema-root", type=Path)
  parser.add_argument("--extract", action="store_true")
  parser.add_argument("--refresh-extraction", action="store_true")
  parser.add_argument("--extracted", required=True, type=Path)
  args = parser.parse_args()
  if args.extract:
    args.extracted.write_text(json.dumps(extract(args.route_dir, args.schema_root or args.james_root)))
    return
  if args.nrdr_root is None:
    parser.error("--nrdr-root is required for replay")
  if args.refresh_extraction or not args.extracted.exists():
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--extract", "--route-dir", str(args.route_dir),
                    "--james-root", str(args.james_root), "--schema-root", str(args.schema_root or args.james_root),
                    "--extracted", str(args.extracted)], check=True)
  print(json.dumps(replay(json.loads(args.extracted.read_text()), args.james_root, args.nrdr_root), indent=2))


if __name__ == "__main__":
  main()
