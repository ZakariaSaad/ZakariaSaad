"""NRDR stopping inputs, isolated from unrelated lateral hooks."""


def stopping_inputs(calibrated_pose, longitudinal_plan) -> tuple[float | None, float | None]:
  pitch = float(calibrated_pose.orientation.xyz[1]) if calibrated_pose is not None else None
  distance = None
  if longitudinal_plan.hasLead and len(longitudinal_plan.leadTrajectoryX0) > 0:
    distance = float(longitudinal_plan.leadTrajectoryX0[0])
  return pitch, distance

