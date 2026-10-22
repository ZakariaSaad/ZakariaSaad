"""Manage the registered local navigation model; no model execution or downloads.

Run as comma with installed system Python, e.g. ``python3 -B -m
openpilot.selfdrive.navigation.manage status``. Both Settings and this CLI use
selection.set_driving_enabled and the same branch-reviewed runtime admission.
"""
import argparse
import os
from pathlib import Path
import sys

from openpilot.selfdrive.navigation.activation import candidate_status
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.selection import set_driving_enabled


def _params():
  if os.geteuid() == 0 or os.geteuid() != Path(__file__).resolve().stat().st_uid:
    raise NavigationModelDisabled('Run this command as the repository owner (comma), without sudo')
  from openpilot.common.params import Params
  return Params()


def _status(params) -> None:
  status = candidate_status(params, candidate_id='big-nav-v2')
  print(f"Model: {status['display_name']} (big-nav-v2)")
  installed = 'installed' if status.get('installed') else 'missing'
  print(f"Local artifact ({installed}): " + (status.get('artifact_path') or 'not found in the configured manifest'))
  print('Registration: ' + ('matching manifest configured' if status.get('registered') else 'missing or mismatched'))
  print('Software compatibility/integration: ' + ('reviewed' if status.get('software_compatible') else 'not reviewed'))
  print('Compiler/artifact evidence: ' + ('reviewed' if status.get('artifact_evidence_reviewed') else 'not reviewed'))
  print('Branch activation approval: ' + ('approved' if status.get('approved') else 'not approved'))
  print('Bigv2 selection: ' + ('selected' if status.get('selected') else 'not selected'))
  print('Navigation Model Driving: ' + ('requested ON' if params.get_bool('NavigationModelDrivingEnabled') else 'OFF'))
  runtime = params.get('NavigationModelRuntimeState')
  print('Runtime: ' + (str(runtime) if runtime else 'inactive (normal while off road)'))
  print('Off-road parameter: ' + str(params.get_bool('IsOffroad')) + '; engaged parameter: ' + str(params.get_bool('IsEngaged')))
  print('Runtime performance evidence: ' + status.get('runtime_compatibility', 'unverified'))
  print('File admission: exact installed hashes, compiler proof and ordinary fallback are rechecked when enabling; no GPU execution')
  blockers = list(status.get('blockers', [status['reason']] if status.get('reason') else []))
  if not params.get_bool('IsOffroad'):
    blockers.append('IsOffroad is false; enabling requires genuine off-road status')
  if params.get_bool('IsEngaged'):
    blockers.append('IsEngaged is true; enabling requires disengaged controls')
  for blocker in blockers:
    print('Blocker: ' + blocker)
  if not blockers:
    print('Eligible for explicit enable; enable also requires fresh off-road/disengaged state and successful file admission.')


def main(argv=None, *, params=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  commands.add_parser('status', help='show registration, approval, selection and activation blockers')
  enable = commands.add_parser('enable', help='explicitly request an approved, registered model while off road')
  enable.add_argument('model', choices=['big-nav-v2'])
  commands.add_parser('disable', help='withdraw navigation driving permission; ordinary model selections are preserved')
  args = parser.parse_args(argv)
  try:
    params = _params() if params is None else params
    if args.command == 'status':
      _status(params)
    elif args.command == 'enable':
      set_driving_enabled(params, True, offroad=True, candidate_id=args.model)
      print('Bigv2 selected; Navigation Model Driving requested ON. This does not engage vehicle controls.')
    else:
      set_driving_enabled(params, False, offroad=True)
      print('Navigation Model Driving OFF. Ordinary model selections are unchanged.')
    return 0
  except (NavigationModelDisabled, OSError, RuntimeError, ImportError, ValueError, TypeError, KeyError) as error:
    print(f'Refused: {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
  raise SystemExit(main())
