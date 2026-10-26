"""Manage the registered local navigation model; no model execution or downloads.

Run as comma with installed system Python, e.g. ``python3 -B -m
openpilot.selfdrive.navigation.manage status``. Both Settings and this CLI use
selection.set_driving_enabled and the same branch-reviewed runtime admission.
"""
import argparse
import os
from pathlib import Path
import sys

from openpilot.selfdrive.navigation.activation import candidate_status, list_candidates
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import require_offroad
from openpilot.selfdrive.navigation.selection import select_candidate, set_driving_enabled


def _params():
  if os.geteuid() == 0 or os.geteuid() != Path(__file__).resolve().stat().st_uid:
    raise NavigationModelDisabled('Run this command as the repository owner (comma), without sudo')
  from openpilot.common.params import Params
  return Params()


def _status(params, candidate_id=None) -> None:
  status = candidate_status(params, candidate_id=candidate_id)
  print(f"Model: {status['display_name']} ({status['id']})")
  installed = 'installed' if status.get('installed') else 'missing'
  print(f"Local artifact ({installed}): " + (status.get('artifact_path') or 'not found in the configured manifest'))
  print('Registration: ' + ('matching manifest configured' if status.get('registered') else 'missing or mismatched'))
  if status.get('manifest_path'):
    print('Manifest: ' + status['manifest_path'])
  print('Software compatibility/integration: ' + ('reviewed' if status.get('software_compatible') else 'not reviewed'))
  print('Compiler/artifact evidence: ' + ('reviewed' if status.get('artifact_evidence_reviewed') else 'not reviewed'))
  print('Branch activation approval: ' + ('approved' if status.get('approved') else 'not approved'))
  print('Navigation model selection: ' + ('selected' if status.get('selected') else 'not selected'))
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


def _register(params, candidate_id):
  # Import registration only for this operation. Status needs no messaging;
  # selection changes use fresh off-road admission with installed system Python.
  from openpilot.selfdrive.navigation.preparation import prepare_existing_model
  require_offroad(params)
  return prepare_existing_model(params, is_offroad=True, model_id=candidate_id)


def main(argv=None, *, params=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  candidates = {entry['id']: entry['display_name'] for entry in list_candidates()}
  status = commands.add_parser('status', help='show registration, approval, selection and activation blockers')
  status.add_argument('model', nargs='?', choices=list(candidates), help='defaults to the selected navigation candidate')
  register = commands.add_parser('register', help='verify and register existing local files while off road; driving stays OFF')
  register.add_argument('model', choices=list(candidates))
  select = commands.add_parser('select', help='prepare and select the approved corrected model while off road; request driving ON')
  select.add_argument('model', choices=list(candidates))
  enable = commands.add_parser('enable', help='explicitly request an approved, registered model while off road')
  enable.add_argument('model', choices=list(candidates))
  commands.add_parser('disable', help='withdraw navigation driving permission; ordinary model selections are preserved')
  args = parser.parse_args(argv)
  try:
    params = _params() if params is None else params
    if args.command == 'status':
      _status(params, args.model)
    elif args.command == 'register':
      receipt = _register(params, args.model)
      print(f"Registered {candidates[args.model]}. Navigation Model Driving remains OFF.")
      print('Manifest: ' + receipt['manifest_path'])
      print('Artifact SHA256: ' + receipt['artifact_sha256'])
      _status(params, args.model)
    elif args.command == 'select':
      select_candidate(params, args.model, offroad=True)
      print(f"Selected {candidates[args.model]}; Navigation Model Driving requested ON. This does not engage vehicle controls.")
    elif args.command == 'enable':
      set_driving_enabled(params, True, offroad=True, candidate_id=args.model)
      print(f'{candidates[args.model]} selected; Navigation Model Driving requested ON. This does not engage vehicle controls.')
    else:
      set_driving_enabled(params, False, offroad=True)
      print('Navigation Model Driving OFF. Ordinary model selections are unchanged.')
    return 0
  except (NavigationModelDisabled, OSError, RuntimeError, ImportError, ValueError, TypeError, KeyError) as error:
    print(f'Refused: {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
  raise SystemExit(main())
