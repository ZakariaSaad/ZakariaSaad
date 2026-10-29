"""Inspect/register corrected Bigv2 or select it through the normal Models path.

The Models UI is the normal user workflow. This optional CLI uses the same
canonical selection; there is no separate navigation-driving enable preference.
"""
import argparse
import os
from pathlib import Path
import sys

from openpilot.selfdrive.navigation.activation import candidate_status, list_candidates
from openpilot.selfdrive.navigation.errors import NavigationModelDisabled
from openpilot.selfdrive.navigation.offroad import require_offroad
from openpilot.selfdrive.navigation.selection import select_candidate


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
  print('Navigation model selection: ' + ('selected' if status.get('selected') else 'not selected'))
  runtime = params.get('NavigationModelRuntimeState')
  print('Runtime: ' + (str(runtime) if runtime else 'inactive (normal while off road)'))
  print('Off-road parameter: ' + str(params.get_bool('IsOffroad')) + '; engaged parameter: ' + str(params.get_bool('IsEngaged')))
  print('Runtime performance evidence: ' + status.get('runtime_compatibility', 'unverified'))
  print('File admission: exact installed hashes, compiler proof and ordinary fallback are rechecked during model selection/loading; no GPU execution')
  blockers = list(status.get('blockers', [status['reason']] if status.get('reason') else []))
  if not params.get_bool('IsOffroad'):
    blockers.append('IsOffroad is false; selection requires genuine off-road status')
  if params.get_bool('IsEngaged'):
    blockers.append('IsEngaged is true; selection requires disengaged controls')
  for blocker in blockers:
    print('Blocker: ' + blocker)
  if not blockers:
    print('Eligible for model selection; selection requires fresh off-road/disengaged state and successful file admission.')


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
  status = commands.add_parser('status', help='show registration, selection, runtime status and compatibility blockers')
  status.add_argument('model', nargs='?', choices=list(candidates), help='defaults to the selected navigation candidate')
  register = commands.add_parser('register', help='verify and register existing local files while off road; current model selection is unchanged')
  register.add_argument('model', choices=list(candidates))
  select = commands.add_parser('select', help='prepare and select corrected Bigv2 as the active Big Model while off road')
  select.add_argument('model', choices=list(candidates))
  args = parser.parse_args(argv)
  try:
    params = _params() if params is None else params
    if args.command == 'status':
      _status(params, args.model)
    elif args.command == 'register':
      receipt = _register(params, args.model)
      print(f"Registered {candidates[args.model]}. Current model selection is unchanged.")
      print('Manifest: ' + receipt['manifest_path'])
      print('Artifact SHA256: ' + receipt['artifact_sha256'])
      _status(params, args.model)
    elif args.command == 'select':
      select_candidate(params, args.model, offroad=True)
      print(f"Selected {candidates[args.model]} as the Big Model. This does not engage vehicle controls.")
    return 0
  except (NavigationModelDisabled, OSError, RuntimeError, ImportError, ValueError, TypeError, KeyError) as error:
    print(f'Refused: {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
  raise SystemExit(main())
