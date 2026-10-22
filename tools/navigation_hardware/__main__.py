"""Read-only discovery or explicit non-actuating physical Chestnut checks."""
import argparse
import json
from pathlib import Path

from tools.navigation_hardware.runner import build_plan, collect_results, run_plan
from tools.navigation_hardware.safety import HardwareAdmissionError, preflight


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest='action', required=True)
  for action in ('inspect', 'plan', 'run'):
    command = subparsers.add_parser(action)
    command.add_argument('--checkout', type=Path, required=True)
    command.add_argument('--workspace', type=Path, required=True)
    if action == 'inspect':
      command.add_argument('--output', type=Path)
    else:
      command.add_argument('--onnx', type=Path, required=True)
      command.add_argument('--recording', type=Path, required=True)
      command.add_argument('--width', type=int, default=1344)
      command.add_argument('--height', type=int, default=760)
      command.add_argument('--long-frames', type=int, default=164)
      command.add_argument('--comparison-atol', type=float, default=1e-3)
      command.add_argument('--comparison-rtol', type=float, default=1e-3)
      command.add_argument('--synthetic-navigation', action='store_true')
      command.add_argument('--allow-repeated-cameras', action='store_true')
      if action == 'run':
        command.add_argument('--non-actuating', action='store_true', required=True)
  archive = subparsers.add_parser('archive')
  archive.add_argument('--workspace', type=Path, required=True)
  args = parser.parse_args()
  try:
    if args.action == 'archive':
      print(collect_results(args.workspace))
    elif args.action == 'inspect':
      report = preflight(args.checkout, args.workspace)
      encoded = json.dumps(report, indent=2) + '\n'
      if args.output:
        args.output.write_text(encoded)
      print(encoded, end='')
      if not report['admitted']:
        raise SystemExit(2)
    else:
      plan = build_plan(args.checkout, args.workspace, args.onnx, args.recording, width=args.width, height=args.height,
                        long_frames=args.long_frames, synthetic_navigation=args.synthetic_navigation,
                        allow_repeated_cameras=args.allow_repeated_cameras,
                        comparison_atol=args.comparison_atol, comparison_rtol=args.comparison_rtol)
      print(run_plan(plan) if args.action == 'run' else json.dumps(plan, indent=2))
  except (HardwareAdmissionError, OSError, ValueError, KeyError) as error:
    parser.exit(2, f'{error}\n')


if __name__ == '__main__':
  main()
