"""Fixed read-only Git inspections under the checkout owner's identity.

AGNOS maintenance supervisors run as root on a read-only root filesystem while
/data/openpilot belongs to comma. Drop credentials for the Git child, including
its recursive submodule children; never alter Git trust, ownership or mounts.
"""
import os
from pathlib import Path
import pwd
import subprocess


class GitInspectionError(ValueError):
  pass


_READ_ONLY_ARGUMENTS = frozenset((
  ('rev-parse', 'HEAD'),
  ('branch', '--show-current'),
  ('status', '--porcelain', '--untracked-files=all'),
  ('status', '--porcelain', '--untracked-files=no'),
  ('submodule', 'status', '--recursive'),
))


def read_only_git(checkout: str | Path, *args: str, timeout: float = 15) -> str:
  if args not in _READ_ONLY_ARGUMENTS:
    raise GitInspectionError('Unsupported read-only Git inspection: ' + ' '.join(args))
  try:
    root = Path(checkout).resolve(strict=True)
    owner = pwd.getpwuid(root.stat().st_uid)
    environment = os.environ.copy()
    environment.update(HOME=owner.pw_dir, USER=owner.pw_name, LOGNAME=owner.pw_name, GIT_OPTIONAL_LOCKS='0')
    # A root supervisor's XDG directory must not override the owner's HOME.
    environment.pop('XDG_CONFIG_HOME', None)
    privileged = os.geteuid() == 0
    if not privileged and os.geteuid() != owner.pw_uid:
      raise GitInspectionError(f'Git inspection requires repository owner {owner.pw_name} (uid {owner.pw_uid}): {root}')
    # Disable optional index writes and fsmonitor hooks, including in child Git
    # processes. subprocess credentials change only this child, not the service.
    result = subprocess.run(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false', '-C', str(root), *args],
                            cwd=root, capture_output=True, text=True, check=False, timeout=timeout, env=environment,
                            user=owner.pw_uid if privileged else None, group=owner.pw_gid if privileged else None,
                            extra_groups=os.getgrouplist(owner.pw_name, owner.pw_gid) if privileged else None)
  except subprocess.TimeoutExpired as error:
    stderr = error.stderr.decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr or ''
    raise GitInspectionError(f'Git inspection timed out in {checkout}: {" ".join(args)}\n' + stderr) from error
  except (OSError, KeyError, subprocess.SubprocessError) as error:
    raise GitInspectionError(f'Git inspection could not run in {checkout}: {" ".join(args)}: {error}') from error
  if result.returncode:
    raise GitInspectionError(f'Git inspection failed (exit {result.returncode}) in {root} as {owner.pw_name}: {" ".join(args)}\n' + result.stderr)
  return result.stdout.strip()
