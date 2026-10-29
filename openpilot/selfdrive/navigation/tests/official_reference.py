"""Offline differential oracle from the exact Bigv2 checkpoint revision.

The fixtures contain unmodified MIT-licensed comma.ai source, not a second
implementation of the desired policy. Every source is SHA256-bound to its
original Git blob in fixtures/official_a03/manifest.json. No model/pickle is
loaded. Imports and the executable main guard are excluded when extracting
definitions so tests can inject inert transports and clocks explicitly.
"""
import ast
import hashlib
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REVISION = 'a03bb1d4e80050cae0007780ecb8eb7cf06d0cb9'
ROOT = Path(__file__).with_name('fixtures') / 'official_a03'
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())


def source(name: str) -> str:
  record = MANIFEST['files'][name]
  data = (ROOT / (name + '.py')).read_bytes()
  assert MANIFEST['revision'] == REVISION
  assert hashlib.sha256(data).hexdigest() == record['sha256'], name
  assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == record['git_blob'], name
  return data.decode()


def load_reference(name: str, *, namespace=None, names=None) -> SimpleNamespace:
  """Execute exact pinned definitions with explicitly supplied dependencies.

``names`` optionally selects top-level function/class/assignment names. When
omitted, all definitions/constants are included. Tests supply e.g. messaging,
RouteTracker or ModelConstants rather than importing a live process/device.
Only imports and module-level executable statements are skipped; function and
class bodies are compiled unchanged from the official source AST.
"""
  tree = ast.parse(source(name), filename=MANIFEST['files'][name]['path'])
  selected = []
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
      labels = {node.name}
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      labels = {child.id for target in targets for child in ast.walk(target) if isinstance(child, ast.Name)}
    else:
      continue
    if names is None or labels.intersection(names):
      selected.append(node)
  values = {'__name__': 'official_a03_' + name, 'np': np, 'time': time, 'os': os,
            'math': math, 'lru_cache': lru_cache, 'Any': Any}
  values.update(namespace or {})
  exec(compile(ast.Module(body=selected, type_ignores=[]), MANIFEST['files'][name]['path'], 'exec'), values)
  return SimpleNamespace(**values)


def navigation_consumer_block() -> str:
  """The actual inline official modeld navigation-input block, unchanged."""
  body = source('modeld')
  start = body.index('    # Map and navigation inputs\n')
  end = body.index('    mt1 = time.perf_counter()', start)
  return '\n'.join(line[4:] if line.startswith('    ') else line for line in body[start:end].splitlines())


class OfficialNavigationInputs:
  """State container executing the exact official inline consumer per frame."""
  def __init__(self):
    self.nav_map = None
    self.nav_map_time = 0.
    self.code = compile(navigation_consumer_block(), 'official_a03_modeld_navigation_inputs', 'exec')
    self.navigation_input = load_reference('helpers').navigation_input

  def read(self, sm, client, now: float, v_ego: float) -> dict:
    values = {'np': np, 'time': SimpleNamespace(monotonic=lambda: now), 'sm': sm,
              'vipc_client_map': client, 'v_ego': v_ego, 'inputs': {},
              'nav_map': self.nav_map, 'nav_map_time': self.nav_map_time,
              'navigation_input': self.navigation_input}
    exec(self.code, values)
    self.nav_map, self.nav_map_time = values['nav_map'], values['nav_map_time']
    return values['inputs']
