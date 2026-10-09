"""Consume a single watchdog restart request without enabling restart loops."""
from openpilot.common.params import Params
from openpilot.system.manager.process import NativeProcess


class V3RecoveryProcess(NativeProcess):
  def start(self):
    if self.proc is not None and self.proc.exitcode is not None:
      params = Params()
      if params.get_bool('CinqueV3RecoveryPending'):
        # Consume before spawning: ordinary missing-model errors never request
        # another restart, and a persistent diagnostic fault is not a request.
        params.put_bool('CinqueV3RecoveryPending', False, block=True)
        self.stop()
    super().start()
