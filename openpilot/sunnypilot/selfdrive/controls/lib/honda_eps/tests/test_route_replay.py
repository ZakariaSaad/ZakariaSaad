from openpilot.sunnypilot.selfdrive.controls.lib.honda_eps.route_replay import DeliveryLimitHistory


def test_recorded_delivery_limit_is_consumed_on_following_control_tick():
  history = DeliveryLimitHistory()
  # A delivery difference is discovered after the current core call. Applying
  # it immediately freezes the wrong integration tick in recorded-route replay.
  assert not history.observe(.4, 0., True)
  assert history.observe(.4, .4, True)
  assert not history.observe(.4, .4, True)
  # Inactive controlsd clears its proxy, even if a stale delivered value differs.
  assert not history.observe(.8, 0., False)
  assert not history.observe(.1, 0., True)
  assert history.observe(.1, .1, True)
