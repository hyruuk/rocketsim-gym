"""Keeping time for a loop that runs a fixed number of times a second."""

from __future__ import annotations

import time

#: How far behind a loop may fall before it stops catching up.
GIVE_UP_SECONDS = 0.1


def wait_for(moment: float) -> float:
    """Wait until ``moment`` on ``time.perf_counter``'s clock, sleeping most of the way and
    spinning the last 2 ms, since a sleep can overshoot by milliseconds; return when the round
    that follows starts, which is now if the loop has fallen :data:`GIVE_UP_SECONDS` behind."""
    delay = moment - time.perf_counter()
    if delay < -GIVE_UP_SECONDS:
        return time.perf_counter()
    if delay > 0.002:
        time.sleep(delay - 0.002)
    while time.perf_counter() < moment:
        pass
    return moment
