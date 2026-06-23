"""Membership-transition schedule used by the main experiment."""

from __future__ import annotations

from typing import Any, Dict, Optional


N_NODES = 20
N_COMMUNITIES = 3
P_IN = 0.7
P_OUT = 0.1
N_CYCLES = 8
CYCLE_LENGTH = 100
WINDOW_SIZES = (1, 2, 5, 10, 20, 50)
GENERATOR_SEED = 42


def build_membership_schedule(
    cycle: int, strategy: str
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Return the fixed transition schedule for one 100-step cycle.

    ``stable`` cycles contain no transitions. ``burst`` cycles contain a
    merge at offset 12, a split at offset 36, and a size-change interval
    spanning offsets 60 through 99 (inclusive).
    """
    if strategy == "stable":
        return None
    if strategy != "burst":
        raise ValueError("strategy must be 'burst' or 'stable'")

    base_time = cycle * CYCLE_LENGTH
    return {
        "Merge": {
            "time": base_time + 12,
            "source": 0,
            "target": 1,
        },
        "Split": {
            "time": base_time + 36,
            "source": 2,
            "target": 0,
        },
        "SizeChange": {
            "start": base_time + 60,
            "end": base_time + 100,
            "community_a": 1,
            "community_b": 2,
            "affected": 2,
        },
    }

