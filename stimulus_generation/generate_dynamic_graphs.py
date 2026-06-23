"""Generate atomic synthetic dynamic graphs for the main experiment."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from membership_schedules import (
    CYCLE_LENGTH,
    GENERATOR_SEED,
    N_COMMUNITIES,
    N_CYCLES,
    N_NODES,
    P_IN,
    P_OUT,
    build_membership_schedule,
)


@dataclass
class EventRecord:
    event: str
    t: int
    cycle: int
    communities: Sequence[int]
    affected: int
    start: Optional[int] = None
    end: Optional[int] = None
    direction: Optional[str] = None


class DynamicSBMGenerator:
    """Dynamic SBM with externally specified community memberships."""

    def __init__(
        self,
        n: int = N_NODES,
        k: int = N_COMMUNITIES,
        p_in: float = P_IN,
        p_out: float = P_OUT,
        seed: int = GENERATOR_SEED,
    ) -> None:
        self.n = n
        self.k = k
        self.p_in = p_in
        self.p_out = p_out
        self.initial_labels = [node_id % k for node_id in range(n)]
        random.seed(seed)
        np.random.seed(seed)

    def _sample_edges(self, labels: Sequence[int]) -> List[List[int]]:
        edges: List[List[int]] = []
        for source in range(self.n):
            for target in range(source + 1, self.n):
                probability = (
                    self.p_in
                    if labels[source] == labels[target]
                    else self.p_out
                )
                if random.random() < probability:
                    edges.append([source, target])
        return edges

    def generate_atomic_sequence(
        self,
        strategy: str,
        num_cycles: int = N_CYCLES,
        cycle_length: int = CYCLE_LENGTH,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if cycle_length != CYCLE_LENGTH:
            raise ValueError(
                f"The released schedule requires cycle_length={CYCLE_LENGTH}."
            )

        frames: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []

        for cycle in range(num_cycles):
            labels = list(self.initial_labels)
            base_time = cycle * cycle_length
            schedule = build_membership_schedule(cycle, strategy)

            for t in range(base_time, base_time + cycle_length):
                old_labels = list(labels)
                event = "Stable"
                affected_community: Optional[int] = None
                direction: Optional[str] = None

                if schedule is not None:
                    merge = schedule["Merge"]
                    split = schedule["Split"]
                    size_change = schedule["SizeChange"]

                    if t == merge["time"]:
                        source, target = merge["source"], merge["target"]
                        labels = [target if label == source else label for label in labels]
                        event = "Merge"
                        affected_community = target
                        events.append(
                            asdict(
                                EventRecord(
                                    event=event,
                                    t=t,
                                    cycle=cycle,
                                    communities=[source, target],
                                    affected=target,
                                )
                            )
                        )
                    elif t == split["time"]:
                        source, target = split["source"], split["target"]
                        source_nodes = [
                            node_id
                            for node_id, label in enumerate(labels)
                            if label == source
                        ]
                        split_count = max(1, len(source_nodes) // 2)
                        for node_id in random.sample(source_nodes, split_count):
                            labels[node_id] = target
                        event = "Split"
                        affected_community = source
                        events.append(
                            asdict(
                                EventRecord(
                                    event=event,
                                    t=t,
                                    cycle=cycle,
                                    communities=[source, target],
                                    affected=source,
                                )
                            )
                        )
                    elif size_change["start"] <= t < size_change["end"]:
                        community_a = size_change["community_a"]
                        community_b = size_change["community_b"]
                        affected = size_change["affected"]
                        step_direction = random.choice(["grow", "shrink"])

                        if affected == community_b:
                            source, target = (
                                (community_a, community_b)
                                if step_direction == "grow"
                                else (community_b, community_a)
                            )
                        else:
                            source, target = (
                                (community_b, community_a)
                                if step_direction == "grow"
                                else (community_a, community_b)
                            )

                        source_nodes = [
                            node_id
                            for node_id, label in enumerate(labels)
                            if label == source
                        ]
                        if source_nodes:
                            migrate_count = 1
                            if len(source_nodes) >= 2 and random.random() < 0.35:
                                migrate_count = 2
                            for node_id in random.sample(
                                source_nodes,
                                k=min(migrate_count, len(source_nodes)),
                            ):
                                labels[node_id] = target
                            direction = step_direction

                        event = "SizeChange"
                        affected_community = affected
                        if t == size_change["start"]:
                            events.append(
                                asdict(
                                    EventRecord(
                                        event=event,
                                        t=t,
                                        cycle=cycle,
                                        start=t,
                                        end=size_change["end"] - 1,
                                        communities=[community_a, community_b],
                                        affected=affected,
                                        direction="mixed",
                                    )
                                )
                            )

                changed = sum(
                    old != new for old, new in zip(old_labels, labels)
                )
                frames.append(
                    {
                        "t": t,
                        "labels": list(labels),
                        "edges": self._sample_edges(labels),
                        "delta_s": float(changed / self.n),
                        "event": event,
                        "affected_community": affected_community,
                        "direction": direction,
                        "cycle": cycle,
                    }
                )

        return frames, events


def generate_all(output_dir: Path) -> None:
    atomic_dir = output_dir / "atomic"
    atomic_dir.mkdir(parents=True, exist_ok=True)
    generator = DynamicSBMGenerator()

    # This order preserves the random-number stream of the archived notebook.
    for strategy in ("burst", "stable"):
        frames, events = generator.generate_atomic_sequence(strategy)
        destination = atomic_dir / f"atomic_{strategy}.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump({"frames": frames, "events": events}, handle, indent=2)
        print(f"[atomic] {strategy}: {len(frames)} frames -> {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    return parser.parse_args()


if __name__ == "__main__":
    generate_all(parse_args().output_dir)
