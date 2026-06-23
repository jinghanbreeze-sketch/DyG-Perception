"""Aggregate atomic graphs and compute Visual Aliasing Entropy (VA-Entropy)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from membership_schedules import N_COMMUNITIES, N_NODES, WINDOW_SIZES


EVENT_TYPES = ("Stable", "SizeChange", "Merge", "Split")


def node_entropy(
    window: Sequence[Dict[str, Any]], node_id: int, k: int = N_COMMUNITIES
) -> float:
    labels = [frame["labels"][node_id] for frame in window]
    counts = Counter(labels)
    probabilities = [counts.get(label, 0) / len(window) for label in range(k)]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def visual_aliasing_entropy(
    window: Sequence[Dict[str, Any]],
    n: int = N_NODES,
    k: int = N_COMMUNITIES,
) -> Tuple[float, List[float]]:
    """Return raw VA-Entropy and the node-level entropy vector."""
    entropies = [node_entropy(window, node_id, k) for node_id in range(n)]
    return float(sum(entropies) / n), entropies


def normalized_vae(raw_vae: float, k: int = N_COMMUNITIES) -> float:
    """Normalize raw VA-Entropy to [0, 1] using log2(k)."""
    return float(raw_vae / math.log2(k))


def detect_window_event(
    window: Sequence[Dict[str, Any]],
) -> Tuple[str, Optional[int]]:
    """Apply the archived presence-based event priority."""
    non_stable = [frame for frame in window if frame["event"] != "Stable"]
    if not non_stable:
        return "Stable", None

    events = {frame["event"] for frame in non_stable}
    for event in ("Split", "Merge", "SizeChange"):
        if event in events:
            affected = next(
                (
                    frame["affected_community"]
                    for frame in non_stable
                    if frame["event"] == event
                ),
                None,
            )
            return event, affected

    return non_stable[0]["event"], non_stable[0]["affected_community"]


def aggregate_sequence(
    frames: Sequence[Dict[str, Any]],
    window_size: int,
    min_delta_for_sizechange: float = 0.005,
    max_delta_for_stable: float = 0.01,
) -> List[Dict[str, Any]]:
    step = window_size if window_size < 20 else max(1, window_size // 2)
    snapshots: List[Dict[str, Any]] = []

    for start in range(0, len(frames) - window_size + 1, step):
        window = frames[start : start + window_size]
        raw_vae, node_entropies = visual_aliasing_entropy(window)
        delta_sigma = float(np.mean([frame["delta_s"] for frame in window]))
        event_type, affected_community = detect_window_event(window)

        if event_type == "SizeChange" and delta_sigma < min_delta_for_sizechange:
            continue
        if event_type == "Stable" and delta_sigma > max_delta_for_stable:
            continue

        edge_weights: Dict[Tuple[int, int], float] = defaultdict(float)
        for frame in window:
            for source, target in frame["edges"]:
                edge_weights[tuple(sorted((source, target)))] += 1.0 / window_size

        majority_labels = []
        for node_id in range(N_NODES):
            labels = [frame["labels"][node_id] for frame in window]
            majority_labels.append(Counter(labels).most_common(1)[0][0])

        snapshots.append(
            {
                "id": len(snapshots),
                "cycle": window[0]["cycle"],
                "range": [start, start + window_size - 1],
                "W": window_size,
                "nodes": [
                    {
                        "id": node_id,
                        "label": int(majority_labels[node_id]),
                        "entropy": float(node_entropies[node_id]),
                    }
                    for node_id in range(N_NODES)
                ],
                "edges": [
                    {"s": source, "t": target, "weight": float(weight)}
                    for (source, target), weight in edge_weights.items()
                ],
                # Logs store raw entropy; analyses divide by log2(k).
                "h_alias": raw_vae,
                "delta_sigma": delta_sigma,
                "event_type": event_type,
                "affected_community": affected_community,
            }
        )

    return snapshots


def aggregate_all(dataset_dir: Path) -> None:
    levels_dir = dataset_dir / "levels"
    levels_dir.mkdir(parents=True, exist_ok=True)

    for strategy in ("burst", "stable"):
        atomic_path = dataset_dir / "atomic" / f"atomic_{strategy}.json"
        with atomic_path.open("r", encoding="utf-8") as handle:
            frames = json.load(handle)["frames"]

        for window_size in WINDOW_SIZES:
            snapshots = aggregate_sequence(frames, window_size)
            destination = levels_dir / f"level_{strategy}_W{window_size}.json"
            with destination.open("w", encoding="utf-8") as handle:
                json.dump({"snapshots": snapshots}, handle, indent=2)
            counts = Counter(snapshot["event_type"] for snapshot in snapshots[1:])
            print(
                f"[aggregate] {strategy} W={window_size}: "
                f"{len(snapshots)} snapshots; candidates={dict(counts)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    return parser.parse_args()


if __name__ == "__main__":
    aggregate_all(parse_args().dataset_dir)

