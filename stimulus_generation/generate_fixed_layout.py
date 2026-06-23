"""Generate the fixed node layout used by the browser experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np


LAYOUT_SEED = 30
SPRING_SEED = 42


def generate_fixed_layout(atomic_path: Path, output_path: Path) -> None:
    with atomic_path.open("r", encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]

    n = len(frames[0]["labels"])
    union_graph = nx.Graph()
    union_graph.add_nodes_from(range(n))

    edge_weights: Dict[tuple, int] = {}
    for frame in frames:
        for source, target in frame["edges"]:
            edge = tuple(sorted((source, target)))
            edge_weights[edge] = edge_weights.get(edge, 0) + 1
    for (source, target), weight in edge_weights.items():
        union_graph.add_edge(source, target, weight=weight)

    radius = 2.0
    centers = {
        0: np.array([0.0, radius]),
        1: np.array([-radius * 0.866, -radius * 0.5]),
        2: np.array([radius * 0.866, -radius * 0.5]),
    }
    initial_labels = frames[0]["labels"]
    initial_positions = {
        node_id: centers[initial_labels[node_id]] + np.random.normal(0, 0.2, 2)
        for node_id in range(n)
    }

    positions = nx.spring_layout(
        union_graph,
        pos=initial_positions,
        weight="weight",
        k=0.2,
        iterations=500,
        seed=SPRING_SEED,
    )

    coordinates = np.array(list(positions.values()))
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    for node_id in positions:
        positions[node_id] = (
            (positions[node_id] - minimum) / (maximum - minimum) * 2 - 1
        )

    layout = {
        node_id: {
            "x": float(positions[node_id][0]),
            "y": float(positions[node_id][1]),
        }
        for node_id in range(n)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(layout, handle, indent=2)
    print(f"[layout] fixed coordinates -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atomic-path",
        type=Path,
        default=Path("dataset/atomic/atomic_burst.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("dataset/layout_default.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(LAYOUT_SEED)
    random.seed(LAYOUT_SEED)
    arguments = parse_args()
    generate_fixed_layout(arguments.atomic_path, arguments.output_path)

