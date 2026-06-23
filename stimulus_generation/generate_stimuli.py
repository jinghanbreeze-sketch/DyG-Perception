"""Run the complete stimulus-data generation pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from aggregate_and_compute_vae import aggregate_all
from generate_dynamic_graphs import generate_all
from generate_fixed_layout import LAYOUT_SEED, generate_fixed_layout


def generate_stimuli(dataset_dir: Path) -> None:
    generate_all(dataset_dir)
    aggregate_all(dataset_dir)

    np.random.seed(LAYOUT_SEED)
    random.seed(LAYOUT_SEED)
    generate_fixed_layout(
        dataset_dir / "atomic" / "atomic_burst.json",
        dataset_dir / "layout_default.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("experiment_ui/dataset")
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_stimuli(parse_args().dataset_dir)
