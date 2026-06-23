"""Reproduce the main behavioral analysis from anonymized logs.

The script reads ``behavioral_logs/anonymized/*.json`` and writes derived
tables to ``analysis_outputs/``. Generated outputs are intentionally ignored by
Git.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


EVENT_ORDER = ["Continue", "SizeChange", "Merge", "Split"]
W_ORDER = [1, 2, 5, 10, 20, 50]
RT_MIN_MS = 200
RT_MAX_MS = 60000


def load_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(log_dir.glob("P*.json")):
        with path.open("r", encoding="utf-8") as handle:
            participant_rows = json.load(handle)
        rows.extend(participant_rows)

    if not rows:
        raise FileNotFoundError(f"No anonymized logs found in {log_dir}")

    frame = pd.DataFrame(rows)
    frame["ground_truth"] = frame["event_type"].replace({"Stable": "Continue"})
    frame["answer"] = frame["response"].replace({"Stable": "Continue"})
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce").astype(int)
    frame["W"] = pd.to_numeric(frame["W"], errors="coerce").astype(int)
    frame["rt_ms"] = pd.to_numeric(frame["rt_ms"], errors="coerce")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["toggle_count"] = pd.to_numeric(frame["toggle_count"], errors="coerce")
    frame["mouse_entropy"] = pd.to_numeric(frame["mouse_entropy"], errors="coerce")
    frame["halias_norm"] = pd.to_numeric(frame["h_alias_norm"], errors="coerce")
    frame["delta_sigma"] = pd.to_numeric(frame["delta_sigma"], errors="coerce")
    frame["ground_truth"] = pd.Categorical(frame["ground_truth"], EVENT_ORDER, ordered=True)
    frame["answer"] = pd.Categorical(frame["answer"], EVENT_ORDER, ordered=True)
    frame["W"] = pd.Categorical(frame["W"], W_ORDER, ordered=True)
    return frame


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "ch5_1_tables": output_dir / "ch5_1" / "tables",
        "ch5_1_figures": output_dir / "ch5_1" / "figures",
        "ch5_2_tables": output_dir / "ch5_2" / "tables",
        "ch5_2_figures": output_dir / "ch5_2" / "figures",
        "ch5_3_tables": output_dir / "ch5_3" / "tables",
        "ch5_3_figures": output_dir / "ch5_3" / "figures",
        "ch5_4_tables": output_dir / "ch5_4" / "tables",
        "ch5_4_figures": output_dir / "ch5_4" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def binomial_ci(accuracy: Iterable[float], n: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    accuracy = np.asarray(list(accuracy), dtype=float)
    n = np.asarray(list(n), dtype=float)
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    low = np.clip(accuracy - 1.96 * se, 0, 1)
    high = np.clip(accuracy + 1.96 * se, 0, 1)
    return low, high


def save_descriptive_tables(frame: pd.DataFrame, paths: dict[str, Path]) -> None:
    master = frame.copy()
    master["ground_truth"] = master["ground_truth"].astype(str)
    master["answer"] = master["answer"].astype(str)
    master.to_csv(paths["ch5_1_tables"] / "ch5_1_trials_master.csv", index=False)

    overall = pd.DataFrame(
        {
            "n_participants": [frame["participant_id"].nunique()],
            "n_trials": [len(frame)],
            "accuracy": [frame["correct"].mean()],
            "mean_rt_ms": [frame["rt_ms"].mean()],
        }
    )
    overall.to_csv(paths["ch5_1_tables"] / "table_5_1_overall_summary.csv", index=False)

    by_event = (
        frame.groupby("ground_truth", observed=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reindex(EVENT_ORDER)
        .reset_index()
    )
    by_event["ci95_lo"], by_event["ci95_hi"] = binomial_ci(
        by_event["accuracy"], by_event["n"]
    )
    by_event.to_csv(paths["ch5_1_tables"] / "table_5_1_by_event.csv", index=False)

    by_w = (
        frame.groupby("W", observed=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reindex(W_ORDER)
        .reset_index()
    )
    by_w["ci95_lo"], by_w["ci95_hi"] = binomial_ci(by_w["accuracy"], by_w["n"])
    by_w.to_csv(paths["ch5_1_tables"] / "table_5_1_by_W.csv", index=False)

    counts = pd.crosstab(frame["ground_truth"], frame["answer"]).reindex(
        index=EVENT_ORDER, columns=EVENT_ORDER, fill_value=0
    )
    counts.to_csv(paths["ch5_1_tables"] / "table_confusion_counts.csv")
    row_norm = counts.div(counts.sum(axis=1), axis=0).reset_index()
    row_norm.to_csv(paths["ch5_1_tables"] / "table_confusion_row_norm.csv", index=False)


def binned_mean(
    frame: pd.DataFrame,
    value_col: str,
    output_col: str,
    bins: int = 8,
    group_col: str | None = None,
) -> pd.DataFrame:
    clean = frame.dropna(subset=["halias_norm", value_col]).copy()
    clean["h_bin"] = pd.cut(clean["halias_norm"], bins=bins)
    groupers = ["h_bin"] if group_col is None else [group_col, "h_bin"]
    grouped = (
        clean.groupby(groupers, observed=True)
        .agg(
            n=(value_col, "size"),
            x_mean=("halias_norm", "mean"),
            **{output_col: (value_col, "mean")},
        )
        .reset_index()
    )
    if output_col == "acc":
        low, high = binomial_ci(grouped[output_col], grouped["n"])
        grouped["ci95_lo"] = low
        grouped["ci95_hi"] = high
    else:
        stats = (
            clean.groupby(groupers, observed=True)[value_col]
            .agg(["std"])
            .reset_index(drop=True)
        )
        se = stats["std"].fillna(0).to_numpy(float) / np.sqrt(grouped["n"].to_numpy(float))
        grouped["ci95_lo"] = grouped[output_col] - 1.96 * se
        grouped["ci95_hi"] = grouped[output_col] + 1.96 * se
    return grouped


def gee_table(result) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "beta": result.params.values,
            "se": result.bse.values,
            "z": result.tvalues.values,
            "p": result.pvalues.values,
            "ci95_lo": result.conf_int()[0].values,
            "ci95_hi": result.conf_int()[1].values,
        }
    )
    table["OR"] = np.exp(table["beta"])
    table["OR_ci95_lo"] = np.exp(table["ci95_lo"])
    table["OR_ci95_hi"] = np.exp(table["ci95_hi"])
    return table


def fit_gee(frame: pd.DataFrame, formula: str, family) -> pd.DataFrame:
    model = smf.gee(
        formula=formula,
        groups="participant_id",
        data=frame,
        family=family,
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return gee_table(model.fit())


def save_h1_tables(frame: pd.DataFrame, paths: dict[str, Path]) -> None:
    binned_mean(frame, "correct", "acc").to_csv(
        paths["ch5_2_tables"] / "table_5_2_binned_acc.csv", index=False
    )
    binned_mean(frame[frame["ground_truth"] != "Continue"], "correct", "acc", group_col="ground_truth").to_csv(
        paths["ch5_2_tables"] / "table_5_2_binned_acc_by_event.csv", index=False
    )
    fit_gee(frame, "correct ~ halias_norm", sm.families.Binomial()).to_csv(
        paths["ch5_2_tables"] / "table_5_2_gee_main.csv", index=False
    )
    fit_gee(frame, "correct ~ halias_norm + C(ground_truth) + C(W)", sm.families.Binomial()).to_csv(
        paths["ch5_2_tables"] / "table_5_2_gee_controls.csv", index=False
    )


def save_h2_tables(frame: pd.DataFrame, paths: dict[str, Path]) -> None:
    trimmed = frame[(frame["rt_ms"] >= RT_MIN_MS) & (frame["rt_ms"] <= RT_MAX_MS)].copy()
    trimmed["log_rt"] = np.log(trimmed["rt_ms"])
    metrics = [
        ("rt_ms", "table_5_3_rt_binned.csv"),
        ("toggle_count", "table_5_3_toggles_binned.csv"),
        ("mouse_entropy", "table_5_3_mouse_entropy_binned.csv"),
        ("confidence", "table_5_3_confidence_binned.csv"),
    ]
    for metric, filename in metrics:
        binned_mean(trimmed, metric, "mean").to_csv(
            paths["ch5_3_tables"] / filename, index=False
        )

    models = [
        ("log_rt", "table_5_3_gee_logrt_main.csv", "log_rt ~ halias_norm"),
        ("toggle_count", "table_5_3_gee_toggles_main.csv", "toggle_count ~ halias_norm"),
        (
            "mouse_entropy",
            "table_5_3_gee_mouse_entropy_main.csv",
            "mouse_entropy ~ halias_norm",
        ),
        ("confidence", "table_5_3_gee_confidence_main.csv", "confidence ~ halias_norm"),
    ]
    for _, filename, formula in models:
        fit_gee(trimmed, formula, sm.families.Gaussian()).to_csv(
            paths["ch5_3_tables"] / filename, index=False
        )

    pd.DataFrame(
        {
            "rt_min_ms": [RT_MIN_MS],
            "rt_max_ms": [RT_MAX_MS],
            "n_before_trim": [len(frame)],
            "n_after_trim": [len(trimmed)],
        }
    ).to_csv(paths["ch5_3_tables"] / "table_5_3_rt_trim_summary.csv", index=False)


def save_h3_tables(frame: pd.DataFrame, paths: dict[str, Path]) -> None:
    interaction = fit_gee(
        frame,
        "correct ~ halias_norm * C(ground_truth) + C(W)",
        sm.families.Binomial(),
    )
    interaction.to_csv(paths["ch5_4_tables"] / "table_5_4_event_dependent_gee.csv", index=False)

    merge = frame[frame["ground_truth"] == "Merge"].copy()
    binned_mean(merge, "correct", "acc").to_csv(
        paths["ch5_4_tables"] / "table_5_4_merge_acc_by_entropy.csv", index=False
    )

    descriptors = fit_gee(
        frame,
        "correct ~ halias_norm + delta_sigma + C(W) + C(ground_truth)",
        sm.families.Binomial(),
    )
    descriptors.to_csv(paths["ch5_4_tables"] / "table_5_4_descriptor_comparison.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("behavioral_logs/anonymized"),
        help="Directory containing anonymized participant JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory for generated analysis tables and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_logs(args.log_dir)
    paths = ensure_dirs(args.output_dir)
    save_descriptive_tables(frame, paths)
    save_h1_tables(frame, paths)
    save_h2_tables(frame, paths)
    save_h3_tables(frame, paths)
    print(f"participants={frame['participant_id'].nunique()}")
    print(f"formal_trials={len(frame)}")
    trimmed = frame[(frame["rt_ms"] >= RT_MIN_MS) & (frame["rt_ms"] <= RT_MAX_MS)]
    print(f"h2_rt_trimmed_trials={len(trimmed)}")
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
