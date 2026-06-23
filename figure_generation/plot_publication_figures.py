"""Publication styling for manuscript Fig. 4-Fig. 10 and VA-Entropy distribution.

This module only reads existing analysis tables and the existing chapter 5.1
master trial table. It does not fit models or write statistical/data outputs.
"""

import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version.*bottleneck")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis_out"
CH51 = ANALYSIS_DIR / "ch5_1"
CH52 = ANALYSIS_DIR / "ch5_2"
CH53 = ANALYSIS_DIR / "ch5_3"
CH54 = ANALYSIS_DIR / "ch5_4"
MANUSCRIPT_DIR = ROOT / "manuscript_figures"

EVENT_ORDER = ["Continue", "SizeChange", "Merge", "Split"]
EVENT_COLORS = {
    "Continue": "#F1C04B",
    "SizeChange": "#BFBFBF",
    "Merge": "#7EBEB9",
    "Split": "#64AB5C",
}
STEELBLUE = "#4682B4"
W_ORDER = [1, 2, 5, 10, 20, 50]
CHANCE_LEVEL = 0.25
PNG_DPI = 600
JITTER_SEED = 20260622


def _set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "errorbar.capsize": 3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required existing analysis table not found: {path}")
    return pd.read_csv(path)


def _load_master_trials():
    frame = _read_csv(CH51 / "tables" / "ch5_1_trials_master.csv")
    frame["ground_truth"] = frame["ground_truth"].replace({"Stable": "Continue"})
    frame["answer"] = frame["answer"].replace({"Stable": "Continue"})
    frame["halias_norm"] = pd.to_numeric(frame["h_alias"], errors="coerce") / math.log2(3)
    return frame


def _finish_axis(axis, grid_axis="y"):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)


def _accuracy_axis(axis):
    axis.set_ylim(0, 1)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.axhline(CHANCE_LEVEL, color="#555555", linestyle="--", linewidth=1.1, zorder=1)
    _finish_axis(axis)


def _panel_label(axis, label):
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")


def _save_figure(figure, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    path = base_path.with_suffix(".png")
    figure.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04)
    return [path]


def _save_manuscript_png(figure, filename):
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = MANUSCRIPT_DIR / filename
    figure.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04)
    return path


def _binomial_ci(accuracy, n):
    accuracy = np.asarray(accuracy, dtype=float)
    n = np.asarray(n, dtype=float)
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    low = np.clip(accuracy - 1.96 * se, 0, 1)
    high = np.clip(accuracy + 1.96 * se, 0, 1)
    return low, high


def _accuracy_note(axis, ci_description):
    axis.text(
        0.985,
        0.965,
        f"{ci_description}\nDashed line: Chance level = 0.25",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555555",
    )


def _plot_ci_line(axis, frame, color, label=None, y_column="acc"):
    clean = frame.dropna(subset=["x_mean", y_column, "ci95_lo", "ci95_hi"]).sort_values("x_mean")
    x = clean["x_mean"].astype(float).to_numpy()
    y = clean[y_column].astype(float).to_numpy()
    low = clean["ci95_lo"].astype(float).to_numpy()
    high = clean["ci95_hi"].astype(float).to_numpy()
    axis.plot(x, y, color=color, marker="o", label=label, zorder=3)
    axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0, zorder=2)


def _plot_event_binned_means(axis, frame, color, label):
    """Plot every existing valid bin mean with its existing 95% CI."""
    ordered = frame.reset_index(drop=True).copy()
    numeric_columns = ["n", "x_mean", "acc"]
    for column in numeric_columns:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    valid = (
        ordered[numeric_columns].notna().all(axis=1) & (ordered["n"] > 0)
    )

    points = ordered.loc[valid]
    if points.empty:
        return
    low = pd.to_numeric(points["ci95_lo"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(points["ci95_hi"], errors="coerce").to_numpy(float)
    means = points["acc"].to_numpy(float)
    axis.errorbar(
        points["x_mean"],
        means,
        yerr=np.vstack([means - low, high - means]),
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.5,
        markersize=5.5,
        label=label,
        zorder=4,
    )

    # Connect only contiguous valid rows. Invalid or suppressed bins break a line.
    segment_start = None
    for position in range(len(ordered) + 1):
        is_valid = position < len(ordered) and bool(valid.iloc[position])
        if is_valid and segment_start is None:
            segment_start = position
        elif not is_valid and segment_start is not None:
            segment = ordered.iloc[segment_start:position]
            if len(segment) >= 2:
                axis.plot(
                    segment["x_mean"],
                    segment["acc"],
                    color=color,
                    marker=None,
                    zorder=3,
                )
            segment_start = None


def figure_4_accuracy_by_event(master):
    """Fig. 4: accuracy across event types."""
    data = _read_csv(CH51 / "tables" / "table_5_1_by_event.csv").set_index("ground_truth")
    data = data.reindex(EVENT_ORDER)
    figure, axis = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    axis.bar(
        np.arange(len(EVENT_ORDER)),
        data["accuracy"],
        width=0.62,
        color=[EVENT_COLORS[event] for event in EVENT_ORDER],
        edgecolor="#000000",
        linewidth=0.6,
        zorder=3,
    )
    low, high = _binomial_ci(data["accuracy"], data["n"])
    for position, event, mean, lo, hi in zip(
        np.arange(len(EVENT_ORDER)), EVENT_ORDER, data["accuracy"], low, high
    ):
        axis.errorbar(
            position,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="none",
            ecolor="#333333",
            markeredgecolor="#222222",
            elinewidth=1.4,
            capsize=3.5,
            zorder=4,
        )
    axis.set_xticks(np.arange(len(EVENT_ORDER)), EVENT_ORDER)
    axis.set_xlabel("Event type")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by event type")
    _accuracy_axis(axis)
    _accuracy_note(axis, "Error bars: 95% CI")
    return _save_figure(figure, CH51 / "figures" / "fig_5_1_acc_by_event"), figure


def figure_5_accuracy_by_window(master):
    """Fig. 5: accuracy across aggregation windows."""
    data = _read_csv(CH51 / "tables" / "table_5_1_by_W.csv").set_index("W").reindex(W_ORDER)
    positions = np.arange(len(W_ORDER))
    figure, axis = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    axis.plot(positions, data["accuracy"], color=STEELBLUE, marker="o", zorder=3)
    low, high = _binomial_ci(data["accuracy"], data["n"])
    means = data["accuracy"].to_numpy(float)
    axis.errorbar(
        positions,
        means,
        yerr=np.vstack([means - low, high - means]),
        fmt="none",
        ecolor=STEELBLUE,
        elinewidth=1.2,
        capsize=3.5,
        zorder=4,
    )
    axis.set_xticks(positions, [f"W = {window}" for window in W_ORDER])
    axis.set_xlabel("Aggregation window size")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by aggregation window size")
    _accuracy_axis(axis)
    _accuracy_note(axis, "Error bars: 95% CI")
    return _save_figure(figure, CH51 / "figures" / "fig_5_1_acc_by_W"), figure


def figure_6_confusion_matrix():
    """Fig. 6: row-normalized confusion matrix."""
    matrix = _read_csv(CH51 / "tables" / "table_confusion_row_norm.csv")
    matrix = matrix.set_index("ground_truth").reindex(index=EVENT_ORDER, columns=EVENT_ORDER)
    cmap = LinearSegmentedColormap.from_list("white_to_steelblue", ["#FFFFFF", STEELBLUE])
    figure, axis = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="equal")
    axis.set_xticks(np.arange(4), EVENT_ORDER, rotation=25, ha="right")
    axis.set_yticks(np.arange(4), EVENT_ORDER)
    axis.set_xlabel("Reported event type")
    axis.set_ylabel("True event type")
    axis.set_title("Row-normalized confusion matrix")
    for row in range(4):
        for column in range(4):
            value = float(matrix.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.52 else "#222222",
                fontsize=9,
                fontweight="bold" if row == column else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Response proportion")
    return _save_figure(figure, CH51 / "figures" / "fig_confusion_row_norm"), figure


def figure_7_accuracy_vs_entropy():
    """Fig. 7: binned accuracy versus normalized VA-Entropy."""
    data = _read_csv(CH52 / "tables" / "table_5_2_binned_acc.csv")
    figure, axis = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    _plot_ci_line(axis, data, STEELBLUE)
    axis.set_xlabel("Normalized VA-Entropy")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy versus normalized VA-Entropy")
    _accuracy_axis(axis)
    _accuracy_note(axis, "Shaded band: 95% CI")
    return _save_figure(figure, CH52 / "figures" / "fig_5_2_acc_vs_halias_binned"), figure


def _continue_entropy_point(master):
    continue_rows = master[master["ground_truth"] == "Continue"].dropna(
        subset=["halias_norm", "correct"]
    )
    n = len(continue_rows)
    accuracy = float(continue_rows["correct"].mean())
    se = math.sqrt(accuracy * (1 - accuracy) / n)
    return pd.DataFrame(
        {
            "n": [n],
            "x_mean": [float(continue_rows["halias_norm"].mean())],
            "acc": [accuracy],
            "ci95_lo": [max(0.0, accuracy - 1.96 * se)],
            "ci95_hi": [min(1.0, accuracy + 1.96 * se)],
            "ground_truth": ["Continue"],
        }
    )


def figure_8_accuracy_vs_entropy_by_event(master):
    """Fig. 8: binned accuracy versus normalized VA-Entropy by event."""
    data = _read_csv(CH52 / "tables" / "table_5_2_binned_acc_by_event.csv")
    data = pd.concat([_continue_entropy_point(master), data], ignore_index=True)
    figure, axis = plt.subplots(figsize=(6.1, 4.1), constrained_layout=True)
    for event in EVENT_ORDER:
        group = data[data["ground_truth"] == event]
        _plot_event_binned_means(axis, group, EVENT_COLORS[event], label=event)
    axis.set_xlabel("Normalized VA-Entropy")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy versus normalized VA-Entropy by event type")
    axis.legend(
        frameon=False,
        ncol=2,
        loc="best",
        title="Error bars: 95% CI",
        title_fontsize=8,
    )
    _accuracy_axis(axis)
    return _save_figure(
        figure, CH52 / "figures" / "fig_5_2_acc_vs_halias_binned_by_event"
    ), figure


def figure_9_behavior_vs_entropy():
    """Fig. 9: behavioral measures versus normalized VA-Entropy."""
    specifications = [
        ("table_5_3_rt_binned.csv", "Response time (ms)", "(A)"),
        ("table_5_3_toggles_binned.csv", "Toggle frequency", "(B)"),
        ("table_5_3_mouse_entropy_binned.csv", "Mouse-movement entropy", "(C)"),
        ("table_5_3_confidence_binned.csv", "Confidence", "(D)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for axis, (filename, ylabel, panel) in zip(axes.flat, specifications):
        data = _read_csv(CH53 / "tables" / filename)
        _plot_ci_line(axis, data, STEELBLUE, y_column="mean")
        axis.set_xlabel("Normalized VA-Entropy")
        axis.set_ylabel(ylabel)
        _panel_label(axis, panel)
        _finish_axis(axis)
    figure.suptitle(
        "Behavioral measures versus normalized VA-Entropy\nShaded bands indicate 95% CI",
        fontsize=11,
    )
    return _save_figure(figure, CH53 / "figures" / "fig_5_3_behavior_vs_halias"), figure


def _merge_summaries(master):
    merge = master[master["ground_truth"] == "Merge"].dropna(
        subset=["halias_norm", "correct", "W"]
    ).copy()
    merge["h_bin"] = pd.cut(merge["halias_norm"], bins=8)
    by_entropy = (
        merge.groupby("h_bin", observed=True)
        .agg(x_mean=("halias_norm", "mean"), accuracy=("correct", "mean"), n=("correct", "count"))
        .reset_index(drop=True)
        .sort_values("x_mean")
    )
    entropy_low, entropy_high = _binomial_ci(by_entropy["accuracy"], by_entropy["n"])
    by_entropy["ci95_lo"] = entropy_low
    by_entropy["ci95_hi"] = entropy_high
    by_window = (
        merge.groupby("W", observed=True)
        .agg(accuracy=("correct", "mean"), n=("correct", "count"))
        .reindex(W_ORDER)
        .reset_index()
    )
    window_low, window_high = _binomial_ci(by_window["accuracy"], by_window["n"])
    by_window["ci95_lo"] = window_low
    by_window["ci95_hi"] = window_high
    return by_entropy, by_window


def _draw_merge_entropy(axis, data):
    axis.errorbar(
        data["x_mean"],
        data["accuracy"],
        yerr=np.vstack(
            [data["accuracy"] - data["ci95_lo"], data["ci95_hi"] - data["accuracy"]]
        ),
        color=EVENT_COLORS["Merge"],
        ecolor=EVENT_COLORS["Merge"],
        marker="o",
        capsize=3,
        zorder=3,
    )
    axis.set_xlabel("Normalized VA-Entropy")
    axis.set_ylabel("Accuracy")
    axis.set_title("Merge accuracy versus normalized VA-Entropy")
    _accuracy_axis(axis)
    _accuracy_note(axis, "Error bars: 95% CI")


def _draw_merge_window(axis, data):
    positions = np.arange(len(W_ORDER))
    axis.errorbar(
        positions,
        data["accuracy"],
        yerr=np.vstack(
            [data["accuracy"] - data["ci95_lo"], data["ci95_hi"] - data["accuracy"]]
        ),
        color=EVENT_COLORS["Merge"],
        ecolor=EVENT_COLORS["Merge"],
        marker="o",
        capsize=3,
        zorder=3,
    )
    axis.set_xticks(positions, [f"W = {window}" for window in W_ORDER])
    axis.set_xlabel("Aggregation window size")
    axis.set_ylabel("Accuracy")
    axis.set_title("Merge accuracy versus aggregation window size")
    _accuracy_axis(axis)
    _accuracy_note(axis, "Error bars: 95% CI")


def figure_10_merge(master, manuscript_path=None):
    """Fig. 10: two-panel Merge accuracy analysis."""
    by_entropy, by_window = _merge_summaries(master)
    outputs = []

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    _draw_merge_entropy(axes[0], by_entropy)
    _draw_merge_window(axes[1], by_window)
    _panel_label(axes[0], "(A)")
    _panel_label(axes[1], "(B)")
    if manuscript_path is not None:
        _save_manuscript_png(figure, manuscript_path)
    outputs.extend(_save_figure(figure, CH54 / "figures" / "fig_5_4_merge_two_panel"))
    plt.close(figure)

    figure_a, axis_a = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    _draw_merge_entropy(axis_a, by_entropy)
    outputs.extend(
        _save_figure(figure_a, CH54 / "figures" / "fig_5_4_merge_accuracy_vs_halias")
    )
    plt.close(figure_a)

    figure_b, axis_b = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    _draw_merge_window(axis_b, by_window)
    outputs.extend(_save_figure(figure_b, CH54 / "figures" / "fig_5_4_merge_accuracy_vs_W"))
    plt.close(figure_b)
    return outputs


def _boxplot(axis, grouped_values, positions, colors, labels):
    boxes = axis.boxplot(
        grouped_values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.3},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
        boxprops={"edgecolor": "#444444", "linewidth": 0.8},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axis.set_xticks(positions, labels)


def figure_vae_distribution(master):
    """Visualize normalized VA-Entropy across W and event types.

    This new figure is intended to show that normalized VA-Entropy is related
    to temporal aggregation while not being reducible to window size alone.
    """
    rng = np.random.default_rng(JITTER_SEED)
    clean = master.dropna(subset=["halias_norm", "W", "ground_truth"]).copy()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True, sharey=True)

    w_positions = np.arange(len(W_ORDER))
    w_values = [clean.loc[clean["W"] == window, "halias_norm"].to_numpy() for window in W_ORDER]
    _boxplot(
        axes[0],
        w_values,
        w_positions,
        [STEELBLUE] * len(W_ORDER),
        [f"W = {window}" for window in W_ORDER],
    )
    for position, values in zip(w_positions, w_values):
        jitter = rng.normal(0, 0.055, len(values))
        axes[0].scatter(
            position + jitter, values, s=7, color=STEELBLUE, alpha=0.10, edgecolors="none", rasterized=True
        )
    axes[0].set_xlabel("Aggregation window size")
    axes[0].set_ylabel("Normalized VA-Entropy")
    axes[0].set_title("VA-Entropy by aggregation window size")
    _panel_label(axes[0], "(A)")

    event_positions = np.arange(len(EVENT_ORDER))
    event_values = [
        clean.loc[clean["ground_truth"] == event, "halias_norm"].to_numpy()
        for event in EVENT_ORDER
    ]
    _boxplot(
        axes[1],
        event_values,
        event_positions,
        [EVENT_COLORS[event] for event in EVENT_ORDER],
        EVENT_ORDER,
    )
    for position, event, values in zip(event_positions, EVENT_ORDER, event_values):
        jitter = rng.normal(0, 0.055, len(values))
        axes[1].scatter(
            position + jitter,
            values,
            s=7,
            color=EVENT_COLORS[event],
            alpha=0.12,
            edgecolors="none",
            rasterized=True,
        )
    axes[1].set_xlabel("Event type")
    axes[1].set_title("VA-Entropy by event type")
    _panel_label(axes[1], "(B)")

    for axis in axes:
        axis.set_ylim(0, 1)
        _finish_axis(axis)
    return _save_figure(figure, CH52 / "figures" / "fig_vae_distribution"), figure


def figure_model_effect_forest():
    """Forest plot of already saved adjusted GEE odds ratios and 95% CIs."""
    data = _read_csv(CH52 / "tables" / "table_5_2_gee_controls.csv")
    term_order = [
        "halias_norm",
        "C(ground_truth)[T.SizeChange]",
        "C(ground_truth)[T.Merge]",
        "C(ground_truth)[T.Split]",
        "C(W)[T.2]",
        "C(W)[T.5]",
        "C(W)[T.10]",
        "C(W)[T.20]",
        "C(W)[T.50]",
    ]
    readable = {
        "halias_norm": "Normalized VA-Entropy (per 0.1 increase)",
        "C(ground_truth)[T.SizeChange]": "SizeChange vs Continue",
        "C(ground_truth)[T.Merge]": "Merge vs Continue",
        "C(ground_truth)[T.Split]": "Split vs Continue",
        "C(W)[T.2]": "W = 2 vs W = 1",
        "C(W)[T.5]": "W = 5 vs W = 1",
        "C(W)[T.10]": "W = 10 vs W = 1",
        "C(W)[T.20]": "W = 20 vs W = 1",
        "C(W)[T.50]": "W = 50 vs W = 1",
    }
    effects = data.set_index("term").reindex(term_order)
    if effects[["OR", "OR_ci95_lo", "OR_ci95_hi"]].isna().any().any():
        raise ValueError("Saved GEE controls table is missing a required forest-plot effect.")

    # Algebraically rescale the saved one-unit effect; the model is not refit.
    entropy_term = "halias_norm"
    effects.loc[entropy_term, "OR"] = math.exp(
        float(effects.loc[entropy_term, "beta"]) * 0.1
    )
    effects.loc[entropy_term, "OR_ci95_lo"] = math.exp(
        float(effects.loc[entropy_term, "ci95_lo"]) * 0.1
    )
    effects.loc[entropy_term, "OR_ci95_hi"] = math.exp(
        float(effects.loc[entropy_term, "ci95_hi"]) * 0.1
    )

    estimates = effects["OR"].to_numpy(float)
    low = effects["OR_ci95_lo"].to_numpy(float)
    high = effects["OR_ci95_hi"].to_numpy(float)
    positions = np.arange(len(effects))
    figure, axis = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack([estimates - low, high - estimates]),
        fmt="o",
        color=STEELBLUE,
        ecolor=STEELBLUE,
        elinewidth=1.4,
        capsize=3,
        markersize=5.5,
        zorder=3,
    )
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.set_xscale("log")
    axis.set_xlim(0.002, 6)
    axis.set_yticks(positions, [readable[term] for term in term_order])
    axis.invert_yaxis()
    axis.set_xlabel("Odds ratio (log scale)")
    axis.set_title("Adjusted GEE model effects (95% CI)")
    axis.text(
        0.99,
        0.985,
        "Dashed line: OR = 1",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    _finish_axis(axis, grid_axis="x")
    return _save_manuscript_png(figure, "fig_model_effect_forest.png"), figure


def main():
    _set_publication_style()
    master = _load_master_trials()
    manuscript_outputs = []
    builders = (
        (lambda: figure_4_accuracy_by_event(master), "fig4_accuracy_by_event.png"),
        (lambda: figure_5_accuracy_by_window(master), "fig5_accuracy_by_window.png"),
        (figure_6_confusion_matrix, "fig6_confusion_matrix.png"),
        (figure_7_accuracy_vs_entropy, "fig7_accuracy_vs_vae.png"),
        (
            lambda: figure_8_accuracy_vs_entropy_by_event(master),
            "fig8_accuracy_vs_vae_by_event.png",
        ),
        (figure_9_behavior_vs_entropy, "fig9_behavior_vs_vae.png"),
    )
    for builder, manuscript_name in builders:
        _, figure = builder()
        manuscript_outputs.append(_save_manuscript_png(figure, manuscript_name))
        plt.close(figure)

    figure_10_merge(master, manuscript_path="fig10_merge_accuracy.png")
    manuscript_outputs.append(MANUSCRIPT_DIR / "fig10_merge_accuracy.png")

    _, figure = figure_vae_distribution(master)
    manuscript_outputs.append(_save_manuscript_png(figure, "fig_vae_distribution.png"))
    plt.close(figure)

    forest_path, figure = figure_model_effect_forest()
    manuscript_outputs.append(forest_path)
    plt.close(figure)

    print(f"Saved {len(manuscript_outputs)} final manuscript PNG files:")
    for path in manuscript_outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
