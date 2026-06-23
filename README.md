# DyG-Perception Reproducibility Materials

## Project Overview

This repository contains reproducibility materials for *Visual Aliasing Entropy
for Perception-Aware Temporal Aggregation in Dynamic Graph Visualization*. The
materials cover the main experimental workflow: synthetic dynamic graph
generation, temporal aggregation, VA-Entropy computation, browser-based event
classification, anonymized participant-level behavioral logs, main statistical
analysis, and figure generation.

## Repository Structure

| Path | Description |
|---|---|
| `stimulus_generation/membership_schedules.py` | Main experimental parameters and fixed membership-transition schedules. |
| `stimulus_generation/generate_dynamic_graphs.py` | Synthetic dynamic graph generator based on a dynamic stochastic block model. |
| `stimulus_generation/aggregate_and_compute_vae.py` | Temporal aggregation, majority community assignment, event labeling, and VA-Entropy computation. |
| `stimulus_generation/generate_fixed_layout.py` | Fixed node-layout generator used by the browser task. |
| `stimulus_generation/generate_stimuli.py` | End-to-end stimulus data generation entry point. |
| `stimuli_release/` | Released trial list, fixed layout, window metadata, and aggregated snapshot files. |
| `experiment_ui/index.html` | Browser experiment interface, visualization, interaction handling, and log export. |
| `experiment_ui/dist/d3.v7.min.js` | Local D3 dependency used by the browser interface. |
| `behavioral_logs/anonymized/` | Complete anonymized behavioral logs for 31 participants. |
| `analysis_scripts/run_main_results.py` | Scripted main analysis from anonymized logs to derived result tables. |
| `figure_generation/plot_publication_figures.py` | Figure-generation script for the main statistical figures. |
| `manuscript_figures/` | Paper figures and conceptual/interface images. |
| `scripts/check_release_integrity.py` | Local integrity checks for data counts, paths, and anonymization. |
| `requirements.txt` | Exact direct Python package versions used for validation. |

## Requirements

The scripts were validated with CPython `3.10.9`.

Install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The browser interface uses the vendored D3 `7.9.0` file in
`experiment_ui/dist/`; no npm installation or remote CDN is required.

## Synthetic Data and Stimuli Generation

The synthetic dynamic graph generator uses the following main settings:

- nodes: `n = 20`
- communities: `k = 3`
- stochastic block model probabilities: `p_in = 0.7`, `p_out = 0.1`
- dynamic sequence length: 800 discrete steps
- cycle structure: 8 cycles x 100 steps
- aggregation windows: `W = {1, 2, 5, 10, 20, 50}`
- graph generator seed: `42`
- trial-list seed: `20260227`
- event categories: `Continue`, `SizeChange`, `Merge`, `Split`

Regenerate the synthetic graph data and browser-loadable stimuli:

```bash
python stimulus_generation/generate_stimuli.py
```

The generated browser dataset is written to `experiment_ui/dataset/`, which is
ignored by Git. The staged release files in `stimuli_release/` provide the fixed
trial list, layout, metadata, and aggregated snapshots used for reproducibility.

Individual generation stages:

```bash
python stimulus_generation/generate_dynamic_graphs.py --output-dir experiment_ui/dataset
python stimulus_generation/aggregate_and_compute_vae.py --dataset-dir experiment_ui/dataset
python stimulus_generation/generate_fixed_layout.py --atomic-path experiment_ui/dataset/atomic/atomic_burst.json --output-path experiment_ui/dataset/layout_default.json
```

For an aggregation window `omega`, VA-Entropy is computed from the node-level
membership distribution before aggregation and then averaged over nodes. The raw
value is stored as `h_alias`; the normalized value is `h_alias / log2(k)`.

## Released Stimuli and Trial List

`stimuli_release/` contains:

- `stimuli_release/trial_list_72_seed20260227.json`
- `stimuli_release/layout_default.json`
- `stimuli_release/window_metadata_with_va_entropy.csv`
- `stimuli_release/aggregated_snapshots_W1.json`
- `stimuli_release/aggregated_snapshots_W2.json`
- `stimuli_release/aggregated_snapshots_W5.json`
- `stimuli_release/aggregated_snapshots_W10.json`
- `stimuli_release/aggregated_snapshots_W20.json`
- `stimuli_release/aggregated_snapshots_W50.json`

The formal trial list contains 72 trials:

```text
6 aggregation windows x 4 event categories x 3 trials = 72 trials
```

All participants were assigned the same fixed randomized trial list to keep
stimulus exposure identical across participants.

## Experimental Interface

After regenerating `experiment_ui/dataset/`, serve the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/experiment_ui/
```

The interface supports the main task flow:

- start page;
- event-classification trials;
- space-key toggling between `t` and `t-1`;
- confidence slider;
- mouse trajectory logging and derived mouse entropy;
- 800 ms masking screen between trials;
- end page and local log export.

The interface logs `h_alias` for analysis but does not display VA-Entropy or
`H_alias` to participants.

## Anonymized Behavioral Logs

`behavioral_logs/anonymized/` contains 31 participant files named
`P001.json` through `P031.json`. Each file contains 72 formal trials.

Each trial record includes:

- `participant_id`
- `trial_id`
- `trial_index`
- `W`
- `dataset_condition`
- `event_type`
- `true_label`
- `response`
- `correct`
- `rt_ms`
- `confidence`
- `toggle_count`
- `mouse_entropy`
- `h_alias`
- `h_alias_norm`
- `delta_sigma`
- `target_community`
- `source_window_id`
- `prev_snapshot_id`
- `curr_snapshot_id`
- `prev_range`
- `curr_range`
- `trajectory_sample_count`
- `layout_id`

## Reproducing the Main Results

Generate derived tables for the main results:

```bash
python analysis_scripts/run_main_results.py
```

By default, outputs are written to `analysis_outputs/`, which is ignored by Git.
The script reports participant count, formal trial count, and H2 RT-trimmed
trial count.

Generate the main statistical figures:

```bash
python figure_generation/plot_publication_figures.py
```

To write figures to a temporary directory instead of `manuscript_figures/`, set
`DYG_FIGURE_DIR`:

```bash
$env:DYG_FIGURE_DIR="analysis_outputs/figure_check"
python figure_generation/plot_publication_figures.py
```

Run local integrity checks:

```bash
python scripts/check_release_integrity.py
```

## Expected Data Checks

- participants = 31
- formal trials per participant = 72
- total formal trials = 2,232
- W values = `{1, 2, 5, 10, 20, 50}`
- event categories = `Continue`, `SizeChange`, `Merge`, `Split`
- H2 RT trimming rule = exclude `rt_ms < 200` or `rt_ms > 60000`
- expected H2 trial count after RT trimming = 2,194

## Data Anonymization

Participant identifiers are replaced with `P001` through `P031`. The released
logs retain trial-level behavioral and stimulus-index fields needed for the main
analyses. Raw subject identifiers, original file names, absolute local paths,
collection times, browser fingerprints, operating-system user names, IP
addresses, email addresses, and full raw mouse trajectories are not included.

Mouse movement is represented by the derived `mouse_entropy` field and
`trajectory_sample_count`.

## Citation

Please cite the associated paper when using these materials:

```text
Visual Aliasing Entropy for Perception-Aware Temporal Aggregation in Dynamic
Graph Visualization.
```
