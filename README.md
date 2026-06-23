# DyG-Perception Reproducibility Materials

This repository contains the reproducibility materials for *Visual Aliasing
Entropy for Perception-Aware Temporal Aggregation in Dynamic Graph
Visualization*. It includes the source code for stimulus construction, the
browser experiment interface, analysis notebooks, figure-generation scripts, two
anonymized example behavioral logs, and the paper figures.

Generated graph datasets, full behavioral records, intermediate analysis tables,
fitted models, notebook outputs, and local cache files are not versioned.

## Repository Contents

| Path | Description |
|---|---|
| `stimulus_generation/membership_schedules.py` | Main experimental parameters and fixed membership-transition schedules for the Stable and Burst conditions. |
| `stimulus_generation/generate_dynamic_graphs.py` | Synthetic dynamic graph generator based on a dynamic stochastic block model. |
| `stimulus_generation/aggregate_and_compute_vae.py` | Temporal aggregation, majority-membership assignment, event labeling, and VA-Entropy computation. |
| `stimulus_generation/generate_fixed_layout.py` | Fixed node-layout generator used by the browser task. |
| `stimulus_generation/generate_stimuli.py` | End-to-end stimulus-generation entry point. |
| `experiment_ui/index.html` | Browser experiment interface, trial construction, visualization, keyboard/mouse interaction handling, and log export. |
| `experiment_ui/dist/d3.v7.min.js` | Local D3 dependency used by the browser interface. |
| `behavioral_logs/examples/` | Two anonymized 72-trial behavioral log examples with participant IDs replaced by `P001` and `P002`. |
| `analysis_notebooks/` | Data preparation, descriptive statistics, and H1/H2/H3 statistical analysis notebooks. Notebook outputs are cleared. |
| `figure_generation/plot_publication_figures.py` | Script for regenerating statistical paper figures from analysis tables. |
| `supplementary_analysis/robustness/run_comment6_checks.py` | Additional robustness-analysis script that runs on the cleaned full-study analysis tables. |
| `manuscript_figures/` | Twelve paper images, including nine statistical result figures and three conceptual/interface figures. |
| `requirements.txt` | Exact direct Python package versions used for validation. |

## Experimental Settings

- Nodes: `n = 20`
- Communities: `k = 3`
- Stochastic block model probabilities: `p_in = 0.7`, `p_out = 0.1`
- Dynamic sequence length: 8 cycles of 100 atomic steps
- Aggregation windows: `W = {1, 2, 5, 10, 20, 50}`
- Graph generator seed: `42`
- Browser trial-list seed: `20260227`
- Browser task length: 72 trials
- Source event labels: `Stable`, `SizeChange`, `Merge`, and `Split`

For an aggregation window `omega`, the stimulus-generation code computes
node-level membership entropy from the pre-aggregation membership distribution
and averages it over nodes. The raw VA-Entropy value is stored as `h_alias`; the
normalized value used in the statistical analyses is `h_alias / log2(k)`.

## Software Environment

The scripts were validated with CPython `3.10.9`. The copied notebooks preserve
their original Python `3.8.5` kernel metadata.

Install the Python dependencies with:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The browser interface uses the vendored D3 `7.9.0` file in
`experiment_ui/dist/`; no npm installation or remote CDN is required. The
interface should be served over HTTP because it loads generated JSON files with
`fetch`.

## Generate Stimuli

From the repository root:

```bash
python stimulus_generation/generate_stimuli.py
```

This creates generated files under `experiment_ui/dataset/`, including:

- atomic dynamic graph sequences;
- temporally aggregated graph snapshots;
- event labels and VA-Entropy fields;
- the fixed node layout used by the task.

`experiment_ui/dataset/` is ignored by Git because it contains generated data.

The stages can also be run separately:

```bash
python stimulus_generation/generate_dynamic_graphs.py --output-dir experiment_ui/dataset
python stimulus_generation/aggregate_and_compute_vae.py --dataset-dir experiment_ui/dataset
python stimulus_generation/generate_fixed_layout.py --atomic-path experiment_ui/dataset/atomic/atomic_burst.json --output-path experiment_ui/dataset/layout_default.json
```

## Run the Browser Task

After generating the stimuli, serve the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/experiment_ui/
```

The interface builds a balanced 72-trial list from the generated stimulus files.
It logs `h_alias` for analysis but does not display VA-Entropy or `H_alias` to
participants.

## Behavioral Logs

Each example JSON file in `behavioral_logs/examples/` is a list of 72 trial
records. Records include:

- pseudonymous participant ID;
- stimulus identifiers and aggregation window;
- true event label, response, confidence, and response time;
- keyboard toggles and mouse-trajectory summaries;
- VA-Entropy and membership-change fields;
- node memberships, fixed screen coordinates, and sampled mouse path.

The example logs do not contain participant names, contact details, IP
addresses, or collection timestamps.

## Analysis Notebooks

The notebooks use the original relative-path convention:

- full authorized behavioral logs are read from `./result/`;
- generated analysis tables are written to `./analysis_out/`.

Both directories are ignored by Git. The two example logs can be copied into a
local `result/` directory to inspect the parsing workflow, but they are not
sufficient to reproduce the full statistical results.

Recommended notebook order:

1. `analysis_notebooks/analyze_results.ipynb`
2. `analysis_notebooks/5_1_overall.ipynb`
3. `analysis_notebooks/5_2_H1.ipynb`
4. `analysis_notebooks/5_3_H2.ipynb`
5. `analysis_notebooks/5_4_H3.ipynb`

## Figure Generation

After the analysis tables exist, regenerate the statistical paper figures with:

```bash
python figure_generation/plot_publication_figures.py
```

The static paper images included in this repository are stored in
`manuscript_figures/`.

## Additional Robustness Analysis

The robustness script expects the full cleaned trial table and H1 analysis tables
under `analysis_out/`:

```bash
python supplementary_analysis/robustness/run_comment6_checks.py
```

Generated robustness outputs are ignored by Git.

## Versioned and Generated Files

The repository versions source code, notebooks with cleared outputs, two example
behavioral logs, and the paper figures. The following are intentionally excluded
from version control:

- generated synthetic graph JSON files;
- generated fixed-layout and trial-list files;
- full behavioral logs and merged full-study tables;
- generated analysis tables, fitted models, and generated analysis figures;
- notebook outputs, Python caches, virtual environments, manuscript source
  files, and local build artifacts.
