"""Local integrity checks for the reproducibility release candidate."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARTICIPANTS = 31
EXPECTED_TRIALS_PER_PARTICIPANT = 72
EXPECTED_TOTAL_TRIALS = 2232
EXPECTED_H2_TRIALS = 2194
EXPECTED_W = {1, 2, 5, 10, 20, 50}
EXPECTED_EVENTS = {"Continue", "SizeChange", "Merge", "Split"}
RT_MIN_MS = 200
RT_MAX_MS = 60000

SENSITIVE_KEYS = {
    "name",
    "email",
    "e-mail",
    "ip",
    "ip_address",
    "timestamp",
    "browser_fingerprint",
    "local_path",
    "username",
    "user_name",
    "os_username",
    "download_path",
    "raw_filename",
    "subject_id",
}

README_FORBIDDEN_TERMS = {
    "review",
    "reviewer",
    "requested",
    "sensitivity",
    "robustness",
    "comment",
}

REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "stimulus_generation/membership_schedules.py",
    "stimulus_generation/generate_dynamic_graphs.py",
    "stimulus_generation/aggregate_and_compute_vae.py",
    "stimulus_generation/generate_fixed_layout.py",
    "stimulus_generation/generate_stimuli.py",
    "stimuli_release/trial_list_72_seed20260227.json",
    "stimuli_release/layout_default.json",
    "stimuli_release/window_metadata_with_va_entropy.csv",
    "experiment_ui/index.html",
    "analysis_scripts/run_main_results.py",
    "figure_generation/plot_publication_figures.py",
    "scripts/check_release_integrity.py",
]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(walk_keys(child))
    return keys


def check_required_paths(errors: list[str]) -> None:
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            fail(f"Missing required path: {relative}", errors)
    for window in sorted(EXPECTED_W):
        path = ROOT / "stimuli_release" / f"aggregated_snapshots_W{window}.json"
        if not path.exists():
            fail(f"Missing aggregated snapshot file: {path.relative_to(ROOT)}", errors)


def load_anonymized_logs(errors: list[str]) -> list[dict[str, Any]]:
    log_dir = ROOT / "behavioral_logs" / "anonymized"
    if not log_dir.exists():
        fail("Missing behavioral_logs/anonymized/", errors)
        return []

    files = sorted(log_dir.glob("P*.json"))
    if len(files) != EXPECTED_PARTICIPANTS:
        fail(f"Expected {EXPECTED_PARTICIPANTS} participant logs, found {len(files)}", errors)

    all_rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if len(rows) != EXPECTED_TRIALS_PER_PARTICIPANT:
            fail(f"{path.name}: expected 72 trials, found {len(rows)}", errors)
        all_rows.extend(rows)

        participant_id = path.stem
        for row in rows:
            if row.get("participant_id") != participant_id:
                fail(f"{path.name}: participant_id mismatch", errors)
                break

        keys = {key.lower() for key in walk_keys(rows)}
        leaked_keys = sorted(keys & SENSITIVE_KEYS)
        if leaked_keys:
            fail(f"{path.name}: sensitive field names present: {leaked_keys}", errors)

        text = json.dumps(rows, ensure_ascii=False)
        if re.search(r"\bS_\d+\b", text):
            fail(f"{path.name}: raw subject identifier pattern found", errors)
        if re.search(r"result_S_\d+", text):
            fail(f"{path.name}: raw result filename pattern found", errors)
        if re.search(r"[A-Za-z]:\\", text):
            fail(f"{path.name}: local Windows path pattern found", errors)

    return all_rows


def check_counts(rows: list[dict[str, Any]], errors: list[str]) -> None:
    if len(rows) != EXPECTED_TOTAL_TRIALS:
        fail(f"Expected {EXPECTED_TOTAL_TRIALS} total trials, found {len(rows)}", errors)

    participants = {row.get("participant_id") for row in rows}
    if len(participants) != EXPECTED_PARTICIPANTS:
        fail(f"Expected {EXPECTED_PARTICIPANTS} participant IDs, found {len(participants)}", errors)

    w_values = {int(row["W"]) for row in rows}
    if w_values != EXPECTED_W:
        fail(f"Unexpected W values: {sorted(w_values)}", errors)

    events = {row["event_type"] for row in rows}
    if events != EXPECTED_EVENTS:
        fail(f"Unexpected event categories: {sorted(events)}", errors)

    h2_count = sum(RT_MIN_MS <= int(row["rt_ms"]) <= RT_MAX_MS for row in rows)
    if h2_count != EXPECTED_H2_TRIALS:
        fail(f"Expected H2 RT-trimmed count {EXPECTED_H2_TRIALS}, found {h2_count}", errors)

    per_participant = Counter(row["participant_id"] for row in rows)
    bad_counts = {
        participant: count
        for participant, count in per_participant.items()
        if count != EXPECTED_TRIALS_PER_PARTICIPANT
    }
    if bad_counts:
        fail(f"Participant trial-count mismatch: {bad_counts}", errors)


def check_trial_list(errors: list[str]) -> None:
    path = ROOT / "stimuli_release" / "trial_list_72_seed20260227.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    trials = data.get("trials", [])
    if len(trials) != EXPECTED_TRIALS_PER_PARTICIPANT:
        fail(f"Trial list should contain 72 trials, found {len(trials)}", errors)
    events = {trial.get("event_type") for trial in trials}
    if events != EXPECTED_EVENTS:
        fail(f"Trial list event categories mismatch: {sorted(events)}", errors)
    w_values = {int(trial.get("W")) for trial in trials}
    if w_values != EXPECTED_W:
        fail(f"Trial list W values mismatch: {sorted(w_values)}", errors)


def check_readme(errors: list[str]) -> None:
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    for term in sorted(README_FORBIDDEN_TERMS):
        if re.search(rf"\b{re.escape(term)}\b", lower):
            fail(f"README contains forbidden term: {term}", errors)

    for relative in REQUIRED_PATHS:
        if relative == "README.md":
            continue
        if relative not in text:
            fail(f"README does not mention required path: {relative}", errors)


def main() -> int:
    errors: list[str] = []
    check_required_paths(errors)
    rows = load_anonymized_logs(errors)
    if rows:
        check_counts(rows, errors)
    check_trial_list(errors)
    check_readme(errors)

    if errors:
        print("Release integrity check: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Release integrity check: PASSED")
    print(f"participants={EXPECTED_PARTICIPANTS}")
    print(f"formal_trials_per_participant={EXPECTED_TRIALS_PER_PARTICIPANT}")
    print(f"total_formal_trials={EXPECTED_TOTAL_TRIALS}")
    print(f"h2_rt_trimmed_trials={EXPECTED_H2_TRIALS}")
    print(f"W_values={sorted(EXPECTED_W)}")
    print(f"event_categories={sorted(EXPECTED_EVENTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
