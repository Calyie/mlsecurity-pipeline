"""A results log: one CSV row per experiment run, so results are never only in a notebook."""
import datetime
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                  # the repository folder
RESULTS_FILE = str(ROOT / "experiments" / "results.csv")   # the same file from any working directory


def log_result(experiment, **fields):
    """Append one row for `experiment` with the given fields (any numbers or text).

    Example: log_result("label_flipping_blobs", poison_rate=0.1, accuracy=0.94)
    """
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "experiment": experiment}
    row.update(fields)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    existing = pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(RESULTS_FILE, index=False)
    return row


def load_results(experiment=None):
    """All logged rows, or only those of one experiment."""
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_FILE)
    if experiment:
        df = df[df["experiment"] == experiment].reset_index(drop=True)
    return df
