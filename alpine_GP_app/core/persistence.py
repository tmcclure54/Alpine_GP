from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def ensure_campaign_dirs(workdir: Path) -> None:
    for d in [workdir / "plans", workdir / "results", workdir / "campaign_jsons", workdir / "plots"]:
        d.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def campaign_latest_path(workdir: Path, campaign_name: str) -> Path:
    return workdir / "campaign_jsons" / f"{campaign_name}_latest.json"


def campaign_snapshot_path(workdir: Path, campaign_name: str, tag: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)
    return workdir / "campaign_jsons" / f"{campaign_name}_{safe}.json"


def run_plan_path(workdir: Path, run_idx: int) -> Path:
    return workdir / "plans" / f"run{run_idx}.csv"


def run_results_path(workdir: Path, run_idx: int) -> Path:
    return workdir / "results" / f"run{run_idx}_results.csv"


def all_runs_path(workdir: Path) -> Path:
    return workdir / "results" / "all_runs.csv"


def append_all_runs(workdir: Path, df: pd.DataFrame, run_idx: int) -> None:
    out = df.copy()
    out.insert(0, "run_idx", run_idx)

    path = all_runs_path(workdir)
    if path.exists():
        prev = pd.read_csv(path)
        merged = pd.concat([prev, out], ignore_index=True)
    else:
        merged = out

    merged.to_csv(path, index=False)


def discover_next_run_idx(workdir: Path) -> int:
    # next plan index based on existing plan files
    plans = list((workdir / "plans").glob("run*.csv"))
    if not plans:
        return 0
    pat = re.compile(r"run(\d+)\.csv$")
    idxs = []
    for p in plans:
        m = pat.search(p.name)
        if m:
            idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0
