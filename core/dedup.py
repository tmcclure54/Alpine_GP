from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

from .persistence import all_runs_path


def measured_keys(workdir: Path, param_cols: List[str]) -> Set[Tuple]:
    """Return a set of tuples representing measured parameter assignments.

    Uses results/all_runs.csv if present. Ignores rows with missing parameter values.
    """
    path = all_runs_path(workdir)
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    # Some rows may include yield/target and run_idx; we only care about param cols
    keep = [c for c in param_cols if c in df.columns]
    if not keep:
        return set()

    sub = df[keep].dropna(how="any")
    return set(map(tuple, sub.itertuples(index=False, name=None)))


def drop_measured(df_rec: pd.DataFrame, keys: Set[Tuple], param_cols: List[str]) -> pd.DataFrame:
    """Drop recommendations that match previously measured points."""
    keep = [c for c in param_cols if c in df_rec.columns]
    if not keep or not keys:
        return df_rec

    mask = []
    for row in df_rec[keep].itertuples(index=False, name=None):
        mask.append(tuple(row) not in keys)
    return df_rec.loc[mask].reset_index(drop=True)
