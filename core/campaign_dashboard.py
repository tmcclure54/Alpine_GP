from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BOOKKEEPING_COLUMNS = {
    "trial_index",
    "yield",
    "objective",
    "response",
    "status",
    "round_index",
    "batch_index",
    "run_index",
    "sem",
    "stderr",
    "std",
    "mean",
    "campaign",
    "timestamp",
    "created_at",
    "updated_at",
    "completed_at",
    "notes",
}


def infer_parameter_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in BOOKKEEPING_COLUMNS]


def split_parameter_types(df: pd.DataFrame, parameter_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    categorical: List[str] = []
    numerical: List[str] = []
    for col in parameter_columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numerical.append(col)
        else:
            categorical.append(col)
    return categorical, numerical


def _candidate_objective_columns(df: pd.DataFrame, parameter_columns: Sequence[str]) -> List[str]:
    preferred = [c for c in ["yield", "objective", "response"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    other_numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in set(parameter_columns) and c not in set(preferred)
    ]
    return preferred + other_numeric


def get_completed_trials(df: pd.DataFrame, objective_col: str, completed_only: bool) -> pd.DataFrame:
    out = df.copy()
    out = out[out[objective_col].notna()].copy()
    if completed_only and "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "completed"].copy()
    return out


def compute_summary_stats(df_all: pd.DataFrame, df_completed: pd.DataFrame, objective_col: str, maximize: bool) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total_trials": int(len(df_all)),
        "completed_trials": int(len(df_completed)),
        "best_objective": np.nan,
        "mean_objective": np.nan,
        "std_objective": np.nan,
        "best_trial_index": None,
    }
    if df_completed.empty:
        return stats

    if maximize:
        best_idx = df_completed[objective_col].idxmax()
        stats["best_objective"] = float(df_completed.loc[best_idx, objective_col])
    else:
        best_idx = df_completed[objective_col].idxmin()
        stats["best_objective"] = float(df_completed.loc[best_idx, objective_col])

    stats["mean_objective"] = float(df_completed[objective_col].mean())
    stats["std_objective"] = float(df_completed[objective_col].std(ddof=1)) if len(df_completed) > 1 else 0.0
    if "trial_index" in df_completed.columns:
        stats["best_trial_index"] = int(df_completed.loc[best_idx, "trial_index"])
    else:
        stats["best_trial_index"] = int(best_idx)
    return stats


def save_figure(fig: plt.Figure, filename: str, campaign_dir: Optional[Path]) -> Optional[Path]:
    if campaign_dir is None:
        return None
    analysis_dir = campaign_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def make_progress_plot(df: pd.DataFrame, objective_col: str, maximize: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = df["trial_index"] if "trial_index" in df.columns else np.arange(len(df))
    y = df[objective_col]

    if "sem" in df.columns:
        sem = pd.to_numeric(df["sem"], errors="coerce")
        ax.errorbar(x, y, yerr=sem, fmt="o", alpha=0.75, label="Observed objective")
    else:
        ax.scatter(x, y, alpha=0.75, label="Observed objective")

    best_so_far = y.cummax() if maximize else y.cummin()
    ax.plot(x, best_so_far, color="tab:red", linewidth=2, label="Best-so-far")
    ax.set_xlabel("Trial index")
    ax.set_ylabel(objective_col)
    ax.set_title("Campaign Progress")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def make_distribution_plot(df: pd.DataFrame, objective_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = df[objective_col].dropna()
    if len(vals) < 10:
        ax.scatter(np.arange(len(vals)), vals, alpha=0.85)
        ax.set_xlabel("Completed trial order")
    else:
        bins = min(30, max(5, int(np.sqrt(len(vals)))))
        ax.hist(vals, bins=bins, alpha=0.8, edgecolor="black")
        ax.set_xlabel(objective_col)
    ax.set_ylabel("Count")
    ax.set_title("Objective Distribution")
    ax.grid(alpha=0.25)
    return fig


def make_status_plot(df: pd.DataFrame) -> Optional[plt.Figure]:
    if "status" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["status"].fillna("unknown").astype(str).str.lower().value_counts()
    ax.bar(counts.index, counts.values, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    ax.set_title("Status Distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    return fig


def make_categorical_param_plot(df: pd.DataFrame, parameter: str, objective_col: str) -> Optional[Tuple[plt.Figure, pd.DataFrame]]:
    series = df[parameter]
    if series.dropna().nunique() == 0:
        return None

    grouped = (
        df.groupby(parameter, dropna=False)[objective_col]
        .agg(["count", "mean", "std", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    x_labels = grouped[parameter].astype(str)
    ax.bar(x_labels, grouped["mean"], color="tab:green", alpha=0.8)
    for i, row in grouped.reset_index(drop=True).iterrows():
        ax.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(parameter)
    ax.set_ylabel(f"Mean {objective_col}")
    ax.set_title(f"Categorical Analysis: {parameter}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    return fig, grouped


def make_numerical_param_plot(df: pd.DataFrame, parameter: str, objective_col: str) -> Optional[plt.Figure]:
    use = df[[parameter, objective_col]].dropna().copy()
    if use.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(use[parameter], use[objective_col], alpha=0.7, label="Observed")

    if use[parameter].nunique() > 1:
        bins = min(10, use[parameter].nunique())
        use["_bin"] = pd.cut(use[parameter], bins=bins, duplicates="drop")
        binned = use.groupby("_bin", observed=True)[objective_col].mean().reset_index()
        binned["center"] = binned["_bin"].apply(lambda x: x.mid)
        ax.plot(binned["center"], binned[objective_col], color="tab:red", linewidth=2, label="Binned mean")

    ax.set_xlabel(parameter)
    ax.set_ylabel(objective_col)
    ax.set_title(f"Numerical Analysis: {parameter}")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def compute_pareto_mask(
    df: pd.DataFrame,
    objective_cols: Sequence[str],
    maximize_flags: Sequence[bool],
) -> np.ndarray:
    """Return a bool mask marking rows on the (first) Pareto front.

    A row is on the Pareto front iff no other row dominates it: another row
    must be at least as good on every objective and strictly better on at
    least one objective.
    """
    if len(objective_cols) != len(maximize_flags):
        raise ValueError("objective_cols and maximize_flags must have equal length.")

    sub = df[list(objective_cols)].to_numpy(dtype=float, copy=True)
    flags = np.asarray(maximize_flags, dtype=bool)
    sub[:, ~flags] = -sub[:, ~flags]

    n = sub.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i?
            if np.all(sub[j] >= sub[i]) and np.any(sub[j] > sub[i]):
                mask[i] = False
                break
    return mask


def make_pareto_plot(
    df: pd.DataFrame,
    objective_x: str,
    objective_y: str,
    maximize_x: bool,
    maximize_y: bool,
    pareto_mask: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = df[objective_x].to_numpy(dtype=float)
    y = df[objective_y].to_numpy(dtype=float)
    ax.scatter(x[~pareto_mask], y[~pareto_mask], alpha=0.55, color="tab:gray", label="Dominated")
    ax.scatter(x[pareto_mask], y[pareto_mask], alpha=0.95, color="tab:red", s=70, label="Pareto front")

    # Draw the Pareto frontier as a step line for visual clarity.
    if pareto_mask.sum() >= 2:
        order = np.argsort(x[pareto_mask] * (1 if maximize_x else -1))
        px = x[pareto_mask][order]
        py = y[pareto_mask][order]
        ax.plot(px, py, color="tab:red", linewidth=1.2, alpha=0.7)

    ax.set_xlabel(f"{objective_x} ({'max' if maximize_x else 'min'})")
    ax.set_ylabel(f"{objective_y} ({'max' if maximize_y else 'min'})")
    ax.set_title("Pareto Front")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _to_summary_json(stats: Dict[str, Any], objective_col: str, maximize: bool, selected_params: Sequence[str]) -> str:
    payload = {
        "objective_column": objective_col,
        "maximize": maximize,
        "selected_parameters": list(selected_params),
        "summary": stats,
    }
    return json.dumps(payload, indent=2)


def render_campaign_dashboard(
    df_trials: pd.DataFrame,
    campaign_dir: Optional[Path] = None,
    target_specs: Optional[Sequence[Dict[str, str]]] = None,
) -> None:
    """Render dashboard. If target_specs (list of {name, mode} dicts) is
    provided and contains >= 2 entries, also render a Pareto front section."""
    st.subheader("Campaign Dashboard")
    if df_trials is None or df_trials.empty:
        st.info("No campaign trials available yet. Ingest results first.")
        return

    df = df_trials.copy()

    if "status" in df.columns:
        df["status"] = df["status"].fillna("unknown").astype(str).str.lower().str.strip()
    if "trial_index" in df.columns:
        df = df.sort_values("trial_index").reset_index(drop=True)

    # Treat configured target columns as bookkeeping for parameter inference,
    # so they are not mistaken for design parameters.
    bookkeeping_extra = set()
    if target_specs:
        bookkeeping_extra.update(t.get("name") for t in target_specs if t.get("name"))
    parameter_columns = [
        c for c in infer_parameter_columns(df) if c not in bookkeeping_extra
    ]
    candidate_objective_columns = _candidate_objective_columns(df, parameter_columns)
    if target_specs:
        configured_names = [t["name"] for t in target_specs if t.get("name") in df.columns]
        # Surface configured targets first, preserving order.
        candidate_objective_columns = configured_names + [
            c for c in candidate_objective_columns if c not in configured_names
        ]
    if not candidate_objective_columns:
        st.warning("No numeric objective-like columns found in the trials table.")
        st.dataframe(df, use_container_width=True)
        return

    objective_col = st.selectbox("Objective column", candidate_objective_columns)
    default_max = True
    if target_specs:
        for t in target_specs:
            if t.get("name") == objective_col:
                default_max = (t.get("mode") != "minimize")
                break
    maximize = st.toggle("Maximize objective", value=default_max)
    completed_only = st.checkbox("Use completed trials only", value=True)
    selected_params = st.multiselect("Parameters to analyze", parameter_columns, default=parameter_columns)

    df_completed = get_completed_trials(df, objective_col=objective_col, completed_only=completed_only)
    if df_completed.empty:
        st.warning("No completed trials available after filtering.")
        st.dataframe(df, use_container_width=True)
        return

    df_completed = df_completed.copy()
    df_completed["best_so_far"] = (
        df_completed[objective_col].cummax() if maximize else df_completed[objective_col].cummin()
    )

    stats = compute_summary_stats(df, df_completed, objective_col=objective_col, maximize=maximize)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total trials", _format_metric(stats["total_trials"]))
    m2.metric("Completed trials", _format_metric(stats["completed_trials"]))
    m3.metric("Best objective", _format_metric(stats["best_objective"]))
    m4.metric("Mean objective", _format_metric(stats["mean_objective"]))
    m5.metric("Std objective", _format_metric(stats["std_objective"]))
    m6.metric("Best trial index", _format_metric(stats["best_trial_index"]))

    progress_fig = make_progress_plot(df_completed, objective_col=objective_col, maximize=maximize)
    st.pyplot(progress_fig)
    save_figure(progress_fig, "progress_plot.png", campaign_dir)

    dist_fig = make_distribution_plot(df_completed, objective_col=objective_col)
    st.pyplot(dist_fig)
    save_figure(dist_fig, "objective_distribution.png", campaign_dir)

    status_fig = make_status_plot(df)
    if status_fig is None:
        st.info("No status column found; skipping status distribution.")
    else:
        st.pyplot(status_fig)
        save_figure(status_fig, "status_counts.png", campaign_dir)

    st.markdown("### Top Trials")
    top_n = int(st.number_input("Top N", min_value=1, max_value=max(1, len(df_completed)), value=min(10, len(df_completed))))
    display_cols: List[str] = [c for c in ["trial_index", objective_col, "status"] if c in df_completed.columns]
    display_cols.extend([c for c in selected_params if c in df_completed.columns and c not in display_cols])
    top_df = df_completed.sort_values(objective_col, ascending=not maximize).head(top_n)
    st.dataframe(top_df[display_cols], use_container_width=True)

    categorical_all, numerical_all = split_parameter_types(df_completed, selected_params)

    st.markdown("### Categorical Parameter Analysis")
    if not categorical_all:
        st.info("No categorical parameters available for analysis.")
    for p in categorical_all:
        out = make_categorical_param_plot(df_completed, parameter=p, objective_col=objective_col)
        if out is None:
            st.info(f"Skipping '{p}' because it has no usable category values.")
            continue
        fig, table = out
        st.pyplot(fig)
        st.dataframe(table, use_container_width=True)
        save_figure(fig, f"parameter_{p}_categorical.png", campaign_dir)

    st.markdown("### Numerical Parameter Analysis")
    if not numerical_all:
        st.info("No numerical parameters available for analysis.")
    for p in numerical_all:
        fig = make_numerical_param_plot(df_completed, parameter=p, objective_col=objective_col)
        if fig is None:
            st.info(f"Skipping '{p}' because all values are missing.")
            continue
        st.pyplot(fig)
        save_figure(fig, f"parameter_{p}_numerical.png", campaign_dir)

    st.markdown("### Round / Batch Analysis")
    grouping_col = next((c for c in ["round_index", "batch_index", "run_index"] if c in df_completed.columns), None)
    if grouping_col is None:
        st.info("No round/batch/run index column found.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        grouped = [g[objective_col].dropna().values for _, g in df_completed.groupby(grouping_col)]
        labels = [str(k) for k, _ in df_completed.groupby(grouping_col)]
        if len(grouped) > 1:
            ax.boxplot(grouped, labels=labels)
        else:
            ax.scatter(np.repeat(labels[0], len(grouped[0])), grouped[0], alpha=0.8)
        ax.set_xlabel(grouping_col)
        ax.set_ylabel(objective_col)
        ax.set_title(f"Objective by {grouping_col}")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        save_figure(fig, f"objective_by_{grouping_col}.png", campaign_dir)

    if target_specs and len([t for t in target_specs if t.get("name") in df_completed.columns]) >= 2:
        _render_pareto_section(df_completed, target_specs, campaign_dir)

    st.markdown("### Raw Data")
    st.dataframe(df_completed, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=df_completed.to_csv(index=False).encode("utf-8"),
        file_name="campaign_dashboard_data.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download JSON summary",
        data=_to_summary_json(stats, objective_col=objective_col, maximize=maximize, selected_params=selected_params),
        file_name="campaign_dashboard_summary.json",
        mime="application/json",
    )


def _render_pareto_section(
    df_completed: pd.DataFrame,
    target_specs: Sequence[Dict[str, str]],
    campaign_dir: Optional[Path],
) -> None:
    available = [t for t in target_specs if t.get("name") in df_completed.columns]
    if len(available) < 2:
        return

    st.markdown("### Pareto Front (multi-objective)")
    st.caption(
        "Computed over completed trials only. A point is on the Pareto front if "
        "no other trial is at least as good on every configured target and strictly "
        "better on at least one."
    )

    name_to_mode = {t["name"]: t.get("mode", "maximize") for t in available}
    target_names = list(name_to_mode.keys())

    objective_cols = list(target_names)
    maximize_flags = [name_to_mode[n] != "minimize" for n in objective_cols]
    valid_rows = df_completed[objective_cols].apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )
    keep = ~valid_rows.isna().any(axis=1)
    df_pareto = df_completed.loc[keep].copy()
    if df_pareto.empty:
        st.info("No completed trials with values for every configured target.")
        return

    df_pareto[objective_cols] = valid_rows.loc[keep]
    pareto_mask = compute_pareto_mask(df_pareto, objective_cols, maximize_flags)
    df_pareto["pareto_optimal"] = pareto_mask

    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X axis target", target_names, index=0, key="pareto_x")
    with col_y:
        default_y = 1 if len(target_names) > 1 else 0
        y_axis = st.selectbox("Y axis target", target_names, index=default_y, key="pareto_y")

    if x_axis == y_axis:
        st.info("Pick two different targets to plot a Pareto scatter.")
    else:
        fig = make_pareto_plot(
            df_pareto,
            objective_x=x_axis,
            objective_y=y_axis,
            maximize_x=name_to_mode[x_axis] != "minimize",
            maximize_y=name_to_mode[y_axis] != "minimize",
            pareto_mask=pareto_mask,
        )
        st.pyplot(fig)
        save_figure(fig, f"pareto_{x_axis}_vs_{y_axis}.png", campaign_dir)

    st.markdown("#### Pareto-optimal trials")
    pareto_rows = df_pareto[pareto_mask]
    show_cols: List[str] = []
    if "trial_index" in pareto_rows.columns:
        show_cols.append("trial_index")
    show_cols.extend(target_names)
    if "status" in pareto_rows.columns:
        show_cols.append("status")
    st.dataframe(pareto_rows[show_cols].reset_index(drop=True), use_container_width=True)
    st.caption(f"{int(pareto_mask.sum())} of {len(df_pareto)} completed trials are Pareto-optimal.")
