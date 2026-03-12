import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.schema import (
    CampaignConfig,
    ParameterSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
    CategoricalSpec,
    SubstanceSpec,
)
from core.baybe_factory import build_campaign, validate_parameter_specs
from core.persistence import (
    ensure_campaign_dirs,
    campaign_latest_path,
    campaign_snapshot_path,
    run_plan_path,
    run_results_path,
    all_runs_path,
    save_text,
    load_text,
    append_all_runs,
    discover_next_run_idx,
)
from core.sobol_init import sobol_initial_design
from core.dedup import measured_keys, drop_measured

APP_TITLE = "Alpine-GP"

ACQ_INFO = {
    "ProbabilityOfImprovement": {
        "score": 0.18,
        "label": "PI",
        "summary": "Greedy; favors regions near the current best.",
    },
    "ExpectedImprovement": {
        "score": 0.33,
        "label": "EI",
        "summary": "Balanced default; rewards likely improvement.",
    },
    "qProbabilityOfImprovement": {
        "score": 0.25,
        "label": "qPI",
        "summary": "Batch PI; still fairly exploitative.",
    },
    "qExpectedImprovement": {
        "score": 0.42,
        "label": "qEI",
        "summary": "Batch EI; strong default for parallel experiments.",
    },
    "qNoisyExpectedImprovement": {
        "score": 0.54,
        "label": "qNEI",
        "summary": "Like qEI, but more robust when measurements are noisy.",
    },
    "UpperConfidenceBound": {
        "score": 0.68,
        "label": "UCB",
        "summary": "Explicit mean–uncertainty tradeoff.",
    },
    "qUpperConfidenceBound": {
        "score": 0.75,
        "label": "qUCB",
        "summary": "Batch UCB; beta tunes exploration strength.",
    },
    "qThompsonSampling": {
        "score": 0.88,
        "label": "qTS",
        "summary": "Posterior sampling; diverse, exploratory batches.",
    },
}

ACQ_ORDER = [
    "ProbabilityOfImprovement",
    "qProbabilityOfImprovement",
    "ExpectedImprovement",
    "qExpectedImprovement",
    "qNoisyExpectedImprovement",
    "UpperConfidenceBound",
    "qUpperConfidenceBound",
    "qThompsonSampling",
]


def _set_page() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")


def _download_button_df(label: str, df: pd.DataFrame, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv")


def _json_download_button(label: str, obj: Any, filename: str) -> None:
    st.download_button(
        label,
        data=json.dumps(obj, indent=2).encode("utf-8"),
        file_name=filename,
        mime="application/json",
    )


def _config_validation_errors(cfg: CampaignConfig) -> List[str]:
    try:
        validate_parameter_specs(cfg.parameters)
    except ValueError as exc:
        return [str(exc)]
    return []


def _default_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="default",
        objective_target="yield",
        objective_mode="maximize",
        batch_size=8,
        init_mode="sobol",
        n_init=8,
        acquisition="qExpectedImprovement",
        acquisition_kwargs={"best_f": None},
        parameters=[
            CategoricalSpec(name="solvent", values=["MeCN", "HFIP"], encoding="OHE"),
            NumericalDiscreteSpec(name="reaction_time", values=[1.0, 2.0, 3.0], unit="h"),
        ],
    )


COLORS = {
    "navy": "#0E2841",
    "teal": "#156082",
    "orange": "#E97132",
    "ltgray": "#E8E8E8",
    "midgray": "#A9B3BC",
    "text": "#0E2841",
    "muted": "#6F7D8A",
    "white": "#FFFFFF",
}


def _beta_to_x(beta: float) -> float:
    return min(0.93, max(0.50, 0.50 + 0.15 * (beta ** 0.5)))


def _render_acquisition_map(selected: str, beta: float | None = None) -> None:
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Aptos", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "figure.facecolor": COLORS["white"],
        "axes.facecolor": COLORS["white"],
    })

    fig, ax = plt.subplots(figsize=(11, 3.8))

    panel = FancyBboxPatch(
        (0.015, 0.08),
        0.97,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0,
        facecolor="#F9FAFB",
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(panel)

    y_axis = -0.03

    ax.hlines(
        y_axis,
        0.06,
        0.94,
        linewidth=3.5,
        color=COLORS["ltgray"],
        zorder=1,
        capstyle="round",
    )

    ax.scatter([0.06, 0.94], [y_axis, y_axis], s=20, color=COLORS["ltgray"], zorder=2)

    ax.text(
        0.06, -0.38, "Exploitative",
        ha="left", va="center",
        fontsize=11,
        color=COLORS["muted"],
        fontweight="medium",
    )
    ax.text(
        0.94, -0.38, "Exploratory",
        ha="right", va="center",
        fontsize=11,
        color=COLORS["muted"],
        fontweight="medium",
    )

    for name in ACQ_ORDER:
        x = ACQ_INFO[name]["score"]
        label = ACQ_INFO[name]["label"]
        is_selected = (name == selected)

        if is_selected:
            ax.scatter(
                x, y_axis,
                s=260,
                color=COLORS["orange"],
                alpha=0.20,
                edgecolor="none",
                zorder=3,
            )
            ax.scatter(
                x, y_axis,
                s=92,
                color=COLORS["navy"],
                edgecolor=COLORS["white"],
                linewidth=1.0,
                zorder=4,
            )
            ax.text(
                x, y_axis - 0.16,
                label,
                ha="center", va="center",
                fontsize=11,
                fontweight="bold",
                color=COLORS["navy"],
            )
        else:
            ax.scatter(
                x, y_axis,
                s=72,
                color=COLORS["midgray"],
                edgecolor=COLORS["white"],
                linewidth=0.8,
                alpha=0.95,
                zorder=2,
            )
            ax.text(
                x, y_axis - 0.16,
                label,
                ha="center", va="center",
                fontsize=9.5,
                color=COLORS["muted"],
            )

    if selected in {"UpperConfidenceBound", "qUpperConfidenceBound"} and beta is not None:
        bx = _beta_to_x(beta)
        ax.vlines(
            bx, y_axis + 0.04, y_axis + 0.22,
            color=COLORS["teal"],
            linewidth=1.4,
            zorder=3,
        )
        ax.scatter(
            bx, y_axis + 0.24,
            s=42,
            marker="D",
            color=COLORS["teal"],
            zorder=4,
        )
        ax.text(
            bx, y_axis + 0.32,
            rf"$\beta = {beta:.1f}$",
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS["teal"],
            fontweight="medium",
        )

    ax.text(
        0.06, 0.84,
        "Acquisition function spectrum",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        0.06, 0.73,
        "A practical guide to exploration versus exploitation in Bayesian optimization",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    if selected in ACQ_INFO:
        summary = ACQ_INFO[selected]["summary"]
        label = ACQ_INFO[selected]["label"]
        annotation = f"{label}: {summary}"

        ax.text(
            0.06, 0.18,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.2,
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=COLORS["white"],
                edgecolor=COLORS["ltgray"],
                linewidth=0.8,
            ),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.48, 0.45)
    ax.axis("off")

    plt.subplots_adjust(top=0.95, bottom=0.10, left=0.03, right=0.97)
    st.pyplot(fig, clear_figure=True)

def main() -> None:
    _set_page()
    st.title(APP_TITLE)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["1) Configure", "2) Initialize", "3) Recommend", "4) Ingest Results", "5) History"],
    )

    st.sidebar.header("Storage")
    workdir_str = st.sidebar.text_input(
        "Campaign folder (WORKDIR)",
        value=str(Path.cwd()),
        help="All plans/, results/, campaign_jsons/ will live here.",
    )
    workdir = Path(workdir_str).expanduser().resolve()

    campaign_name = st.sidebar.text_input("Campaign name", value="default")
    ensure_campaign_dirs(workdir)

    if "config" not in st.session_state:
        st.session_state["config"] = _default_config().to_dict()

    st.session_state["config"]["campaign_name"] = campaign_name

    if page.startswith("1"):
        render_config_page(workdir)
    elif page.startswith("2"):
        render_init_page(workdir)
    elif page.startswith("3"):
        render_recommend_page(workdir)
    elif page.startswith("4"):
        render_ingest_page(workdir)
    else:
        render_history_page(workdir)



def render_config_page(workdir: Path) -> None:
    st.subheader("1) Configure parameters + model settings")
    cfg = CampaignConfig.from_dict(st.session_state["config"])

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### Objective")
        cfg.objective_target = st.text_input("Target column name", value=cfg.objective_target)
        cfg.objective_mode = st.selectbox(
            "Optimize direction",
            options=["maximize", "minimize"],
            index=0 if cfg.objective_mode == "maximize" else 1,
        )

        st.markdown("### Batch")
        cfg.batch_size = int(st.number_input("Batch size", min_value=1, value=int(cfg.batch_size), step=1))

        st.markdown("### Initialization")
        cfg.init_mode = st.selectbox("Init mode", ["sobol", "existing_data"], index=0 if cfg.init_mode == "sobol" else 1)
        cfg.n_init = int(st.number_input("# init points", min_value=0, value=int(cfg.n_init), step=1))

    with colR:
        st.markdown("### Acquisition function (BayBE → BoTorch)")
        st.caption("BayBE wrappers are shown here. The plot places them on a practical exploration ↔ exploitation spectrum.")
        current_index = ACQ_ORDER.index(cfg.acquisition) if cfg.acquisition in ACQ_ORDER else ACQ_ORDER.index("qExpectedImprovement")
        cfg.acquisition = st.selectbox("Acquisition", options=ACQ_ORDER, index=current_index)

        if cfg.acquisition in {"UpperConfidenceBound", "qUpperConfidenceBound"}:
            current_beta = float((cfg.acquisition_kwargs or {}).get("beta", 2.0))
            beta = float(st.slider("UCB beta", min_value=0.1, max_value=10.0, value=current_beta, step=0.1,
                                   help="Larger beta weights uncertainty more heavily, so the optimizer explores more."))
            cfg.acquisition_kwargs = dict(cfg.acquisition_kwargs or {})
            cfg.acquisition_kwargs["beta"] = beta
        else:
            beta = None
            if cfg.acquisition_kwargs is None:
                cfg.acquisition_kwargs = {}
            if "beta" in cfg.acquisition_kwargs:
                cfg.acquisition_kwargs.pop("beta", None)

        _render_acquisition_map(cfg.acquisition, beta)

        with st.expander("Advanced acquisition kwargs"):
            st.caption("Optional JSON passed to the BayBE acquisition wrapper. Leave empty to use defaults.")
            kw_text = st.text_area("", value=json.dumps(cfg.acquisition_kwargs or {}, indent=2), height=140)
            try:
                cfg.acquisition_kwargs = json.loads(kw_text) if kw_text.strip() else {}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.divider()
    st.markdown("### Parameters")
    param_tabs = st.tabs(["Edit", "Add new", "Raw JSON"])

    with param_tabs[0]:
        if not cfg.parameters:
            st.info("No parameters yet. Add one in the 'Add new' tab.")
        else:
            for i, p in enumerate(list(cfg.parameters)):
                with st.expander(f"{i + 1}. {p.name} ({p.kind})", expanded=False):
                    cfg.parameters[i] = render_parameter_editor(p)
                    if st.button(f"Delete parameter '{p.name}'", key=f"del_{p.name}_{i}"):
                        cfg.parameters.pop(i)
                        st.session_state["config"] = cfg.to_dict()
                        st.warning(f"Deleted parameter '{p.name}'.")
                        st.rerun()

    with param_tabs[1]:
        cfg = render_add_parameter(cfg)

    with param_tabs[2]:
        st.caption("This is the persisted config. Editing here overwrites the form values.")
        raw = st.text_area("CampaignConfig JSON", value=json.dumps(cfg.to_dict(), indent=2), height=320)
        if st.button("Load JSON into editor"):
            try:
                st.session_state["config"] = json.loads(raw)
                st.success("Loaded.")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.divider()
    st.session_state["config"] = cfg.to_dict()
    validation_errors = _config_validation_errors(cfg)
    for msg in validation_errors:
        st.error(msg)
    cfg_path = workdir / "campaign_config.json"

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Save config to disk"):
            save_text(cfg_path, json.dumps(cfg.to_dict(), indent=2))
            st.success(f"Saved {cfg_path}")
    with colB:
        _json_download_button("Download config JSON", cfg.to_dict(), "campaign_config.json")



def render_parameter_editor(p: ParameterSpec) -> ParameterSpec:
    name = st.text_input("Name", value=p.name, key=f"pname_{p.name}")

    if isinstance(p, NumericalContinuousSpec):
        lo = float(st.number_input("Lower", value=float(p.lower), key=f"lo_{p.name}"))
        hi = float(st.number_input("Upper", value=float(p.upper), key=f"hi_{p.name}"))
        unit = st.text_input("Unit (optional)", value=p.unit or "", key=f"unit_{p.name}") or None
        return NumericalContinuousSpec(name=name, lower=lo, upper=hi, unit=unit)

    elif isinstance(p, NumericalDiscreteSpec):
        values_str = st.text_input("Values (comma-separated)", value=", ".join(map(str, p.values)), key=f"vals_{p.name}")
        values = [float(x.strip()) for x in values_str.split(",") if x.strip()]
        unit = st.text_input("Unit (optional)", value=p.unit or "", key=f"unit_{p.name}") or None
        return NumericalDiscreteSpec(name=name, values=values, unit=unit)

    elif isinstance(p, CategoricalSpec):
        values_str = st.text_input("Categories (comma-separated)", value=", ".join(p.values), key=f"cats_{p.name}")
        values = [x.strip() for x in values_str.split(",") if x.strip()]
        encoding = st.selectbox(
            "Encoding",
            options=["OHE", "INT"],
            index=0 if (p.encoding or "OHE") == "OHE" else 1,
            key=f"enc_{p.name}",
            help="How the surrogate sees this categorical variable.",
        )
        return CategoricalSpec(name=name, values=values, encoding=encoding)

    elif isinstance(p, SubstanceSpec):
        st.caption("Enter one SMILES per line. These will be used directly as the substance categories.")
        smiles_text = st.text_area(
            "SMILES (one per line)",
            value="\n".join(list(p.smiles)),
            key=f"smiles_{p.name}",
            height=160,
        )
        smiles = [ln.strip() for ln in smiles_text.splitlines() if ln.strip()]

        from baybe.parameters.enum import SubstanceEncoding
        enc_options = [e.name for e in SubstanceEncoding]
        default_enc = p.encoding or "MORDRED"
        enc_index = enc_options.index(default_enc) if default_enc in enc_options else 0
        encoding = st.selectbox(
            "Substance encoding",
            options=enc_options,
            index=enc_index,
            key=f"senc_{p.name}",
            help="How the surrogate encodes molecules (descriptors/fingerprints).",
        )
        decorrelate = st.selectbox(
            "Decorrelate descriptors?",
            options=["True", "False"],
            index=0 if getattr(p, "decorrelate", False) else 1,
            key=f"decor_{p.name}",
        )
        return SubstanceSpec(name=name, smiles=smiles, encoding=encoding, decorrelate=(decorrelate == "True"))

    st.warning("Unknown parameter type; leaving unchanged.")
    return p



def render_add_parameter(cfg: CampaignConfig) -> CampaignConfig:
    kind = st.selectbox(
        "Parameter type",
        options=["categorical", "numerical_discrete", "numerical_continuous", "substance"],
        key="add_kind",
    )
    pname = st.text_input("New parameter name", value="new_param", key="add_pname")

    if kind == "numerical_continuous":
        lo = float(st.number_input("Lower", value=0.0, key="add_nc_lo"))
        hi = float(st.number_input("Upper", value=1.0, key="add_nc_hi"))
        unit = st.text_input("Unit (optional)", value="", key="add_nc_unit") or None
        if st.button("Add numerical continuous", key="btn_add_num_cont"):
            cfg.parameters.append(NumericalContinuousSpec(name=pname, lower=lo, upper=hi, unit=unit))
            st.session_state["config"] = cfg.to_dict()
            st.success(f"Added {pname}")
            st.rerun()

    elif kind == "numerical_discrete":
        values_str = st.text_input("Values (comma-separated)", value="0, 1, 2", key="add_nd_vals")
        unit = st.text_input("Unit (optional)", value="", key="add_nd_unit") or None
        if st.button("Add numerical discrete", key="btn_add_num_disc"):
            values = [float(x.strip()) for x in values_str.split(",") if x.strip()]
            cfg.parameters.append(NumericalDiscreteSpec(name=pname, values=values, unit=unit))
            st.session_state["config"] = cfg.to_dict()
            st.success(f"Added {pname}")
            st.rerun()

    elif kind == "categorical":
        values_str = st.text_input("Categories (comma-separated)", value="A, B, C", key="add_cat_vals")
        encoding = st.selectbox("Encoding", options=["OHE", "INT"], index=0, key="add_cat_enc")
        if st.button("Add categorical", key="btn_add_cat"):
            values = [x.strip() for x in values_str.split(",") if x.strip()]
            cfg.parameters.append(CategoricalSpec(name=pname, values=values, encoding=encoding))
            st.session_state["config"] = cfg.to_dict()
            st.success(f"Added {pname}")
            st.rerun()

    elif kind == "substance":
        st.caption("Enter one SMILES per line. These will be used directly as categories.")
        smiles_text = st.text_area(
            "SMILES (one per line)",
            value="CCO\nc1ccccc1",
            key="add_sub_smiles",
            height=160,
        )
        smiles = [ln.strip() for ln in smiles_text.splitlines() if ln.strip()]
        from baybe.parameters.enum import SubstanceEncoding
        enc_options = [e.name for e in SubstanceEncoding]
        encoding = st.selectbox(
            "Substance encoding",
            options=enc_options,
            index=enc_options.index("MORDRED") if "MORDRED" in enc_options else 0,
            key="add_sub_enc",
        )
        decorrelate = st.selectbox(
            "Decorrelate descriptors?",
            ["True", "False"],
            index=0,
            key="add_sub_decor",
        )
        if st.button("Add substance", key="btn_add_sub"):
            unique_smiles = list(dict.fromkeys(smiles))
            if len(unique_smiles) < 2:
                st.error("Please enter at least two unique SMILES for a substance parameter.")
            else:
                cfg.parameters.append(
                    SubstanceSpec(
                        name=pname,
                        smiles=unique_smiles,
                        encoding=encoding,
                        decorrelate=(decorrelate == "True"),
                    )
                )
                st.session_state["config"] = cfg.to_dict()
                st.success(f"Added {pname}")
                st.rerun()

    return cfg



def render_init_page(workdir: Path) -> None:
    st.subheader("2) Initialize campaign")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    validation_errors = _config_validation_errors(cfg)
    if validation_errors:
        for msg in validation_errors:
            st.error(msg)
        st.info("Fix the campaign configuration on the Configure page before initializing.")
        return
    st.info(
        "Initialization writes an initial plan (run0.csv) and persists a BayBE campaign JSON. "
        "Choose Sobol init for a cold start, or ingest an existing CSV for a warm start."
    )

    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if latest.exists():
        st.warning(f"Existing campaign state found: {latest}")
        if st.button("Archive existing state (snapshot + clear latest)"):
            snap = campaign_snapshot_path(workdir, cfg.campaign_name, tag="archived_before_reinit")
            save_text(snap, load_text(latest))
            latest.unlink()
            st.success(f"Archived to {snap} and removed latest.")

    if cfg.init_mode == "sobol":
        seed = int(st.number_input("Sobol seed", min_value=0, value=0, step=1))
        if st.button("Generate run0.csv via Sobol"):
            plan0 = sobol_initial_design(cfg.parameters, n=cfg.n_init, seed=seed)
            out_path = run_plan_path(workdir, run_idx=0)
            plan0.to_csv(out_path, index=False)
            campaign = build_campaign(cfg)
            save_text(latest, campaign.to_json())
            st.success(f"Wrote {out_path} and initialized campaign JSON at {latest}")
            _download_button_df("Download run0.csv", plan0, "run0.csv")

    else:
        st.caption("Upload a CSV with columns = parameters + target. The first N rows are ingested as initial data.")
        up = st.file_uploader("Upload initial results CSV", type=["csv"], key="initcsv")
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head(20), use_container_width=True)
            if st.button("Initialize from this CSV"):
                campaign = build_campaign(cfg)
                needed = [p.name for p in cfg.parameters] + [cfg.objective_target]
                missing = [c for c in needed if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    st.stop()
                df0 = df[needed].iloc[: cfg.n_init].copy()
                campaign.add_measurements(df0)
                save_text(latest, campaign.to_json())
                init_path = workdir / "results" / "initial_data_results.csv"
                df0.to_csv(init_path, index=False)
                append_all_runs(workdir, df0, run_idx=-1)
                st.success(f"Initialized campaign from {len(df0)} rows. Saved {latest} and {init_path}.")



def render_recommend_page(workdir: Path) -> None:
    st.subheader("3) Recommend next batch")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if not latest.exists():
        st.error("No campaign JSON found. Go to 'Initialize' first.")
        return

    from baybe.campaign import Campaign

    campaign = Campaign.from_json(load_text(latest))
    next_run = discover_next_run_idx(workdir)
    st.markdown(f"Next run index: **{next_run}**")
    st.caption("This writes plans/runN.csv and snapshots the campaign state after recommendation.")

    if st.button("Recommend batch"):
        param_cols = [p.name for p in cfg.parameters]
        keys = measured_keys(workdir, param_cols)
        needed = int(cfg.batch_size)
        collected: List[pd.DataFrame] = []
        attempts = 0
        max_attempts = 6

        while needed > 0 and attempts < max_attempts:
            attempts += 1
            rec = campaign.recommend(batch_size=needed)
            rec2 = drop_measured(rec, keys, param_cols)
            if rec2.empty:
                continue
            collected.append(rec2)
            for row in rec2[[c for c in param_cols if c in rec2.columns]].itertuples(index=False, name=None):
                keys.add(tuple(row))
            needed = int(cfg.batch_size) - int(pd.concat(collected, ignore_index=True).shape[0])

        if not collected:
            st.error(
                "Could not generate any new recommendations. The design space may be exhausted or every point has already been measured."
            )
            st.stop()

        rec_final = pd.concat(collected, ignore_index=True).head(int(cfg.batch_size))
        if rec_final.shape[0] < int(cfg.batch_size):
            st.warning(
                f"Only generated {rec_final.shape[0]} unique recommendations (requested {int(cfg.batch_size)})."
            )

        out_path = run_plan_path(workdir, run_idx=next_run)
        rec_final.to_csv(out_path, index=False)
        save_text(latest, campaign.to_json())
        snap = campaign_snapshot_path(workdir, cfg.campaign_name, tag=f"after_recommend_run{next_run}")
        save_text(snap, campaign.to_json())
        st.success(f"Saved plan -> {out_path}")
        _download_button_df("Download plan CSV", rec_final, f"run{next_run}.csv")
        st.dataframe(rec_final, use_container_width=True)



def render_ingest_page(workdir: Path) -> None:
    st.subheader("4) Ingest results + update campaign")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if not latest.exists():
        st.error("No campaign JSON found. Go to 'Initialize' first.")
        return

    from baybe.campaign import Campaign

    campaign = Campaign.from_json(load_text(latest))
    st.caption(
        "Upload a results CSV (typically a copy of the plan CSV with an extra target column), or point the app to the on-disk results file."
    )
    run_idx = int(st.number_input("Run index for this results file", min_value=0, value=0, step=1))
    up = st.file_uploader("Upload results CSV", type=["csv"], key="resultscsv")
    if up is None:
        st.info("Or place a file at results/runN_results.csv and use the disk-ingest button.")

    colA, colB = st.columns([1, 1])
    with colA:
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head(20), use_container_width=True)
            if st.button("Ingest uploaded results"):
                _ingest_df_and_persist(workdir, cfg, campaign, df, run_idx)
    with colB:
        disk_path = run_results_path(workdir, run_idx)
        st.code(str(disk_path))
        if st.button("Ingest results from disk path"):
            if not disk_path.exists():
                st.error(f"Missing: {disk_path}")
                st.stop()
            df = pd.read_csv(disk_path)
            _ingest_df_and_persist(workdir, cfg, campaign, df, run_idx)



def _ingest_df_and_persist(workdir: Path, cfg: CampaignConfig, campaign, df: pd.DataFrame, run_idx: int) -> None:
    param_cols = [p.name for p in cfg.parameters]
    target_col = cfg.objective_target
    missing = [c for c in (param_cols + [target_col]) if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    df_use = df[param_cols + [target_col]].copy()
    out_path = run_results_path(workdir, run_idx)
    df_use.to_csv(out_path, index=False)
    append_all_runs(workdir, df_use, run_idx=run_idx)
    campaign.add_measurements(df_use)

    latest = campaign_latest_path(workdir, cfg.campaign_name)
    save_text(latest, campaign.to_json())
    snap = campaign_snapshot_path(workdir, cfg.campaign_name, tag=f"after_ingest_run{run_idx}")
    save_text(snap, campaign.to_json())
    st.success(f"Ingested {len(df_use)} rows. Saved {out_path} and updated {latest}.")



def render_history_page(workdir: Path) -> None:
    st.subheader("5) History + saved artifacts")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if latest.exists():
        st.markdown("### Latest campaign JSON")
        st.code(str(latest))
        if st.button("Show latest campaign JSON"):
            st.json(json.loads(load_text(latest)))
    else:
        st.info("No latest campaign JSON found yet.")

    aruns = all_runs_path(workdir)
    if aruns.exists():
        st.markdown("### all_runs.csv")
        df = pd.read_csv(aruns)
        st.dataframe(df, use_container_width=True)
        _download_button_df("Download all_runs.csv", df, "all_runs.csv")
    else:
        st.info("No all_runs.csv yet.")

    st.markdown("### File listing")
    files = []
    for sub in ["plans", "results", "campaign_jsons"]:
        d = workdir / sub
        if d.exists():
            for p in sorted(d.glob("*")):
                files.append({"folder": sub, "name": p.name, "path": str(p)})
    if files:
        st.dataframe(pd.DataFrame(files), use_container_width=True)
    else:
        st.info("No files created yet.")


if __name__ == "__main__":
    main()
