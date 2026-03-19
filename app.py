import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from core.baybe_factory import (
    acquisition_supports_beta,
    default_acquisition_name,
    supported_acquisition_names,
    supported_substance_encodings,
    validate_campaign_config,
    validate_config_payload,
)
from core.campaign_engine import create_campaign_engine, extract_saved_campaign_metadata, load_campaign_engine
from core.persistence import (
    ensure_campaign_dirs,
    campaign_latest_path,
    campaign_snapshot_path,
    campaign_config_path,
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
from core.campaign_dashboard import render_campaign_dashboard

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
def _normalize_smiles_input(lines: List[str]) -> List[str]:
    normalized: List[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if " #" in cleaned:
            cleaned = cleaned.split(" #", 1)[0].rstrip()
        cleaned = cleaned.rstrip(",").strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
        if cleaned:
            normalized.append(cleaned)
    return list(dict.fromkeys(normalized))


def _config_validation_errors(cfg: CampaignConfig) -> List[str]:
    return validate_campaign_config(cfg)
def _extract_campaign_metadata(path: Path) -> Dict[str, Any]:
    return extract_saved_campaign_metadata(path)


def _discover_campaigns(workdir: Path) -> List[Dict[str, Any]]:
    campaign_dir = workdir / "campaign_jsons"
    latest_files = sorted(campaign_dir.glob("*_latest.json"))
    files = latest_files if latest_files else sorted(campaign_dir.glob("*.json"))
    out: List[Dict[str, Any]] = []
    for path in files:
        try:
            out.append(_extract_campaign_metadata(path))
        except Exception:
            continue
    return out


def _build_specs_from_metadata(meta: Dict[str, Any]) -> List[ParameterSpec]:
    specs: List[ParameterSpec] = []
    for p in meta.get("parameters", []):
        p_type = p.get("type", "")
        name = p.get("name")
        values = p.get("values")
        if not name:
            continue
        if p_type == "CategoricalParameter" and isinstance(values, list):
            specs.append(CategoricalSpec(name=name, values=[str(v) for v in values], encoding="OHE"))
        elif p_type == "NumericalDiscreteParameter" and isinstance(values, list):
            specs.append(NumericalDiscreteSpec(name=name, values=[float(v) for v in values]))
        elif p_type == "NumericalContinuousParameter" and isinstance(values, list) and len(values) == 2:
            specs.append(NumericalContinuousSpec(name=name, lower=float(values[0]), upper=float(values[1])))
    return specs


def _compare_config_to_campaign(cfg: CampaignConfig, meta: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    if cfg.campaign_name != meta.get("campaign_name"):
        mismatches.append(f"Campaign name: config '{cfg.campaign_name}' vs loaded '{meta.get('campaign_name')}'")
    if cfg.objective_target != meta.get("objective_target"):
        mismatches.append(f"Objective target: config '{cfg.objective_target}' vs loaded '{meta.get('objective_target')}'")

    cfg_params = {p.name: p for p in cfg.parameters}
    loaded_params = {p.get("name"): p for p in meta.get("parameters", []) if p.get("name")}
    if list(cfg_params.keys()) != list(loaded_params.keys()):
        mismatches.append(
            f"Parameter names/order differ: config {list(cfg_params.keys())} vs loaded {list(loaded_params.keys())}"
        )

    for pname, lp in loaded_params.items():
        cp = cfg_params.get(pname)
        if cp is None:
            continue
        loaded_type = lp.get("type", "")
        cfg_type = type(cp).__name__.replace("Spec", "Parameter")
        if loaded_type and loaded_type != cfg_type:
            mismatches.append(f"Parameter '{pname}' type differs: config {cfg_type} vs loaded {loaded_type}")

        if isinstance(cp, CategoricalSpec) and isinstance(lp.get("values"), list):
            if sorted(cp.values) != sorted([str(v) for v in lp["values"]]):
                mismatches.append(f"Parameter '{pname}' categories differ between config and loaded campaign")
    return mismatches


def _render_campaign_browser(workdir: Path) -> None:
    st.sidebar.subheader("Campaign browser")
    campaigns = _discover_campaigns(workdir)
    if not campaigns:
        st.sidebar.info("No saved campaigns found in campaign_jsons/.")
        return

    label_map = {
        i: f"{c['campaign_name']} ({c['last_modified'].strftime('%Y-%m-%d %H:%M:%S')})"
        for i, c in enumerate(campaigns)
    }
    selected_idx = st.sidebar.selectbox(
        "Saved campaigns",
        options=list(label_map.keys()),
        format_func=lambda x: label_map[x],
    )
    selected = campaigns[selected_idx]

    completed_str = "unknown" if selected["completed_measurements"] < 0 else str(selected["completed_measurements"])
    st.sidebar.markdown("**Selected campaign metadata**")
    st.sidebar.caption(
        f"- name: `{selected['campaign_name']}`\n"
        f"- objective target: `{selected['objective_target']}`\n"
        f"- optimization mode: `{selected['objective_mode']}`\n"
        f"- completed measurements: `{completed_str}`\n"
        f"- parameters: `{', '.join(selected['parameter_names'])}`\n"
        f"- last modified: `{selected['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}`"
    )

    if st.sidebar.button("Load selected campaign"):
        _activate_campaign(workdir, selected)
        st.success(f"Loaded campaign '{selected['campaign_name']}'.")
        st.rerun()


def _check_campaign_config_compatibility(cfg: CampaignConfig, workdir: Path) -> bool:
    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if not latest.exists():
        return True
    meta = _extract_campaign_metadata(latest)
    mismatches = _compare_config_to_campaign(cfg, meta)
    if not mismatches:
        return True

    st.warning("Configuration mismatch with loaded campaign:\n- " + "\n- ".join(mismatches))
    key = f"confirm_mismatch_{cfg.campaign_name}"
    return st.checkbox("I understand and want to proceed with this mismatch.", key=key)


def _default_config() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="default",
        objective_target="yield",
        objective_mode="maximize",
        batch_size=8,
        init_mode="sobol",
        n_init=8,
        acquisition="qExpectedImprovement",
        acquisition_kwargs={},
        parameters=[
            CategoricalSpec(name="solvent", values=["MeCN", "HFIP"], encoding="OHE"),
            NumericalDiscreteSpec(name="reaction_time", values=[1.0, 2.0, 3.0], unit="h"),
        ],
    )


def _persist_campaign_config(workdir: Path, cfg: CampaignConfig) -> Path:
    path = campaign_config_path(workdir, cfg.campaign_name)
    save_text(path, json.dumps(cfg.to_dict(), indent=2))
    return path


def _clear_config_widget_state() -> None:
    prefixes = (
        "cfg_",
        "pname_",
        "lo_",
        "hi_",
        "unit_",
        "vals_",
        "cats_",
        "enc_",
        "smiles_",
        "senc_",
        "decor_",
    )
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def _load_config_for_campaign(workdir: Path, meta: Dict[str, Any]) -> CampaignConfig:
    path = campaign_config_path(workdir, meta["campaign_name"])
    if path.exists():
        try:
            return CampaignConfig.from_dict(json.loads(load_text(path)))
        except Exception:
            pass

    cfg = _default_config()
    cfg.campaign_name = meta["campaign_name"]
    cfg.objective_target = meta.get("objective_target", cfg.objective_target)
    cfg.objective_mode = meta.get("objective_mode", cfg.objective_mode)
    acquisition = meta.get("acquisition")
    if acquisition in ACQ_ORDER:
        cfg.acquisition = acquisition
    inferred_specs = _build_specs_from_metadata(meta)
    if inferred_specs:
        cfg.parameters = inferred_specs
    return cfg


def _activate_campaign(workdir: Path, selected: Dict[str, Any]) -> None:
    _clear_config_widget_state()
    st.session_state["campaign_name"] = selected["campaign_name"]
    st.session_state["active_campaign_path"] = str(selected["path"])
    st.session_state["config"] = _load_config_for_campaign(workdir, selected).to_dict()


def _reset_to_new_campaign() -> None:
    _clear_config_widget_state()
    cfg = _default_config()
    st.session_state["config"] = cfg.to_dict()
    st.session_state["campaign_name"] = cfg.campaign_name
    for key in ["active_campaign_path"]:
        st.session_state.pop(key, None)


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

    if "config" not in st.session_state:
        st.session_state["config"] = _default_config().to_dict()

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "1) Configure",
            "2) Initialize",
            "3) Recommend",
            "4) Ingest Results",
            "5) History",
            "6) Campaign Dashboard",
        ],
    )

    st.sidebar.header("Storage")
    workdir_str = st.sidebar.text_input(
        "Campaign folder (WORKDIR)",
        value=str(Path.cwd()),
        help="All plans/, results/, campaign_jsons/ will live here.",
    )
    workdir = Path(workdir_str).expanduser().resolve()
    ensure_campaign_dirs(workdir)
    _render_campaign_browser(workdir)

    cfg = CampaignConfig.from_dict(st.session_state["config"])
    active_campaign_path = st.session_state.get("active_campaign_path")
    campaign_name = st.sidebar.text_input(
        "Campaign name",
        value=cfg.campaign_name,
        disabled=bool(active_campaign_path),
        help=(
            "Campaign name is locked to the loaded campaign. Select 'Create new campaign' to start a fresh configuration."
            if active_campaign_path
            else "Name for the campaign configuration you are editing."
        ),
    )

    if active_campaign_path and st.sidebar.button("Create new campaign"):
        _reset_to_new_campaign()
        st.rerun()

    st.session_state["campaign_name"] = campaign_name
    st.session_state["config"]["campaign_name"] = campaign_name

    if active_campaign_path:
        st.sidebar.success(f"Active campaign: {st.session_state.get('campaign_name', campaign_name)}")

    if page.startswith("1"):
        render_config_page(workdir)
    elif page.startswith("2"):
        render_init_page(workdir)
    elif page.startswith("3"):
        render_recommend_page(workdir)
    elif page.startswith("4"):
        render_ingest_page(workdir)
    elif page.startswith("5"):
        render_history_page(workdir)
    else:
        render_campaign_dashboard_page(workdir)



def render_config_page(workdir: Path) -> None:
    st.subheader("1) Configure parameters + model settings")
    cfg = CampaignConfig.from_dict(st.session_state["config"])

    if st.session_state.get("active_campaign_path"):
        st.info(
            "This form is populated from the loaded campaign configuration. "
            "Use 'Create new campaign' to reset the editor and start a different campaign."
        )
        if st.button("Create new campaign", key="create_new_campaign_config"):
            _reset_to_new_campaign()
            st.rerun()

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### Objective")
        cfg.objective_target = st.text_input("Target column name", value=cfg.objective_target, key="cfg_objective_target")
        cfg.objective_mode = st.selectbox(
            "Optimize direction",
            options=["maximize", "minimize"],
            index=0 if cfg.objective_mode == "maximize" else 1,
            key="cfg_objective_mode",
        )

        st.markdown("### Batch")
        cfg.batch_size = int(st.number_input("Batch size", min_value=1, value=int(cfg.batch_size), step=1, key="cfg_batch_size"))

        st.markdown("### Initialization")
        cfg.init_mode = st.selectbox(
            "Init mode",
            ["sobol", "existing_data"],
            index=0 if cfg.init_mode == "sobol" else 1,
            key="cfg_init_mode",
        )
        cfg.n_init = int(st.number_input("# init points", min_value=0, value=int(cfg.n_init), step=1, key="cfg_n_init"))

    with colR:
        st.markdown("### Acquisition function (BayBE → BoTorch)")
        st.caption("BayBE wrappers are shown here. The plot places them on a practical exploration ↔ exploitation spectrum.")
        st.caption("Only acquisition functions supported by the current backend and batch size are shown.")
        available_acquisitions = supported_acquisition_names(int(cfg.batch_size))
        if int(cfg.batch_size) > 1:
            st.caption("Batch size > 1 requires batch-compatible acquisition functions (`q...`).")
        if not available_acquisitions:
            st.error(f"No acquisition functions are available for batch size {int(cfg.batch_size)}.")
            return
        if cfg.acquisition not in available_acquisitions:
            st.error(
                f"Configured acquisition '{cfg.acquisition}' is not supported for batch size {int(cfg.batch_size)}. "
                "Choose one of the supported options below."
            )
        cfg.acquisition = st.selectbox(
            "Acquisition",
            options=available_acquisitions,
            index=(
                available_acquisitions.index(cfg.acquisition)
                if cfg.acquisition in available_acquisitions
                else available_acquisitions.index(default_acquisition_name(int(cfg.batch_size)))
            ),
            key="cfg_acquisition",
        )

        if acquisition_supports_beta(cfg.acquisition):
            current_beta = float((cfg.acquisition_kwargs or {}).get("beta", 2.0))
            beta = float(
                st.slider(
                    "UCB beta",
                    min_value=0.1,
                    max_value=10.0,
                    value=current_beta,
                    step=0.1,
                    key="cfg_ucb_beta",
                    help="Larger beta weights uncertainty more heavily, so the optimizer explores more.",
                )
            )
            cfg.acquisition_kwargs = {"beta": beta}
        else:
            beta = None
            cfg.acquisition_kwargs = {}

        _render_acquisition_map(cfg.acquisition, beta)

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
        st.caption(
            "This is the persisted config. Editing here overwrites the form values. "
            "Unsupported keys or unsupported acquisition kwargs are rejected."
        )
        raw = st.text_area("CampaignConfig JSON", value=json.dumps(cfg.to_dict(), indent=2), height=320, key="cfg_raw_json")
        if st.button("Load JSON into editor"):
            try:
                payload = json.loads(raw)
                payload_errors = validate_config_payload(payload)
                if payload_errors:
                    for msg in payload_errors:
                        st.error(msg)
                else:
                    st.session_state["config"] = CampaignConfig.from_dict(payload).to_dict()
                    _clear_config_widget_state()
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
            _persist_campaign_config(workdir, cfg)
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
        smiles = _normalize_smiles_input(smiles_text.splitlines())

        enc_options = supported_substance_encodings()
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
        smiles = _normalize_smiles_input(smiles_text.splitlines())
        enc_options = supported_substance_encodings()
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
    config_is_compatible = _check_campaign_config_compatibility(cfg, workdir)
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
        if st.button("Generate run0.csv via Sobol", disabled=not config_is_compatible):
            plan0 = sobol_initial_design(cfg.parameters, n=cfg.n_init, seed=seed)
            out_path = run_plan_path(workdir, run_idx=0)
            plan0.to_csv(out_path, index=False)
            try:
                engine = create_campaign_engine(cfg)
            except Exception as exc:
                st.error(f"Campaign initialization failed: {exc}")
                return
            engine.save(latest)
            _persist_campaign_config(workdir, cfg)
            st.session_state["campaign_name"] = cfg.campaign_name
            st.session_state["active_campaign_path"] = str(latest)
            st.success(f"Wrote {out_path} and initialized campaign JSON at {latest}")
            _download_button_df("Download run0.csv", plan0, "run0.csv")

    else:
        st.caption(
            "Upload a CSV with columns = parameters + target. The first N rows are ingested as initial data. "
            "Enter yields as fractions (0–1). Example: 0.63 for 63% yield."
        )
        up = st.file_uploader("Upload initial results CSV", type=["csv"], key="initcsv")
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head(20), use_container_width=True)
            if st.button("Initialize from this CSV", disabled=not config_is_compatible):
                try:
                    engine = create_campaign_engine(cfg)
                except Exception as exc:
                    st.error(f"Campaign initialization failed: {exc}")
                    return
                needed = [p.name for p in cfg.parameters] + [cfg.objective_target]
                missing = [c for c in needed if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    st.stop()
                df0 = df[needed].iloc[: cfg.n_init].copy()
                target_error = _validate_fraction_target(df0, cfg.objective_target)
                if target_error:
                    st.error(target_error)
                    return
                try:
                    df0 = engine.ingest(df0)
                except ValueError as exc:
                    st.error(f"Campaign initialization failed: {exc}")
                    return
                engine.save(latest)
                _persist_campaign_config(workdir, cfg)
                st.session_state["campaign_name"] = cfg.campaign_name
                st.session_state["active_campaign_path"] = str(latest)
                init_path = workdir / "results" / "initial_data_results.csv"
                df0.to_csv(init_path, index=False)
                append_all_runs(workdir, df0, run_idx=-1)
                st.success(f"Initialized campaign from {len(df0)} rows. Saved {latest} and {init_path}.")



def render_recommend_page(workdir: Path) -> None:
    st.subheader("3) Recommend next batch")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    validation_errors = _config_validation_errors(cfg)
    if validation_errors:
        for msg in validation_errors:
            st.error(msg)
        st.info("Fix the campaign configuration on the Configure page before requesting recommendations.")
        return
    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if not latest.exists():
        st.error("No campaign JSON found. Go to 'Initialize' first.")
        return
    config_is_compatible = _check_campaign_config_compatibility(cfg, workdir)

    engine = load_campaign_engine(latest, cfg)
    param_cols = [p.name for p in cfg.parameters]
    measured = measured_keys(workdir, param_cols)
    active_info = engine.cached_recommendation_info()
    active_df: Optional[pd.DataFrame] = None
    active_batch_index: Optional[int] = None
    allow_new_batch = True

    if active_info is not None:
        active_batch_index = active_info.get("batch_index")
        decoded_df = active_info.get("dataframe")
        decode_error = active_info.get("decode_error")
        if isinstance(decoded_df, pd.DataFrame):
            active_df = drop_measured(decoded_df, measured, param_cols)
            if active_df.empty:
                active_df = None
            else:
                active_name = (
                    f"run{active_batch_index}.csv"
                    if isinstance(active_batch_index, int) and active_batch_index >= 0
                    else f"{cfg.campaign_name}_active_batch.csv"
                )
                st.info(
                    "Loaded campaign JSON already contains an active recommended batch. "
                    "Re-download that batch below unless you intentionally want to advance to a new batch."
                )
                st.markdown(f"Active batch in saved campaign: **{active_name}**")
                st.caption(
                    "The configured batch size applies per recommendation call. "
                    "Requesting another batch before ingesting results creates additional pending experiments."
                )
                _download_button_df("Download active batch CSV", active_df, active_name)
                st.dataframe(active_df, use_container_width=True)
                allow_new_batch = st.checkbox(
                    "Generate a new batch instead of reusing the active batch",
                    value=False,
                    key=f"allow_new_batch_{cfg.campaign_name}",
                )
        elif decode_error:
            st.warning(
                "This campaign JSON appears to contain a cached recommendation, "
                f"but it could not be decoded in the current environment: {decode_error}"
            )
            allow_new_batch = st.checkbox(
                "Generate a new batch anyway",
                value=False,
                key=f"allow_new_batch_{cfg.campaign_name}",
            )

    next_run = discover_next_run_idx(workdir)
    st.markdown(f"Next new run index: **{next_run}**")
    st.caption("This writes plans/runN.csv and snapshots the campaign state after recommendation.")

    button_label = "Recommend batch" if active_df is None else "Recommend another batch"
    if st.button(button_label, disabled=(not config_is_compatible) or (not allow_new_batch)):
        keys = set(measured)
        if active_df is not None:
            for row in active_df[[c for c in param_cols if c in active_df.columns]].itertuples(index=False, name=None):
                keys.add(tuple(row))
        needed = int(cfg.batch_size)
        collected: List[pd.DataFrame] = []
        attempts = 0
        max_attempts = 6

        while needed > 0 and attempts < max_attempts:
            attempts += 1
            rec = engine.recommend(batch_size=needed)
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
        engine.save(latest)
        snap = campaign_snapshot_path(workdir, cfg.campaign_name, tag=f"after_recommend_run{next_run}")
        engine.save(snap)
        st.success(f"Saved plan -> {out_path}")
        _download_button_df("Download plan CSV", rec_final, f"run{next_run}.csv")
        st.dataframe(rec_final, use_container_width=True)



def _validate_fraction_target(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """Validate target values are numeric fractions in [0, 1]."""
    vals = pd.to_numeric(df[target_col], errors="coerce")
    if vals.isna().any():
        return (
            f"Column '{target_col}' contains non-numeric values. "
            "Enter yields as fractions (0–1). Example: 0.63 for 63% yield."
        )
    out_of_bounds = (~vals.between(0.0, 1.0)).sum()
    if out_of_bounds:
        return (
            f"Column '{target_col}' has {int(out_of_bounds)} value(s) outside [0, 1]. "
            "Enter yields as fractions (0–1). Example: 0.63 for 63% yield."
        )
    return None


def render_ingest_page(workdir: Path) -> None:
    st.subheader("4) Ingest results + update campaign")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    validation_errors = _config_validation_errors(cfg)
    if validation_errors:
        for msg in validation_errors:
            st.error(msg)
        st.info("Fix the campaign configuration on the Configure page before ingesting results.")
        return
    latest = campaign_latest_path(workdir, cfg.campaign_name)
    if not latest.exists():
        st.error("No campaign JSON found. Go to 'Initialize' first.")
        return
    config_is_compatible = _check_campaign_config_compatibility(cfg, workdir)

    engine = load_campaign_engine(latest, cfg)
    st.caption(
        "Upload a results CSV (typically a copy of the plan CSV with an extra target column), or point the app to the on-disk results file. "
        "Enter yields as fractions (0–1). Example: 0.63 for 63% yield."
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
            if st.button("Ingest uploaded results", disabled=not config_is_compatible):
                _ingest_df_and_persist(workdir, cfg, engine, df, run_idx)
    with colB:
        disk_path = run_results_path(workdir, run_idx)
        st.code(str(disk_path))
        if st.button("Ingest results from disk path", disabled=not config_is_compatible):
            if not disk_path.exists():
                st.error(f"Missing: {disk_path}")
                st.stop()
            df = pd.read_csv(disk_path)
            _ingest_df_and_persist(workdir, cfg, engine, df, run_idx)



def _ingest_df_and_persist(workdir: Path, cfg: CampaignConfig, engine, df: pd.DataFrame, run_idx: int) -> None:
    param_cols = [p.name for p in cfg.parameters]
    target_col = cfg.objective_target
    missing = [c for c in (param_cols + [target_col]) if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    df_use = df[param_cols + [target_col]].copy()

    target_error = _validate_fraction_target(df_use, target_col)
    if target_error:
        st.error(target_error)
        return

    try:
        df_use = engine.ingest(df_use)
    except ValueError as exc:
        st.error(f"Failed to ingest results: {exc}")
        return

    out_path = run_results_path(workdir, run_idx)
    df_use.to_csv(out_path, index=False)
    append_all_runs(workdir, df_use, run_idx=run_idx)

    latest = campaign_latest_path(workdir, cfg.campaign_name)
    engine.save(latest)
    snap = campaign_snapshot_path(workdir, cfg.campaign_name, tag=f"after_ingest_run{run_idx}")
    engine.save(snap)
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


def render_campaign_dashboard_page(workdir: Path) -> None:
    st.subheader("6) Campaign Dashboard")
    cfg = CampaignConfig.from_dict(st.session_state["config"])
    runs_path = all_runs_path(workdir)
    if not runs_path.exists():
        st.info("No all_runs.csv found yet. Ingest results to populate the dashboard.")
        return

    df_trials = pd.read_csv(runs_path)
    campaign_dir = workdir / "plots" / cfg.campaign_name
    render_campaign_dashboard(df_trials=df_trials, campaign_dir=campaign_dir)


if __name__ == "__main__":
    main()
