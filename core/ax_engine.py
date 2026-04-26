from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .campaign_engine import CampaignEngine, ModelNotFittedError
from .persistence import load_text, save_text
from .schema import (
    CampaignConfig,
    CategoricalSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
    ParameterSpec,
    SubstanceSpec,
    TargetSpec,
    VALID_TRIAL_STATUSES,
)

# Suppress the sqlalchemy version warning Ax emits on every import.
warnings.filterwarnings("ignore", message=".*sqlalchemy.*", category=UserWarning)

_AX_ENGINE_MARKER = "ax"


# ---------------------------------------------------------------------------
# Parameter-spec → Ax parameter dict conversion
# ---------------------------------------------------------------------------

def _spec_to_ax_param(spec: ParameterSpec) -> dict[str, Any]:
    if isinstance(spec, NumericalContinuousSpec):
        return {
            "name": spec.name,
            "type": "range",
            "bounds": [float(spec.lower), float(spec.upper)],
            "value_type": "float",
        }
    if isinstance(spec, NumericalDiscreteSpec):
        return {
            "name": spec.name,
            "type": "choice",
            "values": [float(v) for v in spec.values],
            "value_type": "float",
            "is_ordered": True,
            "sort_values": True,
        }
    if isinstance(spec, CategoricalSpec):
        return {
            "name": spec.name,
            "type": "choice",
            "values": list(spec.values),
            "value_type": "str",
            "is_ordered": False,
        }
    if isinstance(spec, SubstanceSpec):
        # Ax has no molecular encoding; SMILES are treated as opaque category labels.
        return {
            "name": spec.name,
            "type": "choice",
            "values": list(spec.smiles),
            "value_type": "str",
            "is_ordered": False,
        }
    raise ValueError(f"Unsupported parameter spec type for Ax: {type(spec).__name__}")


def _build_ax_parameters(cfg: CampaignConfig) -> list[dict[str, Any]]:
    return [_spec_to_ax_param(spec) for spec in cfg.parameters]


# ---------------------------------------------------------------------------
# Metadata extraction from a saved Ax JSON snapshot
# ---------------------------------------------------------------------------

def extract_ax_campaign_metadata(path: Path) -> dict[str, Any]:
    raw = json.loads(load_text(path))
    exp = raw.get("experiment", {}) or {}

    name = exp.get("name", path.stem)

    opt_config = exp.get("optimization_config", {}) or {}
    objective = opt_config.get("objective", {}) or {}

    # Detect multi-objective: Ax stores MultiObjective with an "objectives" list
    target_specs: list[dict[str, Any]] = []
    sub_objectives = objective.get("objectives")
    if isinstance(sub_objectives, list) and sub_objectives:
        for sub in sub_objectives:
            metric = sub.get("metric", {}) or {}
            target_specs.append(
                {
                    "name": metric.get("name", "unknown"),
                    "mode": "minimize" if sub.get("minimize", False) else "maximize",
                }
            )
    else:
        metric = objective.get("metric", {}) or {}
        target_specs.append(
            {
                "name": metric.get("name", "unknown"),
                "mode": "minimize" if objective.get("minimize", False) else "maximize",
            }
        )

    target_name = target_specs[0]["name"]
    objective_mode = target_specs[0]["mode"]

    search_space = exp.get("search_space", {}) or {}
    param_meta: list[dict[str, Any]] = []
    for p in search_space.get("parameters", []) or []:
        entry: dict[str, Any] = {"name": p.get("name")}
        ptype = p.get("__type", "")
        if ptype == "RangeParameter":
            entry["type"] = "NumericalContinuousParameter"
            entry["values"] = [p.get("lower"), p.get("upper")]
        elif ptype == "ChoiceParameter":
            vals = p.get("values") or []
            ptype_inner = (p.get("parameter_type") or {}).get("name", "STRING")
            if ptype_inner == "FLOAT" and p.get("is_ordered"):
                entry["type"] = "NumericalDiscreteParameter"
            else:
                entry["type"] = "CategoricalParameter"
            entry["values"] = vals
        param_meta.append(entry)

    # Count completed trials
    completed = sum(
        1
        for t in (exp.get("trials") or {}).values()
        if isinstance(t, dict) and t.get("status") == "COMPLETED"
    )

    from datetime import datetime
    return {
        "path": path,
        "campaign_name": name,
        "objective_target": target_name,
        "objective_mode": objective_mode,
        "targets": target_specs,
        "acquisition": "Ax default (internal)",
        "parameter_names": [p["name"] for p in param_meta if p.get("name")],
        "parameters": param_meta,
        "completed_measurements": completed,
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime),
        "engine": _AX_ENGINE_MARKER,
    }


# ---------------------------------------------------------------------------
# AxEngine
# ---------------------------------------------------------------------------

class AxEngine(CampaignEngine):
    """CampaignEngine backed by Ax (ax-platform).

    Trials are managed through AxClient.  Every ingest event maps to an Ax
    trial status transition:
        completed → complete_trial()
        failed    → log_trial_failure()
        partial   → log_trial_failure()   (recorded, not valid observation)
        invalid   → log_trial_failure()
        abandoned → abandon_trial()

    The pending-trial registry (_pending) maps Ax trial_index → parameter
    dict for trials that have been recommended but not yet ingested.  It is
    serialised into the save file alongside the AxClient snapshot.
    """

    def __init__(self, cfg: CampaignConfig, client: Any, pending: dict[int, dict[str, Any]] | None = None):
        self.cfg = cfg
        self._client = client
        # trial_index → params dict for all open (recommended but unresolved) trials
        self._pending: dict[int, dict[str, Any]] = dict(pending or {})

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: CampaignConfig) -> "AxEngine":
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties

        client = AxClient(verbose_logging=False)
        params = _build_ax_parameters(cfg)
        # Multi-objective: pass one ObjectiveProperties per target. Ax interprets
        # an objectives dict with len > 1 as a Pareto / multi-objective experiment.
        objectives = {
            t.name: ObjectiveProperties(minimize=(t.mode == "minimize"))
            for t in cfg.effective_targets()
        }

        gen_kwargs: dict[str, Any] = {}
        if cfg.init_mode == "existing_data":
            # Skip Sobol init: historical data will be attached via ingest().
            gen_kwargs["num_initialization_trials"] = 0
        elif cfg.n_init > 0:
            gen_kwargs["num_initialization_trials"] = cfg.n_init

        client.create_experiment(
            name=cfg.campaign_name,
            parameters=params,
            objectives=objectives,
            choose_generation_strategy_kwargs=gen_kwargs or None,
        )
        return cls(cfg=cfg, client=client)

    @classmethod
    def load(cls, path: Path, cfg: CampaignConfig | None = None) -> "AxEngine":
        from ax.service.ax_client import AxClient

        raw = json.loads(load_text(path))
        pending_raw = raw.pop("_pending", {})
        pending = {int(k): v for k, v in pending_raw.items()}
        # Remove our marker before passing to Ax
        raw.pop("_engine", None)

        client = AxClient.from_json_snapshot(raw, verbose_logging=False)
        resolved_cfg = cfg or cls._infer_config(client)
        return cls(cfg=resolved_cfg, client=client, pending=pending)

    @classmethod
    def _infer_config(cls, client: Any) -> CampaignConfig:
        exp = client.experiment
        params: list[ParameterSpec] = []
        for p in exp.search_space.parameters.values():
            from ax.core.parameter import RangeParameter, ChoiceParameter, ParameterType
            if isinstance(p, RangeParameter):
                params.append(
                    NumericalContinuousSpec(
                        name=p.name,
                        lower=float(p.lower),
                        upper=float(p.upper),
                    )
                )
            elif isinstance(p, ChoiceParameter):
                vals = list(p.values)
                if p.parameter_type == ParameterType.FLOAT and p.is_ordered:
                    params.append(NumericalDiscreteSpec(name=p.name, values=[float(v) for v in vals]))
                else:
                    params.append(CategoricalSpec(name=p.name, values=[str(v) for v in vals]))

        # Detect multi-objective from the OptimizationConfig
        targets: list[TargetSpec] = []
        opt_cfg = getattr(exp, "optimization_config", None)
        ax_objective = getattr(opt_cfg, "objective", None) if opt_cfg is not None else None
        sub_objectives = getattr(ax_objective, "objectives", None)
        if sub_objectives:
            for sub in sub_objectives:
                metric_name = getattr(getattr(sub, "metric", None), "name", None)
                if metric_name is None:
                    continue
                targets.append(
                    TargetSpec(
                        name=metric_name,
                        mode="minimize" if getattr(sub, "minimize", False) else "maximize",
                    )
                )

        if targets:
            primary = targets[0]
            return CampaignConfig(
                campaign_name=exp.name,
                objective_target=primary.name,
                objective_mode=primary.mode,
                parameters=params,
                targets=targets,
            )

        objective_name = client.objective_name
        minimize = client.objective.minimize if hasattr(client, "objective") else False
        return CampaignConfig(
            campaign_name=exp.name,
            objective_target=objective_name,
            objective_mode="minimize" if minimize else "maximize",
            parameters=params,
        )

    # ------------------------------------------------------------------
    # CampaignEngine interface
    # ------------------------------------------------------------------

    def recommend(self, batch_size: int) -> pd.DataFrame:
        trial_map, _ = self._client.get_next_trials(max_trials=batch_size)
        if not trial_map:
            raise RuntimeError(
                "Ax returned no trials. The design space may be exhausted or the "
                "generation strategy has no more candidates."
            )
        self._pending.update(trial_map)
        rows = []
        for trial_idx, params in trial_map.items():
            row = dict(params)
            row["trial_index"] = trial_idx
            rows.append(row)
        return pd.DataFrame(rows)

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        if "status" not in df.columns:
            raise ValueError(
                "Missing required 'status' column. Every ingestion event must specify a trial status "
                f"({', '.join(sorted(VALID_TRIAL_STATUSES))})."
            )

        df = df.copy()
        df["status"] = df["status"].astype(str).str.lower().str.strip()
        invalid = set(df["status"].unique()) - VALID_TRIAL_STATUSES
        if invalid:
            raise ValueError(
                f"Invalid trial status values: {sorted(invalid)}. "
                f"Supported values: {sorted(VALID_TRIAL_STATUSES)}."
            )

        param_cols = [p.name for p in self.cfg.parameters]
        target_cols = [t.name for t in self.cfg.effective_targets()]
        missing_cols = [c for c in (param_cols + target_cols) if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for ingestion: {missing_cols}.")

        for _, row in df.iterrows():
            status = str(row["status"])
            trial_idx = self._get_or_attach_trial(row, param_cols)

            if status == "completed":
                raw_data: dict[str, tuple[float, None]] = {}
                for target_col in target_cols:
                    raw_val = row[target_col]
                    if pd.isna(raw_val):
                        raise ValueError(
                            f"Row with status='completed' has a missing target value for '{target_col}'. "
                            "Completed trials must have a numeric value for every configured target."
                        )
                    raw_data[target_col] = (float(raw_val), None)
                self._client.complete_trial(trial_index=trial_idx, raw_data=raw_data)
            elif status == "abandoned":
                self._client.abandon_trial(trial_index=trial_idx, reason=status)
            else:
                # failed / partial / invalid → mark as failed in Ax
                self._client.log_trial_failure(trial_index=trial_idx, metadata={"reason": status})

            self._pending.pop(trial_idx, None)

        return df

    def predict(self, candidates: pd.DataFrame) -> pd.DataFrame:
        param_cols = [p.name for p in self.cfg.parameters]
        parameterizations = [
            {col: _coerce_ax_value(row[col], self.cfg.parameters, col) for col in param_cols if col in candidates.columns}
            for _, row in candidates.iterrows()
        ]

        targets = self.cfg.effective_targets()
        target_names = [t.name for t in targets]

        try:
            preds = self._client.get_model_predictions_for_parameterizations(
                parameterizations=parameterizations,
                metric_names=target_names,
            )
        except Exception as exc:
            raise ModelNotFittedError(
                "Ax surrogate predictions not available. The experiment may still be in the "
                "quasi-random initialisation phase. Complete more trials to enter the "
                "Bayesian optimisation phase."
            ) from exc

        out = candidates.copy()

        if len(targets) == 1:
            target = targets[0]
            out["pred_mean"] = [float(p[target.name][0]) for p in preds]
            out["pred_std"] = [float(p[target.name][1]) for p in preds]
            # Ax doesn't expose per-candidate acquisition values for arbitrary points;
            # rank by predicted mean adjusted for optimisation direction.
            if target.mode == "minimize":
                out["acq_score"] = -out["pred_mean"]
            else:
                out["acq_score"] = out["pred_mean"]
            out["rank"] = out["acq_score"].rank(ascending=False, method="first").astype(int)
            return out

        # Multi-objective: per-target prediction columns + Pareto rank.
        means = []
        for t in targets:
            mean_col = f"{t.name}_pred_mean"
            std_col = f"{t.name}_pred_std"
            mean_vals = [float(p[t.name][0]) for p in preds]
            std_vals = [float(p[t.name][1]) for p in preds]
            out[mean_col] = mean_vals
            out[std_col] = std_vals
            means.append(mean_vals)

        import numpy as np
        points = np.array(means).T  # shape (n_candidates, n_targets)
        maximize_mask = np.array([t.mode != "minimize" for t in targets], dtype=bool)
        out["pareto_rank"] = _pareto_ranks(points, maximize_mask)
        return out

    def save(self, path: Path) -> None:
        snapshot = self._client.to_json_snapshot()
        snapshot["_engine"] = _AX_ENGINE_MARKER
        snapshot["_pending"] = {str(k): v for k, v in self._pending.items()}
        save_text(path, json.dumps(snapshot))

    def cached_recommendation_info(self) -> Optional[dict[str, Any]]:
        if not self._pending:
            return None
        rows = [dict(params, trial_index=trial_idx) for trial_idx, params in self._pending.items()]
        df = pd.DataFrame(rows)
        n_completed = self.measurement_count()
        return {
            "dataframe": df,
            "batch_index": n_completed,
            "decode_error": None,
        }

    def measurement_count(self) -> int:
        return sum(
            1
            for t in self._client.experiment.trials.values()
            if t.status.name == "COMPLETED"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_attach_trial(self, row: pd.Series, param_cols: list[str]) -> int:
        """Return the Ax trial_index for this row.

        Resolution order:
        1. Explicit trial_index column that matches a known pending trial.
        2. Parameter-value match against pending trials.
        3. Attach a new trial (warm-start / existing-data path).
        """
        # 1. Explicit trial_index
        if "trial_index" in row.index and pd.notna(row.get("trial_index")):
            idx = int(row["trial_index"])
            if idx in self._pending or idx in self._client.experiment.trials:
                return idx

        # 2. Match by parameter values
        for trial_idx, params in self._pending.items():
            if self._params_match(row, params, param_cols):
                return trial_idx

        # 3. Attach a new trial (warm-start)
        params_dict = {
            col: _coerce_ax_value(row[col], self.cfg.parameters, col)
            for col in param_cols
            if col in row.index
        }
        _, new_idx = self._client.attach_trial(params_dict)
        return new_idx

    @staticmethod
    def _params_match(row: pd.Series, params: dict[str, Any], param_cols: list[str]) -> bool:
        for col in param_cols:
            if col not in row.index:
                continue
            row_val = row[col]
            ax_val = params.get(col)
            # Numeric comparison with tolerance; string exact match
            try:
                if abs(float(row_val) - float(ax_val)) > 1e-9:
                    return False
            except (TypeError, ValueError):
                if str(row_val).strip() != str(ax_val).strip():
                    return False
        return True


def _coerce_ax_value(val: Any, specs: list[ParameterSpec], col_name: str) -> Any:
    """Cast a value to the type Ax expects for the given parameter."""
    spec = next((s for s in specs if s.name == col_name), None)
    if spec is None:
        return val
    if isinstance(spec, NumericalContinuousSpec):
        return float(val)
    if isinstance(spec, NumericalDiscreteSpec):
        return float(val)
    return str(val)


def _pareto_ranks(points: Any, maximize_mask: Any) -> list[int]:
    """Non-dominated sort: rank 1 = first Pareto front, rank 2 = second, etc.

    Args:
        points: (n, m) array-like of objective values for n candidates and m targets.
        maximize_mask: (m,) bool array; True = maximize that objective.

    Returns:
        List of integer ranks length n. A point is on the k-th Pareto front if
        it is not dominated by any point not yet assigned to fronts 1..k-1.
    """
    import numpy as np

    pts = np.asarray(points, dtype=float).copy()
    mask = np.asarray(maximize_mask, dtype=bool)
    # Convert minimize objectives to maximize via negation so domination check is uniform.
    pts[:, ~mask] = -pts[:, ~mask]

    n = pts.shape[0]
    ranks = [0] * n
    remaining = set(range(n))
    front = 1
    while remaining:
        current_front: list[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # j dominates i iff j >= i on all axes and j > i on at least one axis.
                if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                    dominated = True
                    break
            if not dominated:
                current_front.append(i)
        for i in current_front:
            ranks[i] = front
        remaining -= set(current_front)
        front += 1
    return ranks
