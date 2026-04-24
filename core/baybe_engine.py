from __future__ import annotations

import base64
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .baybe_factory import build_campaign
from .campaign_engine import CampaignEngine, ModelNotFittedError
from .persistence import load_text, save_text
from .schema import (
    CampaignConfig,
    CategoricalSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
    ParameterSpec,
    VALID_TRIAL_STATUSES,
)


def _campaign_name_from_path(path: Path) -> str:
    stem = path.stem
    return stem[:-7] if stem.endswith("_latest") else stem


def _build_specs_from_metadata(meta: dict[str, Any]) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    for param in meta.get("parameters", []):
        param_type = param.get("type", "")
        name = param.get("name")
        values = param.get("values")
        if not name:
            continue
        if param_type == "CategoricalParameter" and isinstance(values, list):
            specs.append(CategoricalSpec(name=name, values=[str(value) for value in values], encoding="OHE"))
        elif param_type == "NumericalDiscreteParameter" and isinstance(values, list):
            specs.append(NumericalDiscreteSpec(name=name, values=[float(value) for value in values]))
        elif param_type == "NumericalContinuousParameter" and isinstance(values, list) and len(values) == 2:
            specs.append(NumericalContinuousSpec(name=name, lower=float(values[0]), upper=float(values[1])))
    return specs


def extract_baybe_campaign_metadata(path: Path) -> dict[str, Any]:
    raw = json.loads(load_text(path))
    objective = raw.get("objective", {}) or {}
    target = objective.get("target", {}) or {}
    recommender = raw.get("recommender", {}) or {}
    acquisition = (recommender.get("acquisition_function", {}) or {}).get("type")

    param_meta: list[dict[str, Any]] = []
    searchspace = raw.get("searchspace", {}) or {}
    for param in (searchspace.get("discrete", {}) or {}).get("parameters", []) or []:
        param_meta.append(
            {
                "name": param.get("name"),
                "type": param.get("type", "unknown"),
                "values": param.get("values") or param.get("active_values") or param.get("data"),
            }
        )
    for param in (searchspace.get("continuous", {}) or {}).get("parameters", []) or []:
        param_meta.append(
            {
                "name": param.get("name"),
                "type": param.get("type", "NumericalContinuousParameter"),
                "values": param.get("bounds"),
            }
        )

    completed_measurements = -1
    try:
        from baybe.campaign import Campaign

        completed_measurements = int(Campaign.from_json(load_text(path)).measurements.shape[0])
    except Exception:
        completed_measurements = -1

    return {
        "path": path,
        "campaign_name": _campaign_name_from_path(path),
        "objective_target": target.get("name", "unknown"),
        "objective_mode": "minimize" if target.get("minimize") else "maximize",
        "acquisition": acquisition,
        "parameter_names": [param["name"] for param in param_meta if param.get("name")],
        "parameters": param_meta,
        "completed_measurements": completed_measurements,
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime),
    }


class BayBEEngine(CampaignEngine):
    def __init__(self, cfg: CampaignConfig, campaign: Any):
        self.cfg = cfg
        self._campaign = campaign

    @classmethod
    def from_config(cls, cfg: CampaignConfig) -> "BayBEEngine":
        return cls(cfg=cfg, campaign=build_campaign(cfg))

    @classmethod
    def load(cls, path: Path, cfg: CampaignConfig | None = None) -> "BayBEEngine":
        from baybe.campaign import Campaign

        resolved_cfg = cfg or cls._infer_config_from_path(path)
        return cls(cfg=resolved_cfg, campaign=Campaign.from_json(load_text(path)))

    @classmethod
    def _infer_config_from_path(cls, path: Path) -> CampaignConfig:
        meta = extract_baybe_campaign_metadata(path)
        return CampaignConfig(
            campaign_name=meta["campaign_name"],
            objective_target=meta.get("objective_target", "yield"),
            objective_mode=meta.get("objective_mode", "maximize"),
            acquisition=meta.get("acquisition") or "qExpectedImprovement",
            parameters=_build_specs_from_metadata(meta),
        )

    def recommend(self, batch_size: int) -> pd.DataFrame:
        return self._campaign.recommend(batch_size=batch_size)

    def predict(self, candidates: pd.DataFrame) -> pd.DataFrame:
        from baybe.exceptions import ModelNotTrainedError, NoMeasurementsError

        try:
            stats_df = self._campaign.posterior_stats(candidates, stats=("mean", "std"))
            acq_series = self._campaign.acquisition_values(candidates)
        except (ModelNotTrainedError, NoMeasurementsError) as exc:
            raise ModelNotFittedError(
                "Surrogate predictions not available. Ingest at least one completed "
                "measurement and call recommend() before requesting model metadata."
            ) from exc

        out = candidates.copy()

        target = self.cfg.objective_target
        mean_col = f"{target}_mean"
        std_col = f"{target}_std"

        if mean_col in stats_df.columns:
            out["pred_mean"] = stats_df[mean_col].values
        if std_col in stats_df.columns:
            out["pred_std"] = stats_df[std_col].values

        out["acq_score"] = acq_series.values
        # rank 1 = highest acquisition score
        out["rank"] = acq_series.rank(ascending=False, method="first").astype(int)

        return out

    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        if "status" not in df.columns:
            raise ValueError(
                "Missing required 'status' column. Every ingestion event must specify a trial status "
                f"({', '.join(sorted(VALID_TRIAL_STATUSES))})."
            )

        df = df.copy()
        df["status"] = df["status"].astype(str).str.lower().str.strip()
        invalid_statuses = set(df["status"].unique()) - VALID_TRIAL_STATUSES
        if invalid_statuses:
            raise ValueError(
                f"Invalid trial status values: {sorted(invalid_statuses)}. "
                f"Supported values: {sorted(VALID_TRIAL_STATUSES)}."
            )

        param_cols = [param.name for param in self.cfg.parameters]
        target_col = self.cfg.objective_target
        missing = [c for c in (param_cols + [target_col]) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for ingestion: {missing}.")

        df_completed = df[df["status"] == "completed"].copy()
        if df_completed.empty:
            return df

        df_model = df_completed[param_cols + [target_col]].copy()

        def _normalize_for_campaign(series: pd.Series, is_numeric: bool) -> pd.Series:
            if is_numeric:
                return pd.to_numeric(series, errors="coerce")
            return series.astype("object")

        numeric_param_types = (NumericalContinuousSpec, NumericalDiscreteSpec)
        introduced_nulls: list[str] = []
        for param in self.cfg.parameters:
            before = df_model[param.name]
            after = _normalize_for_campaign(before, isinstance(param, numeric_param_types))
            if ((~before.isna()) & after.isna()).any():
                introduced_nulls.append(param.name)
            df_model[param.name] = after

        target_before = df_model[target_col]
        target_after = _normalize_for_campaign(target_before, is_numeric=True)
        if ((~target_before.isna()) & target_after.isna()).any():
            introduced_nulls.append(target_col)
        df_model[target_col] = target_after

        if introduced_nulls:
            raise ValueError(
                "Datatype normalization introduced nulls in required columns "
                f"{sorted(set(introduced_nulls))}."
            )

        self._normalize_searchspace_for_matching()
        normalized = self._ensure_numpy_backed_dataframe(df_model)
        self._campaign.add_measurements(normalized)
        return df

    def save(self, path: Path) -> None:
        save_text(path, self._campaign.to_json())

    def cached_recommendation_info(self) -> Optional[dict[str, Any]]:
        raw = json.loads(self._campaign.to_json())
        blob = raw.get("cached_recommendation")
        if not isinstance(blob, str) or not blob:
            return None

        batch_index = raw.get("n_batches_done")
        try:
            batch_index = int(batch_index)
        except (TypeError, ValueError):
            batch_index = None

        try:
            df = pickle.loads(base64.b64decode(blob))
        except Exception as exc:
            return {
                "dataframe": None,
                "batch_index": batch_index,
                "decode_error": str(exc),
            }

        if not isinstance(df, pd.DataFrame):
            return {
                "dataframe": None,
                "batch_index": batch_index,
                "decode_error": f"Unsupported cached recommendation type: {type(df).__name__}",
            }

        return {
            "dataframe": df.reset_index(drop=True),
            "batch_index": batch_index,
            "decode_error": None,
        }

    def measurement_count(self) -> int:
        measurements = getattr(self._campaign, "measurements", None)
        if isinstance(measurements, pd.DataFrame):
            return int(measurements.shape[0])
        try:
            return int(len(measurements))
        except Exception:
            return -1

    @staticmethod
    def _ensure_numpy_backed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column in out.columns:
            column_data = out[column]
            if pd.api.types.is_extension_array_dtype(column_data.dtype):
                if pd.api.types.is_numeric_dtype(column_data.dtype):
                    out[column] = pd.to_numeric(column_data, errors="coerce")
                else:
                    out[column] = column_data.astype("object")
        return out

    def _normalize_searchspace_for_matching(self) -> None:
        discrete = getattr(getattr(self._campaign, "searchspace", None), "discrete", None)
        exp_rep = getattr(discrete, "exp_rep", None)
        if isinstance(exp_rep, pd.DataFrame):
            discrete.exp_rep = self._ensure_numpy_backed_dataframe(exp_rep)
