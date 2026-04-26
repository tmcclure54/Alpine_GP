"""Tests for Phase 5: Multi-Objective Optimization.

Covers:
  - TargetSpec serialization round-trip
  - CampaignConfig.effective_targets() bridges single + multi modes
  - validate_campaign_config rejects multi-target BayBE configs
  - validate_config_payload accepts new "targets" key
  - AxEngine.from_config wires multiple ObjectiveProperties to AxClient
  - AxEngine.ingest passes all configured target metrics to complete_trial
  - AxEngine.predict returns per-target prediction columns + pareto_rank
  - dashboard.compute_pareto_mask correctly identifies non-dominated points
  - _pareto_ranks frontiers are layered (rank 1 dominates rank 2)
"""
from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from core.baybe_factory import validate_campaign_config, validate_config_payload
from core.campaign_dashboard import compute_pareto_mask
from core.schema import (
    CampaignConfig,
    CategoricalSpec,
    NumericalContinuousSpec,
    TargetSpec,
)


# ---------------------------------------------------------------------------
# Ax stub helpers (mirrors test_ax_engine.py)
# ---------------------------------------------------------------------------

def _build_ax_stubs(
    trial_params: dict | None = None,
    completed_count: int = 0,
    predictions: list | None = None,
):
    if trial_params is None:
        trial_params = {0: {"solvent": "A"}, 1: {"solvent": "B"}}

    mock_trials = {}
    for i in range(completed_count):
        t = MagicMock()
        t.status.name = "COMPLETED"
        mock_trials[i] = t

    mock_exp = MagicMock()
    mock_exp.name = "mo_test"
    mock_exp.trials = mock_trials

    mock_client = MagicMock()
    mock_client.experiment = mock_exp
    mock_client.objective_name = "yield"
    mock_client.objective.minimize = False
    mock_client.get_next_trials.return_value = (trial_params, None)
    mock_client.attach_trial.return_value = (None, 99)

    if predictions is not None:
        mock_client.get_model_predictions_for_parameterizations.return_value = predictions

    return mock_client


@contextmanager
def _patch_ax(mock_client):
    ax_module = types.ModuleType("ax")
    ax_service_module = types.ModuleType("ax.service")
    ax_client_module = types.ModuleType("ax.service.ax_client")
    ax_service_utils_module = types.ModuleType("ax.service.utils")
    ax_instantiation_module = types.ModuleType("ax.service.utils.instantiation")
    ax_core_module = types.ModuleType("ax.core")
    ax_core_param_module = types.ModuleType("ax.core.parameter")

    class ObjectiveProperties:
        def __init__(self, minimize=False):
            self.minimize = minimize

    class RangeParameter:
        pass

    class ChoiceParameter:
        pass

    class ParameterType:
        FLOAT = "FLOAT"
        STRING = "STRING"

    ax_instantiation_module.ObjectiveProperties = ObjectiveProperties
    ax_core_param_module.RangeParameter = RangeParameter
    ax_core_param_module.ChoiceParameter = ChoiceParameter
    ax_core_param_module.ParameterType = ParameterType

    class AxClient:
        def __new__(cls, *args, **kwargs):
            return mock_client

        @classmethod
        def from_json_snapshot(cls, raw, verbose_logging=False):
            return mock_client

    ax_client_module.AxClient = AxClient

    mods = {
        "ax": ax_module,
        "ax.service": ax_service_module,
        "ax.service.ax_client": ax_client_module,
        "ax.service.utils": ax_service_utils_module,
        "ax.service.utils.instantiation": ax_instantiation_module,
        "ax.core": ax_core_module,
        "ax.core.parameter": ax_core_param_module,
    }
    originals = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield mock_client
    finally:
        for k, v in originals.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _mo_cfg(engine: str = "ax") -> CampaignConfig:
    return CampaignConfig(
        campaign_name="mo_test",
        objective_target="yield",
        objective_mode="maximize",
        batch_size=2,
        init_mode="sobol",
        n_init=4,
        engine=engine,
        parameters=[CategoricalSpec(name="solvent", values=["A", "B", "C"])],
        targets=[
            TargetSpec(name="yield", mode="maximize"),
            TargetSpec(name="cost", mode="minimize"),
        ],
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TargetSpecTests(unittest.TestCase):

    def test_target_spec_roundtrip(self):
        t = TargetSpec(name="yield", mode="maximize")
        self.assertEqual(TargetSpec.from_dict(t.to_dict()), t)

    def test_effective_targets_legacy_single(self):
        cfg = CampaignConfig(
            campaign_name="x",
            objective_target="y",
            objective_mode="minimize",
            parameters=[CategoricalSpec(name="s", values=["A", "B"])],
        )
        targets = cfg.effective_targets()
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].name, "y")
        self.assertEqual(targets[0].mode, "minimize")

    def test_effective_targets_explicit_multi(self):
        cfg = _mo_cfg()
        self.assertEqual([t.name for t in cfg.effective_targets()], ["yield", "cost"])
        self.assertTrue(cfg.is_multi_objective())

    def test_to_dict_omits_targets_when_empty(self):
        cfg = CampaignConfig(
            campaign_name="x",
            parameters=[CategoricalSpec(name="s", values=["A", "B"])],
        )
        self.assertNotIn("targets", cfg.to_dict())

    def test_to_dict_includes_targets_when_set(self):
        cfg = _mo_cfg()
        d = cfg.to_dict()
        self.assertIn("targets", d)
        self.assertEqual(d["targets"], [
            {"name": "yield", "mode": "maximize"},
            {"name": "cost", "mode": "minimize"},
        ])

    def test_from_dict_restores_targets(self):
        original = _mo_cfg()
        restored = CampaignConfig.from_dict(original.to_dict())
        self.assertEqual([t.name for t in restored.targets], ["yield", "cost"])
        self.assertEqual([t.mode for t in restored.targets], ["maximize", "minimize"])


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):

    def test_validate_campaign_config_rejects_multi_target_baybe(self):
        cfg = _mo_cfg(engine="baybe")
        errors = validate_campaign_config(cfg)
        self.assertTrue(
            any("multi-objective" in err.lower() for err in errors),
            f"Expected multi-objective rejection error, got: {errors}",
        )

    def test_validate_campaign_config_accepts_multi_target_ax(self):
        cfg = _mo_cfg(engine="ax")
        errors = validate_campaign_config(cfg)
        # The Ax engine path skips acquisition validation, and multi-target is allowed.
        for err in errors:
            self.assertNotIn("multi-objective", err.lower())

    def test_validate_rejects_duplicate_target_names(self):
        cfg = CampaignConfig(
            campaign_name="x",
            engine="ax",
            parameters=[CategoricalSpec(name="s", values=["A", "B"])],
            targets=[
                TargetSpec(name="yield", mode="maximize"),
                TargetSpec(name="yield", mode="minimize"),
            ],
        )
        errors = validate_campaign_config(cfg)
        self.assertTrue(any("unique" in err.lower() for err in errors))

    def test_validate_rejects_unknown_target_mode(self):
        cfg = CampaignConfig(
            campaign_name="x",
            engine="ax",
            parameters=[CategoricalSpec(name="s", values=["A", "B"])],
            targets=[TargetSpec(name="yield", mode="fastest")],
        )
        errors = validate_campaign_config(cfg)
        self.assertTrue(any("mode" in err.lower() for err in errors))

    def test_validate_config_payload_accepts_targets_key(self):
        payload = {
            "campaign_name": "x",
            "objective_target": "yield",
            "engine": "ax",
            "parameters": [
                {"_type": "CategoricalSpec", "name": "s", "values": ["A", "B"], "encoding": "OHE"},
            ],
            "targets": [
                {"name": "yield", "mode": "maximize"},
                {"name": "cost", "mode": "minimize"},
            ],
        }
        errors = validate_config_payload(payload)
        for err in errors:
            self.assertNotIn("Unsupported config keys", err)


# ---------------------------------------------------------------------------
# Ax engine multi-objective tests
# ---------------------------------------------------------------------------

class AxMultiObjectiveTests(unittest.TestCase):

    def test_from_config_passes_multiple_objectives_to_axclient(self):
        cfg = _mo_cfg()
        mock_client = _build_ax_stubs()
        with _patch_ax(mock_client):
            from core.ax_engine import AxEngine
            AxEngine.from_config(cfg)

        call_kwargs = mock_client.create_experiment.call_args[1]
        objectives = call_kwargs["objectives"]
        self.assertEqual(set(objectives.keys()), {"yield", "cost"})
        self.assertFalse(objectives["yield"].minimize)
        self.assertTrue(objectives["cost"].minimize)

    def test_ingest_completed_passes_all_target_metrics(self):
        cfg = _mo_cfg()
        mock_client = _build_ax_stubs()
        with _patch_ax(mock_client):
            from core.ax_engine import AxEngine
            engine = AxEngine(cfg=cfg, client=mock_client, pending={0: {"solvent": "A"}})
            df = pd.DataFrame([{
                "solvent": "A", "yield": 0.7, "cost": 12.0,
                "status": "completed", "trial_index": 0,
            }])
            engine.ingest(df)

        mock_client.complete_trial.assert_called_once()
        call_kwargs = mock_client.complete_trial.call_args[1]
        self.assertEqual(call_kwargs["raw_data"], {
            "yield": (0.7, None),
            "cost": (12.0, None),
        })

    def test_ingest_completed_missing_one_target_raises(self):
        cfg = _mo_cfg()
        mock_client = _build_ax_stubs()
        with _patch_ax(mock_client):
            from core.ax_engine import AxEngine
            engine = AxEngine(cfg=cfg, client=mock_client, pending={0: {"solvent": "A"}})
            df = pd.DataFrame([{
                "solvent": "A", "yield": 0.7, "cost": float("nan"),
                "status": "completed", "trial_index": 0,
            }])
            with self.assertRaises(ValueError) as ctx:
                engine.ingest(df)
        self.assertIn("cost", str(ctx.exception))

    def test_predict_multi_objective_returns_per_target_columns(self):
        cfg = _mo_cfg()
        # 3 candidates, 2 targets; first candidate is on the front for both
        # (high yield, low cost), second candidate is dominated, third trades.
        predictions = [
            {"yield": (0.9, 0.05), "cost": (5.0, 0.5)},
            {"yield": (0.6, 0.05), "cost": (10.0, 0.5)},   # dominated
            {"yield": (0.7, 0.05), "cost": (3.0, 0.5)},
        ]
        mock_client = _build_ax_stubs(predictions=predictions)
        with _patch_ax(mock_client):
            from core.ax_engine import AxEngine
            engine = AxEngine(cfg=cfg, client=mock_client)
            candidates = pd.DataFrame({"solvent": ["A", "B", "C"]})
            result = engine.predict(candidates)

        self.assertIn("yield_pred_mean", result.columns)
        self.assertIn("yield_pred_std", result.columns)
        self.assertIn("cost_pred_mean", result.columns)
        self.assertIn("cost_pred_std", result.columns)
        self.assertIn("pareto_rank", result.columns)
        # First & third are non-dominated; second is dominated by first.
        self.assertEqual(result.iloc[0]["pareto_rank"], 1)
        self.assertEqual(result.iloc[1]["pareto_rank"], 2)
        self.assertEqual(result.iloc[2]["pareto_rank"], 1)

    def test_predict_single_target_unchanged_columns(self):
        # Sanity: removing targets list leaves single-target predict path intact.
        cfg = _mo_cfg()
        cfg.targets = []  # fall back to legacy single objective_target
        predictions = [{"yield": (0.7, 0.05)}, {"yield": (0.9, 0.03)}]
        mock_client = _build_ax_stubs(predictions=predictions)
        with _patch_ax(mock_client):
            from core.ax_engine import AxEngine
            engine = AxEngine(cfg=cfg, client=mock_client)
            candidates = pd.DataFrame({"solvent": ["A", "B"]})
            result = engine.predict(candidates)

        self.assertIn("pred_mean", result.columns)
        self.assertIn("pred_std", result.columns)
        self.assertIn("acq_score", result.columns)
        self.assertIn("rank", result.columns)


# ---------------------------------------------------------------------------
# Pareto front computation tests
# ---------------------------------------------------------------------------

class ParetoMaskTests(unittest.TestCase):

    def test_dashboard_compute_pareto_mask_two_objectives_max_min(self):
        # yield (max), cost (min). A: (0.9, 5) dominates B: (0.6, 10);
        # C: (0.7, 3) trades off vs A.
        df = pd.DataFrame({
            "yield": [0.9, 0.6, 0.7],
            "cost": [5.0, 10.0, 3.0],
        })
        mask = compute_pareto_mask(df, ["yield", "cost"], maximize_flags=[True, False])
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])

    def test_dashboard_compute_pareto_mask_all_maximize(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        mask = compute_pareto_mask(df, ["a", "b"], maximize_flags=[True, True])
        # All three are non-dominated under maximize/maximize trade-off.
        self.assertTrue(mask.all())

    def test_dashboard_compute_pareto_mask_dominated_point_dropped(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
        mask = compute_pareto_mask(df, ["a", "b"], maximize_flags=[True, True])
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])

    def test_pareto_ranks_layered_fronts(self):
        from core.ax_engine import _pareto_ranks
        # Two layers: front 1 = (3,3) and (4,1), front 2 = (2,2)
        points = np.array([[3.0, 3.0], [4.0, 1.0], [2.0, 2.0]])
        ranks = _pareto_ranks(points, np.array([True, True]))
        self.assertEqual(ranks[0], 1)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[2], 2)


if __name__ == "__main__":
    unittest.main()
