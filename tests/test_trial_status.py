"""Tests for Phase 3: Trial Status System.

Verifies that BayBEEngine.ingest:
  - requires a 'status' column
  - rejects invalid status values
  - only passes completed rows to the model
  - records all rows in the returned dataframe
  - normalises status values (case-insensitive)
"""
from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from core.campaign_engine import create_campaign_engine
from core.schema import CampaignConfig, CategoricalSpec, VALID_TRIAL_STATUSES


@contextmanager
def temporary_modules(module_map: dict[str, types.ModuleType]):
    original = {name: sys.modules.get(name) for name in module_map}
    sys.modules.update(module_map)
    try:
        yield
    finally:
        for name, orig in original.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _build_stub_modules() -> dict[str, types.ModuleType]:
    baybe_module = types.ModuleType("baybe")
    baybe_acquisition_module = types.ModuleType("baybe.acquisition")
    baybe_acqfs_module = types.ModuleType("baybe.acquisition.acqfs")
    baybe_campaign_module = types.ModuleType("baybe.campaign")
    baybe_objectives_module = types.ModuleType("baybe.objectives")
    baybe_parameters_module = types.ModuleType("baybe.parameters")
    baybe_parameters_enum_module = types.ModuleType("baybe.parameters.enum")
    baybe_parameters_substance_module = types.ModuleType("baybe.parameters.substance")
    baybe_recommenders_module = types.ModuleType("baybe.recommenders")
    baybe_searchspace_module = types.ModuleType("baybe.searchspace")
    baybe_targets_module = types.ModuleType("baybe.targets")
    rdkit_module = types.ModuleType("rdkit")

    class AcquisitionBase:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class qExpectedImprovement(AcquisitionBase):
        pass

    class BotorchRecommender:
        def __init__(self, acquisition_function):
            self.acquisition_function = acquisition_function

    class SearchSpace:
        def __init__(self, parameters):
            self.parameters = parameters
            self.discrete = types.SimpleNamespace(exp_rep=None)

        @classmethod
        def from_product(cls, parameters):
            return cls(parameters)

    class NumericalTarget:
        def __init__(self, name, mode):
            self.name = name
            self.mode = mode

    class TargetMode:
        MAX = "MAX"
        MIN = "MIN"

    class SingleTargetObjective:
        def __init__(self, target):
            self.target = target

    class CategoricalParameter:
        def __init__(self, name, values, encoding):
            self.name = name
            self.values = values
            self.encoding = encoding

    class NumericalContinuousParameter:
        def __init__(self, name, bounds, metadata=None):
            self.name = name
            self.bounds = bounds

    class NumericalDiscreteParameter:
        def __init__(self, name, values, metadata=None):
            self.name = name
            self.values = values

    class SubstanceEncoding(Enum):
        MORDRED = "MORDRED"

    class SubstanceParameter:
        def __init__(self, name, data, encoding, decorrelate):
            self.name = name

    class Campaign:
        def __init__(self, searchspace, objective, recommender, measurements=None, n_batches_done=0, cached_recommendation=None):
            self.searchspace = searchspace
            self.objective = objective
            self.recommender = recommender
            self.measurements = measurements if measurements is not None else pd.DataFrame()
            self.n_batches_done = n_batches_done
            self._cached_recommendation = cached_recommendation

        def recommend(self, batch_size: int):
            return pd.DataFrame({"solvent": ["A"] * batch_size})

        def add_measurements(self, df: pd.DataFrame):
            if self.measurements.empty:
                self.measurements = df.copy()
            else:
                self.measurements = pd.concat([self.measurements, df], ignore_index=True)

        def to_json(self):
            import json, base64, pickle
            return json.dumps({
                "objective": {"target": {"name": self.objective.target.name, "minimize": False}},
                "recommender": {"acquisition_function": {"type": "qExpectedImprovement"}},
                "searchspace": {"discrete": {"parameters": []}, "continuous": {"parameters": []}},
                "measurements": self.measurements.to_dict(orient="records"),
                "n_batches_done": self.n_batches_done,
                "cached_recommendation": "",
            })

        @classmethod
        def from_json(cls, raw: str):
            import json
            payload = json.loads(raw)
            searchspace = SearchSpace(parameters=[])
            target = NumericalTarget(name=payload["objective"]["target"]["name"], mode="MAX")
            objective = SingleTargetObjective(target=target)
            recommender = BotorchRecommender(acquisition_function=qExpectedImprovement())
            measurements = pd.DataFrame(payload.get("measurements", []))
            return cls(searchspace=searchspace, objective=objective, recommender=recommender, measurements=measurements)

    baybe_acqfs_module.qExpectedImprovement = qExpectedImprovement
    baybe_recommenders_module.BotorchRecommender = BotorchRecommender
    baybe_searchspace_module.SearchSpace = SearchSpace
    baybe_targets_module.NumericalTarget = NumericalTarget
    baybe_targets_module.TargetMode = TargetMode
    baybe_objectives_module.SingleTargetObjective = SingleTargetObjective
    baybe_campaign_module.Campaign = Campaign
    baybe_parameters_module.CategoricalParameter = CategoricalParameter
    baybe_parameters_module.NumericalContinuousParameter = NumericalContinuousParameter
    baybe_parameters_module.NumericalDiscreteParameter = NumericalDiscreteParameter
    baybe_parameters_enum_module.SubstanceEncoding = SubstanceEncoding
    baybe_parameters_substance_module.SubstanceParameter = SubstanceParameter
    rdkit_module.Chem = type("Chem", (), {"MolFromSmiles": staticmethod(lambda s: object() if s else None)})()

    return {
        "baybe": baybe_module,
        "baybe.acquisition": baybe_acquisition_module,
        "baybe.acquisition.acqfs": baybe_acqfs_module,
        "baybe.campaign": baybe_campaign_module,
        "baybe.objectives": baybe_objectives_module,
        "baybe.parameters": baybe_parameters_module,
        "baybe.parameters.enum": baybe_parameters_enum_module,
        "baybe.parameters.substance": baybe_parameters_substance_module,
        "baybe.recommenders": baybe_recommenders_module,
        "baybe.searchspace": baybe_searchspace_module,
        "baybe.targets": baybe_targets_module,
        "rdkit": rdkit_module,
    }


def _make_cfg() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective_target="yield",
        batch_size=1,
        acquisition="qExpectedImprovement",
        parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
    )


def _row(solvent: str, yield_val: float | None, status: str) -> dict:
    return {"solvent": solvent, "yield": yield_val, "status": status}


class TrialStatusTests(unittest.TestCase):

    def test_ingest_without_status_column_raises(self):
        cfg = _make_cfg()
        df = pd.DataFrame({"solvent": ["A"], "yield": [0.5]})
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            with self.assertRaises(ValueError) as ctx:
                engine.ingest(df)
        self.assertIn("status", str(ctx.exception))

    def test_ingest_with_invalid_status_raises(self):
        cfg = _make_cfg()
        df = pd.DataFrame([_row("A", 0.5, "success")])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            with self.assertRaises(ValueError) as ctx:
                engine.ingest(df)
        self.assertIn("success", str(ctx.exception))

    def test_ingest_all_completed_adds_all_to_model(self):
        cfg = _make_cfg()
        df = pd.DataFrame([_row("A", 0.5, "completed"), _row("B", 0.8, "completed")])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            engine.ingest(df)
            self.assertEqual(engine.measurement_count(), 2)

    def test_ingest_all_failed_adds_none_to_model(self):
        cfg = _make_cfg()
        df = pd.DataFrame([_row("A", None, "failed"), _row("B", None, "failed")])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            engine.ingest(df)
            self.assertEqual(engine.measurement_count(), 0)

    def test_ingest_mixed_status_only_completed_reaches_model(self):
        cfg = _make_cfg()
        df = pd.DataFrame([
            _row("A", 0.7, "completed"),
            _row("B", None, "failed"),
            _row("A", None, "abandoned"),
        ])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            engine.ingest(df)
            self.assertEqual(engine.measurement_count(), 1)

    def test_ingest_abandoned_and_partial_excluded_from_model(self):
        cfg = _make_cfg()
        df = pd.DataFrame([
            _row("A", None, "abandoned"),
            _row("B", None, "partial"),
            _row("A", None, "invalid"),
        ])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            engine.ingest(df)
            self.assertEqual(engine.measurement_count(), 0)

    def test_ingest_returns_all_rows_including_non_completed(self):
        cfg = _make_cfg()
        df = pd.DataFrame([
            _row("A", 0.6, "completed"),
            _row("B", None, "failed"),
        ])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            result = engine.ingest(df)
        self.assertEqual(len(result), 2)
        self.assertIn("status", result.columns)

    def test_ingest_status_normalised_case_insensitive(self):
        cfg = _make_cfg()
        df = pd.DataFrame([_row("A", 0.5, "COMPLETED"), _row("B", None, "FAILED")])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            result = engine.ingest(df)
            self.assertEqual(engine.measurement_count(), 1)
        self.assertTrue((result["status"] == result["status"].str.lower()).all())

    def test_valid_trial_statuses_constant_is_complete(self):
        expected = {"completed", "failed", "abandoned", "partial", "invalid"}
        self.assertEqual(VALID_TRIAL_STATUSES, expected)

    def test_ingest_cumulative_only_counts_completed(self):
        """Two separate ingest calls: first all-failed, then one completed."""
        cfg = _make_cfg()
        df_failed = pd.DataFrame([_row("A", None, "failed")])
        df_completed = pd.DataFrame([_row("B", 0.9, "completed")])
        with temporary_modules(_build_stub_modules()):
            engine = create_campaign_engine(cfg)
            engine.ingest(df_failed)
            self.assertEqual(engine.measurement_count(), 0)
            engine.ingest(df_completed)
            self.assertEqual(engine.measurement_count(), 1)


if __name__ == "__main__":
    unittest.main()
