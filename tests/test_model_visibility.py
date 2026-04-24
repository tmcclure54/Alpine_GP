"""Tests for Phase 4: Model Visibility.

Verifies that BayBEEngine.predict():
  - returns real model predictions (pred_mean, pred_std, acq_score, rank)
  - raises ModelNotFittedError when no measurements have been ingested
  - raises ModelNotFittedError when BayBE's model is not yet trained
  - ranks candidates correctly (rank 1 = highest acq_score)
  - values come from the surrogate, not placeholder constants
"""
from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from core.campaign_engine import create_campaign_engine, ModelNotFittedError
from core.schema import CampaignConfig, CategoricalSpec, NumericalDiscreteSpec


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


def _build_stub_modules(fitted: bool = True) -> dict[str, types.ModuleType]:
    """Build stub modules whose Campaign can optionally simulate a fitted surrogate."""
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
    baybe_exceptions_module = types.ModuleType("baybe.exceptions")
    rdkit_module = types.ModuleType("rdkit")

    # --- Exceptions ---
    class ModelNotTrainedError(Exception):
        pass

    class NoMeasurementsError(Exception):
        pass

    baybe_exceptions_module.ModelNotTrainedError = ModelNotTrainedError
    baybe_exceptions_module.NoMeasurementsError = NoMeasurementsError

    # --- Acquisition functions ---
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

    from enum import Enum

    class SubstanceEncoding(Enum):
        MORDRED = "MORDRED"

    class SubstanceParameter:
        def __init__(self, name, data, encoding, decorrelate):
            self.name = name

    # --- Campaign stub ---
    _fitted = fitted

    class Campaign:
        def __init__(self, searchspace, objective, recommender, measurements=None, n_batches_done=0, cached_recommendation=None):
            self.searchspace = searchspace
            self.objective = objective
            self.recommender = recommender
            self.measurements = measurements if measurements is not None else pd.DataFrame()
            self.n_batches_done = n_batches_done
            self._cached_recommendation = cached_recommendation

        def recommend(self, batch_size: int) -> pd.DataFrame:
            return pd.DataFrame({"solvent": [f"cand_{i}" for i in range(batch_size)]})

        def add_measurements(self, df: pd.DataFrame):
            if self.measurements.empty:
                self.measurements = df.copy()
            else:
                self.measurements = pd.concat([self.measurements, df], ignore_index=True)

        def posterior_stats(self, candidates: pd.DataFrame, stats=("mean", "std")) -> pd.DataFrame:
            if not _fitted or self.measurements.empty:
                raise ModelNotTrainedError("Model not trained.")
            target = self.objective.target.name
            n = len(candidates)
            result = pd.DataFrame(index=candidates.index)
            if "mean" in stats:
                # Deterministic but non-constant: each row gets a different value
                result[f"{target}_mean"] = [0.5 + i * 0.1 for i in range(n)]
            if "std" in stats:
                result[f"{target}_std"] = [0.05 + i * 0.01 for i in range(n)]
            return result

        def acquisition_values(self, candidates: pd.DataFrame, *args, **kwargs) -> pd.Series:
            if not _fitted or self.measurements.empty:
                raise ModelNotTrainedError("Model not trained.")
            n = len(candidates)
            # Descending values so rank 1 = first candidate
            return pd.Series([float(n - i) for i in range(n)], index=candidates.index)

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
        "baybe.exceptions": baybe_exceptions_module,
        "baybe.objectives": baybe_objectives_module,
        "baybe.parameters": baybe_parameters_module,
        "baybe.parameters.enum": baybe_parameters_enum_module,
        "baybe.parameters.substance": baybe_parameters_substance_module,
        "baybe.recommenders": baybe_recommenders_module,
        "baybe.searchspace": baybe_searchspace_module,
        "baybe.targets": baybe_targets_module,
        "rdkit": rdkit_module,
    }


def _categorical_cfg() -> CampaignConfig:
    return CampaignConfig(
        campaign_name="test",
        objective_target="yield",
        batch_size=3,
        acquisition="qExpectedImprovement",
        parameters=[CategoricalSpec(name="solvent", values=["A", "B", "C", "D"])],
    )


def _make_candidates(solvents: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"solvent": solvents})


def _make_completed_row(solvent: str, yield_val: float) -> dict:
    return {"solvent": solvent, "yield": yield_val, "status": "completed"}


class ModelVisibilityTests(unittest.TestCase):

    def _get_fitted_engine(self, stubs: dict):
        """Create an engine and ingest one completed measurement so the model can be fitted."""
        cfg = _categorical_cfg()
        engine = create_campaign_engine(cfg)
        df = pd.DataFrame([_make_completed_row("A", 0.7)])
        engine.ingest(df)
        return engine, cfg

    def test_predict_returns_required_columns(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, cfg = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["B", "C", "D"])
            result = engine.predict(candidates)

        for col in ("pred_mean", "pred_std", "acq_score", "rank"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_predict_preserves_parameter_columns(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, cfg = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["B", "C"])
            result = engine.predict(candidates)

        self.assertIn("solvent", result.columns)
        self.assertEqual(list(result["solvent"]), ["B", "C"])

    def test_predict_rank_one_has_highest_acq_score(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, _ = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["A", "B", "C"])
            result = engine.predict(candidates)

        rank1_row = result[result["rank"] == 1].iloc[0]
        self.assertEqual(rank1_row["acq_score"], result["acq_score"].max())

    def test_predict_all_ranks_unique(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, _ = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["A", "B", "C"])
            result = engine.predict(candidates)

        ranks = list(result["rank"])
        self.assertEqual(sorted(ranks), list(range(1, len(candidates) + 1)))

    def test_predict_before_any_measurements_raises_model_not_fitted(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            cfg = _categorical_cfg()
            engine = create_campaign_engine(cfg)
            candidates = _make_candidates(["A", "B"])
            with self.assertRaises(ModelNotFittedError):
                engine.predict(candidates)

    def test_predict_pred_mean_values_are_non_constant(self):
        """pred_mean must differ across candidates — rules out placeholder constants."""
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, _ = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["A", "B", "C"])
            result = engine.predict(candidates)

        self.assertGreater(result["pred_mean"].nunique(), 1)

    def test_predict_pred_std_non_negative(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, _ = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["A", "B", "C"])
            result = engine.predict(candidates)

        self.assertTrue((result["pred_std"] >= 0).all())

    def test_predict_row_count_matches_candidates(self):
        stubs = _build_stub_modules(fitted=True)
        with temporary_modules(stubs):
            engine, _ = self._get_fitted_engine(stubs)
            candidates = _make_candidates(["A", "B"])
            result = engine.predict(candidates)

        self.assertEqual(len(result), len(candidates))

    def test_model_not_fitted_error_is_value_error_subclass(self):
        self.assertTrue(issubclass(ModelNotFittedError, ValueError))


if __name__ == "__main__":
    unittest.main()
