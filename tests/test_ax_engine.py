"""Tests for Phase 2: AxEngine.

Covers:
  - from_config creates a valid AxEngine
  - recommend returns a DataFrame with parameter columns and trial_index
  - ingest completed rows calls complete_trial
  - ingest failed/abandoned/partial/invalid rows are excluded from model
  - save / load round-trip preserves pending trials
  - predict raises ModelNotFittedError during Sobol phase
  - predict returns required columns after measurements
  - warm-start (existing_data init mode) attaches trials
  - measurement_count counts only COMPLETED trials
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from core.campaign_engine import ModelNotFittedError
from core.schema import (
    CampaignConfig,
    CategoricalSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
)


def _simple_cfg(engine: str = "ax") -> CampaignConfig:
    return CampaignConfig(
        campaign_name="ax_test",
        objective_target="yield",
        objective_mode="maximize",
        batch_size=2,
        init_mode="sobol",
        n_init=4,
        engine=engine,
        parameters=[CategoricalSpec(name="solvent", values=["A", "B", "C"])],
    )


def _build_ax_stubs(
    trial_params: dict | None = None,
    completed_count: int = 0,
    predictions: list | None = None,
    sobol_phase: bool = False,
) -> tuple[types.ModuleType, MagicMock]:
    """Return (ax_module_stub, mock_client) for patching ax imports."""
    if trial_params is None:
        trial_params = {0: {"solvent": "A"}, 1: {"solvent": "B"}}

    ax_module = types.ModuleType("ax")
    ax_service_module = types.ModuleType("ax.service")
    ax_client_module = types.ModuleType("ax.service.ax_client")
    ax_instantiation_module = types.ModuleType("ax.service.utils.instantiation")
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

    # Build mock trial statuses
    mock_trials = {}
    for i in range(completed_count):
        t = MagicMock()
        t.status.name = "COMPLETED"
        mock_trials[i] = t
    for i in range(completed_count, completed_count + 2):
        t = MagicMock()
        t.status.name = "RUNNING"
        mock_trials[i] = t

    mock_exp = MagicMock()
    mock_exp.name = "ax_test"
    mock_exp.trials = mock_trials

    mock_client = MagicMock()
    mock_client.experiment = mock_exp
    mock_client.objective_name = "yield"
    mock_client.objective.minimize = False
    mock_client.get_next_trials.return_value = (trial_params, None)
    mock_client.attach_trial.return_value = (None, 99)

    if sobol_phase:
        mock_client.get_model_predictions_for_parameterizations.side_effect = Exception(
            "Not in BO phase"
        )
    elif predictions is not None:
        mock_client.get_model_predictions_for_parameterizations.return_value = predictions
    else:
        mock_client.get_model_predictions_for_parameterizations.return_value = [
            {"yield": (0.7, 0.05)},
            {"yield": (0.8, 0.03)},
        ]

    class AxClient:
        @classmethod
        def from_json_snapshot(cls, raw, verbose_logging=False):
            return mock_client

        def __new__(cls, *args, **kwargs):
            return mock_client

    ax_client_module.AxClient = AxClient
    ax_module.service = ax_service_module
    ax_service_module.ax_client = ax_client_module

    return ax_module, mock_client


@contextmanager
def _patch_ax(ax_module, mock_client):
    mods = {
        "ax": ax_module,
        "ax.service": types.ModuleType("ax.service"),
        "ax.service.ax_client": _get_client_module(mock_client),
        "ax.service.utils": types.ModuleType("ax.service.utils"),
        "ax.service.utils.instantiation": _get_instantiation_module(),
        "ax.core": types.ModuleType("ax.core"),
        "ax.core.parameter": _get_parameter_module(),
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


def _get_client_module(mock_client):
    mod = types.ModuleType("ax.service.ax_client")

    class AxClient:
        def __new__(cls, *args, **kwargs):
            return mock_client

        @classmethod
        def from_json_snapshot(cls, raw, verbose_logging=False):
            return mock_client

    mod.AxClient = AxClient
    return mod


def _get_instantiation_module():
    mod = types.ModuleType("ax.service.utils.instantiation")

    class ObjectiveProperties:
        def __init__(self, minimize=False):
            self.minimize = minimize

    mod.ObjectiveProperties = ObjectiveProperties
    return mod


def _get_parameter_module():
    mod = types.ModuleType("ax.core.parameter")

    class RangeParameter:
        pass

    class ChoiceParameter:
        pass

    class ParameterType:
        FLOAT = "FLOAT"
        STRING = "STRING"

    mod.RangeParameter = RangeParameter
    mod.ChoiceParameter = ChoiceParameter
    mod.ParameterType = ParameterType
    return mod


def _make_engine(cfg=None, pending=None, completed_count=0, sobol_phase=False, predictions=None):
    """Construct an AxEngine with mocked Ax client."""
    if cfg is None:
        cfg = _simple_cfg()
    trial_params = {0: {"solvent": "A"}, 1: {"solvent": "B"}}
    ax_mod, mock_client = _build_ax_stubs(
        trial_params=trial_params,
        completed_count=completed_count,
        predictions=predictions,
        sobol_phase=sobol_phase,
    )
    with _patch_ax(ax_mod, mock_client):
        from core.ax_engine import AxEngine
        engine = AxEngine(cfg=cfg, client=mock_client, pending=pending or {})
    return engine, mock_client


class TestAxEngineFromConfig(unittest.TestCase):

    def test_from_config_returns_ax_engine(self):
        cfg = _simple_cfg()
        ax_mod, mock_client = _build_ax_stubs()
        with _patch_ax(ax_mod, mock_client):
            from core.ax_engine import AxEngine
            engine = AxEngine.from_config(cfg)
        self.assertIsInstance(engine, AxEngine)
        mock_client.create_experiment.assert_called_once()

    def test_from_config_existing_data_sets_zero_init_trials(self):
        cfg = _simple_cfg()
        cfg.init_mode = "existing_data"
        ax_mod, mock_client = _build_ax_stubs()
        with _patch_ax(ax_mod, mock_client):
            from core.ax_engine import AxEngine
            AxEngine.from_config(cfg)
        call_kwargs = mock_client.create_experiment.call_args[1]
        gen_kwargs = call_kwargs.get("choose_generation_strategy_kwargs") or {}
        self.assertEqual(gen_kwargs.get("num_initialization_trials"), 0)


class TestAxEngineRecommend(unittest.TestCase):

    def test_recommend_returns_dataframe_with_trial_index(self):
        engine, mock_client = _make_engine()
        ax_mod, mock_client2 = _build_ax_stubs(trial_params={0: {"solvent": "A"}, 1: {"solvent": "B"}})
        engine._client = mock_client2
        result = engine.recommend(2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("trial_index", result.columns)
        self.assertIn("solvent", result.columns)

    def test_recommend_updates_pending(self):
        engine, mock_client = _make_engine()
        ax_mod, mock_client2 = _build_ax_stubs(trial_params={0: {"solvent": "A"}, 1: {"solvent": "B"}})
        engine._client = mock_client2
        engine.recommend(2)
        self.assertIn(0, engine._pending)
        self.assertIn(1, engine._pending)

    def test_recommend_raises_if_no_trials_returned(self):
        engine, mock_client = _make_engine()
        engine._client.get_next_trials.return_value = ({}, None)
        with self.assertRaises(RuntimeError):
            engine.recommend(2)


class TestAxEngineIngest(unittest.TestCase):

    def test_ingest_completed_calls_complete_trial(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": 0.8, "status": "completed", "trial_index": 0}])
        engine.ingest(df)
        mock_client.complete_trial.assert_called_once_with(
            trial_index=0, raw_data={"yield": (0.8, None)}
        )

    def test_ingest_failed_calls_log_trial_failure(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": float("nan"), "status": "failed", "trial_index": 0}])
        engine.ingest(df)
        mock_client.log_trial_failure.assert_called_once()

    def test_ingest_abandoned_calls_abandon_trial(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": float("nan"), "status": "abandoned", "trial_index": 0}])
        engine.ingest(df)
        mock_client.abandon_trial.assert_called_once_with(trial_index=0, reason="abandoned")

    def test_ingest_partial_calls_log_trial_failure(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": float("nan"), "status": "partial", "trial_index": 0}])
        engine.ingest(df)
        mock_client.log_trial_failure.assert_called_once()

    def test_ingest_invalid_calls_log_trial_failure(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": float("nan"), "status": "invalid", "trial_index": 0}])
        engine.ingest(df)
        mock_client.log_trial_failure.assert_called_once()

    def test_ingest_missing_status_raises(self):
        engine, _ = _make_engine()
        df = pd.DataFrame([{"solvent": "A", "yield": 0.8}])
        with self.assertRaises(ValueError, msg="Missing 'status' column"):
            engine.ingest(df)

    def test_ingest_invalid_status_raises(self):
        engine, _ = _make_engine()
        df = pd.DataFrame([{"solvent": "A", "yield": 0.8, "status": "done"}])
        with self.assertRaises(ValueError):
            engine.ingest(df)

    def test_ingest_clears_pending(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": 0.8, "status": "completed", "trial_index": 0}])
        engine.ingest(df)
        self.assertNotIn(0, engine._pending)

    def test_ingest_returns_all_rows(self):
        engine, mock_client = _make_engine(pending={0: {"solvent": "A"}, 1: {"solvent": "B"}})
        df = pd.DataFrame([
            {"solvent": "A", "yield": 0.8, "status": "completed", "trial_index": 0},
            {"solvent": "B", "yield": float("nan"), "status": "failed", "trial_index": 1},
        ])
        result = engine.ingest(df)
        self.assertEqual(len(result), 2)

    def test_ingest_completed_missing_target_raises(self):
        engine, _ = _make_engine(pending={0: {"solvent": "A"}})
        df = pd.DataFrame([{"solvent": "A", "yield": float("nan"), "status": "completed", "trial_index": 0}])
        with self.assertRaises(ValueError):
            engine.ingest(df)


class TestAxEnginePredict(unittest.TestCase):

    def test_predict_raises_model_not_fitted_in_sobol_phase(self):
        engine, _ = _make_engine(sobol_phase=True)
        candidates = pd.DataFrame({"solvent": ["A", "B"]})
        with self.assertRaises(ModelNotFittedError):
            engine.predict(candidates)

    def test_predict_returns_required_columns(self):
        engine, _ = _make_engine(
            predictions=[{"yield": (0.7, 0.05)}, {"yield": (0.8, 0.03)}]
        )
        candidates = pd.DataFrame({"solvent": ["A", "B"]})
        result = engine.predict(candidates)
        for col in ("pred_mean", "pred_std", "acq_score", "rank"):
            self.assertIn(col, result.columns)

    def test_predict_rank_1_has_highest_acq_score(self):
        engine, _ = _make_engine(
            predictions=[{"yield": (0.7, 0.05)}, {"yield": (0.9, 0.03)}, {"yield": (0.5, 0.1)}]
        )
        candidates = pd.DataFrame({"solvent": ["A", "B", "C"]})
        result = engine.predict(candidates)
        rank1_row = result[result["rank"] == 1].iloc[0]
        self.assertEqual(rank1_row["acq_score"], result["acq_score"].max())

    def test_predict_all_ranks_unique(self):
        engine, _ = _make_engine(
            predictions=[{"yield": (0.7, 0.05)}, {"yield": (0.8, 0.03)}, {"yield": (0.6, 0.07)}]
        )
        candidates = pd.DataFrame({"solvent": ["A", "B", "C"]})
        result = engine.predict(candidates)
        ranks = sorted(result["rank"].tolist())
        self.assertEqual(ranks, list(range(1, len(candidates) + 1)))

    def test_predict_pred_std_non_negative(self):
        engine, _ = _make_engine(
            predictions=[{"yield": (0.7, 0.05)}, {"yield": (0.8, 0.03)}]
        )
        candidates = pd.DataFrame({"solvent": ["A", "B"]})
        result = engine.predict(candidates)
        self.assertTrue((result["pred_std"] >= 0).all())


class TestAxEngineSaveLoad(unittest.TestCase):

    def test_save_load_round_trip_preserves_pending(self):
        engine, mock_client = _make_engine(pending={3: {"solvent": "C"}})
        snapshot = {"experiment": {}, "_engine": "ax"}
        mock_client.to_json_snapshot.return_value = snapshot.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_campaign.json"
            engine.save(path)

            ax_mod, mock_client2 = _build_ax_stubs()
            with _patch_ax(ax_mod, mock_client2):
                from core.ax_engine import AxEngine
                loaded = AxEngine.load(path)

        self.assertIn(3, loaded._pending)
        self.assertEqual(loaded._pending[3]["solvent"], "C")

    def test_save_writes_engine_marker(self):
        engine, mock_client = _make_engine()
        mock_client.to_json_snapshot.return_value = {"experiment": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_campaign.json"
            engine.save(path)
            raw = json.loads(path.read_text())

        self.assertEqual(raw.get("_engine"), "ax")


class TestAxEngineMeasurementCount(unittest.TestCase):

    def test_measurement_count_returns_completed_count(self):
        engine, mock_client = _make_engine(completed_count=3)
        self.assertEqual(engine.measurement_count(), 3)

    def test_measurement_count_zero_when_no_completed(self):
        engine, mock_client = _make_engine(completed_count=0)
        self.assertEqual(engine.measurement_count(), 0)


class TestAxEngineCachedRecommendation(unittest.TestCase):

    def test_cached_recommendation_info_returns_none_when_empty(self):
        engine, _ = _make_engine(pending={})
        self.assertIsNone(engine.cached_recommendation_info())

    def test_cached_recommendation_info_returns_dict_with_dataframe(self):
        engine, _ = _make_engine(pending={0: {"solvent": "A"}, 1: {"solvent": "B"}})
        result = engine.cached_recommendation_info()
        self.assertIsNotNone(result)
        self.assertIn("dataframe", result)
        self.assertIsInstance(result["dataframe"], pd.DataFrame)
        self.assertEqual(len(result["dataframe"]), 2)


class TestAxEngineWarmStart(unittest.TestCase):

    def test_ingest_warm_start_attaches_new_trial(self):
        """Rows with no matching pending trial should attach a new trial via attach_trial."""
        engine, mock_client = _make_engine(pending={})
        df = pd.DataFrame([{"solvent": "A", "yield": 0.75, "status": "completed"}])
        engine.ingest(df)
        mock_client.attach_trial.assert_called_once()


class TestFactoryRouting(unittest.TestCase):

    def test_create_campaign_engine_routes_to_ax(self):
        cfg = _simple_cfg(engine="ax")
        ax_mod, mock_client = _build_ax_stubs()
        with _patch_ax(ax_mod, mock_client):
            from core.campaign_engine import create_campaign_engine
            from core.ax_engine import AxEngine
            engine = create_campaign_engine(cfg)
        self.assertIsInstance(engine, AxEngine)

    def test_create_campaign_engine_routes_to_baybe_by_default(self):
        """Default engine routing must not touch ax imports."""
        import sys
        from core.campaign_engine import create_campaign_engine

        cfg = _simple_cfg(engine="baybe")

        baybe_mod = types.ModuleType("baybe")
        baybe_campaign_mod = types.ModuleType("baybe.campaign")
        baybe_acquisition_mod = types.ModuleType("baybe.acquisition")
        baybe_acqfs_mod = types.ModuleType("baybe.acquisition.acqfs")
        baybe_objectives_mod = types.ModuleType("baybe.objectives")
        baybe_parameters_mod = types.ModuleType("baybe.parameters")
        baybe_parameters_enum_mod = types.ModuleType("baybe.parameters.enum")
        baybe_parameters_substance_mod = types.ModuleType("baybe.parameters.substance")
        baybe_recommenders_mod = types.ModuleType("baybe.recommenders")
        baybe_searchspace_mod = types.ModuleType("baybe.searchspace")
        baybe_targets_mod = types.ModuleType("baybe.targets")
        baybe_exceptions_mod = types.ModuleType("baybe.exceptions")

        class FakeCampaign:
            def __init__(self, **kwargs): pass

        class FakeSearchSpace:
            def __init__(self, parameters): self.parameters = parameters
            @classmethod
            def from_product(cls, parameters): return cls(parameters)

        class FakeNumericalTarget:
            def __init__(self, name, mode): pass

        class FakeTargetMode:
            MAX = "MAX"; MIN = "MIN"

        class FakeSingleTargetObjective:
            def __init__(self, target): pass

        class FakeBotorchRecommender:
            def __init__(self, acquisition_function): pass

        class FakeqEI:
            def __init__(self, **kwargs): pass

        class FakeCategoricalParameter:
            def __init__(self, **kwargs): pass

        class FakeModelNotTrainedError(Exception): pass
        class FakeNoMeasurementsError(Exception): pass

        baybe_campaign_mod.Campaign = FakeCampaign
        baybe_searchspace_mod.SearchSpace = FakeSearchSpace
        baybe_targets_mod.NumericalTarget = FakeNumericalTarget
        baybe_targets_mod.TargetMode = FakeTargetMode
        baybe_objectives_mod.SingleTargetObjective = FakeSingleTargetObjective
        baybe_recommenders_mod.BotorchRecommender = FakeBotorchRecommender
        baybe_acqfs_mod.qExpectedImprovement = FakeqEI
        baybe_parameters_mod.CategoricalParameter = FakeCategoricalParameter
        baybe_parameters_mod.NumericalContinuousParameter = type("NCP", (), {"__init__": lambda s, **kw: None})
        baybe_parameters_mod.NumericalDiscreteParameter = type("NDP", (), {"__init__": lambda s, **kw: None})
        from enum import Enum
        class FakeSubstanceEncoding(Enum): MORDRED = "MORDRED"
        baybe_parameters_enum_mod.SubstanceEncoding = FakeSubstanceEncoding
        baybe_parameters_substance_mod.SubstanceParameter = type("SP", (), {"__init__": lambda s, **kw: None})
        baybe_exceptions_mod.ModelNotTrainedError = FakeModelNotTrainedError
        baybe_exceptions_mod.NoMeasurementsError = FakeNoMeasurementsError

        mods = {
            "baybe": baybe_mod,
            "baybe.acquisition": baybe_acquisition_mod,
            "baybe.acquisition.acqfs": baybe_acqfs_mod,
            "baybe.campaign": baybe_campaign_mod,
            "baybe.exceptions": baybe_exceptions_mod,
            "baybe.objectives": baybe_objectives_mod,
            "baybe.parameters": baybe_parameters_mod,
            "baybe.parameters.enum": baybe_parameters_enum_mod,
            "baybe.parameters.substance": baybe_parameters_substance_mod,
            "baybe.recommenders": baybe_recommenders_mod,
            "baybe.searchspace": baybe_searchspace_mod,
            "baybe.targets": baybe_targets_mod,
        }
        originals = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            from core.baybe_engine import BayBEEngine
            engine = create_campaign_engine(cfg)
            self.assertIsInstance(engine, BayBEEngine)
        finally:
            for k, v in originals.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
