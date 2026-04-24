import base64
import json
import pickle
import sys
import types
import unittest
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from core.campaign_engine import CampaignEngine, create_campaign_engine, extract_saved_campaign_metadata, load_campaign_engine
from core.schema import CampaignConfig, CategoricalSpec


@contextmanager
def temporary_modules(module_map: dict[str, types.ModuleType]):
    original_modules = {name: sys.modules.get(name) for name in module_map}
    sys.modules.update(module_map)
    try:
        yield
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def build_engine_stub_modules() -> dict[str, types.ModuleType]:
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
            for key, value in kwargs.items():
                setattr(self, key, value)

    class qExpectedImprovement(AcquisitionBase):
        pass

    class UpperConfidenceBound(AcquisitionBase):
        def __init__(self, beta: float = 1.0):
            super().__init__(beta=beta)

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
            self.metadata = metadata

    class NumericalDiscreteParameter:
        def __init__(self, name, values, metadata=None):
            self.name = name
            self.values = values
            self.metadata = metadata

    class SubstanceEncoding(Enum):
        MORDRED = "MORDRED"

    class SubstanceParameter:
        def __init__(self, name, data, encoding, decorrelate):
            self.name = name
            self.data = data
            self.encoding = encoding
            self.decorrelate = decorrelate

    class Campaign:
        def __init__(self, searchspace, objective, recommender, measurements=None, n_batches_done=0, cached_recommendation=None):
            self.searchspace = searchspace
            self.objective = objective
            self.recommender = recommender
            self.measurements = measurements if measurements is not None else pd.DataFrame()
            self.n_batches_done = n_batches_done
            self._cached_recommendation = cached_recommendation

        def recommend(self, batch_size: int):
            acqf = self.recommender.acquisition_function
            token = f"{type(acqf).__name__}:{getattr(acqf, 'beta', 'default')}"
            df = pd.DataFrame(
                {
                    "solvent": [f"{token}:{index}" for index in range(batch_size)],
                }
            )
            self._cached_recommendation = df.copy()
            self.n_batches_done += 1
            return df

        def add_measurements(self, df: pd.DataFrame):
            if self.measurements.empty:
                self.measurements = df.copy()
            else:
                self.measurements = pd.concat([self.measurements, df], ignore_index=True)

        def to_json(self):
            discrete_parameters = []
            continuous_parameters = []
            for parameter in self.searchspace.parameters:
                entry = {"name": parameter.name, "type": type(parameter).__name__}
                if hasattr(parameter, "values"):
                    entry["values"] = list(parameter.values)
                    discrete_parameters.append(entry)
                elif hasattr(parameter, "bounds"):
                    entry["bounds"] = list(parameter.bounds)
                    continuous_parameters.append(entry)
            cached_blob = ""
            if self._cached_recommendation is not None:
                cached_blob = base64.b64encode(pickle.dumps(self._cached_recommendation)).decode("ascii")
            payload = {
                "objective": {
                    "target": {
                        "name": self.objective.target.name,
                        "minimize": self.objective.target.mode == "MIN",
                    }
                },
                "recommender": {
                    "acquisition_function": {
                        "type": type(self.recommender.acquisition_function).__name__,
                    }
                },
                "searchspace": {
                    "discrete": {"parameters": discrete_parameters},
                    "continuous": {"parameters": continuous_parameters},
                },
                "measurements": self.measurements.to_dict(orient="records"),
                "n_batches_done": self.n_batches_done,
                "cached_recommendation": cached_blob,
            }
            return json.dumps(payload)

        @classmethod
        def from_json(cls, raw: str):
            payload = json.loads(raw)
            searchspace = SearchSpace(parameters=[])
            target = NumericalTarget(
                name=payload["objective"]["target"]["name"],
                mode="MIN" if payload["objective"]["target"].get("minimize") else "MAX",
            )
            objective = SingleTargetObjective(target=target)
            acq_name = payload["recommender"]["acquisition_function"]["type"]
            acquisition_class = getattr(baybe_acqfs_module, acq_name)
            recommender = BotorchRecommender(acquisition_function=acquisition_class())
            cached = None
            blob = payload.get("cached_recommendation")
            if blob:
                cached = pickle.loads(base64.b64decode(blob))
            measurements = pd.DataFrame(payload.get("measurements", []))
            return cls(
                searchspace=searchspace,
                objective=objective,
                recommender=recommender,
                measurements=measurements,
                n_batches_done=int(payload.get("n_batches_done", 0)),
                cached_recommendation=cached,
            )

    class ChemModule:
        @staticmethod
        def MolFromSmiles(value):
            return object() if value else None

    baybe_acqfs_module.qExpectedImprovement = qExpectedImprovement
    baybe_acqfs_module.UpperConfidenceBound = UpperConfidenceBound
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
    rdkit_module.Chem = ChemModule

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


class CampaignEngineTests(unittest.TestCase):
    def test_create_campaign_engine_returns_interface(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            batch_size=1,
            acquisition="qExpectedImprovement",
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )

        with temporary_modules(build_engine_stub_modules()):
            engine = create_campaign_engine(cfg)

        self.assertIsInstance(engine, CampaignEngine)

    def test_baybe_engine_save_load_and_cached_recommendation_roundtrip(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            batch_size=2,
            acquisition="qExpectedImprovement",
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )

        with temporary_modules(build_engine_stub_modules()):
            engine = create_campaign_engine(cfg)
            rec = engine.recommend(batch_size=2)

            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "campaign.json"
                engine.save(path)
                loaded = load_campaign_engine(path, cfg)
                cached = loaded.cached_recommendation_info()

        self.assertEqual(list(rec.columns), ["solvent"])
        self.assertIsNotNone(cached)
        self.assertEqual(cached["dataframe"].shape, rec.shape)
        self.assertEqual(cached["batch_index"], 1)

    def test_baybe_engine_ingest_returns_normalized_dataframe_and_updates_measurements(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            objective_target="yield",
            batch_size=1,
            acquisition="qExpectedImprovement",
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )
        df = pd.DataFrame(
            {
                "solvent": pd.Series(["A"], dtype="string"),
                "yield": pd.Series([0.5], dtype="Float64"),
                "status": pd.Series(["completed"], dtype="string"),
            }
        )

        with temporary_modules(build_engine_stub_modules()):
            engine = create_campaign_engine(cfg)
            result = engine.ingest(df)

        self.assertEqual(engine.measurement_count(), 1)
        self.assertIn("status", result.columns)

    def test_extract_saved_campaign_metadata_is_backend_agnostic_to_ui(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            batch_size=1,
            acquisition="qExpectedImprovement",
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )

        with temporary_modules(build_engine_stub_modules()):
            engine = create_campaign_engine(cfg)
            engine.recommend(batch_size=1)

            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "demo_latest.json"
                engine.save(path)
                metadata = extract_saved_campaign_metadata(path)

        self.assertEqual(metadata["campaign_name"], "demo")
        self.assertEqual(metadata["acquisition"], "qExpectedImprovement")
        self.assertEqual(metadata["completed_measurements"], 0)


if __name__ == "__main__":
    unittest.main()
