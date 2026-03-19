import sys
import types
import unittest
from contextlib import contextmanager
from enum import Enum
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from core import baybe_factory
from core.schema import CampaignConfig, CategoricalSpec, SubstanceSpec


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


def build_stub_backend_modules() -> dict[str, types.ModuleType]:
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

    class ProbabilityOfImprovement(AcquisitionBase):
        pass

    class ExpectedImprovement(AcquisitionBase):
        pass

    class qProbabilityOfImprovement(AcquisitionBase):
        pass

    class qExpectedImprovement(AcquisitionBase):
        pass

    class qNoisyExpectedImprovement(AcquisitionBase):
        pass

    class UpperConfidenceBound(AcquisitionBase):
        def __init__(self, beta: float = 1.0):
            super().__init__(beta=beta)

    class qUpperConfidenceBound(AcquisitionBase):
        def __init__(self, beta: float = 1.0):
            super().__init__(beta=beta)

    class qThompsonSampling(AcquisitionBase):
        pass

    class BotorchRecommender:
        def __init__(self, acquisition_function):
            self.acquisition_function = acquisition_function

    class SearchSpace:
        def __init__(self, parameters):
            self.parameters = parameters

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

    class Campaign:
        def __init__(self, searchspace, objective, recommender):
            self.searchspace = searchspace
            self.objective = objective
            self.recommender = recommender

        def recommend(self, batch_size: int):
            acqf = self.recommender.acquisition_function
            beta = getattr(acqf, "beta", None)
            if beta is None:
                return {"strategy": type(acqf).__name__, "batch_size": batch_size}
            return {"strategy": type(acqf).__name__, "beta": beta, "batch_size": batch_size}

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

    class ChemModule:
        @staticmethod
        def MolFromSmiles(value):
            return object() if value else None

    baybe_acqfs_module.ProbabilityOfImprovement = ProbabilityOfImprovement
    baybe_acqfs_module.ExpectedImprovement = ExpectedImprovement
    baybe_acqfs_module.qProbabilityOfImprovement = qProbabilityOfImprovement
    baybe_acqfs_module.qExpectedImprovement = qExpectedImprovement
    baybe_acqfs_module.qNoisyExpectedImprovement = qNoisyExpectedImprovement
    baybe_acqfs_module.UpperConfidenceBound = UpperConfidenceBound
    baybe_acqfs_module.qUpperConfidenceBound = qUpperConfidenceBound
    baybe_acqfs_module.qThompsonSampling = qThompsonSampling
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


class TruthfulControlsTests(unittest.TestCase):
    def test_validate_campaign_config_rejects_unsupported_kwargs(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            batch_size=1,
            acquisition="ExpectedImprovement",
            acquisition_kwargs={"beta": 2.0},
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )

        errors = baybe_factory.validate_campaign_config(cfg)

        self.assertTrue(any("does not support kwargs" in error for error in errors))

    def test_validate_campaign_config_rejects_single_point_acquisition_for_batch(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            batch_size=3,
            acquisition="UpperConfidenceBound",
            acquisition_kwargs={"beta": 2.0},
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )

        errors = baybe_factory.validate_campaign_config(cfg)

        self.assertTrue(any("only supports batch_size <= 1" in error for error in errors))

    def test_validate_campaign_config_rejects_invalid_categorical_encoding(self):
        cfg = CampaignConfig(
            campaign_name="demo",
            batch_size=1,
            acquisition="qExpectedImprovement",
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"], encoding="BAD")],
        )

        errors = baybe_factory.validate_campaign_config(cfg)

        self.assertTrue(any("unsupported encoding" in error for error in errors))

    def test_build_campaign_wires_ucb_beta_into_backend_object(self):
        cfg_low = CampaignConfig(
            campaign_name="demo",
            batch_size=1,
            acquisition="UpperConfidenceBound",
            acquisition_kwargs={"beta": 1.5},
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )
        cfg_high = CampaignConfig(
            campaign_name="demo",
            batch_size=1,
            acquisition="UpperConfidenceBound",
            acquisition_kwargs={"beta": 6.0},
            parameters=[CategoricalSpec(name="solvent", values=["A", "B"])],
        )

        with temporary_modules(build_stub_backend_modules()):
            campaign_low = baybe_factory.build_campaign(cfg_low)
            campaign_high = baybe_factory.build_campaign(cfg_high)

        self.assertEqual(campaign_low.recommender.acquisition_function.beta, 1.5)
        self.assertEqual(campaign_high.recommender.acquisition_function.beta, 6.0)
        self.assertNotEqual(campaign_low.recommend(batch_size=1), campaign_high.recommend(batch_size=1))

    def test_build_parameters_wires_substance_decorrelate(self):
        spec = SubstanceSpec(
            name="ligand",
            smiles=["CCO", "CCN"],
            encoding="MORDRED",
            decorrelate=False,
        )

        with temporary_modules(build_stub_backend_modules()):
            parameters = baybe_factory.build_parameters([spec])

        self.assertFalse(parameters[0].decorrelate)

    def test_validate_config_payload_rejects_unknown_keys(self):
        payload = {
            "campaign_name": "demo",
            "batch_size": 1,
            "acquisition": "qExpectedImprovement",
            "parameters": [],
            "future_flag": True,
        }

        errors = baybe_factory.validate_config_payload(payload)

        self.assertTrue(any("Unsupported config keys" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
