from __future__ import annotations

from typing import List

from baybe.acquisition.utils import str_to_acqf
from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.enum import SubstanceEncoding
from baybe.parameters.substance import SubstanceParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

from .schema import (
    CampaignConfig,
    ParameterSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
    CategoricalSpec,
    SubstanceSpec,
)


def build_parameters(specs: List[ParameterSpec]):
    params = []

    for s in specs:
        if isinstance(s, NumericalContinuousSpec):
            md = {"unit": s.unit} if getattr(s, "unit", None) else None
            kwargs = {"metadata": md} if md is not None else {}
            params.append(
                NumericalContinuousParameter(
                    name=s.name,
                    bounds=(float(s.lower), float(s.upper)),
                    **kwargs,
                )
            )

        elif isinstance(s, NumericalDiscreteSpec):
            md = {"unit": s.unit} if getattr(s, "unit", None) else None
            kwargs = {"metadata": md} if md is not None else {}
            params.append(
                NumericalDiscreteParameter(
                    name=s.name,
                    values=[float(v) for v in s.values],
                    tolerance=float(getattr(s, "tolerance", 0.0) or 0.0),
                    **kwargs,
                )
            )

        elif isinstance(s, CategoricalSpec):
            enc = s.encoding if s.encoding in ("OHE", "INT") else "OHE"
            params.append(
                CategoricalParameter(
                    name=s.name,
                    values=list(s.values),
                    encoding=enc,
                )
            )

        elif isinstance(s, SubstanceSpec):
            enc = SubstanceEncoding[s.encoding]
            params.append(
                SubstanceParameter(
                    name=s.name,
                    data={sm: sm for sm in s.smiles},
                    encoding=enc,
                )
            )
        else:
            raise ValueError(f"Unsupported parameter spec: {type(s)}")

    return params


def build_recommender(cfg: CampaignConfig) -> BotorchRecommender:
    alias = {
        "EI": "qExpectedImprovement",
        "qEI": "qExpectedImprovement",
        "UCB": "qUpperConfidenceBound",
        "qUCB": "qUpperConfidenceBound",
        "TS": "qThompsonSampling",
        "qTS": "qThompsonSampling",
        "PI": "qProbabilityOfImprovement",
        "qPI": "qProbabilityOfImprovement",
    }
    acq_name = alias.get(cfg.acquisition, cfg.acquisition)
    acqf = str_to_acqf(acq_name, **(cfg.acquisition_kwargs or {}))
    return BotorchRecommender(acquisition_function=acqf)


def build_campaign(cfg: CampaignConfig) -> Campaign:
    params = build_parameters(cfg.parameters)
    searchspace = SearchSpace.from_product(params)
    target = NumericalTarget(name=cfg.objective_target, mode=cfg.objective_mode)
    objective = SingleTargetObjective(target=target)
    recommender = build_recommender(cfg)
    return Campaign(searchspace=searchspace, objective=objective, recommender=recommender)
