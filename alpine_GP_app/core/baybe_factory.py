from __future__ import annotations

from typing import List
import warnings

from rdkit import Chem

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
from baybe.targets import NumericalTarget, TargetMode

from .schema import (
    CampaignConfig,
    ParameterSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
    CategoricalSpec,
    SubstanceSpec,
)


def _unique_in_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _clean_smiles_entry(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""

    # Allow users to paste python-list style lines, e.g. "<smiles>",  # comment
    if " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].rstrip()
    cleaned = cleaned.rstrip(",").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_unique_smiles(values: List[str]) -> List[str]:
    cleaned = [_clean_smiles_entry(sm) for sm in values]
    return _unique_in_order([sm for sm in cleaned if sm])


def validate_parameter_specs(specs: List[ParameterSpec]) -> None:
    for s in specs:
        if isinstance(s, SubstanceSpec):
            unique_smiles = _normalize_unique_smiles(s.smiles)
            if len(unique_smiles) < 2:
                raise ValueError(
                    f"Substance parameter '{s.name}' needs at least 2 unique SMILES entries; "
                    f"got {len(unique_smiles)}."
                )
            invalid = [sm for sm in unique_smiles if Chem.MolFromSmiles(sm) is None]
            if invalid:
                shown = ", ".join(invalid[:5])
                extra = f" (+{len(invalid) - 5} more)" if len(invalid) > 5 else ""
                raise ValueError(
                    f"Substance parameter '{s.name}' contains invalid SMILES: {shown}{extra}. "
                    "Use one raw SMILES per line (no quotes, trailing commas, or inline comments)."
                )


def build_parameters(specs: List[ParameterSpec]):
    validate_parameter_specs(specs)
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
            unique_smiles = _normalize_unique_smiles(s.smiles)
            params.append(
                SubstanceParameter(
                    name=s.name,
                    data={sm: sm for sm in unique_smiles},
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
    if cfg.acquisition_kwargs:
        warnings.warn(
            "acquisition_kwargs are ignored because this BayBE version's str_to_acqf does not accept kwargs.",
            RuntimeWarning,
            stacklevel=2,
        )
    acqf = str_to_acqf(acq_name)
    return BotorchRecommender(acquisition_function=acqf)


def build_campaign(cfg: CampaignConfig) -> Campaign:
    params = build_parameters(cfg.parameters)
    searchspace = SearchSpace.from_product(params)
    mode_alias = {
        "maximize": TargetMode.MAX,
        "max": TargetMode.MAX,
        "maximise": TargetMode.MAX,
        "minimize": TargetMode.MIN,
        "min": TargetMode.MIN,
        "minimise": TargetMode.MIN,
    }
    mode = mode_alias.get(str(cfg.objective_mode).strip().lower(), cfg.objective_mode)
    target = NumericalTarget(name=cfg.objective_target, mode=mode)
    objective = SingleTargetObjective(target=target)
    recommender = build_recommender(cfg)
    return Campaign(searchspace=searchspace, objective=objective, recommender=recommender)
