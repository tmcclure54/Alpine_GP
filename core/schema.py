from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type


# ------------------------
# Parameter specs (UI-facing)
# ------------------------


@dataclass
class ParameterSpec:
    name: str

    @property
    def kind(self) -> str:
        return self.__class__.__name__.replace("Spec", "").lower()

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["_type"] = self.__class__.__name__
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParameterSpec":
        t = d.get("_type")
        mapping: Dict[str, Type[ParameterSpec]] = {
            "NumericalContinuousSpec": NumericalContinuousSpec,
            "NumericalDiscreteSpec": NumericalDiscreteSpec,
            "CategoricalSpec": CategoricalSpec,
            "SubstanceSpec": SubstanceSpec,
        }
        if t not in mapping:
            raise ValueError(f"Unknown ParameterSpec _type: {t}")
        dd = dict(d)
        dd.pop("_type", None)
        return mapping[t](**dd)  # type: ignore[arg-type]


@dataclass
class NumericalContinuousSpec(ParameterSpec):
    lower: float
    upper: float
    unit: Optional[str] = None


@dataclass
class NumericalDiscreteSpec(ParameterSpec):
    values: List[float]
    unit: Optional[str] = None


@dataclass
class CategoricalSpec(ParameterSpec):
    values: List[str]
    encoding: Literal["OHE", "INT"] = "OHE"


@dataclass
class SubstanceSpec(ParameterSpec):
    smiles: List[str] = field(default_factory=list)
    encoding: str = "MORDRED"
    decorrelate: bool = True


# ------------------------
# Campaign config
# ------------------------


@dataclass
class CampaignConfig:
    campaign_name: str
    objective_target: str = "yield"
    objective_mode: Literal["maximize", "minimize"] = "maximize"

    batch_size: int = 8

    init_mode: Literal["sobol", "existing_data"] = "sobol"
    n_init: int = 8

    acquisition: str = "qExpectedImprovement"
    acquisition_kwargs: Dict[str, Any] = field(default_factory=dict)

    parameters: List[ParameterSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_name": self.campaign_name,
            "objective_target": self.objective_target,
            "objective_mode": self.objective_mode,
            "batch_size": self.batch_size,
            "init_mode": self.init_mode,
            "n_init": self.n_init,
            "acquisition": self.acquisition,
            "acquisition_kwargs": self.acquisition_kwargs,
            "parameters": [p.to_dict() for p in self.parameters],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CampaignConfig":
        params = [ParameterSpec.from_dict(x) for x in d.get("parameters", [])]
        return cls(
            campaign_name=d.get("campaign_name", "default"),
            objective_target=d.get("objective_target", "yield"),
            objective_mode=d.get("objective_mode", "maximize"),
            batch_size=int(d.get("batch_size", 8)),
            init_mode=d.get("init_mode", "sobol"),
            n_init=int(d.get("n_init", 0)),
            acquisition=d.get("acquisition", "qExpectedImprovement"),
            acquisition_kwargs=d.get("acquisition_kwargs", {}) or {},
            parameters=params,
        )
