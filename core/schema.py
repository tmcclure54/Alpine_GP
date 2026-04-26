from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Type

EngineType = Literal["baybe", "ax"]
ObjectiveMode = Literal["maximize", "minimize"]

TrialStatus = Literal["completed", "failed", "abandoned", "partial", "invalid"]
VALID_TRIAL_STATUSES: FrozenSet[str] = frozenset({"completed", "failed", "abandoned", "partial", "invalid"})


# ------------------------
# Target spec (multi-objective)
# ------------------------


@dataclass
class TargetSpec:
    name: str
    mode: ObjectiveMode = "maximize"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mode": self.mode}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetSpec":
        return cls(name=d["name"], mode=d.get("mode", "maximize"))


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
    objective_mode: ObjectiveMode = "maximize"

    batch_size: int = 8

    init_mode: Literal["sobol", "existing_data"] = "sobol"
    n_init: int = 8

    acquisition: str = "qExpectedImprovement"
    acquisition_kwargs: Dict[str, Any] = field(default_factory=dict)

    engine: EngineType = "baybe"

    parameters: List[ParameterSpec] = field(default_factory=list)

    # Multi-objective: when non-empty, this list is the source of truth for the
    # campaign's optimization targets. When empty, a single target is derived
    # from objective_target/objective_mode (legacy single-objective behaviour).
    targets: List[TargetSpec] = field(default_factory=list)

    def effective_targets(self) -> List[TargetSpec]:
        """Always returns a list of >= 1 TargetSpec.

        Bridges single-target (objective_target/objective_mode) and explicit
        multi-target configurations into a uniform list.
        """
        if self.targets:
            return list(self.targets)
        return [TargetSpec(name=self.objective_target, mode=self.objective_mode)]

    def is_multi_objective(self) -> bool:
        return len(self.effective_targets()) > 1

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "campaign_name": self.campaign_name,
            "objective_target": self.objective_target,
            "objective_mode": self.objective_mode,
            "batch_size": self.batch_size,
            "init_mode": self.init_mode,
            "n_init": self.n_init,
            "acquisition": self.acquisition,
            "engine": self.engine,
            "parameters": [p.to_dict() for p in self.parameters],
        }
        if self.acquisition_kwargs:
            data["acquisition_kwargs"] = self.acquisition_kwargs
        if self.targets:
            data["targets"] = [t.to_dict() for t in self.targets]
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CampaignConfig":
        params = [ParameterSpec.from_dict(x) for x in d.get("parameters", [])]
        targets = [TargetSpec.from_dict(x) for x in d.get("targets", []) or []]
        return cls(
            campaign_name=d.get("campaign_name", "default"),
            objective_target=d.get("objective_target", "yield"),
            objective_mode=d.get("objective_mode", "maximize"),
            batch_size=int(d.get("batch_size", 8)),
            init_mode=d.get("init_mode", "sobol"),
            n_init=int(d.get("n_init", 0)),
            acquisition=d.get("acquisition", "qExpectedImprovement"),
            acquisition_kwargs=d.get("acquisition_kwargs", {}) or {},
            engine=d.get("engine", "baybe"),
            parameters=params,
            targets=targets,
        )
