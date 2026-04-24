from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import (
    CampaignConfig,
    ParameterSpec,
    NumericalContinuousSpec,
    NumericalDiscreteSpec,
    CategoricalSpec,
    SubstanceSpec,
)


@dataclass(frozen=True)
class AcquisitionSpec:
    name: str
    module_path: str
    class_name: str
    max_batch_size: int | None = None
    supported_kwargs: tuple[str, ...] = ()

    def supports_batch_size(self, batch_size: int) -> bool:
        if batch_size < 1:
            return False
        if self.max_batch_size is None:
            return True
        return batch_size <= self.max_batch_size


ACQUISITION_SPECS: dict[str, AcquisitionSpec] = {
    "ProbabilityOfImprovement": AcquisitionSpec(
        name="ProbabilityOfImprovement",
        module_path="baybe.acquisition.acqfs",
        class_name="ProbabilityOfImprovement",
        max_batch_size=1,
    ),
    "ExpectedImprovement": AcquisitionSpec(
        name="ExpectedImprovement",
        module_path="baybe.acquisition.acqfs",
        class_name="ExpectedImprovement",
        max_batch_size=1,
    ),
    "qProbabilityOfImprovement": AcquisitionSpec(
        name="qProbabilityOfImprovement",
        module_path="baybe.acquisition.acqfs",
        class_name="qProbabilityOfImprovement",
    ),
    "qExpectedImprovement": AcquisitionSpec(
        name="qExpectedImprovement",
        module_path="baybe.acquisition.acqfs",
        class_name="qExpectedImprovement",
    ),
    "qNoisyExpectedImprovement": AcquisitionSpec(
        name="qNoisyExpectedImprovement",
        module_path="baybe.acquisition.acqfs",
        class_name="qNoisyExpectedImprovement",
    ),
    "UpperConfidenceBound": AcquisitionSpec(
        name="UpperConfidenceBound",
        module_path="baybe.acquisition.acqfs",
        class_name="UpperConfidenceBound",
        max_batch_size=1,
        supported_kwargs=("beta",),
    ),
    "qUpperConfidenceBound": AcquisitionSpec(
        name="qUpperConfidenceBound",
        module_path="baybe.acquisition.acqfs",
        class_name="qUpperConfidenceBound",
        supported_kwargs=("beta",),
    ),
    "qThompsonSampling": AcquisitionSpec(
        name="qThompsonSampling",
        module_path="baybe.acquisition.acqfs",
        class_name="qThompsonSampling",
    ),
}

CAMPAIGN_CONFIG_KEYS = {
    "campaign_name",
    "objective_target",
    "objective_mode",
    "batch_size",
    "init_mode",
    "n_init",
    "acquisition",
    "acquisition_kwargs",
    "engine",
    "parameters",
}
PARAMETER_SPEC_KEYS = {
    "NumericalContinuousSpec": {"_type", "name", "lower", "upper", "unit"},
    "NumericalDiscreteSpec": {"_type", "name", "values", "unit"},
    "CategoricalSpec": {"_type", "name", "values", "encoding"},
    "SubstanceSpec": {"_type", "name", "smiles", "encoding", "decorrelate"},
}


def _load_symbol(module_path: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, symbol_name)


def supported_acquisition_names(batch_size: int) -> list[str]:
    return [name for name, spec in ACQUISITION_SPECS.items() if spec.supports_batch_size(batch_size)]


def acquisition_supports_beta(acquisition_name: str) -> bool:
    spec = ACQUISITION_SPECS.get(acquisition_name)
    return spec is not None and "beta" in spec.supported_kwargs


def supported_substance_encodings() -> list[str]:
    substance_encoding_enum = _load_symbol("baybe.parameters.enum", "SubstanceEncoding")
    return [member.name for member in substance_encoding_enum]


def default_acquisition_name(batch_size: int) -> str:
    supported = supported_acquisition_names(batch_size)
    if "qExpectedImprovement" in supported:
        return "qExpectedImprovement"
    if supported:
        return supported[0]
    raise ValueError(f"No supported acquisition functions are available for batch size {batch_size}.")


def _normalize_acquisition_kwargs(acquisition_name: str, raw_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    spec = ACQUISITION_SPECS.get(acquisition_name)
    if spec is None:
        raise ValueError(f"Unsupported acquisition function '{acquisition_name}'.")

    kwargs = dict(raw_kwargs or {})
    unexpected = sorted(set(kwargs) - set(spec.supported_kwargs))
    if unexpected:
        supported = ", ".join(spec.supported_kwargs) if spec.supported_kwargs else "none"
        raise ValueError(
            f"Acquisition '{acquisition_name}' does not support kwargs {unexpected}. Supported kwargs: {supported}."
        )

    normalized: dict[str, Any] = {}
    if "beta" in kwargs:
        try:
            beta = float(kwargs["beta"])
        except (TypeError, ValueError) as exc:
            raise ValueError("UCB beta must be numeric.") from exc
        if beta <= 0.0:
            raise ValueError("UCB beta must be greater than 0.")
        normalized["beta"] = beta

    return normalized


def validate_acquisition_config(acquisition_name: str, raw_kwargs: Mapping[str, Any] | None, batch_size: int) -> list[str]:
    errors: list[str] = []
    spec = ACQUISITION_SPECS.get(acquisition_name)
    if spec is None:
        supported = ", ".join(ACQUISITION_SPECS)
        return [f"Unsupported acquisition function '{acquisition_name}'. Supported values: {supported}."]

    if not spec.supports_batch_size(batch_size):
        errors.append(
            f"Acquisition '{acquisition_name}' only supports batch_size <= {spec.max_batch_size}; "
            f"received batch_size={batch_size}."
        )

    try:
        _normalize_acquisition_kwargs(acquisition_name, raw_kwargs)
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def validate_config_payload(payload: Mapping[str, Any]) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["Campaign config payload must be a JSON object."]

    errors: list[str] = []
    unknown_top_level = sorted(set(payload) - CAMPAIGN_CONFIG_KEYS)
    if unknown_top_level:
        errors.append(f"Unsupported config keys: {unknown_top_level}.")

    raw_parameters = payload.get("parameters", [])
    if not isinstance(raw_parameters, list):
        errors.append("'parameters' must be a list.")
    else:
        for index, raw_parameter in enumerate(raw_parameters):
            if not isinstance(raw_parameter, Mapping):
                errors.append(f"Parameter #{index + 1} must be an object.")
                continue
            spec_type = raw_parameter.get("_type")
            if spec_type not in PARAMETER_SPEC_KEYS:
                errors.append(
                    f"Parameter #{index + 1} has unsupported _type '{spec_type}'. "
                    f"Supported values: {sorted(PARAMETER_SPEC_KEYS)}."
                )
                continue
            unknown_parameter_keys = sorted(set(raw_parameter) - PARAMETER_SPEC_KEYS[spec_type])
            if unknown_parameter_keys:
                errors.append(
                    f"Parameter #{index + 1} ({spec_type}) has unsupported keys: {unknown_parameter_keys}."
                )

    if errors:
        return errors

    try:
        cfg = CampaignConfig.from_dict(dict(payload))
    except Exception as exc:
        return [f"Invalid campaign config payload: {exc}."]

    return validate_campaign_config(cfg)


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _clean_smiles_entry(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""

    if " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].rstrip()
    cleaned = cleaned.rstrip(",").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_unique_smiles(values: list[str]) -> list[str]:
    cleaned = [_clean_smiles_entry(sm) for sm in values]
    return _unique_in_order([sm for sm in cleaned if sm])


def validate_parameter_specs(specs: list[ParameterSpec]) -> None:
    for spec in specs:
        if isinstance(spec, NumericalContinuousSpec):
            if float(spec.lower) >= float(spec.upper):
                raise ValueError(
                    f"Continuous parameter '{spec.name}' needs lower < upper; "
                    f"got {spec.lower} >= {spec.upper}."
                )

        elif isinstance(spec, NumericalDiscreteSpec):
            if not spec.values:
                raise ValueError(f"Numerical discrete parameter '{spec.name}' needs at least one value.")

        elif isinstance(spec, CategoricalSpec):
            if not spec.values:
                raise ValueError(f"Categorical parameter '{spec.name}' needs at least one category.")
            if spec.encoding not in {"OHE", "INT"}:
                raise ValueError(
                    f"Categorical parameter '{spec.name}' has unsupported encoding '{spec.encoding}'. "
                    "Supported values: ['OHE', 'INT']."
                )

        elif isinstance(spec, SubstanceSpec):
            unique_smiles = _normalize_unique_smiles(spec.smiles)
            if len(unique_smiles) < 2:
                raise ValueError(
                    f"Substance parameter '{spec.name}' needs at least 2 unique SMILES entries; "
                    f"got {len(unique_smiles)}."
                )
            try:
                chem_module = _load_symbol("rdkit", "Chem")
            except ModuleNotFoundError as exc:
                raise ValueError(
                    "RDKit is required to validate substance parameters before initializing the campaign."
                ) from exc
            invalid = [sm for sm in unique_smiles if chem_module.MolFromSmiles(sm) is None]
            if invalid:
                shown = ", ".join(invalid[:5])
                extra = f" (+{len(invalid) - 5} more)" if len(invalid) > 5 else ""
                raise ValueError(
                    f"Substance parameter '{spec.name}' contains invalid SMILES: {shown}{extra}. "
                    "Use one raw SMILES per line (no quotes, trailing commas, or inline comments)."
                )
            if not isinstance(spec.decorrelate, bool):
                raise ValueError(
                    f"Substance parameter '{spec.name}' decorrelate must be a boolean in this UI."
                )


def validate_campaign_config(cfg: CampaignConfig) -> list[str]:
    errors: list[str] = []

    if cfg.init_mode not in {"sobol", "existing_data"}:
        errors.append(
            f"Unsupported init_mode '{cfg.init_mode}'. Supported values: ['sobol', 'existing_data']."
        )

    if cfg.objective_mode not in {"maximize", "minimize"}:
        errors.append(
            f"Unsupported objective_mode '{cfg.objective_mode}'. Supported values: ['maximize', 'minimize']."
        )

    if int(cfg.batch_size) < 1:
        errors.append("Batch size must be at least 1.")

    if int(cfg.n_init) < 0:
        errors.append("Number of initial points must be >= 0.")

    try:
        validate_parameter_specs(cfg.parameters)
    except ValueError as exc:
        errors.append(str(exc))

    if cfg.engine != "ax":
        errors.extend(
            validate_acquisition_config(
                acquisition_name=cfg.acquisition,
                raw_kwargs=cfg.acquisition_kwargs,
                batch_size=int(cfg.batch_size),
            )
        )

    return errors


def build_parameters(specs: list[ParameterSpec]) -> list[Any]:
    validate_parameter_specs(specs)
    categorical_parameter = _load_symbol("baybe.parameters", "CategoricalParameter")
    numerical_continuous_parameter = _load_symbol("baybe.parameters", "NumericalContinuousParameter")
    numerical_discrete_parameter = _load_symbol("baybe.parameters", "NumericalDiscreteParameter")
    substance_encoding_enum = _load_symbol("baybe.parameters.enum", "SubstanceEncoding")
    substance_parameter = _load_symbol("baybe.parameters.substance", "SubstanceParameter")

    params: list[Any] = []
    for spec in specs:
        if isinstance(spec, NumericalContinuousSpec):
            metadata = {"unit": spec.unit} if getattr(spec, "unit", None) else None
            kwargs = {"metadata": metadata} if metadata is not None else {}
            params.append(
                numerical_continuous_parameter(
                    name=spec.name,
                    bounds=(float(spec.lower), float(spec.upper)),
                    **kwargs,
                )
            )

        elif isinstance(spec, NumericalDiscreteSpec):
            metadata = {"unit": spec.unit} if getattr(spec, "unit", None) else None
            kwargs = {"metadata": metadata} if metadata is not None else {}
            params.append(
                numerical_discrete_parameter(
                    name=spec.name,
                    values=[float(value) for value in spec.values],
                    **kwargs,
                )
            )

        elif isinstance(spec, CategoricalSpec):
            params.append(
                categorical_parameter(
                    name=spec.name,
                    values=list(spec.values),
                    encoding=spec.encoding,
                )
            )

        elif isinstance(spec, SubstanceSpec):
            encoding = substance_encoding_enum[spec.encoding]
            unique_smiles = _normalize_unique_smiles(spec.smiles)
            params.append(
                substance_parameter(
                    name=spec.name,
                    data={sm: sm for sm in unique_smiles},
                    encoding=encoding,
                    decorrelate=spec.decorrelate,
                )
            )
        else:
            raise ValueError(f"Unsupported parameter spec: {type(spec)}")

    return params


def _build_acquisition_function(cfg: CampaignConfig) -> Any:
    spec = ACQUISITION_SPECS.get(cfg.acquisition)
    if spec is None:
        raise ValueError(f"Unsupported acquisition function '{cfg.acquisition}'.")

    kwargs = _normalize_acquisition_kwargs(cfg.acquisition, cfg.acquisition_kwargs)
    acquisition_class = _load_symbol(spec.module_path, spec.class_name)

    signature = inspect.signature(acquisition_class)
    for kwarg_name in kwargs:
        if kwarg_name not in signature.parameters:
            raise ValueError(
                f"Installed BayBE acquisition '{cfg.acquisition}' does not accept kwarg '{kwarg_name}'."
            )

    return acquisition_class(**kwargs)


def build_recommender(cfg: CampaignConfig) -> Any:
    validation_errors = validate_campaign_config(cfg)
    if validation_errors:
        raise ValueError("Invalid campaign config:\n- " + "\n- ".join(validation_errors))

    botorch_recommender = _load_symbol("baybe.recommenders", "BotorchRecommender")
    acquisition_function = _build_acquisition_function(cfg)
    return botorch_recommender(acquisition_function=acquisition_function)


def build_campaign(cfg: CampaignConfig) -> Any:
    validation_errors = validate_campaign_config(cfg)
    if validation_errors:
        raise ValueError("Invalid campaign config:\n- " + "\n- ".join(validation_errors))

    params = build_parameters(cfg.parameters)
    search_space_cls = _load_symbol("baybe.searchspace", "SearchSpace")
    numerical_target_cls = _load_symbol("baybe.targets", "NumericalTarget")
    target_mode_enum = _load_symbol("baybe.targets", "TargetMode")
    single_target_objective_cls = _load_symbol("baybe.objectives", "SingleTargetObjective")
    campaign_cls = _load_symbol("baybe.campaign", "Campaign")

    searchspace = search_space_cls.from_product(params)
    mode_alias = {
        "maximize": target_mode_enum.MAX,
        "max": target_mode_enum.MAX,
        "maximise": target_mode_enum.MAX,
        "minimize": target_mode_enum.MIN,
        "min": target_mode_enum.MIN,
        "minimise": target_mode_enum.MIN,
    }
    mode = mode_alias.get(str(cfg.objective_mode).strip().lower(), cfg.objective_mode)
    target = numerical_target_cls(name=cfg.objective_target, mode=mode)
    objective = single_target_objective_cls(target=target)
    recommender = build_recommender(cfg)
    return campaign_cls(searchspace=searchspace, objective=objective, recommender=recommender)
