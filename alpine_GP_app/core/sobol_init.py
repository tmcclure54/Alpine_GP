from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .schema import ParameterSpec, NumericalContinuousSpec, NumericalDiscreteSpec, CategoricalSpec, SubstanceSpec


def _pick_from_list(u: np.ndarray, choices: List):
    # u in [0,1)
    idx = np.floor(u * len(choices)).astype(int)
    idx = np.clip(idx, 0, len(choices) - 1)
    return [choices[i] for i in idx]


def sobol_initial_design(parameters: List[ParameterSpec], n: int, seed: int = 0) -> pd.DataFrame:
    """Generate an initial design table (experimental representation) using a Sobol sequence.

    Notes:
    - For numerical continuous parameters, we scale Sobol u∈[0,1) into [lower, upper].
    - For numerical discrete, categorical, substance: we use Sobol u to index into the allowed list.

    This is a pragmatic design generator for "cold starts" in the lab; it is *not* a full DOE engine.
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if not parameters:
        raise ValueError("No parameters provided")

    d = len(parameters)
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = eng.random(n)

    cols = {}

    for j, p in enumerate(parameters):
        uj = u[:, j]

        if isinstance(p, NumericalContinuousSpec):
            lo, hi = float(p.lower), float(p.upper)
            cols[p.name] = lo + uj * (hi - lo)

        elif isinstance(p, NumericalDiscreteSpec):
            cols[p.name] = _pick_from_list(uj, [float(v) for v in p.values])

        elif isinstance(p, CategoricalSpec):
            cols[p.name] = _pick_from_list(uj, list(p.values))

        elif isinstance(p, SubstanceSpec):
            labels = list(p.smiles)
            cols[p.name] = _pick_from_list(uj, labels)

        else:
            raise ValueError(f"Unsupported parameter type for Sobol init: {type(p)}")

    return pd.DataFrame(cols)
