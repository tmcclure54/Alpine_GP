from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .schema import CampaignConfig


class ModelNotFittedError(ValueError):
    """Raised when predictions are requested before the surrogate model has been fitted.

    This occurs when no completed measurements have been ingested yet (e.g., during
    the Sobol initialisation phase). Ingest at least one completed result and call
    recommend() before requesting model metadata.
    """


class CampaignEngine(ABC):
    @abstractmethod
    def recommend(self, batch_size: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def predict(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Attach surrogate model predictions to a candidates DataFrame.

        Adds columns: pred_mean, pred_std, acq_score, rank.

        Raises:
            ModelNotFittedError: If no completed measurements have been ingested yet.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path, cfg: CampaignConfig | None = None) -> "CampaignEngine":
        raise NotImplementedError

    @abstractmethod
    def cached_recommendation_info(self) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def measurement_count(self) -> int:
        raise NotImplementedError


def create_campaign_engine(cfg: CampaignConfig) -> CampaignEngine:
    if cfg.engine == "ax":
        from .ax_engine import AxEngine
        return AxEngine.from_config(cfg)
    from .baybe_engine import BayBEEngine
    return BayBEEngine.from_config(cfg)


def load_campaign_engine(path: Path, cfg: CampaignConfig | None = None) -> CampaignEngine:
    import json
    from .persistence import load_text

    try:
        raw = json.loads(load_text(path))
    except Exception:
        raw = {}

    # Detect Ax saves by the _engine marker written by AxEngine.save()
    if raw.get("_engine") == "ax" or (cfg is not None and cfg.engine == "ax"):
        from .ax_engine import AxEngine
        return AxEngine.load(path, cfg)

    from .baybe_engine import BayBEEngine
    return BayBEEngine.load(path, cfg)


def extract_saved_campaign_metadata(path: Path) -> dict[str, Any]:
    import json
    from .persistence import load_text

    try:
        raw = json.loads(load_text(path))
    except Exception:
        raw = {}

    if raw.get("_engine") == "ax":
        from .ax_engine import extract_ax_campaign_metadata
        return extract_ax_campaign_metadata(path)

    from .baybe_engine import extract_baybe_campaign_metadata
    return extract_baybe_campaign_metadata(path)
