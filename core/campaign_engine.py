from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .schema import CampaignConfig


class CampaignEngine(ABC):
    @abstractmethod
    def recommend(self, batch_size: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def ingest(self, df: pd.DataFrame) -> pd.DataFrame:
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
    from .baybe_engine import BayBEEngine

    return BayBEEngine.from_config(cfg)


def load_campaign_engine(path: Path, cfg: CampaignConfig | None = None) -> CampaignEngine:
    from .baybe_engine import BayBEEngine

    return BayBEEngine.load(path, cfg)


def extract_saved_campaign_metadata(path: Path) -> dict[str, Any]:
    from .baybe_engine import extract_baybe_campaign_metadata

    return extract_baybe_campaign_metadata(path)
