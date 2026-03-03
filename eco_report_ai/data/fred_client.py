"""FRED API client for downloading macroeconomic time series.

Uses the `fredapi` library. Requires `FRED_API_KEY` environment variable.
If the key is absent or any error occurs, callers should fall back to the
local sample CSV via `loaders.py`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FREDClient:
    """Thin wrapper around the fredapi library.

    Args:
        api_key: FRED API key. Reads from `FRED_API_KEY` env var if not provided.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("FRED_API_KEY", "")
        self._fred = None

    def _get_fred(self):
        """Lazy-initialize the fredapi.Fred client."""
        if self._fred is None:
            try:
                from fredapi import Fred  # type: ignore[import-untyped]

                self._fred = Fred(api_key=self._api_key)
            except ImportError as exc:
                raise ImportError(
                    "fredapi is not installed. Run: pip install fredapi"
                ) from exc
        return self._fred

    def is_available(self) -> bool:
        """Return True if an API key is configured.

        Returns:
            True when FRED_API_KEY is non-empty.
        """
        return bool(self._api_key)

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch a single FRED series as a pandas Series.

        Args:
            series_id: FRED series identifier (e.g., "CPIAUCSL").
            start_date: ISO date string for series start.
            end_date: ISO date string for series end.

        Returns:
            Pandas Series with DatetimeIndex and float values.

        Raises:
            RuntimeError: If the API call fails.
        """
        fred = self._get_fred()
        logger.info("Fetching FRED series: %s", series_id)
        try:
            kwargs: dict = {}
            if start_date:
                kwargs["observation_start"] = start_date
            if end_date:
                kwargs["observation_end"] = end_date
            series = fred.get_series(series_id, **kwargs)
            series.name = series_id
            logger.info(
                "Fetched %s: %d observations (%s to %s)",
                series_id,
                len(series),
                series.index.min().date(),
                series.index.max().date(),
            )
            return series
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch FRED series '{series_id}': {exc}") from exc

    def fetch_all(
        self,
        series_ids: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch multiple FRED series and combine into a single DataFrame.

        Args:
            series_ids: List of FRED series identifiers.
            start_date: ISO date string for series start.
            end_date: ISO date string for series end.

        Returns:
            DataFrame with DatetimeIndex and one column per series.

        Raises:
            RuntimeError: If any series fetch fails.
        """
        series_list: list[pd.Series] = []
        for sid in series_ids:
            s = self.fetch_series(sid, start_date=start_date, end_date=end_date)
            series_list.append(s)

        df = pd.concat(series_list, axis=1)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
