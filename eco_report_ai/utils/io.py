"""File I/O utilities for EcoReport AI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Serialize a dict to a JSON file.

    Args:
        data: Dictionary to serialize.
        path: Output file path.
        indent: JSON indentation level.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)
    logger.debug("Saved JSON to %s", p)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default.

    Handles numpy types, pandas Timestamps, and NaN/Inf values.
    """
    import math

    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return str(obj)
