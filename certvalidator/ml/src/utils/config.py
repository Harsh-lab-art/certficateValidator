"""
ml/src/utils/config.py

Loads ml/config.yaml and provides typed access to all hyperparameters.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


_CONFIG_PATH = Path(__file__).parents[3] / "config.yaml"
_cache: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    global _cache
    if _cache is not None and path is None:
        return _cache
    p = Path(path) if path else _CONFIG_PATH
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _cache = cfg
    return cfg


def get(key_path: str, default: Any = None) -> Any:
    """Dot-notation key access. E.g. get('forgery_model.training.lr')"""
    cfg = load_config()
    parts = key_path.split(".")
    node = cfg
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node
