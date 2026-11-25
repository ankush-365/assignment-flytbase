"""Config loading. The YAML is kept as a plain nested dict wrapped for attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class Cfg(dict):
    """dict with attribute access (cfg.rules.loitering.seconds)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _wrap(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _wrap(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_wrap(v) for v in d]
    return d


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return _wrap(data)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
