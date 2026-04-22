"""Configuration loader.

API keys live in ~/.claude/mcp-servers/refcheck/.env (outside this repo).
Pipeline params live in config/*.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
REFCHECK_ENV = Path.home() / ".claude" / "mcp-servers" / "refcheck" / ".env"


def _env() -> dict[str, str]:
    vals: dict[str, str] = {}
    if REFCHECK_ENV.exists():
        vals.update({k: v for k, v in dotenv_values(REFCHECK_ENV).items() if v is not None})
    vals.update({k: v for k, v in os.environ.items() if v})
    return vals


def elsevier_key() -> str | None:
    return _env().get("ELSEVIER_KEY")


def elsevier_insttoken() -> str | None:
    return _env().get("ELSEVIER_INSTTOKEN")


def ieee_key() -> str | None:
    return _env().get("IEEE_API_KEY")


def crossref_email() -> str | None:
    return _env().get("CROSSREF_EMAIL")


def s2_key() -> str | None:
    return _env().get("S2_API_KEY")


def load_venues(*, include_disabled: bool = False) -> list[dict[str, Any]]:
    """Load venue list. Disabled entries (enabled: false) filtered by default."""
    with (CONFIG_DIR / "venues.yaml").open() as f:
        venues = yaml.safe_load(f)["venues"]
    if include_disabled:
        return venues
    return [v for v in venues if v.get("enabled", True)]


def load_pipeline() -> dict[str, Any]:
    with (CONFIG_DIR / "pipeline.yaml").open() as f:
        return yaml.safe_load(f)


def data_dir(subdir: str = "") -> Path:
    p = REPO_ROOT / "data" / subdir if subdir else REPO_ROOT / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def site_dir() -> Path:
    p = REPO_ROOT / "site"
    p.mkdir(parents=True, exist_ok=True)
    return p
