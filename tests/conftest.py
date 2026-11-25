"""Shared fixtures. Everything runs offline: text-simulated frames, mock VLM, hash embeddings, a fresh
SQLite file per test session. No GPU, no Ollama, no Groq, and the real data/drone_security.db is never touched."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent.config import load_config  # noqa: E402
from drone_agent.db import Database  # noqa: E402
from drone_agent.pipeline import Pipeline  # noqa: E402
from drone_agent.sources import build_source  # noqa: E402
from drone_agent.vlm import HashEmbedder  # noqa: E402

NIGHT_START = "2026-11-24T23:59:30"   # the demo crosses midnight
DAY_START = "2026-11-24T14:00:00"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def run_sim(cfg, db: Database, start: str, scenario: str = "demo", duration: float = 100.0) -> int:
    pipe = Pipeline(cfg, db=db, embedder=HashEmbedder(), write_video=False, save_images=False)
    return pipe.run(build_source(cfg, sim=scenario, sim_duration_s=duration, start_time=start), f"sim-{scenario}")


@pytest.fixture(scope="session")
def demo(cfg, tmp_path_factory):
    """The scripted demo (blue F150 twice, loiterer at the main gate, intruder in the backyard, ...)
    starting just before midnight. Returns (db, run_id)."""
    db = Database(str(tmp_path_factory.mktemp("db") / "demo.db"))
    return db, run_sim(cfg, db, NIGHT_START)


@pytest.fixture(scope="session")
def demo_day(cfg, tmp_path_factory):
    """Same demo in the afternoon: no night alerts expected."""
    db = Database(str(tmp_path_factory.mktemp("db") / "day.db"))
    return db, run_sim(cfg, db, DAY_START)
