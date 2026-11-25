"""Q&A retrieval: strict colour/type matching, place filter, time parsing, follow-ups. The LLM is a stub."""
import re
from datetime import datetime

import pytest

from drone_agent.search_agent import SecurityAgent, format_context, parse_time_window, retrieve
from drone_agent.vlm import HashEmbedder

T0 = datetime(2026, 11, 24, 23, 59, 30)


@pytest.mark.parametrize("q,window", [
    ("what happened between 10 and 15 seconds", (10.0, 15.0)),
    ("from 5s to 8s", (5.0, 8.0)),
    ("anything at 12s?", (10.5, 13.5)),
    ("in the first 20 seconds", (0.0, 20.0)),
    ("who was there at 23:59:45", (13.5, 16.5)),
    ("show me the red car", (None, None)),
])
def test_parse_time_window(q, window):
    assert parse_time_window(q, T0) == window


def ids(ctx):
    return sorted(t["track_id"] for t in ctx["tracks"])


def test_colour_and_type_both_required(demo, cfg):
    db, rid = demo
    ctx = retrieve(db, rid, "show me the blue truck", cfg, HashEmbedder())
    assert len(ctx["tracks"]) == 2 and all("F150" in t["description"] for t in ctx["tracks"])
    none = retrieve(db, rid, "show me the red truck", cfg, HashEmbedder())
    assert none["tracks"] == [] and "No object matches all of: red, truck" in none["note"]


def test_place_narrows_and_reports_absence(demo, cfg):
    db, rid = demo
    at_gate = retrieve(db, rid, "who was at the main gate", cfg, HashEmbedder())
    assert [t["description"] for t in at_gate["tracks"]] == ["man in red jacket"]
    nothing = retrieve(db, rid, "trucks at the main gate", cfg, HashEmbedder())
    assert nothing["tracks"] == [] and "main gate" in nothing["note"]


def test_time_window_retrieval(demo, cfg):
    db, rid = demo
    ctx = retrieve(db, rid, "what happened between 70 and 80 seconds", cfg, HashEmbedder())
    descs = {t["description"] for t in ctx["tracks"]}
    assert "blue Ford F150 pickup truck" in descs and "white Toyota Corolla sedan" not in descs


def test_context_is_compact(demo, cfg):
    db, rid = demo
    text = format_context(retrieve(db, rid, "show me the blue truck", cfg, HashEmbedder()))
    assert len(text) < 1500
    assert text.count("\n  #") == 2


class StubLLM:
    """Returns every object listed in the context, like a well-behaved LLM would for 'show me ...'."""
    used = "stub"

    def chat_json(self, messages):
        ctx = messages[-1]["content"]
        return {"answer": "ok", "show_images": False,
                "track_ids": [int(i) for i in re.findall(r"^  #(\d+)", ctx, re.M)], "scene_ids": []}


def test_follow_up_searches_previous_results(demo, cfg):
    db, rid = demo
    agent = SecurityAgent(db, rid, cfg, embedder=HashEmbedder(), llm=StubLLM())
    first = agent.ask("show me the vehicles", save_page=False)
    assert len(first.tracks) == 3                             # F150 twice + Corolla
    follow = agent.ask("which of those were on the driveway?", save_page=False)
    assert "Follow-up: searched only within the 3 objects" in follow.context
    assert [t["description"] for t in follow.tracks] == ["white Toyota Corolla sedan"]
    fresh = agent.ask("show me the dog", save_page=False)      # no reference word -> whole run again
    assert "Follow-up" not in fresh.context and fresh.tracks[0]["description"] == "brown dog"
