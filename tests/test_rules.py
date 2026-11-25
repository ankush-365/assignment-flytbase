"""Rules engine as pure functions: does each alert fire at the right time, once, and not when it shouldn't?"""
from datetime import datetime, timedelta

import pytest

from drone_agent.rules import RulesEngine, in_window, restricted_keyword, same_vehicle, where
from drone_agent.types import TrackState

T0 = datetime(2026, 11, 24, 14, 30, 0)


def track(category="person", first=T0, seen_s=0.0, location="at the main gate", verified=True, **kw):
    tr = TrackState(track_id=kw.pop("track_id", 1), label=kw.pop("label", category), category=category,
                    first_ts=first, last_ts=first + timedelta(seconds=seen_s), zone=location,
                    zone_enter_ts=first, verified=verified, **kw)
    tr.description = kw.get("description", "") or f"test {category}"
    return tr


def rules_of(alerts):
    return [a.rule for a in alerts]


# ------------------------------------------------------------------ night activity
@pytest.mark.parametrize("hhmm,expected", [("23:30", True), ("00:01", True), ("04:59", True),
                                           ("05:00", False), ("14:30", False), ("21:59", False)])
def test_night_window_crosses_midnight(hhmm, expected):
    h, m = map(int, hhmm.split(":"))
    assert in_window(T0.replace(hour=h, minute=m), "22:00", "05:00") is expected


def test_night_alert_at_midnight(cfg):
    first = datetime(2026, 11, 25, 0, 1, 0)
    tr = track(first=first, seen_s=2)
    alerts = RulesEngine(cfg.rules).evaluate(tr, tr.last_ts)
    night = [a for a in alerts if a.rule == "night_activity"]
    assert len(night) == 1
    assert night[0].ts == first                    # timestamped when the person appeared, not later
    assert "00:01:00" in night[0].message and "at the main gate" in night[0].message


def test_no_night_alert_in_afternoon(cfg):
    tr = track(first=T0, seen_s=2)
    assert "night_activity" not in rules_of(RulesEngine(cfg.rules).evaluate(tr, tr.last_ts))


# ------------------------------------------------------------------ loitering
def test_loitering_after_threshold(cfg):
    eng, limit = RulesEngine(cfg.rules), cfg.rules.loitering.seconds
    tr = track(seen_s=limit)
    alerts = [a for a in eng.evaluate(tr, tr.last_ts) if a.rule == "loitering"]
    assert len(alerts) == 1
    assert alerts[0].ts == T0 + timedelta(seconds=limit)
    assert "loitering at the main gate" in alerts[0].message


def test_no_loitering_before_threshold(cfg):
    tr = track(seen_s=cfg.rules.loitering.seconds - 1)
    assert "loitering" not in rules_of(RulesEngine(cfg.rules).evaluate(tr, tr.last_ts))


def test_loitering_fires_once(cfg):
    eng, tr = RulesEngine(cfg.rules), track(seen_s=0)
    fired = []
    for s in range(0, 120):                      # evaluated every second for 2 minutes
        tr.last_ts = T0 + timedelta(seconds=s)
        fired += [a for a in eng.evaluate(tr, tr.last_ts) if a.rule == "loitering"]
    assert len(fired) == 1


def test_loitering_only_for_people(cfg):
    tr = track(category="vehicle", seen_s=60)
    assert "loitering" not in rules_of(RulesEngine(cfg.rules).evaluate(tr, tr.last_ts))


# ------------------------------------------------------------------ guards
def test_unverified_track_never_alerts(cfg):
    tr = track(first=datetime(2026, 11, 25, 0, 1), seen_s=60, verified=None)
    assert RulesEngine(cfg.rules).evaluate(tr, tr.last_ts) == []


def test_short_blip_never_alerts(cfg):
    """A 0.2 s detection (e.g. a tiny false 'person' at night) must not raise an alert."""
    tr = track(first=datetime(2026, 11, 25, 0, 1), seen_s=0.2, location="in the backyard")
    assert RulesEngine(cfg.rules).evaluate(tr, tr.last_ts) == []


# ------------------------------------------------------------------ restricted area (VLM location text)
@pytest.mark.parametrize("location,hit", [("in the backyard", "backyard"), ("near the Backyard fence", "backyard"),
                                          ("in front of the garage", None), ("backyards of houses", None),
                                          ("", None), ("unknown", None)])
def test_restricted_keyword(location, hit):
    assert restricted_keyword(location, ["backyard"]) == hit


def test_restricted_alert(cfg):
    tr = track(location="in the backyard", seen_s=2)
    alerts = [a for a in RulesEngine(cfg.rules).evaluate(tr, tr.last_ts) if a.rule == "restricted_zone"]
    assert len(alerts) == 1 and alerts[0].severity == "critical"


# ------------------------------------------------------------------ repeat vehicle
def test_repeat_vehicle_second_visit(cfg):
    eng = RulesEngine(cfg.rules)
    attrs = {"color": "blue", "type": "pickup truck"}
    first = track("vehicle", seen_s=20, location="in front of the garage", track_id=1, attributes=dict(attrs))
    assert eng.on_verified(first) == []
    later = T0 + timedelta(seconds=20 + cfg.rules.repeat_vehicle.min_gap_s + 5)
    again = track("vehicle", first=later, seen_s=10, location="in front of the garage", track_id=2,
                  attributes=dict(attrs))
    alerts = eng.on_verified(again)
    assert rules_of(alerts) == ["repeat_vehicle"]
    assert "2nd sighting" in alerts[0].message


def test_no_repeat_within_same_visit(cfg):
    """A track id switch a few seconds later is the same visit, not a second sighting."""
    eng = RulesEngine(cfg.rules)
    attrs = {"color": "blue", "type": "pickup truck"}
    eng.on_verified(track("vehicle", seen_s=10, track_id=1, attributes=dict(attrs)))
    soon = track("vehicle", first=T0 + timedelta(seconds=12), seen_s=5, track_id=2, attributes=dict(attrs))
    assert eng.on_verified(soon) == []


def test_different_colour_is_not_same_vehicle():
    a = track("vehicle", attributes={"color": "blue", "type": "pickup truck"})
    b = track("vehicle", attributes={"color": "white", "type": "pickup truck"})
    assert not same_vehicle(a, b, 0.9)


# ------------------------------------------------------------------ message wording
@pytest.mark.parametrize("loc,text", [("at the main gate", " at the main gate"), ("on the main road", " on the main road"),
                                      ("main road", " at main road"), ("unknown", ""), ("", "")])
def test_where_phrasing(loc, text):
    assert where(loc) == text


# ------------------------------------------------------------------ unusual activity (VLM scene field)
@pytest.mark.parametrize("text", ["none", "None", "", "no unusual activity", "nothing", "normal", "N/A",
                                  "None observed."])
def test_unusual_nothing_does_not_alert(cfg, text):
    assert RulesEngine(cfg.rules).on_scene({"unusual": text}, T0) == []


def test_unusual_activity_alert_with_cooldown(cfg):
    eng = RulesEngine(cfg.rules)
    first = eng.on_scene({"unusual": "car crashing through the gate"}, T0)
    assert rules_of(first) == ["unusual_activity"] and first[0].severity == "critical"
    assert first[0].message == "Unusual activity: car crashing through the gate, 14:30:00"
    assert first[0].ts == T0
    cool = cfg.rules.unusual_activity.cooldown_s
    # same incident, new detail: no second alert, the first one is extended
    assert eng.on_scene({"unusual": "heavy smoke"}, T0 + timedelta(seconds=1)) == []
    assert eng.updated == first
    assert first[0].message.endswith("; then heavy smoke, 14:30:01")
    eng.updated.clear()
    assert eng.on_scene({"unusual": "heavy smoke"}, T0 + timedelta(seconds=2)) == [] and eng.updated == []
    # quiet for longer than the cooldown -> a new incident
    later = T0 + timedelta(seconds=2 + cool + 1)
    assert rules_of(eng.on_scene({"unusual": "heavy smoke"}, later)) == ["unusual_activity"]
