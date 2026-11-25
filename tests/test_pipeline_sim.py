"""End-to-end on the text-simulated demo (the brief's own examples): logs, alerts, frame index."""
import re

from conftest import NIGHT_START


def alerts_by_rule(db, rid):
    out = {}
    for a in db.alerts(rid):
        out.setdefault(a["rule"], []).append(a)
    return out


# ------------------------------------------------------------------ logging
def test_truck_logged_correctly(demo):
    db, rid = demo
    spotted = [e for e in db.events(rid, event_type="spotted") if "F150" in e["message"]]
    assert len(spotted) == 2                                  # it came twice
    first = spotted[0]
    assert first["message"] == "Blue Ford F150 pickup truck spotted in front of the garage, 23:59:35"
    assert first["category"] == "vehicle" and first["video_t"] == 5.0


def test_every_object_logged_with_location(demo):
    db, rid = demo
    msgs = [e["message"] for e in db.events(rid, event_type="spotted")]
    for expected in ("Man in red jacket spotted at the main gate", "White Toyota Corolla sedan spotted on the driveway",
                     "Brown dog spotted on the lawn", "spotted in the backyard"):
        assert any(expected in m for m in msgs), expected


def test_left_view_event(demo):
    db, rid = demo
    left = [e["message"] for e in db.events(rid, event_type="left")]
    assert any(m.startswith("Man in red jacket left view at the main gate after 50s") for m in left)


# ------------------------------------------------------------------ alerts
def test_loitering_alert_at_main_gate(demo):
    db, rid = demo
    lo = [a for a in alerts_by_rule(db, rid)["loitering"] if "red jacket" in a["message"]]
    assert len(lo) == 1
    assert lo[0]["message"] == "Man in red jacket loitering at the main gate for over 10s, 23:59:50"
    assert lo[0]["video_t"] == 20.0                           # appeared at 10 s + 10 s threshold


def test_night_alerts_after_midnight(demo):
    db, rid = demo
    night = alerts_by_rule(db, rid)["night_activity"]
    assert {a["ts"][11:19] for a in night} == {"23:59:40", "00:00:32"}   # both people; vehicles excluded


def test_restricted_backyard_alert(demo):
    db, rid = demo
    rz = alerts_by_rule(db, rid)["restricted_zone"]
    assert len(rz) == 1 and "in the backyard" in rz[0]["message"] and rz[0]["severity"] == "critical"


def test_repeat_vehicle_alert(demo):
    db, rid = demo
    rep = alerts_by_rule(db, rid)["repeat_vehicle"]
    assert len(rep) == 1
    assert "Blue Ford F150" in rep[0]["message"] and "2nd sighting, first at 23:59:35" in rep[0]["message"]
    assert rep[0]["video_t"] == 75.0


def test_no_alert_for_the_dog_or_the_corolla(demo):
    db, rid = demo
    assert not [a for a in db.alerts(rid) if "dog" in a["message"].lower() or "Corolla" in a["message"]]


def test_no_night_alerts_in_daytime(demo_day):
    db, rid = demo_day
    rules = alerts_by_rule(db, rid)
    assert "night_activity" not in rules
    assert "loitering" in rules and "repeat_vehicle" in rules   # time-independent rules still fire


# ------------------------------------------------------------------ frame index
def test_every_frame_indexed_with_text_and_telemetry(demo, cfg):
    db, rid = demo
    frames = db.q("SELECT * FROM frames WHERE run_id=? ORDER BY id", (rid,))
    assert len(frames) == 100 * cfg.source.sample_fps
    assert frames[0]["ts"].startswith(NIGHT_START)
    assert re.match(r"Frame 1: nothing detected \| Time 23:59:30, Lat [\d.]+, Lon [\d.]+, Alt \d+ m, Battery \d+%",
                    frames[0]["caption"])
    assert all(f["altitude_m"] and f["lat"] for f in frames)


def test_query_by_object_and_time(demo):
    db, rid = demo
    trucks = db.events(rid, label="truck", event_type="spotted")
    assert len(trucks) == 2
    window = db.events(rid, start_s=60, end_s=80)              # by video seconds
    assert {e["track_id"] for e in window} >= {e["track_id"] for e in trucks if e["video_t"] >= 60}
    by_clock = db.events(rid, start="2026-11-25T00:00:00", end="2026-11-25T00:00:30")
    assert by_clock and all(e["ts"] >= "2026-11-25T00:00:00" for e in by_clock)
