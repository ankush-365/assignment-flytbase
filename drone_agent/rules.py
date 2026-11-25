"""Rules engine. Each rule is a pure function of (track state, time, config) -> alerts.

Rules only fire on VLM-*verified* tracks (or all tracks when the VLM is disabled), and each rule fires
at most once per track, so a 2-minute loiterer produces one alert, not 480.
"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta

import numpy as np

from .types import Alert, TrackState
from .vlm import cosine


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def in_window(ts: datetime, start: str, end: str) -> bool:
    """True if ts's time-of-day is inside [start, end); handles windows crossing midnight."""
    t, s, e = ts.time(), _hhmm(start), _hhmm(end)
    return s <= t < e if s <= e else (t >= s or t < e)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


_PREP = re.compile(r"^(at|on|in|near|by|inside|outside|behind|beside|next|along|across|under|over|crossing)\b")


def where(location: str) -> str:
    """' on the main road' / ' at the gate' / '' - so messages read naturally for free-text locations."""
    loc = (location or "").strip()
    if not loc or loc == "unknown":
        return ""
    return f" {loc}" if _PREP.match(loc.lower()) else f" at {loc}"


def fmt_time(ts: datetime) -> str:
    return ts.strftime("%H:%M:%S")


class RulesEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.updated: list[Alert] = []   # stored alerts whose message changed (ongoing incidents)
        # history of finished/active verified vehicles for repeat matching: (track_id, last_ts, emb, attrs, desc)
        self.vehicle_history: list[TrackState] = []

    @staticmethod
    def _eligible(tr: TrackState) -> bool:
        return tr.verified is True

    def evaluate(self, tr: TrackState, now: datetime) -> list[Alert]:
        """Called every frame for every active track (and again when a late VLM verdict arrives).

        Alert timestamps are the moment the condition became true (e.g. first seen + threshold), not the
        moment the asynchronous VLM verdict arrived, so logs and time-range searches stay accurate."""
        if not self._eligible(tr):
            return []
        # a 1-2 frame blip (e.g. a tiny false detection the VLM also got wrong) never raises an alert;
        # the alert timestamp is still the moment the condition began
        if (tr.last_ts - tr.first_ts).total_seconds() < self.cfg.get("min_track_s", 0.0):
            return []
        out: list[Alert] = []
        r = self.cfg
        name = tr.display_name()

        lo = r.loitering
        if lo.enabled and tr.category in lo.categories and "loitering" not in tr.alerted_rules:
            # time in view, not time per location: VLM location wording varies between calls
            dwell = (now - tr.first_ts).total_seconds()
            if dwell >= lo.seconds:
                tr.alerted_rules.add("loitering")
                at = tr.first_ts + timedelta(seconds=lo.seconds)
                out.append(Alert("loitering", "warning",
                                 f"{cap(name)} loitering{where(tr.zone)} for over {lo.seconds:.0f}s, {fmt_time(at)}",
                                 tr.track_id, at, tr.zone))

        na = r.night_activity
        if na.enabled and tr.category in na.categories and "night_activity" not in tr.alerted_rules:
            if in_window(tr.first_ts, na.start, na.end):
                tr.alerted_rules.add("night_activity")
                out.append(Alert("night_activity", "warning",
                                 f"{cap(name)} detected{where(tr.zone)} during night hours, {fmt_time(tr.first_ts)}",
                                 tr.track_id, tr.first_ts, tr.zone))

        rz = r.restricted_zone
        hit = restricted_keyword(tr.zone, rz.get("keywords", [])) if rz.enabled else None
        key = f"restricted:{hit}"
        if hit and tr.category in rz.categories and key not in tr.alerted_rules:
            tr.alerted_rules.add(key)
            out.append(Alert("restricted_zone", "critical",
                             f"{cap(name)} in restricted area ({tr.zone}), {fmt_time(tr.zone_enter_ts)}",
                             tr.track_id, tr.zone_enter_ts, tr.zone))
        return out

    def on_scene(self, attrs: dict, ts: datetime) -> list[Alert]:
        """Unusual activity: the VLM's per-second `unusual` field (e.g. "car crashing through the gate",
        "heavy smoke") describes something no object rule can see. One alert, then a cooldown so a
        10-second incident is not 10 alerts."""
        ua = self.cfg.get("unusual_activity")
        if not (ua and ua.enabled):
            return []
        text = str((attrs or {}).get("unusual", "") or "").strip().rstrip(".")
        if is_nothing(text):
            return []
        cur = getattr(self, "_unusual", None)          # the open incident: (alert, last seen ts, texts)
        if cur is not None and (ts - cur[1]).total_seconds() < ua.get("cooldown_s", 10):
            alert, _, texts = cur
            self._unusual = (alert, ts, texts)
            if text.lower() not in texts:              # same incident, new detail -> extend the alert
                texts.add(text.lower())
                alert.message += f"; then {text}, {fmt_time(ts)}"
                self.updated.append(alert)
            return []
        alert = Alert("unusual_activity", ua.get("severity", "critical"),
                      f"Unusual activity: {text}, {fmt_time(ts)}", None, ts, "")
        self._unusual = (alert, ts, {text.lower()})
        return [alert]

    def on_verified(self, tr: TrackState) -> list[Alert]:
        """Called once when a vehicle track is verified: check whether we've seen it before."""
        rv = self.cfg.repeat_vehicle
        if not (rv.enabled and tr.category == "vehicle" and tr.verified):
            return []
        matches = []
        for prev in self.vehicle_history:
            if prev.track_id == tr.track_id:
                continue
            gap = (tr.first_ts - prev.last_ts).total_seconds()
            if gap < rv.min_gap_s:
                continue
            if same_vehicle(tr, prev, rv.similarity):
                matches.append(prev)
        self.vehicle_history.append(tr)
        if not matches:
            return []
        tr.alerted_rules.add("repeat_vehicle")
        n = len(matches) + 1
        first = min(m.first_ts for m in matches)
        return [Alert("repeat_vehicle", "info",
                      f"{cap(tr.display_name())} seen again{where(tr.zone)} ({_ordinal(n)} sighting, "
                      f"first at {fmt_time(first)}), {fmt_time(tr.first_ts)}", tr.track_id, tr.first_ts, tr.zone)]


_NOTHING = re.compile(r"^(none|no|nothing|n/?a|normal|null|unknown|not applicable)\b|^no (unusual|suspicious|anomal)",
                      re.I)


def is_nothing(text: str) -> bool:
    """True for the VLM's ways of saying 'nothing unusual'."""
    return not text or bool(_NOTHING.match(text.strip()))


def restricted_keyword(location: str, keywords) -> str | None:
    """First restricted keyword found as a whole word in the (VLM / simulated) location text."""
    loc = (location or "").lower()
    for k in keywords or []:
        if re.search(rf"\b{re.escape(k.lower())}\b", loc):
            return k
    return None


def same_vehicle(a: TrackState, b: TrackState, threshold: float) -> bool:
    """Colour must agree (when both known); then description embeddings must be similar.
    Without embeddings fall back to exact (colour, type) attribute match."""
    ca, cb = a.attributes.get("color", "unknown"), b.attributes.get("color", "unknown")
    if "unknown" not in (ca, cb) and ca != cb:
        return False
    if a.embedding is not None and b.embedding is not None:
        return cosine(np.asarray(a.embedding), np.asarray(b.embedding)) >= threshold
    ta = a.attributes.get("type") or a.vlm_label or a.label
    tb = b.attributes.get("type") or b.vlm_label or b.label
    return ca == cb and ca != "unknown" and ta == tb


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
