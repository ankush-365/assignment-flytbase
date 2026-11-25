"""SQLite storage: every analysed frame is indexed, plus detections, tracks, VLM scenes, events, alerts
and embeddings. Every row carries two clocks:

  ts       - simulated wall-clock time (ISO string)  -> "what happened between 00:00 and 00:05?"
  video_t  - seconds into the source video          -> jump to that point of the clip

SQLite (WAL mode) is the system of record: structured queries are plain SQL with indexes on (run_id, ts).
Semantic search uses embeddings stored as BLOBs + brute-force cosine in numpy - fine up to ~10^5-10^6 rows;
see the report for the pgvector / FAISS upgrade path.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT, started_at TEXT, finished_at TEXT, sim_start TEXT, sim_end TEXT,
    frames INTEGER DEFAULT 0, config TEXT, summary TEXT, stats TEXT
);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, frame_idx INTEGER, video_t REAL, ts TEXT,
    lat REAL, lon REAL, altitude_m REAL, heading_deg REAL, battery_pct REAL,
    n_objects INTEGER, labels TEXT, caption TEXT, image_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_frames_run_ts ON frames(run_id, ts);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, frame_id INTEGER, ts TEXT, video_t REAL, track_id INTEGER,
    label TEXT, category TEXT, conf REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, zone TEXT
);
CREATE INDEX IF NOT EXISTS ix_det_run_label ON detections(run_id, label, ts);
CREATE INDEX IF NOT EXISTS ix_det_run_track ON detections(run_id, track_id);
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, frame_id INTEGER, ts TEXT, video_t REAL,
    lat REAL, lon REAL, altitude_m REAL,
    scene TEXT, attributes TEXT, objects TEXT, n_objects INTEGER,
    model TEXT, latency_s REAL, image_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_scenes_run_ts ON scenes(run_id, ts);
CREATE TABLE IF NOT EXISTS tracks (
    run_id INTEGER, track_id INTEGER, label TEXT, category TEXT,
    verified INTEGER, verify_source TEXT, vlm_label TEXT, color TEXT, obj_type TEXT, state TEXT,
    description TEXT, attributes TEXT,
    first_ts TEXT, last_ts TEXT, first_video_t REAL, last_video_t REAL,
    verified_ts TEXT, verified_video_t REAL,
    zones TEXT, hits INTEGER, best_conf REAL, thumb_path TEXT,
    PRIMARY KEY (run_id, track_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, ts TEXT, video_t REAL, type TEXT, track_id INTEGER, label TEXT, category TEXT,
    zone TEXT, message TEXT, frame_id INTEGER
);
CREATE INDEX IF NOT EXISTS ix_events_run_ts ON events(run_id, ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, ts TEXT, video_t REAL, rule TEXT, severity TEXT, track_id INTEGER,
    zone TEXT, message TEXT, frame_id INTEGER, snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_run_ts ON alerts(run_id, ts);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER, kind TEXT, ref_id INTEGER, ts TEXT, video_t REAL, text TEXT, vector BLOB
);
CREATE INDEX IF NOT EXISTS ix_emb_run ON embeddings(run_id, kind);
"""

SYNONYMS = {"people": "person", "persons": "person", "man": "person", "men": "person", "woman": "person",
            "women": "person", "pedestrian": "person", "pedestrians": "person", "vehicles": "vehicle",
            "buses": "bus", "bikes": "bike", "lorry": "truck", "lorries": "truck", "pickup": "truck"}


def _ts(v: datetime | str | None) -> str | None:
    if v is None:
        return None
    return v.isoformat(timespec="seconds") if isinstance(v, datetime) else str(v)


def _r(v: float | None, nd: int = 2) -> float | None:
    return None if v is None else round(float(v), nd)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self._pending = 0

    # ------------------------------------------------------------------ writes
    def _exec(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, tuple(params))
            self._pending += 1
            if self._pending >= 200:
                self.conn.commit()
                self._pending = 0
            return cur.lastrowid

    def commit(self) -> None:
        with self.lock:
            self.conn.commit()
            self._pending = 0

    def start_run(self, source: str, config: dict, sim_start: datetime) -> int:
        rid = self._exec("INSERT INTO runs(source, started_at, sim_start, config) VALUES (?,?,?,?)",
                         (source, _ts(datetime.now()), _ts(sim_start), json.dumps(config, default=str)))
        self.commit()
        return rid

    def finish_run(self, run_id: int, frames: int, sim_end: datetime | None, stats: dict) -> None:
        self._exec("UPDATE runs SET finished_at=?, frames=?, sim_end=?, stats=? WHERE id=?",
                   (_ts(datetime.now()), frames, _ts(sim_end), json.dumps(stats), run_id))
        self.commit()

    def set_summary(self, run_id: int, summary: str) -> None:
        self._exec("UPDATE runs SET summary=? WHERE id=?", (summary, run_id))
        self.commit()

    def add_frame(self, run_id, pkt, n_objects: int, labels: list[str], caption: str | None = None,
                  image_path: str | None = None) -> int:
        t = pkt.telemetry
        return self._exec(
            "INSERT INTO frames(run_id, frame_idx, video_t, ts, lat, lon, altitude_m, heading_deg, battery_pct,"
            " n_objects, labels, caption, image_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, pkt.idx, _r(pkt.video_t, 3), _ts(pkt.ts), t.lat, t.lon, t.altitude_m, t.heading_deg,
             t.battery_pct, n_objects, ",".join(sorted(set(labels))), caption, image_path))

    def set_frame_caption(self, frame_id: int, caption: str, image_path: str | None = None) -> None:
        self._exec("UPDATE frames SET caption=?, image_path=COALESCE(?, image_path) WHERE id=?",
                   (caption, image_path, frame_id))

    def add_detections(self, run_id, frame_id, ts, video_t, dets) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT INTO detections(run_id, frame_id, ts, video_t, track_id, label, category, conf,"
                " x1, y1, x2, y2, zone) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, frame_id, _ts(ts), _r(video_t, 3), d.track_id, d.label, d.category, round(d.conf, 3),
                  *[round(v, 1) for v in d.bbox], d.zone) for d in dets])
            self._pending += 1

    def add_scene(self, run_id, frame_id, ts, video_t, telemetry, scene: str, attributes: dict, objects: dict,
                  model: str, latency_s: float, image_path: str | None) -> int:
        t = telemetry
        return self._exec(
            "INSERT INTO scenes(run_id, frame_id, ts, video_t, lat, lon, altitude_m, scene, attributes, objects,"
            " n_objects, model, latency_s, image_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, frame_id, _ts(ts), _r(video_t), t.lat if t else None, t.lon if t else None,
             t.altitude_m if t else None, scene, json.dumps(attributes), json.dumps(objects, default=str),
             len(objects), model, _r(latency_s), image_path))

    def upsert_track(self, run_id, tr) -> None:
        verified = None if tr.verified is None else int(tr.verified)
        a = tr.attributes or {}
        self._exec(
            "INSERT INTO tracks(run_id, track_id, label, category, verified, verify_source, vlm_label, color,"
            " obj_type, state, description, attributes, first_ts, last_ts, first_video_t, last_video_t,"
            " verified_ts, verified_video_t, zones, hits, best_conf, thumb_path)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(run_id, track_id) DO UPDATE SET category=excluded.category, verified=excluded.verified,"
            " verify_source=excluded.verify_source, vlm_label=excluded.vlm_label, color=excluded.color,"
            " obj_type=excluded.obj_type, state=excluded.state, description=excluded.description,"
            " attributes=excluded.attributes, last_ts=excluded.last_ts, last_video_t=excluded.last_video_t,"
            " verified_ts=excluded.verified_ts, verified_video_t=excluded.verified_video_t,"
            " zones=excluded.zones, hits=excluded.hits, best_conf=excluded.best_conf,"
            " thumb_path=COALESCE(NULLIF(excluded.thumb_path, ''), tracks.thumb_path)",
            (run_id, tr.track_id, tr.label, tr.category, verified, tr.verify_source, tr.vlm_label,
             a.get("color"), a.get("type"), a.get("state"), tr.description, json.dumps(a, default=str),
             _ts(tr.first_ts), _ts(tr.last_ts), _r(tr.first_video_t), _r(tr.last_video_t),
             _ts(tr.verified_ts), _r(tr.verified_video_t), ",".join(tr.zones_visited), tr.hits,
             round(tr.best_conf, 3), tr.thumb_path))

    def add_event(self, run_id, ev, frame_id=None) -> int:
        return self._exec(
            "INSERT INTO events(run_id, ts, video_t, type, track_id, label, category, zone, message, frame_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, _ts(ev.ts), _r(ev.video_t), ev.type, ev.track_id, ev.label, ev.category, ev.zone, ev.message,
             frame_id))

    def add_alert(self, run_id, al, frame_id=None, snapshot_path=None) -> int:
        return self._exec(
            "INSERT INTO alerts(run_id, ts, video_t, rule, severity, track_id, zone, message, frame_id,"
            " snapshot_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, _ts(al.ts), _r(al.video_t), al.rule, al.severity, al.track_id, al.zone, al.message, frame_id,
             snapshot_path))

    def update_alert_message(self, alert_id: int, message: str) -> None:
        self._exec("UPDATE alerts SET message=? WHERE id=?", (message, alert_id))
        self.commit()

    def add_embedding(self, run_id, kind: str, ref_id: int, ts, video_t, text: str, vec: np.ndarray) -> None:
        self._exec("INSERT INTO embeddings(run_id, kind, ref_id, ts, video_t, text, vector) VALUES (?,?,?,?,?,?,?)",
                   (run_id, kind, ref_id, _ts(ts), _r(video_t), text, np.asarray(vec, np.float32).tobytes()))

    # ------------------------------------------------------------------ reads
    def q(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def latest_run_id(self) -> int | None:
        rows = self.q("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
        return rows[0]["id"] if rows else None

    def runs(self) -> list[dict]:
        return self.q("SELECT * FROM runs ORDER BY id DESC")

    def run(self, run_id: int) -> dict | None:
        rows = self.q("SELECT * FROM runs WHERE id=?", (run_id,))
        return rows[0] if rows else None

    @staticmethod
    def _label_clause(label: str | None, cols: tuple[str, ...]) -> tuple[str, list]:
        if not label:
            return "", []
        label = label.lower().strip()
        label = SYNONYMS.get(label, label)
        if label.endswith("s") and label not in ("bus", "glass"):
            label = label[:-1]
        parts = " OR ".join(f"LOWER({c}) LIKE ?" for c in cols)
        return f" AND ({parts})", [f"%{label}%"] * len(cols)

    @staticmethod
    def _time_clause(start=None, end=None, start_s=None, end_s=None, ts_col="ts", vt_col="video_t"):
        sql, params = "", []
        if start:
            sql += f" AND {ts_col} >= ?"
            params.append(start)
        if end:
            sql += f" AND {ts_col} <= ?"
            params.append(end)
        if start_s is not None:
            sql += f" AND {vt_col} >= ?"
            params.append(float(start_s))
        if end_s is not None:
            sql += f" AND {vt_col} <= ?"
            params.append(float(end_s))
        return sql, params

    def events(self, run_id: int, label: str | None = None, start: str | None = None, end: str | None = None,
               event_type: str | None = None, zone: str | None = None, start_s: float | None = None,
               end_s: float | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM events WHERE run_id=?"
        params: list[Any] = [run_id]
        clause, p = self._label_clause(label, ("label", "category", "message"))
        tc, tp = self._time_clause(start, end, start_s, end_s)
        sql += clause + tc
        params += p + tp
        if event_type:
            sql += " AND type = ?"
            params.append(event_type)
        if zone:
            sql += " AND LOWER(zone) LIKE ?"
            params.append(f"%{zone.lower()}%")
        sql += " ORDER BY ts, id LIMIT ?"
        params.append(limit)
        return self.q(sql, params)

    def alerts(self, run_id: int, rule: str | None = None, start: str | None = None, end: str | None = None,
               start_s: float | None = None, end_s: float | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM alerts WHERE run_id=?"
        params: list[Any] = [run_id]
        if rule:
            sql += " AND rule LIKE ?"
            params.append(f"%{rule}%")
        tc, tp = self._time_clause(start, end, start_s, end_s)
        sql += tc + " ORDER BY ts, id LIMIT ?"
        params += tp + [limit]
        return self.q(sql, params)

    def tracks(self, run_id: int, label: str | None = None, verified_only: bool = True, start: str | None = None,
               end: str | None = None, start_s: float | None = None, end_s: float | None = None) -> list[dict]:
        """Tracks visible at any point inside the time window (overlap test)."""
        sql = "SELECT * FROM tracks WHERE run_id=?"
        params: list[Any] = [run_id]
        if verified_only:
            sql += " AND verified=1"
        clause, p = self._label_clause(label, ("label", "category", "vlm_label", "obj_type", "description"))
        sql += clause
        params += p
        if start:
            sql += " AND last_ts >= ?"
            params.append(start)
        if end:
            sql += " AND first_ts <= ?"
            params.append(end)
        if start_s is not None:
            sql += " AND last_video_t >= ?"
            params.append(float(start_s))
        if end_s is not None:
            sql += " AND first_video_t <= ?"
            params.append(float(end_s))
        return self.q(sql + " ORDER BY first_ts, track_id", params)

    def scenes(self, run_id: int, start: str | None = None, end: str | None = None, start_s: float | None = None,
               end_s: float | None = None, text: str | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM scenes WHERE run_id=?"
        params: list[Any] = [run_id]
        tc, tp = self._time_clause(start, end, start_s, end_s)
        sql += tc
        params += tp
        if text:
            sql += " AND (LOWER(scene) LIKE ? OR LOWER(objects) LIKE ?)"
            params += [f"%{text.lower()}%"] * 2
        sql += " ORDER BY ts, id LIMIT ?"
        params.append(limit)
        rows = self.q(sql, params)
        for r in rows:
            r["attributes"] = json.loads(r["attributes"] or "{}")
            r["objects"] = json.loads(r["objects"] or "{}")
        return rows

    def frames(self, run_id: int, start: str | None = None, end: str | None = None, label: str | None = None,
               start_s: float | None = None, end_s: float | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM frames WHERE run_id=?"
        params: list[Any] = [run_id]
        tc, tp = self._time_clause(start, end, start_s, end_s)
        sql += tc
        params += tp
        if label:
            sql += " AND (labels LIKE ? OR LOWER(caption) LIKE ?)"
            params += [f"%{label.lower()}%"] * 2
        sql += " ORDER BY frame_idx LIMIT ?"
        params.append(limit)
        return self.q(sql, params)

    def object_counts(self, run_id: int) -> list[dict]:
        return self.q(
            "SELECT category, COALESCE(NULLIF(vlm_label,''), label) AS object, COUNT(*) AS n FROM tracks"
            " WHERE run_id=? AND verified=1 GROUP BY category, object ORDER BY n DESC", (run_id,))

    def stats(self, run_id: int) -> dict:
        one = lambda sql: self.q(sql, (run_id,))[0]  # noqa: E731
        return {
            "frames": one("SELECT COUNT(*) n FROM frames WHERE run_id=?")["n"],
            "detections": one("SELECT COUNT(*) n FROM detections WHERE run_id=?")["n"],
            "tracks_total": one("SELECT COUNT(*) n FROM tracks WHERE run_id=?")["n"],
            "tracks_verified": one("SELECT COUNT(*) n FROM tracks WHERE run_id=? AND verified=1")["n"],
            "tracks_rejected": one("SELECT COUNT(*) n FROM tracks WHERE run_id=? AND verified=0")["n"],
            "tracks_unverified": one("SELECT COUNT(*) n FROM tracks WHERE run_id=? AND verified IS NULL")["n"],
            "vlm_scenes": one("SELECT COUNT(*) n FROM scenes WHERE run_id=?")["n"],
            "vlm_avg_latency_s": one("SELECT ROUND(AVG(latency_s),2) n FROM scenes WHERE run_id=?")["n"],
            "events": one("SELECT COUNT(*) n FROM events WHERE run_id=?")["n"],
            "alerts": one("SELECT COUNT(*) n FROM alerts WHERE run_id=?")["n"],
            "time_range": [one("SELECT MIN(ts) a FROM frames WHERE run_id=?")["a"],
                           one("SELECT MAX(ts) b FROM frames WHERE run_id=?")["b"]],
        }

    def embeddings(self, run_id: int, kind: str | None = None) -> tuple[list[dict], np.ndarray | None]:
        sql = "SELECT id, kind, ref_id, ts, video_t, text, vector FROM embeddings WHERE run_id=?"
        params: list[Any] = [run_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        rows = self.q(sql, params)
        if not rows:
            return [], None
        mat = np.stack([np.frombuffer(r.pop("vector"), dtype=np.float32) for r in rows])
        return rows, mat

    def close(self) -> None:
        self.commit()
        self.conn.close()
