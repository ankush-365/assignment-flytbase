"""Main pipeline: frames -> YOLO+ByteTrack -> track state -> per-second VLM (+object locations) (async) -> rules -> SQLite + vectors.

 frames+telemetry ─► YOLO+ByteTrack ─► track state (dwell, best crop) ─► rules ─► alerts
                                            │                                   ▲
                       once per video second│(if objects present)               │ verified tracks only
                                            ▼                                   │
                     VLM worker: full frame + numbered crop sheet ─► scene + per-object verdict/attributes
                                            │
                                            ▼
                      SQLite: frames, detections, scenes, tracks, events, alerts, embeddings (all timestamped)

VLM cost is bounded by video length (<= 1 call per second that contains objects), not by object count.
Objects that never got a VLM slot (busy seconds) fall back to a class-level verdict: their YOLO class was
confirmed by the VLM elsewhere in the run, and their colour comes from a cheap OpenCV estimate.
"""
from __future__ import annotations

import logging
import time as _time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import cv2
import numpy as np

from .config import Cfg, resolve
from .db import Database
from .rules import RulesEngine, cap, fmt_time, where
from .types import Alert, Detection, Event, FramePacket, TrackState
from .vlm import Embedder, OllamaVLM, crop_object, dominant_color

log = logging.getLogger(__name__)

COLORS = {True: (60, 200, 60), None: (0, 165, 255), False: (90, 90, 200)}
PRIORITY = {"person": 0, "vehicle": 1, "object": 2}
VEHICLE_WORDS = ("car", "truck", "bus", "van", "pickup", "suv", "sedan", "hatchback", "motorcycle", "motorbike",
                 "scooter", "bicycle", "bike", "lorry", "jeep", "taxi", "vehicle", "minibus", "trailer")
PERSON_WORDS = ("person", "man", "woman", "pedestrian", "people", "child", "cyclist", "worker", "boy", "girl")


def output_video_path(cfg, run_id: int):
    """data/output/annotated.mp4 -> data/output/annotated_run_<id>.mp4"""
    p = resolve(cfg.storage.output_video)
    return p.with_name(f"{p.stem}_run_{run_id}{p.suffix}")


def text_category(text: str) -> str | None:
    words = text.lower().replace("-", " ").split()
    if any(w in PERSON_WORDS for w in words):
        return "person"
    if any(w.rstrip("s") in VEHICLE_WORDS or w in VEHICLE_WORDS for w in words):
        return "vehicle"
    return None


@dataclass
class StepResult:
    packet: FramePacket
    annotated: np.ndarray | None
    events: list[Event] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    scenes: list[dict] = field(default_factory=list)
    active_tracks: int = 0
    pending_vlm: int = 0


class Pipeline:
    def __init__(self, cfg: Cfg, db: Database | None = None, vlm=None, embedder=None, detector=None,
                 use_vlm: bool | None = None, use_embeddings: bool | None = None, write_video: bool = True,
                 save_images: bool = True):
        self.cfg = cfg
        self.db = db or Database(resolve(cfg.storage.db))
        self.rules = RulesEngine(cfg.rules)
        self.use_vlm = cfg.vlm.enabled if use_vlm is None else use_vlm
        self.use_emb = cfg.embeddings.enabled if use_embeddings is None else use_embeddings
        self._vlm = vlm
        self._embedder = embedder
        self._detector = detector
        self.write_video = write_video
        self.save_images = save_images
        self.tracks: dict[int, TrackState] = {}
        # tracks that left view before getting a VLM verdict: their saved best crop is sent in later calls
        self.backlog: dict[int, TrackState] = {}
        self.backlog_calls = 0
        self.gave_up: list[int] = []   # tracks dropped after max_attempts failed VLM calls
        self.futures: list[tuple[str, object, Future]] = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
        self.recent_alerts: list[tuple[float, Alert]] = []
        self.confirmed_labels: set[str] = set()   # YOLO classes the VLM confirmed at least once
        self.rejected_labels: dict[str, int] = {}
        self.timings = {"detect": 0.0, "vlm_wait": 0.0, "total": 0.0}
        self.detect_frames = 0
        self.vlm_log: list[dict] = []   # per-call: video_t, latency_s, n_objects, tokens, truncated
        self.t0: datetime | None = None
        self.stats: dict = {}

    # ------------------------------------------------------------------ lazy components
    @property
    def vlm(self):
        if self._vlm is None and self.use_vlm:
            v = self.cfg.vlm
            self._vlm = OllamaVLM(v.model, v.host, v.timeout_s, v.get("frame_max_side", 896), v.get("num_ctx", 4096),
                                  v.get("max_tokens", 200), v.get("tokens_per_object", 70),
                                  v.get("max_tokens_cap", 1600))
        return self._vlm

    @property
    def embedder(self):
        if self._embedder is None and self.use_emb:
            self._embedder = Embedder(self.cfg.embeddings.model, self.cfg.vlm.host)
        return self._embedder

    @property
    def detector(self):
        if self._detector is None:
            from .detector import YoloTracker

            self._detector = YoloTracker(self.cfg.detector, self.cfg.categories)
        return self._detector

    def _vt(self, ts: datetime) -> float:
        """video seconds for a simulated timestamp (sim clock = start + video_t)."""
        return (ts - self.t0).total_seconds() if self.t0 else 0.0

    # ------------------------------------------------------------------ run
    def run(self, source, source_name: str = "", progress_every: int = 0) -> int:
        for i, step in enumerate(self.run_iter(source, source_name)):
            if progress_every and i % progress_every == 0:
                log.info("t=%5.1fs %s objects=%d tracks=%d vlm_queue=%d", step.packet.video_t,
                         fmt_time(step.packet.ts), len(step.packet.sim_detections or []) or step.active_tracks,
                         step.active_tracks, step.pending_vlm)
        return self.run_id

    def run_iter(self, source, source_name: str = "") -> Iterator[StepResult]:
        cfg = self.cfg
        t_start = _time.perf_counter()
        self.t0 = getattr(source, "start", datetime.now())
        self.run_id = self.db.start_run(source_name or getattr(source, "path", type(source).__name__),
                                        dict(cfg), self.t0)
        self.img_dir = resolve(cfg.storage.frames_dir) / f"run_{self.run_id}"
        if self.save_images:
            self.img_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        last_vlm_sec = None
        last_pkt = None
        n = 0
        try:
            for pkt in source:
                n += 1
                last_pkt = pkt
                step = StepResult(packet=pkt, annotated=None)

                # 1. detect + track
                if pkt.sim_detections is not None:
                    dets = pkt.sim_detections
                else:
                    t0 = _time.perf_counter()
                    dets = self.detector(pkt.image)
                    self.timings["detect"] += _time.perf_counter() - t0
                    self.detect_frames += 1
                # location: video detections get it later from the VLM ("on the main road near the traffic
                # lights" - works while the drone moves); text-simulated detections carry it already

                # 2. index the frame + raw detections (every analysed frame)
                frame_id = self.db.add_frame(self.run_id, pkt, len(dets), [d.label for d in dets], caption=pkt.frame_text)
                if dets:
                    self.db.add_detections(self.run_id, frame_id, pkt.ts, pkt.video_t, dets)

                # 3. update track state
                for d in dets:
                    self._update_track(d, pkt, frame_id)

                # 4. VLM once per video second (only if something was detected)
                sec = int(pkt.video_t // cfg.vlm.get("call_every_s", 1.0))
                if pkt.sim_detections is not None:
                    self._sim_verify()
                    if pkt.sim_caption:
                        self._store_scene(frame_id, pkt.ts, pkt.video_t, pkt.telemetry, pkt.sim_caption, {}, {},
                                          "sim", 0.0, None)
                elif dets and sec != last_vlm_sec and self.use_vlm:
                    if len(self.futures) < cfg.vlm.max_in_flight:
                        last_vlm_sec = sec
                        self._submit_second(pkt, frame_id, dets)
                elif not self.use_vlm:
                    self._accept_without_vlm()

                # 5. apply finished VLM jobs
                step.events += self._collect(block=False)

                # 6. rules on tracks visible in this frame
                for tr in self.tracks.values():
                    if tr.last_ts == pkt.ts:
                        step.alerts += self.rules.evaluate(tr, pkt.ts)

                # 7. tracks that disappeared
                step.events += self._expire_tracks(pkt.ts)

                annotated = None
                if pkt.image is not None:
                    annotated = self._annotate(pkt, dets, step.alerts)
                    if self.write_video:
                        vid = annotated
                        if vid.shape[1] > 1920:  # 4K -> 1080p output video
                            vid = cv2.resize(vid, (1920, int(vid.shape[0] * 1920 / vid.shape[1])),
                                             interpolation=cv2.INTER_AREA)
                        if writer is None:
                            out = self.video_path = output_video_path(cfg, self.run_id)   # one file per run, never overwritten
                            out.parent.mkdir(parents=True, exist_ok=True)
                            fps = getattr(source, "effective_fps", cfg.source.sample_fps)
                            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                     (vid.shape[1], vid.shape[0]))
                        writer.write(vid)
                for al in step.alerts:
                    snap = self._save(annotated, f"alert_{pkt.idx:06d}_{al.rule}_{al.track_id}.jpg") \
                        if annotated is not None else None
                    self._write_alert(al, frame_id, snap)

                step.annotated = annotated
                step.active_tracks = len(self.tracks)
                step.pending_vlm = len(self.futures)
                yield step
        finally:
            end_ts = last_pkt.ts if last_pkt else self.t0
            t0 = _time.perf_counter()
            # drain outstanding VLM jobs; late verdicts can still trigger rules (timestamps stay correct)
            while self.futures:
                self._collect(block=True)
                for tr in self.tracks.values():
                    for al in self.rules.evaluate(tr, tr.last_ts):
                        self._write_alert(al, tr.best_frame_id)
            self._expire_tracks(end_ts, force=True)   # every remaining track: finalize or -> backlog
            self._drain_backlog(last_pkt)
            self.timings["vlm_wait"] += _time.perf_counter() - t0
            if writer is not None:
                writer.release()
            self.timings["total"] = _time.perf_counter() - t_start
            stats = self.db.stats(self.run_id)
            stats.update({"video_frames_analysed": n, "wall_s": round(self.timings["total"], 2),
                          "timing": self.timing_report(last_pkt),
                          "vlm_drain_wait_s": round(self.timings["vlm_wait"], 2),
                          "backlog_calls": self.backlog_calls,
                          "vlm_gave_up": self.gave_up,
                          "confirmed_labels": sorted(self.confirmed_labels),
                          "rejected_labels": self.rejected_labels})
            self.db.finish_run(self.run_id, n, end_ts, stats)
            self.pool.shutdown(wait=True)
            self.stats = stats

    def _drain_backlog(self, last_pkt) -> None:
        """End of run: extra crop-only calls for objects that never got a verdict (bounded by
        max_backlog_calls), then a class-level decision for whatever is still unverified."""
        cfg = self.cfg
        while self.use_vlm and last_pkt is not None and last_pkt.image is not None \
                and self.backlog_calls < cfg.vlm.get("max_backlog_calls", 5):
            picked = self._pick_for_vlm(set())
            if not picked:
                break
            self.backlog_calls += 1
            self._submit_items(picked, last_pkt, None, store_scene=False)
            while self.futures:
                self._collect(block=True)
        for tr in list(self.backlog.values()):
            self._class_level(tr)
            self._finalize(tr)

    def timing_report(self, last_pkt=None) -> dict:
        """YOLO vs VLM time breakdown for the run (also stored in runs.stats)."""
        lat = [c["latency_s"] for c in self.vlm_log]
        video_s = (last_pkt.video_t if last_pkt else 0.0) + 1.0 / max(self.cfg.source.sample_fps, 1e-6)
        det = self.timings["detect"]
        return {
            "video_duration_s": round(video_s, 2),
            "yolo_frames": self.detect_frames,
            "yolo_total_s": round(det, 2),
            "yolo_avg_ms_per_frame": round(1000 * det / max(self.detect_frames, 1), 1),
            "vlm_calls": len(lat),
            "vlm_total_s": round(sum(lat), 2),
            "vlm_avg_s": round(sum(lat) / len(lat), 2) if lat else None,
            "vlm_max_s": round(max(lat), 2) if lat else None,
            "vlm_truncated_calls": sum(c["truncated"] for c in self.vlm_log),
            "vlm_failed_calls": sum(bool(c["error"]) for c in self.vlm_log),
            "wall_s": round(self.timings["total"], 2),
            "realtime_factor": round(video_s / max(self.timings["total"], 1e-6), 2),  # >1 = faster than real time
            "calls": self.vlm_log,
        }

    # ------------------------------------------------------------------ tracks
    def _update_track(self, d: Detection, pkt: FramePacket, frame_id: int) -> None:
        v = self.cfg.vlm
        tr = self.tracks.get(d.track_id)
        if tr is None:
            tr = TrackState(track_id=d.track_id, label=d.label, category=d.category, first_ts=pkt.ts,
                            last_ts=pkt.ts, zone=d.zone or "unknown", zone_enter_ts=pkt.ts,
                            first_video_t=pkt.video_t,
                            last_video_t=pkt.video_t, best_conf=d.conf, best_bbox=d.bbox,
                            zones_visited=[d.zone] if d.zone else [], sim_description=d.sim_description,
                            sim_attributes=d.sim_attributes, best_frame_id=frame_id)
            if pkt.image is not None:
                tr.best_crop = crop_object(pkt.image, d.bbox, v.crop_padding, v.min_crop_px)
            self.tracks[d.track_id] = tr
            return
        tr.hits += 1
        tr.last_ts, tr.last_video_t = pkt.ts, pkt.video_t
        if d.zone and d.zone != tr.zone:
            self._set_location(tr, d.zone, pkt.ts)
        # keep the largest / most confident view for the VLM until it has been sent
        if not tr.verify_submitted and tr.verified is None:
            area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            best_area = (tr.best_bbox[2] - tr.best_bbox[0]) * (tr.best_bbox[3] - tr.best_bbox[1])
            if d.conf * area > tr.best_conf * best_area:
                tr.best_conf, tr.best_bbox, tr.best_frame_id = d.conf, d.bbox, frame_id
                if pkt.image is not None:
                    tr.best_crop = crop_object(pkt.image, d.bbox, v.crop_padding, v.min_crop_px)

    @staticmethod
    def _set_location(tr: TrackState, location: str, ts: datetime) -> None:
        if not location or location == "unknown" or location == tr.zone:
            return
        tr.zone, tr.zone_enter_ts = location, ts
        if location not in tr.zones_visited:
            tr.zones_visited.append(location)

    def _eligible(self, tr: TrackState) -> bool:
        return (tr.verified is None and not tr.verify_submitted and tr.best_crop is not None
                and tr.hits >= self.cfg.tracking.min_hits_before_verify
                and tr.category in self.cfg.vlm.verify_categories)

    def _pick_for_vlm(self, visible_ids: set[int]) -> list[TrackState]:
        """Up to N crops per call: unverified tracks visible now first (people > vehicles > objects), then
        spare slots go to the backlog (tracks that already left view without a verdict), oldest first."""
        n = self.cfg.vlm.max_objects_per_call
        cand = [tr for tid, tr in self.tracks.items() if tid in visible_ids and self._eligible(tr)]
        cand.sort(key=lambda t: (PRIORITY.get(t.category, 3), -t.best_conf))
        picked = cand[:n]
        if len(picked) < n:
            back = [tr for tr in self.backlog.values() if self._eligible(tr)]
            back.sort(key=lambda t: (PRIORITY.get(t.category, 3), t.first_ts))
            picked += back[: n - len(picked)]
        return picked

    def _submit_second(self, pkt: FramePacket, frame_id: int, dets: list[Detection]) -> None:
        self._submit_items(self._pick_for_vlm({d.track_id for d in dets}), pkt, frame_id, store_scene=True)

    def _submit_items(self, picked: list[TrackState], pkt: FramePacket, frame_id, store_scene: bool) -> None:
        for tr in picked:
            tr.verify_submitted = True
            if self.save_images:
                tr.thumb_path = self._save(tr.best_crop, f"track_{tr.track_id}.jpg") or ""
        items = [(tr.track_id, tr.label, tr.best_crop) for tr in picked]
        img_path = self._save(pkt.image, f"scene_{pkt.video_t:07.2f}s.jpg") if store_scene else None
        key = {"frame_id": frame_id, "ts": pkt.ts, "video_t": pkt.video_t, "telemetry": pkt.telemetry,
               "image_path": img_path, "ids": [tr.track_id for tr in picked], "store_scene": store_scene,
               # counted at submit time: how many crops in this call had already left view
               "n_backlog": sum(1 for tr in picked if tr.track_id in self.backlog)}
        self._submit("second", key, self._job_second, pkt.image.copy(), items, fmt_time(pkt.ts),
                     pkt.telemetry.altitude_m)

    def _sim_verify(self) -> None:
        for tr in self.tracks.values():
            if tr.verified is None and not tr.verify_submitted and tr.sim_description is not None:
                tr.verify_submitted = True
                self._submit("sim", tr.track_id, self._job_sim, tr.sim_description, dict(tr.sim_attributes or {}))

    def _accept_without_vlm(self) -> None:
        for tr in self.tracks.values():
            if tr.verified is None and tr.hits >= self.cfg.tracking.min_hits_before_verify:
                self._class_level(tr, source="none", trust_yolo=True)
                self._announce(tr)

    def _class_level(self, tr: TrackState, source: str = "class-level", trust_yolo: bool = False) -> None:
        """Verdict for a track that never got its own VLM look. Runs only after all VLM verdicts are in.
        It can ACCEPT (the VLM confirmed this YOLO class on other objects) but never REJECT: without
        evidence the track stays unverified. Only an explicit VLM is_match=false rejects."""
        tr.verified = True if (trust_yolo or tr.label in self.confirmed_labels) else None
        color = dominant_color(tr.best_crop) if tr.best_crop is not None else "unknown"
        tr.vlm_label = tr.label
        tr.description = f"{color} {tr.label}" if color != "unknown" else tr.label
        tr.attributes = {"color": color, "type": tr.label, "state": "unknown", "yolo_label": tr.label,
                         "source": source}
        tr.verify_source = source
        tr.verified_ts, tr.verified_video_t = tr.last_ts, tr.last_video_t
        if tr.verified and tr.embedding is None:
            tr.embedding = self._embed(self._track_text(tr))

    def _expire_tracks(self, now: datetime, force: bool = False) -> list[Event]:
        """Tracks unseen for lost_after_s leave the active set. Without a verdict yet (in flight, or never
        sent) they go to the backlog and are finalized when their verdict arrives."""
        events = []
        lost_after = self.cfg.tracking.lost_after_s
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if not force and (now - tr.last_ts).total_seconds() < lost_after:
                continue
            del self.tracks[tid]
            if tr.verified is None and (tr.verify_submitted or (self.use_vlm and self._eligible(tr))):
                self.backlog[tid] = tr
                continue
            events += self._finalize(tr)
        return events

    def _finalize(self, tr: TrackState) -> list[Event]:
        """Track is gone and has its final verdict: write spotted/rejected (if not yet), rules, 'left'."""
        events = self._announce(tr)
        if tr.verified:
            for al in self.rules.evaluate(tr, tr.last_ts):
                self._write_alert(al, tr.best_frame_id)
        self.backlog.pop(tr.track_id, None)
        self.db.upsert_track(self.run_id, tr)
        if tr.verified:
            dur = (tr.last_ts - tr.first_ts).total_seconds()
            ev = Event("left", f"{cap(tr.display_name())} left view{where(tr.zone)} after {dur:.0f}s, "
                               f"{fmt_time(tr.last_ts)}", tr.last_ts, tr.track_id, tr.vlm_label or tr.label,
                       tr.category, tr.zone, tr.last_video_t)
            self.db.add_event(self.run_id, ev)
            events.append(ev)
        return events

    # ------------------------------------------------------------------ worker-thread jobs
    def _submit(self, kind: str, key, fn, *args) -> None:
        self.futures.append((kind, key, self.pool.submit(fn, *args)))

    def _embed(self, text: str):
        if not (self.use_emb and self.embedder and text):
            return None
        try:
            return self.embedder([text])[0]
        except Exception as exc:  # embeddings are best-effort
            log.debug("embedding failed: %s", exc)
            return None

    def _job_second(self, frame, items, time_str, altitude):
        t = _time.perf_counter()
        try:
            res = self.vlm.analyze_second(frame, items, time_str, altitude)
        except Exception as exc:
            log.error("VLM call failed: %s", exc)
            res = {"scene": "", "scene_attributes": {}, "objects": {}, "missing": [i for i, _, _ in items],
                   "error": str(exc)}
        res["latency_s"] = _time.perf_counter() - t
        res["scene_embedding"] = self._embed(res["scene"]) if res.get("scene") else None
        return res

    def _job_sim(self, desc, attrs):
        return {"description": desc, "attributes": attrs, "embedding": self._embed(desc)}

    # ------------------------------------------------------------------ apply results (main thread)
    def _collect(self, block: bool) -> list[Event]:
        """Apply finished jobs in submission order. block=True waits for the oldest one."""
        events: list[Event] = []
        while self.futures and (block or self.futures[0][2].done()):
            kind, key, fut = self.futures.pop(0)
            events += self._handle(kind, key, fut.result())
            if block:
                break
        return events

    def _handle(self, kind: str, key, res: dict) -> list[Event]:
        if kind == "sim":
            tr = self.tracks.get(key) or self.backlog.get(key)
            if tr is None:
                return []
            tr.verified, tr.verify_source = True, "sim"
            tr.description = res["description"]
            tr.attributes = res["attributes"]
            tr.vlm_label = tr.attributes.get("type") or tr.label
            tr.embedding = res["embedding"]
            tr.verified_ts, tr.verified_video_t = tr.last_ts, tr.last_video_t
            return self._finalize(tr) if key in self.backlog else self._announce(tr)

        # one VLM call = one scene record + verdicts for the crops in it
        scene_id = None
        if res.get("scene") and key.get("store_scene", True):
            scene_id = self._store_scene(key["frame_id"], key["ts"], key["video_t"], key["telemetry"], res["scene"],
                                         res["scene_attributes"], res["objects"], self.cfg.vlm.model,
                                         res["latency_s"], key["image_path"], res.get("scene_embedding"))
            for al in self.rules.on_scene(res.get("scene_attributes") or {}, key["ts"]):
                self._write_alert(al, key["frame_id"], key["image_path"])
            for al in self.rules.updated:          # ongoing incident got a new detail
                if al.db_id is not None:
                    self.db.update_alert_message(al.db_id, al.message)
                    log.warning("ALERT [%s] updated: %s", al.rule, al.message)
            self.rules.updated.clear()
        self.vlm_log.append({"video_t": round(key["video_t"], 2), "latency_s": round(res["latency_s"], 2),
                             "n_objects": len(key["ids"]), "n_backlog": key.get("n_backlog", 0),
                             "returned": len(res["objects"]), "tokens": res.get("tokens"),
                             "budget": res.get("budget"), "truncated": bool(res.get("truncated")),
                             "kind": "scene" if key.get("store_scene", True) else "backlog-only",
                             "error": res.get("error")})
        log.info("VLM t=%.1fs %s: %d/%d objects, %s tokens in %.1fs | %s", key["video_t"], fmt_time(key["ts"]),
                 len(res["objects"]), len(key["ids"]), res.get("tokens"), res["latency_s"],
                 res.get("scene", "")[:90])
        events: list[Event] = []
        for tid in key["ids"]:
            tr = self.tracks.get(tid) or self.backlog.get(tid)
            if tr is None:
                continue
            obj = res["objects"].get(tid)
            if obj is None:          # skipped or lost to truncation
                tr.vlm_attempts += 1
                if tr.vlm_attempts < self.cfg.vlm.get("max_attempts", 2):
                    tr.verify_submitted = False      # retry in a later call
                else:                                # this crop keeps failing: stop wasting calls on it,
                    self.gave_up.append(tid)         # it gets the class-level decision at the end
                    log.info("VLM gave up on track #%d after %d attempts", tid, tr.vlm_attempts)
                continue
            self._apply_object(tr, obj, key["ts"], key["video_t"], scene_id)
            events += self._finalize(tr) if tid in self.backlog else self._announce(tr)
        return events

    def _apply_object(self, tr: TrackState, obj: dict, ts: datetime, video_t: float, scene_id) -> None:
        verified = obj["is_match"]
        corrected = text_category(f"{obj['object']} {obj['type']}")
        if not verified and corrected in ("person", "vehicle"):
            # YOLO got the class wrong (e.g. "cell phone" that is really a car): keep it, relabel it
            verified = True
            tr.category = corrected
        tr.verified = verified
        tr.verify_source = "vlm"
        tr.verified_ts, tr.verified_video_t = ts, video_t
        tr.vlm_label = obj["object"] or tr.label
        tr.description = obj["description"] or f"{obj['color']} {tr.vlm_label}"
        tr.attributes = {"color": obj["color"], "type": obj["type"], "state": obj["state"],
                         "location": obj.get("location", "unknown"),
                         "object": obj["object"], "yolo_label": tr.label, "scene_id": scene_id}
        self._set_location(tr, obj.get("location", ""), ts)   # where the VLM saw it at this call
        if verified:
            self.confirmed_labels.add(tr.label)
            tr.embedding = self._embed(self._track_text(tr))
        else:
            self.rejected_labels[tr.label] = self.rejected_labels.get(tr.label, 0) + 1

    @staticmethod
    def _track_text(tr: TrackState) -> str:
        a = tr.attributes
        return (f"{tr.description} ({tr.category}: {a.get('color', '')} {a.get('type', '')}, "
                f"{a.get('state', '')}){where(tr.zone)}")

    def _announce(self, tr: TrackState) -> list[Event]:
        """Write the 'spotted' / 'rejected' event once a verdict exists."""
        if tr.announced or tr.verified is None:
            return []
        tr.announced = True
        self.db.upsert_track(self.run_id, tr)
        if tr.verified:
            note = (f" [class-level: '{tr.label}' confirmed by VLM on other objects]"
                    if tr.verify_source == "class-level" else "")
            ev = Event("spotted", f"{cap(tr.description)} spotted{where(tr.zone)}, {fmt_time(tr.first_ts)}{note}",
                       tr.first_ts, tr.track_id, tr.vlm_label or tr.label, tr.category, tr.zone, tr.first_video_t)
            self.db.add_event(self.run_id, ev, tr.best_frame_id)
            if tr.embedding is not None:
                self.db.add_embedding(self.run_id, "track", tr.track_id, tr.first_ts, tr.first_video_t,
                                      self._track_text(tr), tr.embedding)
            for al in self.rules.on_verified(tr):
                self._write_alert(al, tr.best_frame_id)
        else:
            ev = Event("rejected", f"YOLO '{tr.label}' rejected by VLM: it is a {tr.vlm_label or 'unknown object'}"
                                   f" ({tr.description}), {fmt_time(tr.first_ts)}", tr.first_ts, tr.track_id, tr.label, tr.category,
                       tr.zone, tr.first_video_t)
            self.db.add_event(self.run_id, ev, tr.best_frame_id)
        return [ev]

    def _store_scene(self, frame_id, ts, video_t, telemetry, scene, attrs, objects, model, latency, img_path,
                     emb=None):
        sid = self.db.add_scene(self.run_id, frame_id, ts, video_t, telemetry, scene, attrs, objects, model,
                                latency, img_path)
        self.db.set_frame_caption(frame_id, scene, img_path)
        if emb is None:
            emb = self._embed(scene)
        if emb is not None:
            self.db.add_embedding(self.run_id, "scene", sid, ts, video_t, scene, emb)
        return sid

    def _write_alert(self, al: Alert, frame_id=None, snapshot=None) -> None:
        al.video_t = self._vt(al.ts)
        al.db_id = self.db.add_alert(self.run_id, al, frame_id, snapshot)
        self.recent_alerts.append((al.video_t, al))
        log.warning("ALERT [%s] t=%.1fs %s", al.rule, al.video_t, al.message)

    # ------------------------------------------------------------------ drawing
    def _save(self, img, name: str) -> str | None:
        if not self.save_images or img is None:
            return None
        path = self.img_dir / name
        cv2.imwrite(str(path), img)
        return str(path)

    def _annotate(self, pkt: FramePacket, dets: list[Detection], new_alerts: list[Alert]) -> np.ndarray:
        img = pkt.image.copy()
        scale = max(1.0, img.shape[1] / 1280)   # readable text on 4K
        fs, th_ = 0.45 * scale, max(1, int(scale))
        for d in dets:
            tr = self.tracks.get(d.track_id)
            verified = tr.verified if tr else None
            color = COLORS[verified]
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, th_ + 1) if verified is not False else th_)
            name = tr.description if tr and tr.verified and tr.description else d.label
            tag = f"#{d.track_id} {name[:30]}"
            if tr and tr.verified and tr.category == "person":
                tag += f" {tr.dwell_s(pkt.ts):.0f}s"
            if verified is None:
                tag += " ?"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, tag, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th_, cv2.LINE_AA)
        # telemetry HUD
        t = pkt.telemetry
        hud = (f"{pkt.ts:%Y-%m-%d %H:%M:%S}  t={pkt.video_t:5.1f}s  {t.lat:.5f},{t.lon:.5f}  "
               f"alt {t.altitude_m:.0f}m  hdg {t.heading_deg:.0f}  bat {t.battery_pct:.0f}%  "
               f"vlm queue {len(self.futures)}")
        hh = int(26 * scale)
        cv2.rectangle(img, (0, 0), (img.shape[1], hh), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, int(hh * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_, cv2.LINE_AA)
        # alert banner: alerts raised in the last 4 s of video
        self.recent_alerts = [(vt, a) for vt, a in self.recent_alerts if 0 <= pkt.video_t - vt <= 4.0]
        y = img.shape[0] - int(10 * scale)
        for _, al in reversed(self.recent_alerts[-3:]):
            txt = f"ALERT {al.rule}: {al.message}"[:120]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
            cv2.rectangle(img, (0, y - th - 8), (tw + 12, y + 6), (0, 0, 200), -1)
            cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_, cv2.LINE_AA)
            y -= th + int(16 * scale)
        return img
