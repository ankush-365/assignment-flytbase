"""Frame sources.

* VideoSource      - real video file / RTSP stream, sampled at `sample_fps`, paired with simulated telemetry.
* TextSimSource    - frames described as text (no pixels): scripted scenario + random traffic. Used for
                     unit tests and the scalability test (hours of "footage" in seconds).
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Iterator

import cv2

from .telemetry import TelemetrySimulator
from .types import Detection, FramePacket


def parse_start(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value
    if value:
        return datetime.fromisoformat(str(value))
    return datetime.now().replace(microsecond=0)


class VideoSource:
    def __init__(self, path: str, sample_fps: float, start_time: datetime, telemetry: TelemetrySimulator,
                 max_seconds: float | None = None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = fps if fps and fps > 1 else 25.0
        self.sample_fps = min(sample_fps, self.src_fps)
        self.step = max(1, round(self.src_fps / self.sample_fps))
        self.start = start_time
        self.telemetry = telemetry
        self.max_seconds = max_seconds
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_s = n / self.src_fps if n and n > 0 else None

    @property
    def effective_fps(self) -> float:
        return self.src_fps / self.step

    def __iter__(self) -> Iterator[FramePacket]:
        raw_idx, out_idx = 0, 0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                if raw_idx % self.step == 0:
                    t = raw_idx / self.src_fps
                    if self.max_seconds and t > self.max_seconds:
                        break
                    tel = self.telemetry.at(t)
                    h, w = frame.shape[:2]
                    yield FramePacket(idx=out_idx, video_t=t, ts=tel.ts, telemetry=tel, width=w, height=h,
                                      image=frame)
                    out_idx += 1
                raw_idx += 1
        finally:
            self.cap.release()


# --------------------------------------------------------------------------------------------
# Text simulation
# --------------------------------------------------------------------------------------------
VEHICLES = [
    ("truck", {"color": "blue", "type": "pickup truck", "make": "Ford F150"}),
    ("car", {"color": "white", "type": "sedan", "make": "Toyota Corolla"}),
    ("car", {"color": "red", "type": "hatchback", "make": "Maruti Swift"}),
    ("truck", {"color": "white", "type": "delivery van", "make": "Tata Ace"}),
    ("motorcycle", {"color": "black", "type": "motorcycle", "make": "Royal Enfield"}),
]
PEOPLE = [
    {"color": "red", "type": "person", "desc": "man in red jacket"},
    {"color": "black", "type": "person", "desc": "person in black hoodie carrying a backpack"},
    {"color": "yellow", "type": "person", "desc": "delivery worker in yellow uniform"},
    {"color": "white", "type": "person", "desc": "woman in white kurta walking a dog"},
]


SIM_LOCATIONS = ["at the main gate", "on the driveway", "in front of the garage", "on the lawn",
                 "on the street outside", "in the backyard"]


class _SimActor:
    def __init__(self, track_id, label, category, desc, attrs, zone_xy, enter_t, leave_t, drift, location=""):
        self.track_id, self.label, self.category = track_id, label, category
        self.location = location
        self.desc, self.attrs = desc, attrs
        self.x, self.y = zone_xy
        self.enter_t, self.leave_t, self.drift = enter_t, leave_t, drift


class TextSimSource:
    """Generates frames as structured text descriptions.

    scenario="demo"  : scripted events that exercise every rule (vehicle twice, loiterer at the gate at night).
    scenario="random": Poisson arrivals of random people / vehicles for `duration_s` (scalability test).
    """

    W, H = 1280, 720

    def __init__(self, duration_s: float, sample_fps: float, start_time: datetime,
                 telemetry: TelemetrySimulator, scenario: str = "demo", seed: int = 1,
                 arrivals_per_min: float = 2.0):
        self.duration_s = duration_s
        self.sample_fps = sample_fps
        self.start = start_time
        self.telemetry = telemetry
        self.rng = random.Random(seed)
        self.actors: list[_SimActor] = []
        self._next_id = 1
        if scenario == "demo":
            self._script_demo()
        else:
            self._script_random(arrivals_per_min)

    @property
    def effective_fps(self) -> float:
        return self.sample_fps

    def _add(self, label, category, desc, attrs, xy, enter, leave, drift=(0.0, 0.0), location=""):
        self.actors.append(_SimActor(self._next_id, label, category, desc, attrs, xy, enter, leave, drift,
                                     location))
        self._next_id += 1

    def _script_demo(self):
        blue_truck = VEHICLES[0]
        # Blue F150 arrives at the garage, leaves, comes back later -> repeat vehicle alert
        self._add("truck", "vehicle", "blue Ford F150 pickup truck", blue_truck[1], (0.75, 0.6), 5, 25,
                  location="in front of the garage")
        self._add("person", "person", "man in red jacket", {"color": "red", "type": "person"}, (0.2, 0.5), 10, 60,
                  drift=(0.0005, 0.0), location="at the main gate")  # lingers -> loitering
        self._add("car", "vehicle", "white Toyota Corolla sedan", VEHICLES[1][1], (0.3, 0.7), 30, 40,
                  location="on the driveway")
        self._add("truck", "vehicle", "blue Ford F150 pickup truck", blue_truck[1], (0.75, 0.6), 75, 90,
                  location="in front of the garage")
        # walks into the backyard (restricted keyword) -> restricted-area alert
        self._add("person", "person", "person in black hoodie carrying a backpack",
                  {"color": "black", "type": "person"}, (0.85, 0.3), 62, 80, location="in the backyard")
        self._add("dog", "object", "brown dog", {"color": "brown", "type": "dog"}, (0.6, 0.8), 50, 55, location="on the lawn")

    def _script_random(self, per_min: float):
        t = 0.0
        while t < self.duration_s:
            t += self.rng.expovariate(per_min / 60.0)
            if t >= self.duration_s:
                break
            stay = self.rng.uniform(5, 90)
            xy = (self.rng.uniform(0.05, 0.95), self.rng.uniform(0.2, 0.95))
            if self.rng.random() < 0.5:
                p = self.rng.choice(PEOPLE)
                self._add("person", "person", p["desc"], {"color": p["color"], "type": "person"}, xy, t, t + stay,
                          drift=(self.rng.uniform(-0.002, 0.002), 0.0), location=self.rng.choice(SIM_LOCATIONS))
            else:
                label, attrs = self.rng.choice(VEHICLES)
                self._add(label, "vehicle", f"{attrs['color']} {attrs['make']} {attrs['type']}", attrs, xy, t,
                          t + min(stay, 30), location=self.rng.choice(SIM_LOCATIONS))

    def __iter__(self) -> Iterator[FramePacket]:
        n = int(self.duration_s * self.sample_fps)
        # actors sorted by enter time; keep an active window so 24h simulations stay O(frames + actors)
        actors = sorted(self.actors, key=lambda a: a.enter_t)
        ptr, active = 0, []
        for i in range(n):
            t = i / self.sample_fps
            while ptr < len(actors) and actors[ptr].enter_t <= t:
                active.append(actors[ptr])
                ptr += 1
            active = [a for a in active if a.leave_t >= t]
            tel = self.telemetry.at(t)
            dets = []
            for a in active:
                dt = t - a.enter_t
                cx = min(max(a.x + a.drift[0] * dt * self.sample_fps, 0.02), 0.98)
                cy = min(max(a.y + a.drift[1] * dt * self.sample_fps, 0.02), 0.98)
                bw, bh = (0.05, 0.15) if a.category == "person" else (0.15, 0.12)
                box = ((cx - bw / 2) * self.W, (cy - bh) * self.H, (cx + bw / 2) * self.W, cy * self.H)
                dets.append(Detection(track_id=a.track_id, label=a.label, category=a.category, conf=0.9,
                                      bbox=box, zone=a.location, sim_description=a.desc,
                                      sim_attributes=dict(a.attrs)))
            caption = None
            if dets and i % int(5 * self.sample_fps) == 0:
                caption = ("Drone view of property: "
                           + ", ".join(f"{d.sim_description} {d.zone}".strip() for d in dets) + ".")
            seen = "; ".join(f"{d.sim_description} {d.zone}".strip() for d in dets) or "nothing detected"
            text = (f"Frame {i + 1}: {seen} | Time {tel.ts:%H:%M:%S}, Lat {tel.lat:.5f}, Lon {tel.lon:.5f}, "
                    f"Alt {tel.altitude_m:.0f} m, Battery {tel.battery_pct:.0f}%")
            yield FramePacket(idx=i, video_t=t, ts=tel.ts, telemetry=tel, width=self.W, height=self.H,
                              sim_detections=dets, sim_caption=caption, frame_text=text)


def build_source(cfg, video: str | None = None, sim: str | None = None, sim_duration_s: float = 120.0,
                 arrivals_per_min: float = 2.0, start_time: str | None = None):
    from .config import resolve

    start = parse_start(start_time or cfg.source.start_time)
    tel = TelemetrySimulator(cfg.telemetry, start)
    if sim:
        return TextSimSource(sim_duration_s, cfg.source.sample_fps, start, tel, scenario=sim,
                             arrivals_per_min=arrivals_per_min)
    path = video or cfg.source.video
    if not str(path).startswith(("rtsp://", "http://", "https://")):
        path = resolve(path)
    return VideoSource(str(path), cfg.source.sample_fps, start, tel, cfg.source.get("max_seconds"))

