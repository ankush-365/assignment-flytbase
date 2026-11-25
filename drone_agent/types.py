"""Shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Telemetry:
    ts: datetime
    lat: float
    lon: float
    altitude_m: float
    heading_deg: float
    speed_mps: float
    battery_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(timespec="seconds"),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "altitude_m": round(self.altitude_m, 1),
            "heading_deg": round(self.heading_deg, 1),
            "speed_mps": round(self.speed_mps, 2),
            "battery_pct": round(self.battery_pct, 1),
        }


@dataclass
class Detection:
    track_id: int
    label: str                 # YOLO class name
    category: str              # person | vehicle | object
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    zone: str = ""
    # Text-simulation mode: the "ground truth" description replaces the VLM.
    sim_description: str | None = None
    sim_attributes: dict | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the object touches the ground."""
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2, y2


@dataclass
class FramePacket:
    idx: int                   # analysed-frame index (0..N)
    video_t: float             # seconds into the source video
    ts: datetime               # simulated wall-clock time
    telemetry: Telemetry
    width: int
    height: int
    image: np.ndarray | None = None               # BGR frame (None in text-simulation mode)
    sim_detections: list[Detection] | None = None  # text-simulation mode
    sim_caption: str | None = None
    frame_text: str | None = None   # text-simulation: 'Frame 12: ... | Time ..., Alt ...'


@dataclass
class TrackState:
    track_id: int
    label: str
    category: str
    first_ts: datetime
    last_ts: datetime
    zone: str
    zone_enter_ts: datetime
    first_video_t: float = 0.0  # seconds into the source video
    last_video_t: float = 0.0
    hits: int = 1
    best_conf: float = 0.0
    best_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    best_crop: np.ndarray | None = None
    best_frame_id: int | None = None
    # verification state: None = not yet verified, True = confirmed, False = rejected
    verified: bool | None = None
    verify_submitted: bool = False   # included in a pending VLM call
    vlm_attempts: int = 0            # VLM calls this crop was sent in without getting a verdict back
    verified_ts: datetime | None = None   # time of the VLM call that confirmed / rejected it
    verified_video_t: float | None = None
    verify_source: str = ""          # vlm | class-level | sim | none
    vlm_label: str = ""
    description: str = ""
    attributes: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None
    zones_visited: list[str] = field(default_factory=list)
    alerted_rules: set[str] = field(default_factory=set)
    announced: bool = False    # "spotted" event written
    thumb_path: str = ""
    sim_description: str | None = None
    sim_attributes: dict | None = None

    def dwell_s(self, now: datetime) -> float:
        return (now - self.zone_enter_ts).total_seconds()

    def display_name(self) -> str:
        if self.description:
            return self.description
        return self.vlm_label or self.label


@dataclass
class Alert:
    rule: str
    severity: str              # info | warning | critical
    message: str
    track_id: int | None
    ts: datetime
    zone: str = ""
    video_t: float = 0.0
    db_id: int | None = None   # set once stored, so an ongoing incident can be updated in place


@dataclass
class Event:
    type: str                  # spotted | left | rejected | caption
    message: str
    ts: datetime
    track_id: int | None = None
    label: str = ""
    category: str = ""
    zone: str = ""
    video_t: float = 0.0
