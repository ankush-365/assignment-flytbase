"""Simulated drone telemetry: a docked drone flying a circular patrol around the property."""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from .types import Telemetry

EARTH_R = 6_371_000.0


class TelemetrySimulator:
    def __init__(self, cfg, start_time: datetime, seed: int = 7):
        self.cfg = cfg
        self.start = start_time
        self.rng = random.Random(seed)

    def at(self, t: float) -> Telemetry:
        """Telemetry `t` seconds after start."""
        c = self.cfg
        phase = 2 * math.pi * (t / c.patrol_period_s)
        r = c.patrol_radius_m
        north, east = r * math.cos(phase), r * math.sin(phase)
        lat = c.home_lat + math.degrees(north / EARTH_R)
        lon = c.home_lon + math.degrees(east / (EARTH_R * math.cos(math.radians(c.home_lat))))
        speed = 2 * math.pi * r / c.patrol_period_s
        heading = (math.degrees(phase) + 90.0) % 360.0
        altitude = c.altitude_m + self.rng.uniform(-c.altitude_jitter_m, c.altitude_jitter_m)
        battery = max(0.0, c.battery_start_pct - c.battery_drain_pct_per_min * t / 60.0)
        return Telemetry(
            ts=self.start + timedelta(seconds=t),
            lat=lat,
            lon=lon,
            altitude_m=altitude,
            heading_deg=heading,
            speed_mps=speed,
            battery_pct=battery,
        )
