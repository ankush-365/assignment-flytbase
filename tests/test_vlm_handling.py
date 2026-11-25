"""How the pipeline handles VLM output: truncated JSON, missing objects, give-up, YOLO class correction."""
from datetime import datetime, timedelta

import numpy as np

from drone_agent.db import Database
from drone_agent.pipeline import Pipeline, text_category
from drone_agent.telemetry import TelemetrySimulator
from drone_agent.types import FramePacket, TrackState
from drone_agent.vlm import HashEmbedder, normalize_second, salvage_json

T0 = datetime(2026, 11, 24, 14, 30)


def test_salvage_truncated_json():
    text = ('{"scene": "A street at night.", "scene_attributes": {"setting": "street"}, "objects": ['
            '{"id": 3, "is_match": true, "object": "car", "color": "red"}, {"id": 4, "is_ma')
    out = salvage_json(text)
    assert out["scene"] == "A street at night."
    assert [o["id"] for o in out["objects"]] == [3]          # the complete entry survives, the cut one is dropped
    assert out["_salvaged"]


def test_normalize_marks_missing_ids():
    raw = {"scene": " s ", "objects": [{"id": 1, "is_match": True, "object": "Car", "location": "On The Main Road"}]}
    out = normalize_second(raw, [1, 2])
    assert out["missing"] == [2]
    assert out["objects"][1]["object"] == "car" and out["objects"][1]["location"] == "on the main road"


def test_text_category():
    assert text_category("pickup truck") == "vehicle"
    assert text_category("woman walking") == "person"
    assert text_category("rooftop") is None


class _NeverAnswers:
    """VLM stub whose answer never contains the requested objects (like the looping 2B-model calls)."""
    calls = 0

    def analyze_second(self, frame, items, time_str="", altitude=0.0):
        self.calls += 1
        return {"scene": "", "scene_attributes": {}, "objects": {}, "missing": [i for i, _, _ in items]}


def _pipe(cfg, tmp_path, vlm):
    db = Database(str(tmp_path / "t.db"))
    p = Pipeline(cfg, db=db, vlm=vlm, embedder=HashEmbedder(), use_vlm=True, write_video=False,
                 save_images=False)
    p.t0 = T0
    p.run_id = db.start_run("test", {}, T0)
    return p


def _packet(cfg, t=5.0):
    tel = TelemetrySimulator(cfg.telemetry, T0).at(t)
    return FramePacket(0, t, T0 + timedelta(seconds=t), tel, 64, 64, np.zeros((64, 64, 3), np.uint8))


def test_crop_dropped_after_max_attempts(cfg, tmp_path):
    vlm = _NeverAnswers()
    p = _pipe(cfg, tmp_path, vlm)
    tr = TrackState(7, "car", "vehicle", T0, T0 + timedelta(seconds=3), "unknown", T0, hits=5,
                    best_crop=np.zeros((32, 32, 3), np.uint8))
    p.backlog[7] = tr
    p._drain_backlog(_packet(cfg))
    p.pool.shutdown()
    assert vlm.calls == cfg.vlm.max_attempts                   # 2 calls, not max_backlog_calls (5)
    assert p.gave_up == [7]
    assert tr.verify_source == "class-level" and 7 not in p.backlog   # finalized by the fallback


def test_yolo_class_corrected_by_vlm(cfg, tmp_path):
    p = _pipe(cfg, tmp_path, vlm=None)
    tr = TrackState(9, "cell phone", "object", T0, T0, "unknown", T0)
    obj = {"is_match": False, "object": "car", "color": "white", "type": "sedan", "state": "parked",
           "location": "on the main road", "description": "white sedan parked on the main road"}
    p._apply_object(tr, obj, T0, 0.0, None)
    p.pool.shutdown()
    assert tr.verified is True and tr.category == "vehicle"   # kept and recategorised, not rejected
    assert tr.zone == "on the main road"                      # location comes from the VLM


def test_vlm_rejection(cfg, tmp_path):
    p = _pipe(cfg, tmp_path, vlm=None)
    tr = TrackState(10, "train", "vehicle", T0, T0, "unknown", T0)
    p._apply_object(tr, {"is_match": False, "object": "rooftop", "color": "red", "type": "unknown",
                         "state": "unknown", "location": "unknown", "description": "red rooftop"}, T0, 0.0, None)
    p.pool.shutdown()
    assert tr.verified is False and p.rejected_labels == {"train": 1}
