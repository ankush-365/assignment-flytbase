"""YOLO + ByteTrack detector: every object gets a persistent track id."""
from __future__ import annotations

import numpy as np

from .config import resolve
from .types import Detection


def categorize(label: str, categories) -> str:
    for cat, labels in categories.items():
        if label in labels:
            return cat
    return "object"


class YoloTracker:
    def __init__(self, cfg, categories):
        from ultralytics import YOLO

        model_path = resolve(cfg.model)
        self.model = YOLO(str(model_path) if model_path.exists() else cfg.model)
        self.cfg = cfg
        self.categories = categories
        self.names = self.model.names

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        c = self.cfg
        res = self.model.track(
            frame,
            persist=True,
            tracker=c.tracker,
            conf=c.conf,
            iou=c.iou,
            imgsz=c.imgsz,
            device=c.device,
            classes=c.classes,
            verbose=False,
        )[0]
        dets: list[Detection] = []
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        for box, tid, cls, conf in zip(xyxy, ids, clss, confs):
            x1, y1, x2, y2 = map(float, box)
            if min(x2 - x1, y2 - y1) < c.min_box_px:
                continue
            label = self.names[int(cls)]
            dets.append(Detection(track_id=int(tid), label=label, category=categorize(label, self.categories),
                                  conf=float(conf), bbox=(x1, y1, x2, y2)))
        return dets
