"""Vision-language model stage (local + free: Ollama small VLM, e.g. qwen2.5vl:3b / qwen3-vl:2b).

One VLM call per second of video (only for seconds where YOLO found something), not one per object:
  image 1 = the full frame (downscaled)      -> scene description + scene attributes
  image 2 = numbered contact sheet of crops  -> per-object verification + colour/type/state + description
            of objects first seen recently

Why verify with a VLM?
  * YOLO (COCO-trained on ground-level photos) produces false positives on aerial footage
    (e.g. rooftops as "cell phone"/"train"). The VLM rejects them, so they never become events or alerts.
  * The VLM adds attributes YOLO can't ("white sedan, moving north on the main road"), which power
    descriptive logs, semantic search and repeat-vehicle matching.
  * Cost is bounded by video duration (<= 1 call / second), not by object count.
"""
from __future__ import annotations

import json
import logging
import math
import re
import zlib

import cv2
import numpy as np

log = logging.getLogger(__name__)

SECOND_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "scene_attributes": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "time_of_day": {"type": "string"},
                "activity_level": {"type": "string"},
                "people_count": {"type": "integer"},
                "vehicle_count": {"type": "integer"},
                "unusual": {"type": "string"},
                "landmarks": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["setting", "time_of_day", "activity_level", "people_count", "vehicle_count", "unusual",
                         "landmarks"],
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "is_match": {"type": "boolean"},
                    "object": {"type": "string"},
                    "color": {"type": "string"},
                    "type": {"type": "string"},
                    "state": {"type": "string"},
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "is_match", "object", "color", "type", "state", "location", "description"],
            },
        },
    },
    "required": ["scene", "scene_attributes", "objects"],
}

SECOND_PROMPT = """You are the analysis stage of a drone security system. You get two images.

IMAGE 1: the full drone camera frame (time {time}, altitude {alt:.0f} m).
IMAGE 2: numbered close-up crops of objects an automatic detector found. Each crop is labelled "#id label".
Detector claims for the crops: {claims}
(Some crops may come from a few seconds earlier and may no longer be visible in IMAGE 1.)

Return JSON with:
- scene: 1-2 sentences describing the whole frame for a security log (setting, people, vehicles, activity, anything unusual).
- scene_attributes: setting, time_of_day, activity_level (none/low/moderate/high), people_count, vehicle_count (approximate counts in IMAGE 1),
    unusual: a short phrase for anything a security guard must act on (e.g. vehicle crashing through a gate or fence, forced entry, fire or smoke, person climbing a fence, fight, abandoned bag); "none" for normal activity such as traffic, parked cars or people walking,
    landmarks: 3-6 short names of fixed places visible in IMAGE 1 that a guard would use to say where something is (e.g. "main road", "traffic lights", "gate", "parking lot", "west sidewalk", "intersection", "building entrance").
- objects: one entry for EVERY crop id listed above, with
    id: the crop number,
    is_match: true if the crop really shows the detector's label (car/truck/van/bus count as the same vehicle family), false if it is something else (roof, shadow, sign) or unrecognisable,
    object: what it really is (1-3 words),
    color: main colour or "unknown",
    type: finer type (e.g. sedan, hatchback, SUV, pickup, van, bus, motorcycle, pedestrian, cyclist) or "unknown",
    state: parked / moving / standing / walking / unknown,
    location: where it is in IMAGE 1, named by the landmarks above, e.g. "on the main road near the traffic lights", "at the gate", "west sidewalk"; "unknown" if it is not visible in IMAGE 1,
    description: short phrase for a log, e.g. "white sedan parked in the lot", "red hatchback driving on the road".
Do not invent brands. Be concise."""


# ---------------------------------------------------------------------------------------------- images
def crop_object(frame: np.ndarray, bbox, padding: float = 0.25, min_px: int = 160) -> np.ndarray:
    """Crop with context padding, upscaled so the short side is >= min_px."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.zeros((min_px, min_px, 3), np.uint8)
    ch, cw = crop.shape[:2]
    scale = min_px / max(1, min(ch, cw))
    if scale > 1:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    return crop


def contact_sheet(items: list[tuple[int, str, np.ndarray]], tile: int = 160, cols: int = 4) -> np.ndarray:
    """Grid of crops, each with a '#id label' header, so the VLM can refer to objects by number."""
    n = max(1, len(items))
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    head = 26
    sheet = np.full((rows * (tile + head), cols * tile, 3), 255, np.uint8)
    for k, (tid, label, crop) in enumerate(items):
        r, c = divmod(k, cols)
        h, w = crop.shape[:2]
        s = tile / max(h, w)
        img = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        y0, x0 = r * (tile + head), c * tile
        sheet[y0 + head:y0 + head + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.rectangle(sheet, (x0, y0), (x0 + tile - 1, y0 + head), (0, 0, 0), -1)
        cv2.putText(sheet, f"#{tid} {label}", (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0 + tile - 1, y0 + tile + head - 1), (0, 0, 0), 1)
    return sheet


_COLOR_NAMES = [  # (name, hue range) for saturated colours
    ("red", (0, 10)), ("orange", (10, 22)), ("yellow", (22, 35)), ("green", (35, 85)),
    ("blue", (85, 130)), ("purple", (130, 160)), ("red", (160, 180)),
]


def dominant_color(crop: np.ndarray) -> str:
    """Cheap OpenCV colour name from the centre of a crop (fallback when an object gets no VLM call)."""
    h, w = crop.shape[:2]
    c = crop[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if c.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if val < 60:
        return "black"
    if sat < 40:
        return "white" if val > 170 else "grey" if val > 100 else "dark grey"
    for name, (lo, hi) in _COLOR_NAMES:
        if lo <= hue < hi:
            return name
    return "unknown"


def _jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return salvage_json(text)


def salvage_json(text: str) -> dict:
    """Recover what is complete from a truncated response: the scene, scene attributes and every object
    entry whose closing brace was emitted. Lost entries are retried in a later call."""
    out: dict = {}
    m = re.search(r'"scene"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        out["scene"] = json.loads(f'"{m.group(1)}"')
    m = re.search(r'"scene_attributes"\s*:\s*(\{[^{}]*\})', text)
    if m:
        try:
            out["scene_attributes"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    objs = []
    start = text.find('"objects"')
    for m in re.finditer(r'\{[^{}]*"id"\s*:\s*\d+[^{}]*\}', text[start:] if start >= 0 else ""):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    out["objects"] = objs
    out["_salvaged"] = True
    return out


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def normalize_second(data: dict, ids: list[int]) -> dict:
    """Make the VLM output safe to consume: every requested id gets an entry."""
    scene = str(data.get("scene", "")).strip()
    attrs = data.get("scene_attributes") or {}
    objs = {}
    for o in data.get("objects") or []:
        try:
            oid = int(o.get("id"))
        except (TypeError, ValueError):
            continue
        objs[oid] = {
            "is_match": bool(o.get("is_match", False)),
            "object": str(o.get("object", "")).strip().lower(),
            "color": str(o.get("color", "unknown")).strip().lower() or "unknown",
            "type": str(o.get("type", "unknown")).strip().lower() or "unknown",
            "state": str(o.get("state", "unknown")).strip().lower() or "unknown",
            "location": str(o.get("location", "unknown")).strip().lower() or "unknown",
            "description": str(o.get("description", "")).strip(),
        }
    return {"scene": scene, "scene_attributes": attrs, "objects": {i: objs[i] for i in ids if i in objs},
            "missing": [i for i in ids if i not in objs]}


# ---------------------------------------------------------------------------------------------- models
class OllamaVLM:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout_s: float = 120,
                 frame_max_side: int = 896, num_ctx: int = 4096, max_tokens: int = 200,
                 tokens_per_object: int = 70, max_tokens_cap: int = 1600):
        import ollama

        self.model = model
        self.frame_max_side = frame_max_side
        self.num_ctx = num_ctx
        # output budget = base + per-object, capped: enough for a full answer, short enough that a
        # runaway (repetition loop) generation ends in seconds instead of filling the whole context
        self.max_tokens = max_tokens
        self.tokens_per_object = tokens_per_object
        self.max_tokens_cap = max_tokens_cap
        self.client = ollama.Client(host=host, timeout=timeout_s)

    def analyze_second(self, frame: np.ndarray, items: list[tuple[int, str, np.ndarray]], time_str: str = "",
                       altitude: float = 0.0) -> dict:
        """items: [(track_id, yolo_label, crop)]. Returns normalised dict (see normalize_second)."""
        ids = [i for i, _, _ in items]
        budget = min(self.max_tokens + self.tokens_per_object * len(items), self.max_tokens_cap)
        claims = ", ".join(f"#{i} {lbl}" for i, lbl, _ in items) or "(no crops this second)"
        images = [_jpeg(_downscale(frame, self.frame_max_side))]
        if items:
            images.append(_jpeg(contact_sheet(items)))
        prompt = SECOND_PROMPT.format(time=time_str, alt=altitude, claims=claims)
        if not items:
            # scene-only call (e.g. the context call at the start of a video): no crops to judge
            prompt = (prompt.replace("You get two images.", "You get one image.")
                      + "\nThere are NO crops in this call: return \"objects\": [] and only describe the scene.")
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": images}],
            format=SECOND_SCHEMA,
            options={"temperature": 0, "num_ctx": self.num_ctx, "num_predict": budget},
            keep_alive="10m",
        )
        raw = _parse_json(resp.message.content or "")
        out = normalize_second(raw, ids)
        out["tokens"] = getattr(resp, "eval_count", None)
        out["budget"] = budget
        out["truncated"] = getattr(resp, "done_reason", "") == "length"
        out["salvaged"] = bool(raw.get("_salvaged"))
        if out["truncated"]:
            log.warning("VLM output hit the %d-token budget: salvaged %d/%d objects, %d without a verdict",
                        budget, len(out["objects"]), len(ids), len(out["missing"]))
        return out


class Embedder:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        import ollama

        self.model = model
        self.client = ollama.Client(host=host)

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        # nomic-embed-text expects task prefixes
        resp = self.client.embed(model=self.model, input=[f"search_document: {t}" for t in texts])
        return [np.asarray(v, dtype=np.float32) for v in resp.embeddings]

    def query(self, text: str) -> np.ndarray:
        resp = self.client.embed(model=self.model, input=[f"search_query: {text}"])
        return np.asarray(resp.embeddings[0], dtype=np.float32)


class MockVLM:
    """Deterministic stand-in used by tests / offline runs. Rejects labels in `reject_labels`."""

    def __init__(self, reject_labels: tuple[str, ...] = ("cell phone", "train")):
        self.reject = set(reject_labels)
        self.calls = 0

    def analyze_second(self, frame, items, time_str: str = "", altitude: float = 0.0) -> dict:
        self.calls += 1
        objs = {}
        for tid, label, crop in items:
            ok = label not in self.reject
            color = dominant_color(crop) if crop is not None else "unknown"
            objs[tid] = {"is_match": ok, "object": label if ok else "rooftop", "color": color,
                         "type": label if ok else "unknown", "state": "unknown",
                         "description": f"{color} {label}" if ok else "rooftop, not an object"}
        return {"scene": f"Mock scene with {len(items)} new objects.",
                "scene_attributes": {"setting": "mock", "time_of_day": "unknown", "activity_level": "low",
                                     "people_count": 0, "vehicle_count": len(items), "unusual": "none"},
                "objects": objs, "missing": []}


class HashEmbedder:
    """Cheap bag-of-words embedder for tests / offline use (no model needed)."""

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        return v

    def __call__(self, texts):
        return [self._vec(t) for t in texts]

    def query(self, text):
        return self._vec(text)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
