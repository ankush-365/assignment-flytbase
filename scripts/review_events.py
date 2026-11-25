#!/usr/bin/env python3
"""Build a local HTML page to manually verify stored events against the video.

For every event: the DB record (time, type, message), the track's VLM verdict + attributes, and an image
cut from the source video at that exact timestamp with the object's box drawn.

  python scripts/review_events.py [--run N] [--video path]
  -> data/output/review_run_N/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drone_agent.config import load_config, resolve  # noqa: E402
from drone_agent.db import Database  # noqa: E402

TYPE_COLORS = {"spotted": "#1a7f37", "rejected": "#cf222e", "left": "#6e7781"}


def context_crop(img, box, min_side=360, factor=3.0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(min_side, (x2 - x1) * factor, (y2 - y1) * factor)
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[b:d, a:c]
    s = 320 / max(crop.shape[:2])
    return cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int)
    ap.add_argument("--video")
    args = ap.parse_args()
    cfg = load_config()
    db = Database(resolve(cfg.storage.db))
    rid = args.run or db.latest_run_id()
    run = db.run(rid)
    video = args.video or run["source"]
    if not Path(video).is_absolute():
        video = str(resolve(video))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = resolve(cfg.storage.frames_dir).parent / "output" / f"review_run_{rid}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    tracks = {t["track_id"]: t for t in db.tracks(rid, verified_only=False)}
    events = db.events(rid, limit=100000)

    frame_cache: dict[int, object] = {}

    def frame_at(video_t):
        idx = int(round(video_t * fps))
        if idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, im = cap.read()
            frame_cache[idx] = im if ok else None
        return frame_cache[idx]

    cards = []
    for ev in events:
        tid = ev["track_id"]
        tr = tracks.get(tid, {})
        # the detection of this track closest to the event time
        det = db.q("SELECT * FROM detections WHERE run_id=? AND track_id=? ORDER BY ABS(video_t - ?) LIMIT 1",
                   (rid, tid, ev["video_t"] or 0))
        img_rel = full_rel = ""
        if det:
            d = det[0]
            im = frame_at(d["video_t"])
            if im is not None:
                im = im.copy()
                box = (d["x1"], d["y1"], d["x2"], d["y2"])
                col = (60, 200, 60) if ev["type"] == "spotted" else (40, 40, 220) if ev["type"] == "rejected" \
                    else (150, 150, 150)
                cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 4)
                name = f"ev{ev['id']}_t{tid}"
                cv2.imwrite(str(out / "img" / f"{name}.jpg"), context_crop(im, box))
                small = cv2.resize(im, (960, int(im.shape[0] * 960 / im.shape[1])))
                cv2.imwrite(str(out / "img" / f"{name}_full.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_rel, full_rel = f"img/{name}.jpg", f"img/{name}_full.jpg"
        thumb = tr.get("thumb_path") or ""
        thumb_rel = ""
        if thumb and Path(thumb).exists():
            thumb_rel = f"img/vlm_crop_{tid}.jpg"
            if not (out / thumb_rel).exists():
                (out / thumb_rel).write_bytes(Path(thumb).read_bytes())
        verdict = {None: "unverified", 1: "verified", 0: "rejected"}.get(tr.get("verified"), "?")
        attrs = {k: tr.get(k) for k in ("label", "vlm_label", "color", "obj_type", "state", "verify_source",
                                        "verified_video_t", "first_video_t", "last_video_t", "zones")}
        cards.append(f"""
<div class="card" data-type="{ev['type']}" data-src="{html.escape(tr.get('verify_source') or '')}">
  <div class="imgs">
    {'<a href="' + full_rel + '" target="_blank"><img src="' + img_rel + '" title="frame at event time (click: full frame)"></a>' if img_rel else '<div class="noimg">no image</div>'}
    {'<img class="vlm" src="' + thumb_rel + '" title="crop the VLM actually saw">' if thumb_rel else ''}
  </div>
  <div class="meta">
    <div><span class="badge" style="background:{TYPE_COLORS.get(ev['type'], '#555')}">{ev['type']}</span>
      <b>event #{ev['id']}</b> &middot; t={ev['video_t']}s &middot; {ev['ts'][11:]} &middot; track #{tid}</div>
    <div class="msg">{html.escape(ev['message'])}</div>
    <div class="small">YOLO: <b>{html.escape(str(tr.get('label')))}</b> &rarr; VLM: <b>{html.escape(str(tr.get('vlm_label')))}</b>
      &middot; verdict: <b>{verdict}</b> ({html.escape(str(tr.get('verify_source')))})</div>
    <div class="small">{html.escape(tr.get('description') or '')}</div>
    <details><summary>stored attributes</summary><pre>{html.escape(json.dumps(attrs, indent=1))}</pre></details>
    <label class="chk"><input type="checkbox" class="wrong" data-ev="{ev['id']}"> wrong</label>
  </div>
</div>""")

    counts = {t: sum(1 for e in events if e["type"] == t) for t in ("spotted", "rejected", "left")}
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Event Review run {rid}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#f6f8fa;color:#1f2328}}
.bar{{position:sticky;top:0;background:#fff;padding:10px;border:1px solid #d0d7de;border-radius:8px;margin-bottom:12px;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}}
.card{{display:flex;gap:10px;background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px}}
.imgs{{display:flex;flex-direction:column;gap:6px}} .imgs img{{width:220px;border-radius:4px;display:block}}
.imgs img.vlm{{width:110px;border:2px dashed #888}} .noimg{{width:220px;height:120px;background:#eee}}
.badge{{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px}} .msg{{margin:6px 0;font-weight:600}}
.small{{font-size:13px;color:#555;margin:2px 0}} pre{{font-size:11px;white-space:pre-wrap}}
.chk{{color:#cf222e;font-size:13px}} .card.hide{{display:none}}
</style></head><body>
<div class="bar"><b>Run {rid}</b> &middot; {html.escape(video)} &middot; {len(events)} events
 ({', '.join(f'{k}: {v}' for k, v in counts.items())})<br>
 Show: <label><input type="checkbox" class="f" value="spotted" checked> spotted</label>
 <label><input type="checkbox" class="f" value="rejected" checked> rejected</label>
 <label><input type="checkbox" class="f" value="left"> left</label>
 &nbsp;|&nbsp; <button id="exp">copy ids marked wrong</button> <span id="cnt"></span>
 <div class="small">Big image = source video at the event time, box = the stored detection (click for full frame).
 Dashed small image = the crop the VLM actually judged.</div></div>
<div class="grid">{''.join(cards)}</div>
<script>
function apply(){{const on=[...document.querySelectorAll('.f:checked')].map(x=>x.value);
 document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hide',!on.includes(c.dataset.type)));}}
document.querySelectorAll('.f').forEach(x=>x.addEventListener('change',apply));apply();
document.getElementById('exp').onclick=()=>{{const ids=[...document.querySelectorAll('.wrong:checked')].map(x=>x.dataset.ev);
 const t='wrong events: '+ids.join(', ');document.getElementById('cnt').textContent=t;
 try{{navigator.clipboard.writeText(t)}}catch(e){{}}}};
</script></body></html>"""
    (out / "index.html").write_text(page)
    print(out / "index.html")


if __name__ == "__main__":
    main()
