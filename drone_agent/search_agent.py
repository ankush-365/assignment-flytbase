"""Q&A / semantic search over a stored run, answered by a free LLM API (Groq) with DB context.

Flow for one question:
  1. retrieve  - pull a small, relevant slice of the DB: embedding hits (tracks + scenes), tracks whose
                 colour/label/type words appear in the question, and a time window if the question names
                 one ("at 12s", "between 5 and 10 seconds", "14:30:05"). Always add run-level counts + alerts.
  2. answer    - question + that context -> LLM -> JSON {answer, show_images, track_ids, scene_ids}.
                 The LLM decides whether the user wants to SEE things (show_images).
  3. evidence  - ids are checked against the context (no hallucinated ids), matching events are read
                 from the DB, and when show_images is true the timestamps are used to cut the frame
                 (box drawn) from the source video + the crop the VLM judged.

Scalability: only the retrieved slice goes to the LLM (capped by agent.max_tracks / max_scenes), so the
prompt size does not grow with the length of the footage; the vector search itself is brute force and
would move to pgvector / FAISS for days of footage.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Cfg, resolve
from .db import Database
from .search import semantic_search

log = logging.getLogger(__name__)

SYSTEM = """You are a drone security analyst assistant. You answer questions about ONE patrol video that \
was already analysed: YOLO detected and tracked objects, a VLM verified each one and described it, and the \
VLM described the whole scene once per second. Everything is timestamped: `t` = seconds into the video, \
clock = simulated wall-clock time.

The "Matching objects" in the CONTEXT were already filtered by the question's colour / object / place / time \
words: they ALL match. When the user asks to see or find something, return all of them (several objects can \
fit "the red car"; mention how many). Use ONLY the CONTEXT. If it does not contain the answer, say so. Cite times (e.g. "t=12.4s, 14:30:12") and \
track numbers (#id). Be concise. Colours and types come from a small VLM and can be wrong - do not invent \
details beyond the context.

Return a JSON object:
{"answer": str,                 # the reply to the user
 "show_images": bool,           # true if the user wants to SEE something (show / display / find / where / \
which / what did X look like ...), false for counts, yes/no, summaries
 "track_ids": [int],            # ALL listed objects that match the question, even if the user wrote \
"the red car" (best first, max 12); [] if none
 "scene_ids": [int]}            # scene records (per-second VLM descriptions) that support the answer, max 4"""

COLORS = {"red", "blue", "green", "white", "black", "grey", "gray", "silver", "yellow", "orange", "brown",
          "purple", "pink", "beige", "dark", "light", "tan", "maroon", "gold"}
STOP = {"show", "me", "the", "all", "any", "was", "were", "is", "are", "did", "does", "detected", "seen",
        "there", "what", "which", "where", "when", "how", "many", "of", "in", "on", "at", "a", "an", "and",
        "or", "find", "display", "give", "list", "video", "clip", "footage", "image", "images", "picture",
        "pictures", "frame", "frames", "with", "that", "this", "time", "from", "to", "between", "near",
        "next", "by", "side", "around", "close", "anyone", "anything", "something", "someone"}


VISUAL = re.compile(r"\b(show|display|see|picture|pictures|image|images|photo|photos|look like|looks like)\b",
                    re.I)


def _sing(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


# ------------------------------------------------------------------------------------------- retrieval
def parse_time_window(q: str, t0: datetime | None) -> tuple[float | None, float | None]:
    """'between 5 and 10 s' / 'from 5s to 10s' / 'at 12s' / '14:30:05' -> (start_s, end_s) in video seconds."""
    ql = q.lower()
    m = re.search(r"(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s+(?:and|to|-)\s+(\d+(?:\.\d+)?)", ql)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(?:first|last)\s+(\d+(?:\.\d+)?)\s*(?:s\b|sec|seconds)", ql)
    if m and "first" in m.group(0):
        return 0.0, float(m.group(1))
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|secs\b|seconds\b)", ql)
    if m and not re.search(r"(?:for|over|longer than|more than)\s+" + re.escape(m.group(0)), ql):
        t = float(m.group(1))
        return max(0.0, t - 1.5), t + 1.5
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", ql)
    if m and t0 is not None:
        clock = t0.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3) or 0))
        t = (clock - t0).total_seconds()
        return (max(0.0, t - 1.5), t + 1.5) if m.group(3) else (max(0.0, t), t + 60)
    return None, None


OBJECT_SYNONYMS = {"people": "person", "persons": "person", "pedestrian": "person", "man": "person",
                   "woman": "person", "men": "person", "women": "person", "vehicle": "vehicle",
                   "automobile": "car", "lorry": "truck", "bike": "bicycle", "motorbike": "motorcycle"}
SCENE_Q = re.compile(r"\b(happen|happened|going on|scene|describe|summar|area|overall|weather|time of day|busy|"
                     r"activity|landmark)", re.I)
TRACK_FIELDS = ("label", "vlm_label", "obj_type", "category", "state", "description", "zones")
FOLLOW_UP = re.compile(r"\b(those|these|them|they|their|it|its|that one|this one|the first|the last|which of|"
                       r"among|of them|same one|same ones|the others?|each of)\b", re.I)


def _matches(t: dict, colors: list[str], objects: list[str], places: list[str]) -> tuple[int, int]:
    """(object terms hit, place terms hit). Colours must be the track's colour field (so 'red' does not hit
    'a traffic light showing red'); object words hit label/type/category/description; place words hit the
    VLM location."""
    col = (t.get("color") or "").lower()
    text = " ".join(str(t.get(f) or "") for f in TRACK_FIELDS[:6]).lower()
    loc = (t.get("zones") or "").lower()
    return (sum(1 for c in colors if re.search(rf"\b{c}\b", col))
            + sum(1 for o in objects if re.search(rf"\b{o}", text)),
            sum(1 for p in places if re.search(rf"\b{p}", loc)))


def retrieve(db: Database, run_id: int, question: str, cfg: Cfg, embedder=None,
             within: list[int] | None = None) -> dict:
    """Small, relevant context for one question:
    attribute words  -> tracks matching ALL of them (red AND car); fall back to ANY only if none match all
    time window      -> restrict to tracks visible in it (+ the scenes of that window)
    neither          -> embedding search, only hits close to the best one
    Scenes are added only for scene-type questions or time windows.
    within: follow-up question ("which of those were parked?") -> search only these track ids."""
    a = cfg.agent
    max_tracks, max_scenes = a.get("max_tracks", 15), a.get("max_scenes", 3)
    run = db.run(run_id)
    t0 = datetime.fromisoformat(run["sim_start"]) if run.get("sim_start") else None
    all_tracks = {t["track_id"]: t for t in db.tracks(run_id, verified_only=False)
                  if t["verified"] == 1 or t["verify_source"] == "vlm"}

    words = [OBJECT_SYNONYMS.get(w, _sing(w)) for w in re.findall(r"[a-z]+", question.lower()) if w not in STOP]
    obj_vocab: set[str] = set()
    place_vocab: set[str] = set()
    for t in all_tracks.values():
        for f in TRACK_FIELDS[:5]:
            obj_vocab |= {_sing(w) for w in re.findall(r"[a-z]+", (t.get(f) or "").lower())}
        place_vocab |= {_sing(w) for w in re.findall(r"[a-z]+", (t.get("zones") or "").lower())}
    colors = [w for w in words if w in COLORS]
    objects = [w for w in words if w not in COLORS and w in obj_vocab and w not in place_vocab]
    places = [w for w in words if w in place_vocab and w not in COLORS and w not in objects and len(w) > 2]
    terms = colors + objects + places

    s, e = parse_time_window(question, t0)
    pool = all_tracks
    if within:
        pool = {tid: all_tracks[tid] for tid in within if tid in all_tracks}
    if s is not None:
        pool = {t["track_id"]: pool[t["track_id"]]
                for t in db.tracks(run_id, verified_only=False, start_s=s, end_s=e) if t["track_id"] in pool}

    scores: dict[int, float] = {}
    n_matching, note = 0, ""
    if terms:
        hits = {tid: _matches(t, colors, objects, places) for tid, t in pool.items()}
        need = len(colors) + len(objects)
        if need:   # colour + object type are mandatory: "red car" never returns a red traffic light
            chosen = [tid for tid, (o, _) in hits.items() if o == need]
            if not chosen:
                note = f"No object matches all of: {', '.join(colors + objects)}."
        else:
            chosen = list(pool)
        if places:  # the place narrows the result; if nothing is there, say so instead of listing others
            at = [tid for tid in chosen if hits[tid][1] == len(places)]
            if chosen and not at:
                note = (f"None of the matching objects ({', '.join(colors + objects) or 'any'}) has a location "
                        f"containing: {' '.join(places)}.")
            chosen = at
        n_matching = len(chosen)
        scores = {tid: float(sum(hits[tid])) for tid in chosen}
    elif within:    # follow-up without new filters ("when was the first one seen?"): all previous results
        scores = {tid: -(t["first_video_t"] or 0) for tid, t in pool.items()}   # earliest first
        n_matching = len(pool)
    elif SCENE_Q.search(question) and s is None:
        pass        # "describe the area": scene descriptions only, no object list
    elif s is not None:
        scores = {tid: 1.0 for tid in pool}
        n_matching = len(pool)

    sem_scenes = []
    try:
        hits = semantic_search(db, run_id, question, cfg, k=60, embedder=embedder)
        tr_hits = [h for h in hits if h["kind"] == "track" and h["ref_id"] in pool]
        if within:
            pass        # keep the follow-up order
        elif scores:    # rank the attribute/time matches by similarity, never add new tracks
            for h in tr_hits:
                if h["ref_id"] in scores:
                    scores[h["ref_id"]] += max(0.0, h["score"])
        elif tr_hits and not terms and not SCENE_Q.search(question):   # open question ("anyone carrying a bag?"): only hits close to the best one
            best = tr_hits[0]["score"]
            close = [h for h in tr_hits if h["score"] >= best - a.get("semantic_margin", 0.05)][:max_tracks]
            scores = {h["ref_id"]: h["score"] for h in close}
            n_matching = len(close)
        sem_scenes = [h["ref_id"] for h in hits if h["kind"] == "scene"]
    except Exception as exc:  # embeddings are optional context
        log.debug("semantic search failed: %s", exc)

    ranked = sorted(scores, key=lambda tid: -scores[tid])[:max_tracks]
    tracks = [all_tracks[t] for t in ranked]          # best match first

    scenes = []
    if s is not None:
        scenes = db.scenes(run_id, start_s=s, end_s=e)[:max_scenes]
    elif SCENE_Q.search(question) or not tracks:
        rows = {r["id"]: r for r in db.scenes(run_id)}
        scenes = [rows[i] for i in sem_scenes if i in rows][:max_scenes]
    scenes.sort(key=lambda r: r["video_t"])
    return {"run": run, "counts": db.object_counts(run_id), "alerts": db.alerts(run_id), "tracks": tracks,
            "n_matching": n_matching, "note": note, "within": len(within or []), "scenes": scenes, "time_window": (s, e), "terms": terms}


def format_context(ctx: dict) -> str:
    """Compact: one short line per object, only fields that carry information."""
    run = ctx["run"]
    lines = [f"Video '{Path(run['source']).stem.replace('_', ' ')}', clock starts {run['sim_start']}. "
             "Verified object counts: "
             + ", ".join(f"{c['object']}={c['n']}" for c in ctx["counts"]) + "."]
    if ctx["alerts"]:
        lines.append("Alerts:")
        lines += [f"  t={a['video_t']}s {a['ts'][11:19]} {a['rule']}: {a['message']}" for a in ctx["alerts"]]
    else:
        lines.append("Alerts: none.")
    if ctx.get("within"):
        lines.append(f"Follow-up: searched only within the {ctx['within']} objects of the previous answer.")
    if ctx.get("note"):
        lines.append("Search note: " + ctx["note"])
    if ctx["tracks"]:
        more = ctx.get("n_matching", 0) - len(ctx["tracks"])
        lines.append(f"Matching objects ({len(ctx['tracks'])}{f', {more} more not shown' if more > 0 else ''}):")
    for t in ctx["tracks"]:
        typ = t["obj_type"] if t["obj_type"] not in ("unknown", None, "") else ""
        what = " ".join(x for x in (t["color"], typ) if x and x != "unknown")
        cls = t["vlm_label"] or t["label"]
        what = f"{what} ({cls})" if typ and typ != cls else f"{what} {cls}".strip() if cls not in what else what
        extra = f" (YOLO said {t['label']})" if t["label"] not in (cls, typ) else ""
        flag = "" if t["verified"] == 1 and t["verify_source"] == "vlm" else \
            f" [{'REJECTED by VLM' if t['verified'] == 0 else t['verify_source']}]"
        lines.append(f"  #{t['track_id']} {what}{extra}, {t['state']}, {t['zones'] or 'location unknown'}, "
                     f"t={t['first_video_t']:.1f}-{t['last_video_t']:.1f}s {t['first_ts'][11:19]}: "
                     f"\"{t['description']}\"{flag}")
    if ctx["scenes"]:
        lines.append("Scene descriptions:")
        lines += [f"  scene {s['id']} t={s['video_t']:.1f}s {s['ts'][11:19]}: {s['scene']}" for s in ctx["scenes"]]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- LLM
def _load_env() -> None:
    """Project .env (GROQ_API_KEY=...) wins over a stale key exported in the shell."""
    try:
        from dotenv import load_dotenv

        load_dotenv(resolve(".env"), override=True)
    except ImportError:
        pass


class LLM:
    """Groq (free tier) chat completion in JSON mode; falls back to the local Ollama model on any error."""

    def __init__(self, cfg: Cfg):
        a = cfg.agent
        self.provider = a.get("provider", "groq")
        self.model = a.get("groq_model", "openai/gpt-oss-120b")
        self.fallback = a.get("model", "llama3.1:latest")
        self.host = cfg.vlm.host
        self.temperature = a.get("temperature", 0)
        # fail over to the local model quickly instead of the SDK default (60 s x 3 attempts)
        self.timeout_s = a.get("timeout_s", 20)
        self.used = ""

    def chat_json(self, messages: list[dict]) -> dict:
        _load_env()
        if self.provider == "groq" and os.environ.get("GROQ_API_KEY"):
            try:
                from groq import Groq

                r = Groq(timeout=self.timeout_s, max_retries=1).chat.completions.create(model=self.model, messages=messages,
                                                   temperature=self.temperature,
                                                   response_format={"type": "json_object"}, max_tokens=2500)
                self.used = f"groq:{self.model}"
                return json.loads(r.choices[0].message.content)
            except Exception as exc:
                log.warning("Groq failed (%s) - falling back to local %s", exc, self.fallback)
        import ollama

        r = ollama.Client(host=self.host).chat(model=self.fallback, messages=messages, format="json",
                                               options={"temperature": self.temperature})
        self.used = f"ollama:{self.fallback}"
        return json.loads(r.message.content)


# ------------------------------------------------------------------------------------------- agent
@dataclass
class Answer:
    question: str
    answer: str
    show_images: bool
    tracks: list[dict] = field(default_factory=list)
    scenes: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)   # {track_id, video_t, frame, crop}
    model: str = ""
    context: str = ""
    page: str = ""


class SecurityAgent:
    def __init__(self, db: Database, run_id: int, cfg: Cfg, embedder=None, llm: LLM | None = None,
                 history_turns: int = 4):
        self.db, self.run_id, self.cfg = db, run_id, cfg
        self.embedder = embedder
        self.llm = llm or LLM(cfg)
        self.history: list[dict] = []      # follow-up questions keep the last few Q/A turns
        self.last_ids: list[int] = []      # objects of the previous answer, for "which of those ..."
        self.history_turns = history_turns
        self.out_dir = resolve(cfg.storage.frames_dir).parent / "output" / f"ask_run_{run_id}"

    def ask(self, question: str, save_page: bool = True) -> Answer:
        follow = bool(self.last_ids) and bool(FOLLOW_UP.search(question))
        ctx = retrieve(self.db, self.run_id, question, self.cfg, self.embedder,
                       within=self.last_ids if follow else None)
        context = format_context(ctx)
        msgs = [{"role": "system", "content": SYSTEM}] + self.history[-2 * self.history_turns:]
        msgs.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})
        try:
            out = self.llm.chat_json(msgs)
        except Exception as exc:
            out = {"answer": f"LLM unavailable ({exc}).", "show_images": False, "track_ids": [], "scene_ids": []}
        by_id = {t["track_id"]: t for t in ctx["tracks"]}
        sc_by_id = {s["id"]: s for s in ctx["scenes"]}
        tids = [int(i) for i in out.get("track_ids") or [] if str(i).lstrip("#").isdigit()
                and int(i) in by_id][:12]          # only ids that were in the context
        sids = [int(i) for i in out.get("scene_ids") or [] if str(i).isdigit() and int(i) in sc_by_id][:4]
        # the LLM decides; an explicit visual request ("show", "see", "picture" ...) is a safety net for
        # small fallback models that miss it
        show = bool(out.get("show_images")) or bool(VISUAL.search(question))
        ans = Answer(question, str(out.get("answer", "")).strip(), show,
                     [by_id[i] for i in tids], [sc_by_id[i] for i in sids], model=self.llm.used, context=context)
        if tids:
            ph = ",".join("?" * len(tids))
            ans.events = self.db.q(f"SELECT * FROM events WHERE run_id=? AND track_id IN ({ph})"
                                   f" AND type != 'left' ORDER BY video_t", (self.run_id, *tids))
        if ans.show_images:
            ans.images = self._images(ans.tracks)
        if tids or ctx["tracks"]:   # what the next "those" refers to
            self.last_ids = tids or [t["track_id"] for t in ctx["tracks"]]
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": json.dumps({"answer": ans.answer, "track_ids": tids})}]
        if save_page:
            ans.page = str(self._page(ans))
        return ans

    # -- evidence images: frame at the detection time (box drawn) + the crop the VLM judged
    def _images(self, tracks: list[dict]) -> list[dict]:
        import cv2

        run = self.db.run(self.run_id)
        video = run["source"] if Path(run["source"]).is_absolute() else str(resolve(run["source"]))
        img_dir = self.out_dir / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = []
        for t in tracks:
            vt = t["verified_video_t"] if t["verified_video_t"] is not None else t["first_video_t"]
            det = self.db.q("SELECT * FROM detections WHERE run_id=? AND track_id=? ORDER BY ABS(video_t-?) "
                            "LIMIT 1", (self.run_id, t["track_id"], vt or 0))
            item = {"track_id": t["track_id"], "video_t": vt, "ts": t["verified_ts"] or t["first_ts"],
                    "frame": "", "context": "", "crop": t.get("thumb_path") or ""}
            if det and cap.isOpened():
                d = det[0]
                item["video_t"], item["ts"] = d["video_t"], d["ts"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(d["video_t"] * fps)))
                ok, im = cap.read()
                if ok:
                    x1, y1, x2, y2 = (int(d[k]) for k in ("x1", "y1", "x2", "y2"))
                    cv2.rectangle(im, (x1, y1), (x2, y2), (60, 220, 60), 6)
                    h, w = im.shape[:2]
                    side = max(480, (x2 - x1) * 3, (y2 - y1) * 3)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    ctx_im = im[max(0, cy - side // 2):min(h, cy + side // 2),
                                max(0, cx - side // 2):min(w, cx + side // 2)]
                    s = 360 / max(ctx_im.shape[:2])
                    name = f"t{t['track_id']}_{d['video_t']:.2f}s"
                    cv2.imwrite(str(img_dir / f"{name}_ctx.jpg"), cv2.resize(ctx_im, None, fx=s, fy=s))
                    cv2.imwrite(str(img_dir / f"{name}_full.jpg"), cv2.resize(im, (1280, int(h * 1280 / w))),
                                [cv2.IMWRITE_JPEG_QUALITY, 82])
                    item["context"], item["frame"] = str(img_dir / f"{name}_ctx.jpg"), str(img_dir / f"{name}_full.jpg")
            out.append(item)
        cap.release()
        return out

    def _page(self, ans: Answer) -> Path:
        """One HTML page per question (new file each time, nothing overwritten)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", ans.question.lower()).strip("_")[:40] or "question"
        path = self.out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.html"
        rel = lambda p: os.path.relpath(p, self.out_dir) if p else ""  # noqa: E731
        cards = []
        for im in ans.images:
            t = next((x for x in ans.tracks if x["track_id"] == im["track_id"]), {})
            cards.append(f"""<div class="card">
 {'<a href="' + rel(im['frame']) + '" target="_blank"><img src="' + rel(im['context']) + '"></a>' if im['context'] else ''}
 {'<img class="vlm" src="' + rel(im['crop']) + '" title="crop the VLM judged">' if im['crop'] else ''}
 <div><b>#{im['track_id']}</b> {html.escape(t.get('description') or '')}<br>
 <span class="small">t={im['video_t']}s &middot; {str(im['ts'])[11:19]} &middot; {html.escape(str(t.get('color')))}
 {html.escape(str(t.get('vlm_label')))} &middot; {html.escape(str(t.get('zones')))}</span></div></div>""")
        ev_rows = "".join(f"<tr><td>{e['video_t']}s</td><td>{e['ts'][11:19]}</td><td>{e['type']}</td>"
                          f"<td>#{e['track_id']}</td><td>{html.escape(e['message'])}</td></tr>" for e in ans.events)
        sc_rows = "".join(f"<li><b>t={s['video_t']}s</b> {html.escape(s['scene'])}</li>" for s in ans.scenes)
        path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Drone Q&amp;A</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;max-width:1100px;background:#f6f8fa;color:#1f2328}}
.q{{font-size:20px;font-weight:600}} .a{{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:12px;
white-space:pre-wrap;margin:10px 0}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:8px}} .card img{{width:100%;border-radius:4px}}
.card img.vlm{{width:90px;border:2px dashed #888}} .small{{color:#57606a;font-size:13px}}
table{{border-collapse:collapse;background:#fff;width:100%}} td{{border:1px solid #d0d7de;padding:4px 8px;font-size:14px}}
</style></head><body><div class="q">Q: {html.escape(ans.question)}</div>
<div class="a">{html.escape(ans.answer)}</div>
<div class="small">model {html.escape(ans.model)} &middot; run {self.run_id} &middot; show_images={ans.show_images}</div>
{'<h3>Images</h3><div class="grid">' + ''.join(cards) + '</div>' if cards else ''}
{'<h3>Events</h3><table>' + ev_rows + '</table>' if ev_rows else ''}
{'<h3>Supporting scenes</h3><ul>' + sc_rows + '</ul>' if sc_rows else ''}
<details><summary class="small">context sent to the LLM</summary><pre>{html.escape(ans.context)}</pre></details>
</body></html>""")
        return path


def summarize_run(db: Database, run_id: int, cfg: Cfg) -> str:
    """One-paragraph summary of the whole run from counts, alerts and the per-second scene descriptions."""
    ctx = {"run": db.run(run_id), "stats": db.stats(run_id), "counts": db.object_counts(run_id),
           "alerts": db.alerts(run_id), "tracks": [], "scenes": db.scenes(run_id)[:: max(1, len(db.scenes(run_id)) // 8)]}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{format_context(ctx)}\n\nQUESTION: Summarise this patrol "
                                        "video in 2-3 sentences (what the area is, what was seen, any alerts)."}]
    try:
        return str(LLM(cfg).chat_json(msgs).get("answer", "")).strip()
    except Exception as exc:
        return f"LLM unavailable ({exc})."
