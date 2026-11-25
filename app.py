"""Drone Security Analyst - dashboard for an analysed run: logs, alerts, indexed frames and a Q&A chat.

    .venv/bin/python -m streamlit run app.py     # venv python: has pyarrow + groq

Tabs map to the brief's expected output:
  Logs            "Blue Ford F150 spotted at garage, 12:00."       (events table)
  Alerts          "Person loitering at main gate, 00:01."          (alerts table)
  Indexed frames  queryable by time or object ("show all truck events")  (frames + detections + scenes)
  Chat            free-text questions answered by the LLM from the same DB
The app only reads the DB; it never writes or deletes anything.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, time as dtime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from drone_agent.config import load_config, resolve
from drone_agent.db import Database
from drone_agent.pipeline import Pipeline
from drone_agent.search_agent import SecurityAgent
from drone_agent.sources import build_source

st.set_page_config(page_title="Drone Security Analyst", page_icon="🛸", layout="wide")

EXAMPLES = ["show me the red car", "how many people were detected?", "what happened between 10 and 15 seconds",
            "were there any alerts?"]
SEVERITY = {"critical": st.error, "warning": st.warning, "info": st.info}
BOX = {1: (60, 200, 60), 0: (60, 60, 220), None: (0, 165, 255)}   # BGR: verified / rejected / unverified


MAX_UPLOAD_S = 120
UPLOAD_DIR = "data/videos/uploads"


@st.cache_resource
def gpu_lock() -> threading.Lock:
    """One analysis at a time across all browser sessions (one GPU, one VLM)."""
    return threading.Lock()


@st.cache_resource
def get_cfg_db():
    cfg = load_config()
    return cfg, Database(resolve(cfg.storage.db))


cfg, db = get_cfg_db()
runs = db.runs()
if not runs:
    st.error("No runs in the database yet - run `python run.py analyze` first.")
    st.stop()


# ------------------------------------------------------------------ helpers
def video_name(source: str) -> str:
    """data/videos/gate_breach.mp4 -> 'Gate breach' (the UI shows videos by name, not run id)."""
    return Path(source or "").stem.replace("_", " ").replace("-", " ").strip().capitalize() or "Video"


def clock(ts: str) -> str:
    return (ts or "")[11:19]


def video_path(run: dict) -> Path | None:
    p = Path(run["source"])
    p = p if p.is_absolute() else resolve(p)
    return p if p.exists() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv") else None


@st.cache_data(max_entries=300, show_spinner=False)
def read_frame(path: str, video_t: float):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(video_t * fps)))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def nearest_frame(rid: int, video_t: float) -> dict | None:
    rows = db.q("SELECT * FROM frames WHERE run_id=? ORDER BY ABS(video_t-?) LIMIT 1", (rid, video_t))
    return rows[0] if rows else None


def frame_image(run: dict, frame: dict, tracks: dict, highlight: int | None = None, width: int = 1280):
    """The source-video frame at this indexed frame's time, with its stored detections drawn on it."""
    vp = video_path(run)
    if vp is None or frame is None:
        return None
    img = read_frame(str(vp), frame["video_t"])
    if img is None:
        return None
    img = img.copy()
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)))
    for d in db.q("SELECT * FROM detections WHERE frame_id=?", (frame["id"],)):
        tr = tracks.get(d["track_id"], {})
        hit = d["track_id"] == highlight
        color = (0, 0, 255) if hit else BOX.get(tr.get("verified"), BOX[None])
        x1, y1, x2, y2 = (int(d[k] * s) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4 if hit else 2)
        name = tr.get("vlm_label") or d["label"]
        tag = f"#{d['track_id']} {name}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_frame(run, frame, tracks, highlight=None, caption=""):
    img = frame_image(run, frame, tracks, highlight)
    if img is not None:
        st.image(img, caption=caption, width="stretch")
    elif frame is not None and frame.get("caption"):
        st.info(frame["caption"])       # text-simulated run: the frame IS a text description
    else:
        st.caption("no image for this run")


def time_slider(key: str, max_t: float):
    if max_t <= 0:
        return 0.0, 0.0
    return st.slider("Video time (s)", 0.0, float(max_t), (0.0, float(max_t)), step=0.2, key=key)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛸 Drone Security Analyst")
    run_ids = [r["id"] for r in runs]
    if "run_next" in st.session_state:              # set by a finished upload, applied before the widget exists
        st.session_state["run_sel"] = st.session_state.pop("run_next")
    if st.session_state.get("run_sel") not in run_ids:
        st.session_state["run_sel"] = max(run_ids)
    rid = st.selectbox("Video", run_ids, key="run_sel", format_func=lambda i: video_name(db.run(i)["source"]))
    run = db.run(rid)
    stats = db.stats(rid)
    st.caption(f"Clock start {run['sim_start']}")
    c1, c2 = st.columns(2)
    c1.metric("Frames", stats.get("frames", 0))
    c2.metric("Verified objects", stats.get("tracks_verified", "?"))
    c1.metric("VLM scenes", stats.get("vlm_scenes", 0))
    c2.metric("Alerts", len(db.alerts(rid)))
    counts = db.object_counts(rid)
    if counts:
        st.markdown("**Objects (VLM-verified)**")
        st.dataframe(pd.DataFrame(counts)[["object", "n"]], hide_index=True, width="stretch")
    st.markdown(f"**LLM:** Groq `{cfg.agent.get('groq_model')}`  \nfallback: local `{cfg.agent.get('model')}`")
    if st.button("Clear chat", width="stretch"):
        st.session_state.pop(f"chat_{rid}", None)
        st.session_state.pop(f"agent_{rid}", None)
        st.rerun()

tracks = {t["track_id"]: t for t in db.tracks(rid, verified_only=False)}
max_t = (db.q("SELECT MAX(video_t) m FROM frames WHERE run_id=?", (rid,))[0]["m"] or 0.0)

# a view switcher instead of st.tabs: only a top-level st.chat_input is pinned to the bottom of the page
VIEWS = ["📋 Logs", "🚨 Alerts", "🎞️ Indexed frames", "💬 Chat", "⬆️ Upload video"]
if "view_next" in st.session_state:          # set by a finished upload
    st.session_state["view"] = st.session_state.pop("view_next")
if not st.session_state.get("view"):           # initial value via session state only (no `default=`), so
    st.session_state["view"] = VIEWS[0]        # setting it programmatically doesn't trigger a Streamlit warning
view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or VIEWS[0]

# ------------------------------------------------------------------ logs
if view == "📋 Logs":
    st.subheader("Logs")
    st.caption("Every object the VLM confirmed: when it was spotted, where, and when it left view.")
    events = db.events(rid, limit=100000)
    types = sorted({e["type"] for e in events})
    f1, f2, f3 = st.columns([2, 2, 3])
    sel_types = f1.multiselect("Type", types, default=[t for t in types if t != "left"] or types, key="log_types")
    obj = f2.text_input("Object (e.g. truck, person, red)", key="log_obj").strip()
    t0, t1 = (f3.container().slider("Video time (s)", 0.0, float(max_t), (0.0, float(max_t)), step=0.2,
                                    key="log_t") if max_t > 0 else (0.0, 0.0))
    rows = db.events(rid, label=obj or None, start_s=t0, end_s=t1, limit=100000)
    rows = [e for e in rows if e["type"] in sel_types]
    if not rows:
        st.info("No logs match these filters.")
    else:
        df = pd.DataFrame([{"t (s)": e["video_t"], "clock": clock(e["ts"]), "type": e["type"],
                            "track": e["track_id"], "object": e["label"], "location": e["zone"],
                            "log": e["message"]} for e in rows])
        st.caption(f"{len(rows)} logs")
        st.dataframe(df, hide_index=True, width="stretch")
        pick = st.selectbox("Show the frame for a log", range(len(rows)), key="log_pick",
                            format_func=lambda i: f"t={rows[i]['video_t']}s · {rows[i]['message'][:110]}")
        e = rows[pick]
        a, b = st.columns([3, 1])
        with a:
            show_frame(run, nearest_frame(rid, e["video_t"] or 0), tracks, e["track_id"],
                       f"t={e['video_t']}s · {clock(e['ts'])} · track #{e['track_id']} in red")
        with b:
            tr = tracks.get(e["track_id"], {})
            if tr.get("thumb_path") and Path(tr["thumb_path"]).exists():
                st.image(tr["thumb_path"], caption="crop the VLM judged", width="stretch")
            st.markdown(f"**#{e['track_id']}** {tr.get('description') or ''}")
            st.caption(f"YOLO: {tr.get('label')} → VLM: {tr.get('vlm_label')}  \ncolour: {tr.get('color')} · "
                       f"type: {tr.get('obj_type')} · state: {tr.get('state')}  \nlocation: {tr.get('zones')}  \n"
                       f"seen {tr.get('first_video_t')}–{tr.get('last_video_t')}s · verified by "
                       f"{tr.get('verify_source')} at {tr.get('verified_video_t')}s")

# ------------------------------------------------------------------ alerts
if view == "🚨 Alerts":
    st.subheader("Alerts")
    st.caption("Raised by the rules (night activity, loitering, restricted area, repeat vehicle) on VLM-verified "
               "objects. Time = when the condition happened.")
    alerts = db.alerts(rid)
    if not alerts:
        st.success("No alerts in this run.")
    rules = sorted({a["rule"] for a in alerts})
    sel_rules = st.multiselect("Rule", rules, default=rules, key="al_rules") if rules else []
    for al in [a for a in alerts if a["rule"] in sel_rules]:
        with st.container(border=True):
            a, b = st.columns([2, 3])
            with a:
                SEVERITY.get(al["severity"], st.info)(f"**{al['rule'].replace('_', ' ').upper()}** "
                                                      f"({al['severity']})  \n{al['message']}")
                tr = tracks.get(al["track_id"], {})
                who = f" · track #{al['track_id']}" if al["track_id"] is not None else " · whole scene (VLM)"
                st.caption(f"t={al['video_t']}s · {clock(al['ts'])}{who} · {al['zone'] or ''}")
                if tr.get("thumb_path") and Path(tr["thumb_path"]).exists():
                    st.image(tr["thumb_path"], caption="crop the VLM judged", width=160)
            with b:
                show_frame(run, nearest_frame(rid, al["video_t"] or 0), tracks, al["track_id"],
                           "frame at the alert time" + (f" · track #{al['track_id']} in red"
                                                        if al["track_id"] is not None else ""))

# ------------------------------------------------------------------ indexed frames
if view == "🎞️ Indexed frames":
    st.subheader("Indexed frames")
    st.caption("Every analysed frame is a DB row keyed by its video time (and clock time), with the drone "
               "telemetry and what YOLO detected in it; the VLM adds a scene description once per second. "
               "Query by time or by object.")
    g1, g2 = st.columns([3, 2])
    with g1:
        ft0, ft1 = time_slider("fr_t", max_t)
    fobj = g2.text_input("Object in frame (e.g. truck, person)", key="fr_obj").strip()
    frows = db.frames(rid, start_s=ft0, end_s=ft1, label=fobj or None, limit=100000)
    if not frows:
        st.info("No frames match these filters.")
    else:
        st.caption(f"{len(frows)} frames")
        fdf = pd.DataFrame([{"frame": f["frame_idx"], "t (s)": f["video_t"], "clock": clock(f["ts"]),
                             "lat": round(f["lat"], 5), "lon": round(f["lon"], 5), "alt m": f["altitude_m"],
                             "battery %": f["battery_pct"], "objects": f["n_objects"], "labels": f["labels"],
                             **({"text": f["caption"]} if any(x["caption"] for x in frows) else {})}
                            for f in frows])
        st.dataframe(fdf, hide_index=True, width="stretch", height=260)
        idx = st.select_slider("Open frame (video time, s)", options=list(range(len(frows))), key="fr_pick",
                               format_func=lambda i: f"{frows[i]['video_t']:.1f}")
        fr = frows[idx]
        a, b = st.columns([3, 2])
        with a:
            show_frame(run, fr, tracks, caption=f"frame {fr['frame_idx']} · t={fr['video_t']}s · {clock(fr['ts'])}")
        with b:
            st.markdown(f"**Frame {fr['frame_idx']}** · t={fr['video_t']}s · {clock(fr['ts'])}")
            st.caption(f"lat {fr['lat']:.5f}, lon {fr['lon']:.5f} · alt {fr['altitude_m']:.0f} m · "
                       f"heading {fr['heading_deg']:.0f}° · battery {fr['battery_pct']:.0f}%")
            dets = db.q("SELECT track_id, label, conf FROM detections WHERE frame_id=?", (fr["id"],))
            if dets:
                st.dataframe(pd.DataFrame([{"track": d["track_id"], "YOLO": d["label"], "conf": round(d["conf"], 2),
                                            "VLM": tracks.get(d["track_id"], {}).get("description") or "-"}
                                           for d in dets]), hide_index=True, width="stretch")
            sc = db.q("SELECT * FROM scenes WHERE run_id=? AND video_t<=? ORDER BY video_t DESC LIMIT 1",
                      (rid, fr["video_t"] + 1e-6))
            if sc:
                st.markdown(f"**VLM scene (t={sc[0]['video_t']}s):** {sc[0]['scene']}")


# ------------------------------------------------------------------ upload + analyse a new video
def probe_video(path: Path) -> tuple[float, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (n / fps if fps else 0.0), w, h, fps


def save_upload(up) -> Path:
    folder = resolve(UPLOAD_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(up.name).stem.replace(" ", "_"), Path(up.name).suffix.lower()
    dest, k = folder / f"{stem}{suffix}", 2
    while dest.exists():                       # never overwrite an earlier upload (its run points to it)
        dest, k = folder / f"{stem}_{k}{suffix}", k + 1
    dest.write_bytes(up.getbuffer())
    return dest


def analyse(path: Path, start: datetime, vehicles_too: bool, width: int, duration: float) -> int:
    """Runs the existing pipeline unchanged (same code as `run.py analyze`), with per-run config overrides."""
    imgsz = min(1920, max(640, -(-width // 32) * 32))          # native width, capped like the 4K setting
    over = {"detector": {"imgsz": imgsz}}
    if vehicles_too:
        over["rules"] = {"night_activity": {"categories": ["person", "vehicle"]},
                         "loitering": {"categories": ["person", "vehicle"]}}
    run_cfg = load_config(overrides=over)
    rel = str(path.relative_to(resolve(".")))
    pipe = Pipeline(run_cfg, db=db, write_video=False)
    bar = st.progress(0.0, text="Starting YOLO + VLM ...")
    status, alert_box = st.empty(), st.container()
    shown = 0
    for i, step in enumerate(pipe.run_iter(build_source(run_cfg, video=str(path), start_time=start.isoformat()),
                                           rel)):
        t = step.packet.video_t
        bar.progress(min(t / duration, 1.0) if duration else 0.0,
                     text=f"Video {t:.1f} / {duration:.1f} s · VLM calls waiting: {step.pending_vlm}")
        if i % 5 == 0:
            alerts_now = db.alerts(pipe.run_id)
            for al in alerts_now[shown:]:
                alert_box.warning(f"**{al['rule'].replace('_', ' ')}** · {al['message']}")
            shown = len(alerts_now)
            status.caption(f"objects tracked now: {step.active_tracks}")
        if t >= duration - 0.5:
            bar.progress(1.0, text="Video done · waiting for the last VLM verdicts ...")
    st_ = pipe.stats or {}
    for al in db.alerts(pipe.run_id)[shown:]:
        alert_box.warning(f"**{al['rule'].replace('_', ' ')}** · {al['message']}")
    status.empty()
    bar.progress(1.0, text=f"Done in {st_.get('wall_s', 0):.0f} s")
    return pipe.run_id


if view == "⬆️ Upload video":
    st.subheader("Analyse a new video")
    st.caption(f"Runs the full pipeline (YOLO + tracking → VLM verification → rules → database) on your clip. "
               f"Max {MAX_UPLOAD_S // 60} minutes. It becomes a new entry in the Video list; earlier videos are "
               "not changed.")
    up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"], key=f"upload_file_{st.session_state.get('up_n', 0)}")
    if up is not None:
        if st.session_state.get("upload_saved_name") != up.name + str(up.size):
            path = save_upload(up)
            st.session_state["upload_path"] = str(path)
            st.session_state["upload_saved_name"] = up.name + str(up.size)
        path = Path(st.session_state["upload_path"])
        dur, w, h, fps = probe_video(path)
        if dur <= 0:
            st.error("Could not read this video (unknown length or codec).")
        elif dur > MAX_UPLOAD_S:
            st.error(f"This video is {dur:.0f} s long. Please upload a clip under {MAX_UPLOAD_S} s.")
        else:
            st.success(f"{path.name}: {dur:.1f} s · {w}×{h} · {fps:.0f} fps")
            st.video(str(path))
            c1, c2, c3 = st.columns(3)
            d = c1.date_input("Clock start date", value=date.today(), key="up_date")
            tm = c2.time_input("Clock start time", value=dtime(datetime.now().hour, datetime.now().minute),
                               step=60, key="up_time",
                               help="The simulated wall-clock time of the first frame. Night rules use it "
                                    f"({cfg.rules.night_activity.start}-{cfg.rules.night_activity.end}).")
            veh = c3.checkbox("Night activity + loitering also for vehicles", value=False, key="up_veh")
            est = dur * 4.5
            st.caption(f"Estimated time: about {est / 60:.0f}-{est * 1.6 / 60:.0f} min "
                       "(one VLM call per second of video with detections).")
            lock = gpu_lock()
            busy = lock.locked()
            if busy:
                st.warning("Another video is being analysed right now. Please wait for it to finish.")
            if st.button("▶ Analyse", type="primary", disabled=busy, key="up_go"):
                with lock:
                    try:
                        new_rid = analyse(path, datetime.combine(d, tm), veh, w, dur)
                    except Exception as exc:          # e.g. Ollama not running
                        st.error(f"Analysis failed: {exc}")
                        st.stop()
                st.session_state["run_next"] = new_rid
                st.session_state["view_next"] = "📋 Logs"
                st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1   # fresh, empty uploader
                for k in ("upload_saved_name", "upload_path"):
                    st.session_state.pop(k, None)
                st.success("Done. Opening its logs ...")
                st.rerun()

# ------------------------------------------------------------------ chat
akey, ckey = f"agent_{rid}", f"chat_{rid}"
if akey not in st.session_state:
    st.session_state[akey] = SecurityAgent(db, rid, cfg)
chat: list[dict] = st.session_state.setdefault(ckey, [])
agent: SecurityAgent = st.session_state[akey]


def render_answer(ans) -> None:
    st.markdown(ans.answer or "_(no answer)_")
    if ans.images:
        st.markdown(f"**Images** ({len(ans.images)})")
        cols = st.columns(3)
        by_id = {t["track_id"]: t for t in ans.tracks}
        for i, im in enumerate(ans.images):
            t = by_id.get(im["track_id"], {})
            with cols[i % 3]:
                if im.get("context") and Path(im["context"]).exists():
                    st.image(im["context"], width="stretch")
                if im.get("crop") and Path(im["crop"]).exists():
                    st.image(im["crop"], width=90, caption="crop the VLM judged")
                st.caption(f"**#{im['track_id']}** {t.get('description') or ''}  \n"
                           f"t={im['video_t']:.2f}s · {str(im['ts'])[11:19]} · {t.get('color')} "
                           f"{t.get('vlm_label')} · {t.get('zones')}")
                if im.get("frame") and Path(im["frame"]).exists():
                    with st.popover("full frame"):
                        st.image(im["frame"], width="stretch")
    if ans.events:
        st.markdown(f"**Events** ({len(ans.events)})")
        df = pd.DataFrame(ans.events)[["video_t", "ts", "type", "track_id", "message"]]
        df["ts"] = df["ts"].str[11:19]
        st.dataframe(df.rename(columns={"video_t": "t (s)", "ts": "clock", "track_id": "track"}),
                     hide_index=True, width="stretch")
    if ans.scenes:
        with st.expander(f"Supporting scenes ({len(ans.scenes)})"):
            for s in ans.scenes:
                st.markdown(f"**t={s['video_t']}s** {s['ts'][11:19]} - {s['scene']}")
                if s.get("image_path") and Path(s["image_path"]).exists():
                    st.image(s["image_path"], width=420)
    with st.expander("Context sent to the LLM"):
        st.code(ans.context, language=None)
    st.caption(f"{ans.model} · show_images={ans.show_images}")


if view == "💬 Chat":
    st.subheader(f"Ask about: {video_name(run['source'])}")
    st.caption("Answers come only from the stored detections, VLM descriptions, events and alerts. "
               "Follow-up questions keep the conversation context.")
    for msg in chat:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["text"])
            else:
                render_answer(msg["answer"])
    pending = st.session_state.pop("pending_q", None)
    if not chat and not pending:
        st.markdown("**Try:**")
        cols = st.columns(len(EXAMPLES))
        for c, ex in zip(cols, EXAMPLES):
            if c.button(ex, width="stretch", key=f"ex_{ex}"):
                st.session_state["pending_q"] = ex
                st.rerun()
    question = pending
    # main page body, not inside a tab/column/container -> Streamlit pins it to the bottom of the page
    question = st.chat_input("e.g. show me the red car") or question
    if question:
        chat.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the footage..."):
                ans = agent.ask(question, save_page=False)
            render_answer(ans)
        chat.append({"role": "assistant", "answer": ans})
