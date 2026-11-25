#!/usr/bin/env python3
"""Drone Security Analyst Agent - command line.

  python run.py analyze --video data/videos/clip.mp4      # full pipeline on a video
  python run.py analyze --sim demo                       # scripted text-simulation (no video / GPU needed)
  python run.py ask "show all truck events"              # Q&A over the latest run
  python run.py summary                                  # one-sentence video summary
  python run.py query --label truck                      # structured query, no LLM
  python run.py search "man in red jacket"               # semantic search over frames / tracks
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from drone_agent.config import load_config, resolve
from drone_agent.db import Database


def _overrides(pairs: list[str]) -> dict:
    """['detector.imgsz=768', 'rules.loitering.seconds=8'] -> nested dict (values parsed as YAML)."""
    import yaml

    out: dict = {}
    for pair in pairs:
        key, _, val = pair.partition("=")
        node = out
        *parents, leaf = key.strip().split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(val)
    return out


def _run_id(db: Database, arg) -> int:
    rid = arg or db.latest_run_id()
    if rid is None:
        sys.exit("No runs in the database yet - run `python run.py analyze` first.")
    return int(rid)


def cmd_analyze(args, cfg):
    from drone_agent.pipeline import Pipeline, output_video_path
    from drone_agent.sources import build_source

    overrides_vlm = False if args.no_vlm else None
    db = Database(resolve(cfg.storage.db))
    src = build_source(cfg, video=args.video, sim=args.sim, sim_duration_s=args.duration,
                       arrivals_per_min=args.arrivals, start_time=args.start)
    vlm = embedder = None
    if args.mock:
        from drone_agent.vlm import HashEmbedder, MockVLM

        vlm, embedder = MockVLM(), HashEmbedder()
    pipe = Pipeline(cfg, db=db, vlm=vlm, embedder=embedder, use_vlm=overrides_vlm,
                    use_embeddings=False if args.no_embed else None, write_video=not args.no_video)
    name = f"sim:{args.sim}" if args.sim else (args.video or cfg.source.video)
    rid = pipe.run(src, source_name=name, progress_every=args.progress)
    st = dict(pipe.stats)
    tm = st.pop("timing")
    calls = tm.pop("calls")
    print(f"\nRun {rid} finished: {json.dumps(st, indent=2)}")
    print("\nTiming:")
    print(f"  video            {tm['video_duration_s']} s")
    print(f"  YOLO             {tm['yolo_frames']} frames, {tm['yolo_total_s']} s total, "
          f"{tm['yolo_avg_ms_per_frame']} ms/frame")
    print(f"  VLM              {tm['vlm_calls']} calls, {tm['vlm_total_s']} s total, avg {tm['vlm_avg_s']} s, "
          f"max {tm['vlm_max_s']} s, truncated {tm['vlm_truncated_calls']}, failed {tm['vlm_failed_calls']}")
    print(f"  wall clock       {tm['wall_s']} s  (realtime factor {tm['realtime_factor']}x)")
    for c in calls:
        print(f"    {c.get('kind', 'scene'):12} @ {c['video_t']:5.1f}s: {c['n_objects']:2d} crops "
              f"({c.get('n_backlog', 0)} backlog), {c.get('returned', '?')} returned, "
              f"{c['tokens']}/{c.get('budget')} tokens, {c['latency_s']} s"
              f"{'  TRUNCATED' if c['truncated'] else ''}{'  ERROR' if c['error'] else ''}")
    print("\nEvents:")
    print("\nVLM scenes (one per second):")
    for sc in db.scenes(rid):
        print(f"  [{sc['video_t']:5.1f}s {sc['ts'][11:]}] ({sc['n_objects']} obj, {sc['latency_s']}s) {sc['scene']}")
    for e in db.events(rid):
        print(f"  [{e['video_t']:5.1f}s {e['ts'][11:]}] {e['type']:8} {e['message']}")
    print("\nAlerts:")
    for a in db.alerts(rid):
        print(f"  [{a['video_t']:5.1f}s {a['ts'][11:]}] {a['rule']}: {a['message']}")
    if args.summary:
        from drone_agent.search_agent import summarize_run

        print("\nSummary:", summarize_run(db, rid, cfg))
    if not args.no_video and not args.sim:
        print(f"\nAnnotated video: {output_video_path(cfg, rid)}")


def _print_answer(ans):
    print(f"\n{ans.answer}\n")
    if ans.events:
        print("Events:")
        for e in ans.events:
            print(f"  [{e['video_t']:5.1f}s {e['ts'][11:19]}] {e['type']:9} #{e['track_id']} {e['message']}")
    for im in ans.images:
        print(f"  image #{im['track_id']} t={im['video_t']}s: {im['context'] or im['crop']}")
    print(f"({ans.model}; show_images={ans.show_images}) page: {ans.page}")


def cmd_ask(args, cfg):
    from drone_agent.search_agent import SecurityAgent

    db = Database(resolve(cfg.storage.db))
    agent = SecurityAgent(db, _run_id(db, args.run), cfg)
    if args.question:
        _print_answer(agent.ask(" ".join(args.question)))
        return
    print("Ask about the footage (empty line to quit). Follow-up questions keep context.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        _print_answer(agent.ask(q))


def cmd_summary(args, cfg):
    from drone_agent.search_agent import summarize_run

    db = Database(resolve(cfg.storage.db))
    print(summarize_run(db, _run_id(db, args.run), cfg))


def cmd_query(args, cfg):
    db = Database(resolve(cfg.storage.db))
    rid = _run_id(db, args.run)
    tw = dict(start=args.start, end=args.end, start_s=args.from_s, end_s=args.to_s)
    if args.scenes:
        rows = db.scenes(rid, text=args.label, **tw)
        for r in rows:
            print(f"[{r['video_t']:5.1f}s {r['ts']}] {r['scene']}")
    elif args.tracks:
        rows = db.tracks(rid, label=args.label, **tw)
        for r in rows:
            print(f"[{r['first_video_t']:5.1f}-{r['last_video_t']:5.1f}s] #{r['track_id']} {r['category']:8}"
                  f" {r['description']}  (color={r['color']}, type={r['obj_type']}, state={r['state']},"
                  f" verified by {r['verify_source']} at {r['verified_video_t']}s)")
    else:
        rows = (db.alerts(rid, rule=args.rule, **tw) if args.alerts else
                db.events(rid, label=args.label, event_type=args.type, zone=args.zone, **tw))
        for r in rows:
            print(f"[{r['video_t']:5.1f}s {r['ts']}] {r.get('type') or r.get('rule'):15} {r['message']}")
    print(f"({len(rows)} rows)")


def cmd_search(args, cfg):
    from drone_agent.search import semantic_search

    db = Database(resolve(cfg.storage.db))
    for hit in semantic_search(db, _run_id(db, args.run), " ".join(args.text), cfg, k=args.k):
        print(f"{hit['score']:.3f}  [{hit.get('video_t') or 0:5.1f}s {hit['ts']}]  [{hit['kind']}] {hit['text']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value for this run, e.g. --set detector.imgsz=768 (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="run the detection/verification pipeline")
    a.add_argument("--video", help="video file or RTSP url (default: config source.video)")
    a.add_argument("--sim", choices=["demo", "random"], help="text-simulation mode instead of video")
    a.add_argument("--duration", type=float, default=120.0, help="simulation length in seconds")
    a.add_argument("--arrivals", type=float, default=2.0, help="random sim: objects per minute")
    a.add_argument("--start", help="simulated start time, ISO format (default: config)")
    a.add_argument("--no-vlm", action="store_true", help="skip VLM verification (YOLO only)")
    a.add_argument("--no-embed", action="store_true", help="skip embeddings")
    a.add_argument("--mock", action="store_true", help="mock VLM + hash embeddings (offline)")
    a.add_argument("--no-video", action="store_true", help="don't write annotated video")
    a.add_argument("--summary", action="store_true", help="print LLM one-sentence summary at the end")
    a.add_argument("--progress", type=int, default=20, help="log every N frames")
    a.set_defaults(fn=cmd_analyze)

    q = sub.add_parser("ask", help="ask the agent about a run (interactive if no question)")
    q.add_argument("question", nargs="*")
    q.add_argument("--run", type=int)
    q.set_defaults(fn=cmd_ask)

    s = sub.add_parser("summary", help="one-sentence summary of a run")
    s.add_argument("--run", type=int)
    s.set_defaults(fn=cmd_summary)

    qq = sub.add_parser("query", help="structured query (no LLM)")
    qq.add_argument("--label")
    qq.add_argument("--type", choices=["spotted", "left", "rejected"])
    qq.add_argument("--zone")
    qq.add_argument("--start")
    qq.add_argument("--end")
    qq.add_argument("--from-s", type=float, help="video seconds, window start")
    qq.add_argument("--to-s", type=float, help="video seconds, window end")
    qq.add_argument("--alerts", action="store_true")
    qq.add_argument("--tracks", action="store_true", help="list objects (with attributes) instead of events")
    qq.add_argument("--scenes", action="store_true", help="list VLM scene descriptions")
    qq.add_argument("--rule")
    qq.add_argument("--run", type=int)
    qq.set_defaults(fn=cmd_query)

    se = sub.add_parser("search", help="semantic search")
    se.add_argument("text", nargs="+")
    se.add_argument("-k", type=int, default=8)
    se.add_argument("--run", type=int)
    se.set_defaults(fn=cmd_search)

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "ultralytics", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.fn(args, load_config(args.config, _overrides(args.set)))


if __name__ == "__main__":
    main()
