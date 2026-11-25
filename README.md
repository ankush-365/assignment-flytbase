# Drone Security Analyst Agent

Processes drone video + telemetry, logs objects and events, raises security alerts, indexes every frame in
SQLite, and answers questions in a chat. YOLO11 + a local VLM (qwen3-vl:2b via Ollama) do the vision;
Groq (free tier) answers the chat.

Architecture and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Setup

Needs Python 3.12 and an NVIDIA GPU (~4 GB VRAM).

```bash
git clone https://github.com/ankush-365/assignment-flytbase.git
cd assignment-flytbase
python -m venv .venv && source .venv/bin/activate

# PyTorch for your CUDA version (https://pytorch.org/get-started/locally/), e.g. CUDA 12.4:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# YOLO weights
mkdir -p models && wget -P models https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

# Ollama (https://ollama.com): VLM, embeddings, local chat fallback
ollama pull qwen3-vl:2b-instruct
ollama pull nomic-embed-text:v1.5
ollama pull llama3.1

# Groq key for the chat (https://console.groq.com/keys)
echo "GROQ_API_KEY=gsk_..." > .env
```

Videos are not in the repo. The database already holds three analysed clips; to see their frames, put them in
`data/videos/` as `drone_street.mp4`, `night_loiter.mp4`
([source](https://www.gettyimages.in/detail/video/alone-traveler-with-surgical-mask-walking-on-the-lonely-stock-footage/1339983520))
and `gate_breach.mp4`
([source](https://www.gettyimages.in/detail/video/silver-2004-dodge-stratus-4-door-sedan-crashes-stock-video-footage/1208736653)).

## What to run

| To | Run |
|---|---|
| Open the app (logs, alerts, indexed frames, chat, upload a video) | `python -m streamlit run app.py` |
| Analyse a video | `python run.py analyze --video data/videos/night_loiter.mp4 --start 2026-11-24T23:59:00` |
| Analyse with text-simulated frames (no video/GPU) | `python run.py analyze --sim demo --start 2026-11-24T23:59:30` |
| Override a setting for one run | `python run.py --set detector.imgsz=768 analyze --video ...` |
| Show logs | `python run.py query --run 6` |
| Show alerts | `python run.py query --run 6 --alerts` |
| Query by object ("show all truck events") | `python run.py query --run 2 --label truck` |
| Query by video time / clock time | `python run.py query --run 6 --from-s 25 --to-s 55` · `--start 2026-11-24T23:59:00 --end 2026-11-24T23:59:05` |
| Per-second VLM scene descriptions | `python run.py query --run 4 --scenes` |
| Semantic search | `python run.py search --run 2 "person carrying a bag"` |
| Ask a question / chat in the terminal | `python run.py ask --run 6 "what happened at the gate?"` · `python run.py ask --run 6` |
| One-line video summary | `python run.py summary --run 6` |
| Review logs against the video (HTML page) | `python scripts/review_events.py --run 6` |
| Run the tests (no GPU/API needed) | `python -m pytest` |

Runs in the database: **2** = drone street (daytime), **4** = night loitering, **6** = gate breach.
Settings (thresholds, models, rules) are in [`config.yaml`](config.yaml).

## Design decisions

```
video + telemetry → YOLO11 + ByteTrack → VLM check (1 call/s, async) → rules → SQLite → app / CLI / chat
```

- **YOLO finds, the VLM verifies.** YOLO is fast but makes false detections on drone footage (rooftops as
  "train"). A small local VLM checks every object and adds colour, type, state, location and a description.
- **One VLM call per second, not per object.** Cost stays bounded by video length; the same call describes
  the whole scene, which powers the "unusual activity" alert (e.g. a car crashing through a gate).
- **qwen3-vl:2b on Ollama.** Free and local, fits in 3 GB VRAM, 3-5 s per call. 3B/7B models spilled to the
  CPU (23-47 s per call).
- **Async VLM.** It runs on a background thread so the video never waits; logs and alerts keep the time the
  event happened, not the time the VLM answered.
- **Locations as text, not drawn zones.** The drone moves, so fixed image regions point at different ground;
  the VLM names the place from landmarks ("at the gate").
- **Alerts only on verified objects seen ≥ 1 s**, once per object, so false detections and blips never alert.
- **SQLite with timestamps on every row.** One file, no server; every frame queryable by time or object.
- **Chat = filters in code + one LLM call.** Colour/object/place/time filters pick a small context (≤ 15
  objects), so answers are fast, cheap and grounded; follow-ups search within the previous results.

Details, diagrams and measured performance: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## AI tools

**In the system**

| Tool | Used for |
|---|---|
| YOLO11s + ByteTrack (Ultralytics) | detecting and tracking objects |
| qwen3-vl:2b-instruct (Ollama, local) | verifying objects, attributes, locations, per-second scene description |
| nomic-embed-text (Ollama, local) | embeddings for semantic search and repeat-vehicle matching |
| openai/gpt-oss-120b (Groq, free tier) | chat answers and summaries (fallback: local llama3.1) |

**In development**

- **Claude Code** was the pair programmer: it proposed the architecture and wrote most of the code (pipeline,
  VLM prompt + schema, rules, database, chat, Streamlit app, tests, docs).
- **Impact:** a full prototype in days instead of weeks. Most of my time went into deciding scope, choosing
  models, and checking results on real videos. Each problem found there was fixed and re-tested the same
  day, e.g. the VLM cutting its JSON short, a 0.2 s blip raising a night alert, the crash detail being
  dropped by the alert cooldown.
