# Architecture: Drone Security Analyst Agent

A prototype for a docked drone that patrols a property. It processes video frames and drone telemetry,
identifies and logs objects and events with context, raises security alerts from predefined rules,
indexes every frame in a database, and answers questions about the footage.

All diagrams below are Mermaid (rendered by GitHub). PNG versions are in [`docs/img/`](img/).

---

## 1. System overview

![System overview](img/1_system_overview.png)

```mermaid
flowchart LR
    subgraph IN["Inputs"]
        V["Video file or RTSP stream<br/>(5 frames/s sampled)"]
        S["Text simulation<br/>'Frame 12: man in red jacket<br/>at the main gate'"]
        T["Simulated telemetry<br/>GPS, altitude, heading, battery"]
    end

    subgraph PIPE["Processing pipeline (drone_agent/pipeline.py)"]
        D["YOLO11s + ByteTrack<br/>detect + track<br/>~20-50 ms/frame, GPU"]
        TS["Track state<br/>first/last seen, best crop,<br/>verdict, location"]
        Q[["Async VLM queue<br/>1 worker thread"]]
        VLM["Local VLM: qwen3-vl:2b<br/>(Ollama, 3.1 GB VRAM)<br/>1 call per second:<br/>scene + every new object"]
        EMB["Embeddings<br/>nomic-embed-text"]
        R["Rules engine<br/>night activity, loitering,<br/>restricted area, repeat vehicle,<br/>unusual activity"]
    end

    DB[("SQLite (WAL)<br/>runs, frames, detections,<br/>scenes, tracks, events,<br/>alerts, embeddings")]

    subgraph OUT["Interfaces"]
        CLI["CLI: run.py<br/>analyze / query / search / ask"]
        APP["Streamlit app<br/>Logs, Alerts, Indexed frames,<br/>Chat, Upload video"]
        QA["Q&A agent<br/>Groq gpt-oss-120b<br/>(fallback: local llama3.1)"]
    end

    V --> D
    T --> D
    S -->|"detections from text"| TS
    D --> TS
    TS -->|"crops of new objects"| Q --> VLM
    VLM -->|"verdict, colour, type,<br/>state, location, scene"| TS
    VLM --> EMB
    TS --> R
    VLM -->|"'unusual' field"| R
    D -->|"every frame + box"| DB
    TS -->|"spotted / left logs"| DB
    R -->|"alerts"| DB
    VLM -->|"scene per second"| DB
    EMB --> DB
    DB --> CLI
    DB --> APP
    DB --> QA
    QA --> APP
```

| Component | File | Role |
|---|---|---|
| Sources | `sources.py` | Video reader (file or RTSP, resampled to 5 fps) and the text simulator (scripted demo or random arrivals) |
| Telemetry | `telemetry.py` | Simulated docked-drone patrol: circular route, altitude jitter, battery drain, clock time per frame |
| Detector | `detector.py` | YOLO11s with ByteTrack (`model.track(persist=True)`); maps COCO labels to person / vehicle / object |
| VLM stage | `vlm.py` | One structured-JSON call per second: full frame + numbered crop sheet; salvage of cut-off JSON |
| Pipeline | `pipeline.py` | Track state, VLM scheduling, retries, verdicts, logs, rule evaluation, timing report |
| Rules | `rules.py` | Pure functions: (track state, time, config) → alerts |
| Storage | `db.py` | SQLite schema, writers, time / object / text queries |
| Q&A | `search_agent.py`, `search.py` | Retrieval from the DB + LLM answer + image evidence, follow-up context |
| UI | `app.py`, `run.py` | Streamlit dashboard and command line |

---

## 2. Per-frame flow (YOLO and VLM in parallel)

YOLO takes about 20-50 ms per frame; one VLM call takes 3-5 s. The VLM therefore runs on a background
thread and never blocks the frame loop. Every log and alert is stamped with the time the thing
**happened** in the video, not the time the VLM answered.

![Per-frame flow](img/2_per_frame_flow.png)

```mermaid
sequenceDiagram
    autonumber
    participant SRC as Video source
    participant P as Pipeline (main thread)
    participant Y as YOLO + ByteTrack
    participant W as VLM worker thread
    participant DB as SQLite
    participant R as Rules

    loop every sampled frame (5 per second)
        SRC->>P: frame + telemetry + clock time
        P->>Y: detect + track
        Y-->>P: boxes with track ids
        P->>DB: frames row (time, GPS, altitude, battery, labels)<br/>+ detections rows
        P->>P: update track state, keep best crop per object
        opt first frame of a new video second, objects detected
            P->>W: submit: full frame + crops of up to 12<br/>not-yet-verified objects
        end
        P->>P: collect finished VLM answers (non-blocking)
        W-->>P: scene + per-object verdicts (3-5 s later)
        P->>DB: scenes row, tracks row, "spotted" log
        P->>R: evaluate verified objects
        R-->>DB: alerts (timestamped when the condition began)
        P->>P: objects unseen for 3 s: "left view" log,<br/>or backlog if still unverified
    end
    P->>W: end of video: wait for the queue, then<br/>extra calls for objects that never got a slot
```

---

## 3. VLM verification

YOLO is trained on ground-level photos (COCO). On drone footage it produces false positives (rooftops as
"train", signs as "cell phone"). Every object is therefore checked by a small local VLM before it becomes a
log entry or can raise an alert. The VLM also adds what YOLO cannot see: colour, finer type, state, location
and a description.

![VLM verification](img/3_vlm_verification.png)

```mermaid
flowchart TD
    A["New track seen in >= 2 frames"] --> B["Keep the best crop<br/>(confidence x box area)"]
    B --> C{"Next video second<br/>with detections?"}
    C -->|yes| D["Pick up to 12 objects:<br/>visible + unverified first<br/>(people, then vehicles, then others),<br/>spare slots: backlog, oldest first"]
    D --> E["One VLM call<br/>image 1: full frame (896 px)<br/>image 2: numbered crop sheet<br/>JSON schema enforced"]
    E --> F{"Answer complete?"}
    F -->|"cut off at the token limit"| G["Salvage complete entries<br/>(scene + finished objects)"]
    F -->|yes| H["Per object: is_match, object,<br/>colour, type, state, location,<br/>description"]
    G --> H
    H --> I{"Object in the answer?"}
    I -->|no| J{"2 attempts used?"}
    J -->|no| C
    J -->|yes| K["Give up on the crop"]
    I -->|yes| L{"is_match?"}
    L -->|yes| M["VERIFIED<br/>'spotted' log + embedding<br/>rules can fire"]
    L -->|"no, but the VLM says it is<br/>a person or vehicle"| N["Class corrected<br/>e.g. 'cell phone' -> car<br/>VERIFIED"]
    L -->|no| O["REJECTED<br/>'rejected' log, never alerts"]
    K --> P["End of video: class-level check"]
    P --> Q{"VLM confirmed this YOLO class<br/>on other objects?"}
    Q -->|yes| R2["Accepted (marked class-level)"]
    Q -->|no| S2["Stays unverified<br/>(never rejected without evidence)"]
```

Token budget per call = 200 + 85 per crop, capped at 1600, so a runaway generation stops within seconds.
The first call of a video has no crops: it records the scene as context for later questions.

---

## 4. Alert rules

Rules only run on **VLM-verified** objects, fire **once** per object, and require the object to have been
seen for at least 1 s (a 1-2 frame blip never alerts). Every threshold is in `config.yaml` and can be
overridden per run (`run.py --set rules.loitering.seconds=30`).

![Alert rules](img/4_alert_rules.png)

```mermaid
flowchart LR
    T["Verified object<br/>(seen >= 1 s)"] --> N{"First seen between<br/>22:00 and 05:00?"}
    N -->|yes| NA["night_activity (warning)<br/>'Person detected at the main gate<br/>during night hours, 00:01'"]
    T --> L{"Person in view<br/>> 10 s?"}
    L -->|yes| LA["loitering (warning)<br/>'... loitering at the main gate<br/>for over 10s, 23:59:50'"]
    T --> Z{"VLM location contains a<br/>restricted word<br/>(e.g. backyard)?"}
    Z -->|yes| ZA["restricted_zone (critical)"]
    T --> RV{"Vehicle with same colour +<br/>similar description seen<br/>>= 30 s earlier?"}
    RV -->|yes| RVA["repeat_vehicle (info)<br/>'Blue Ford F150 seen again<br/>(2nd sighting)'"]
    SC["VLM scene, every second"] --> U{"'unusual' field<br/>not 'none'?"}
    U -->|"yes, no open incident"| UA["unusual_activity (critical)<br/>'car crashing through a fence'"]
    U -->|"yes, within 10 s<br/>of the last one"| UB["same alert extended<br/>'...; then car crashing<br/>through a fence, 23:58:28'"]
```

| Rule | Applies to | Condition | Timestamp |
|---|---|---|---|
| `night_activity` | people (vehicles optional) | first seen inside the night window | first seen |
| `loitering` | people (vehicles optional) | in view longer than 10 s | first seen + 10 s |
| `restricted_zone` | people, vehicles | VLM location text contains a configured keyword | entered that location |
| `repeat_vehicle` | vehicles | colour matches + description embedding similarity >= 0.90, previous sighting ended >= 30 s earlier | first seen this time |
| `unusual_activity` | whole scene | the VLM's per-second `unusual` field is not "none" | that second |

Locations are text from the VLM ("on the main road near the traffic lights", "at the gate"), not pixel
polygons: a moving drone's view changes every second, so fixed image regions would point at different ground.

---

## 5. Q&A agent

![Q&A flow](img/5_qa_flow.png)

```mermaid
flowchart TD
    Q["Question<br/>'show me the red car'"] --> F{"Refers to the previous answer?<br/>(those, them, which of ...)"}
    F -->|yes| W["Search only within<br/>the previous results"]
    F -->|no| A["Search the whole video"]
    W --> P
    A --> P["Parse the question"]
    P --> C1["Colour + object words<br/>(both must match:<br/>'red car' != red traffic light)"]
    P --> C2["Place words<br/>(VLM location text)"]
    P --> C3["Time window<br/>'between 10 and 15 s', '23:59:45'"]
    P --> C4["Open question:<br/>embedding similarity,<br/>only hits close to the best"]
    C1 & C2 & C3 & C4 --> CTX["Compact context<br/>counts + alerts + up to 15 objects<br/>+ up to 3 scene descriptions<br/>(~100-700 tokens)"]
    CTX --> LLM["Groq gpt-oss-120b, JSON mode<br/>20 s timeout, then local llama3.1"]
    LLM --> J["answer, show_images,<br/>track_ids, scene_ids"]
    J --> V["Keep only ids that were<br/>in the context"]
    V --> E["Logs of those objects"]
    V --> I{"Wants to see it?"}
    I -->|yes| IMG["Frame cut from the video at the<br/>stored time, box drawn<br/>+ the crop the VLM judged"]
```

The prompt size does not grow with the length of the footage: at most 15 objects and 3 scenes are sent per
question, chosen by the filters above.

---

## 6. Database schema (frame-by-frame index)

![Database schema](img/6_database_schema.png)

```mermaid
erDiagram
    runs ||--o{ frames : has
    runs ||--o{ tracks : has
    runs ||--o{ events : has
    runs ||--o{ alerts : has
    runs ||--o{ scenes : has
    runs ||--o{ embeddings : has
    frames ||--o{ detections : contains
    tracks ||--o{ detections : "seen as"
    tracks ||--o{ events : logs
    tracks ||--o{ alerts : raises
    frames ||--o| scenes : "described by (1/s)"

    runs {
        int id PK
        text source
        text sim_start
        text config
        text stats
    }
    frames {
        int id PK
        int run_id FK
        real video_t
        text ts
        real lat
        real lon
        real altitude_m
        real battery_pct
        text labels
        text caption
    }
    detections {
        int id PK
        int frame_id FK
        int track_id
        text label
        real conf
        real x1_y1_x2_y2
    }
    scenes {
        int id PK
        int frame_id FK
        real video_t
        text scene
        text attributes
    }
    tracks {
        int run_id PK
        int track_id PK
        text label
        int verified
        text vlm_label
        text color
        text obj_type
        text state
        text zones
        text description
        real first_video_t
        real verified_video_t
    }
    events {
        int id PK
        int track_id FK
        real video_t
        text ts
        text type
        text message
    }
    alerts {
        int id PK
        int track_id FK
        real video_t
        text ts
        text rule
        text severity
        text message
    }
    embeddings {
        int id PK
        text kind
        int ref_id
        real video_t
        blob vector
    }
```

Every row carries both `video_t` (seconds into the clip) and `ts` (wall-clock time), so the index is
queryable by either: `run.py query --label truck`, `--from-s 10 --to-s 15`, `--start 23:59:00`.

---

## 7. Design decisions

| Decision | Why | Alternative considered |
|---|---|---|
| YOLO11s + ByteTrack for detection | Fast (20-50 ms/frame on an RTX 3080), all 80 COCO classes, stable track ids across frames, so each object is logged once | VLM-only detection: too slow for 5 frames/s |
| VLM verifies every object | COCO-trained YOLO makes false positives on drone views; the VLM removes them and adds colour, type, state, location and a description | Trusting YOLO: false logs and alerts |
| qwen3-vl:2b via Ollama (local, free) | 3.1 GB VRAM, fully on the 10 GB GPU, 3-5 s per call, best colours of the models tested | qwen2.5vl:3b spilled to CPU (~47 s/call); qwen2.5vl:7b needed 13 GB (~23 s/call on CPU) |
| One VLM call per second, not per object | Cost is bounded by video length (at most 1 call/s) instead of object count; one call also gives the scene description | One call per object: 10-50 calls per second on busy streets |
| Async VLM on a worker thread | YOLO and logging keep up with the video; verdicts arrive a few seconds later with correct timestamps | Blocking calls: about 4 s pause per second of video |
| Locations as VLM text, not zone polygons | The drone moves, so pixel regions point at different ground every second | Polygons (tried: wrong); segmentation model; GPS geofencing (needs real telemetry) |
| SQLite (WAL) + embeddings as BLOBs | Zero setup, single file, enough for one property's footage; SQL time and label queries | Postgres + pgvector at scale |
| Groq gpt-oss-120b for Q&A | Free tier, fast (~1-3 s), JSON mode; retrieval keeps the prompt small | Local llama3.1 8B (kept as fallback; weaker answers) |
| Retrieval in code, one LLM call per question | Predictable, cheap on the free tier, follow-ups handled by remembering result ids | LangChain tool-calling agent (2-5 calls per question) |

## 8. Measured performance (RTX 3080 10 GB)

| Video | Length | YOLO | VLM | Total | vs real time |
|---|---|---|---|---|---|
| Street, 4K, 30 fps | 23.6 s | 52.8 ms/frame (118 frames) | 24 calls, avg 4.6 s | 137 s | 5.8x slower |
| Night street, 768x432 | 23.0 s | 33.5 ms/frame (115 frames) | 23 calls, avg 3.1 s | 96 s | 4.2x slower |
| Gate breach, 768x432 | 54.2 s | 19.9 ms/frame (260 frames) | 31 calls, avg 3.0 s | 125 s | 2.3x slower |

The VLM is the bottleneck (about 95% of the time). Ways to scale:
- Run several VLM workers (`OLLAMA_NUM_PARALLEL`, or more GPUs); calls are independent.
- For live streams, set a small `vlm.max_in_flight` so seconds are skipped instead of queued.
- Use a vector index (FAISS / pgvector) instead of brute-force similarity once there are days of footage.
- Use a detector trained on aerial images (e.g. VisDrone) to cut false positives before the VLM.
- Use real telemetry and camera angles to place objects on a map (GPS geofences instead of text locations).
