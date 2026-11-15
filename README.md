# VisionGuard Safety AI 🦺

**Workplace video analytics prototype for human-reviewed safety investigations.**

> ### 💬 *"Is this workplace safe right now?"*

VisionGuard analyzes video for PPE evidence, restricted-zone entries, fall cues and calibrated worker-to-vehicle distances. It combines detector output and configured rules into a **0–100 risk score**, dashboard and timestamped report. The score helps prioritize a human review. It is a prototype with no safety certification or field-validated accuracy guarantee. See the [privacy and operating limits](PRIVACY.md).

The repository includes model-free pipeline tests, detector performance tools and [recorded limitations](#-engineering-field-notes).

> [External demonstration Space](https://huggingface.co/spaces/khatem/VisionGuard-Safety-AI), published under the `khatem` account. Availability, deployed commit, current operator and data-handling settings could not be verified on 2026-11-15. Use the local setup for footage and questions requiring controlled handling.

![Generated VisionGuard replay showing a tracked figure and a helmet alert](docs/images/synthetic_pipeline.gif)

The replay uses drawn shapes and scripted detections. Live tracking, the PPE rule and annotated output run through the same application pipeline. It illustrates the alert flow without footage of real people or a claim about detector accuracy.

[CI runs](https://github.com/Ameen-GAMAL/VisionGuard-Safety-AI/actions/workflows/ci.yml) · [Security](SECURITY.md) · [Third-party sources](THIRD_PARTY_NOTICES.md)

---

## 📑 Table of Contents

- [Why This Project Stands Out](#-why-this-project-stands-out)
- [Feature Overview](#-feature-overview-13-features-3-phases)
- [Dashboard Tour](#-dashboard-tour)
- [Results & Evidence](#-results--evidence)
- [Edge Deployment Benchmarks](#-edge-deployment-benchmarks)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Models Used](#-models-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Testing & Verification](#-testing--verification)
- [Engineering Field Notes](#-engineering-field-notes)
- [Technologies Used](#-technologies-used)
- [Attribution](#-attribution)

---

## 🌟 Why This Project Stands Out

Most portfolio CV projects stop at "YOLO draws boxes." VisionGuard is built as a **production-shaped safety analytics system**:

| Capability | The common shortcut | What VisionGuard does instead |
|---|---|---|
| Distance | Pixel distance (meaningless with perspective) | **Camera calibration + ground-plane homography** → distances in actual meters |
| Identity | Per-frame detections with no memory | ByteTrack IDs **plus cross-camera Re-ID** via appearance embeddings |
| Alerts | One alert per frame (alarm spam) | Temporal smoothing, debouncing, hysteresis, and cooldowns on **every** rule engine |
| Behavior | Single-frame if-statements | A **GRU sequence model** over 30-frame pose windows + stay-down confirmation logic |
| Deployment | "It runs on my GPU" | **Measured benchmark table**: PyTorch/ONNX/INT8, latency, FPS, size, platform trade-offs |
| Output | A list of detections | A single interpretable **0–100 risk score**, evidence screenshots, PDF reports, and a **RAG assistant** |
| Quality | No tests | **76 unit tests** (GPU-free, ~12 s) + a **live-footage verification suite** that caught real bugs |

---

## 🧩 Feature Overview (13 features, 3 phases)

### Phase 1 — Core MVP

| # | Feature | How it works |
|---|---------|--------------|
| 1 | **Multi-class detection** | YOLO11s fine-tuned on construction data detects workers, helmets, vests, their *absence* (`NO-Hardhat`, `NO-Safety Vest`), machinery, and vehicles. Detecting absence directly beats inferring it from missing overlaps |
| 2 | **Multi-object tracking** | ByteTrack assigns persistent IDs (`Worker #12`) with trajectory trails; a Kalman motion model carries tracks through short occlusions |
| 3 | **PPE compliance engine** | PPE evidence is associated to workers *anatomically* (helmets at head height, vests on the torso), smoothed over a rolling 2 s window to kill detector flicker, debounced with hysteresis + cooldown. Outputs per-video compliance % |
| 4 | **Restricted zones** | Draw polygon zones on the camera view with an interactive editor; entry and loitering alerts use each worker's **ground point** (feet), with re-entry cooldown against boundary flicker |
| 5 | **Fall detection** | YOLO11-pose keypoints → torso-angle state machine; a fall is confirmed only when a person goes horizontal **and stays down** N seconds (bending ≠ fall), with box-shape fallback when keypoints fail |
| 6 | **Safety Command Center** | Streamlit dashboard: KPI row, annotated playback, filterable alert table with evidence photos, incident timeline, danger heatmap, one-click report export |
| 7 | **PDF incident reports** | Executive summary + one evidence page per incident: screenshot, timestamps, involved IDs, confidence, and a recommended action |

### Phase 2 — Depth

| # | Feature | How it works |
|---|---------|--------------|
| 8 | **Calibrated proximity** | A ground-plane homography (calibrated once per camera by clicking 4+ known points) converts pixels → meters. Worker–vehicle distances are measured for real: *"Machinery #16 within 0.2 m of Worker #3"*. Two debounced tiers (warning ≤ 5 m, critical ≤ 2 m) + near-miss counting |
| 9 | **Safety Risk Score (0–100)** | Every event adds a configured weight that decays linearly over a rolling 60 s window — the score spikes on incidents and cools as the site behaves. Bands: 0–30 Safe · 31–60 Moderate · 61–80 High · 81–100 Critical. Live on the video HUD, charted in the dashboard, summarized in the PDF |
| 10 | **AI Safety Assistant (RAG)** | Ask in plain language: *"How many helmet violations today?"* Exact numbers come from **SQL aggregates** (never LLM guesses), relevant incidents from **semantic search** (sentence-transformers + FAISS), and **Claude** synthesizes the grounded answer — with a deterministic fallback mode that works without any API key |

### Phase 3 — Standout Engineering

| # | Feature | How it works |
|---|---------|--------------|
| 11 | **Cross-camera Re-ID** | Each worker's appearance is embedded (CLIP image encoder); an identity gallery matches tracks against running-mean centroids by cosine similarity, so one worker keeps one **global ID** across cameras and occlusions. Demo script produces visual proof montages |
| 12 | **Edge optimization + benchmarks** | ONNX export + INT8 dynamic quantization, with measured latency/FPS/size across five configurations on real frames ([table below](#-edge-deployment-benchmarks)) |
| 13 | **Temporal behavior model** | A ~50K-parameter GRU classifies 30-frame pose sequences (walk / bend / **fall**) using translation- and scale-invariant body-centric features. Trained on procedurally generated skeleton sequences in ~30 s — a transparent stand-in whose featurization and training loop transfer unchanged to real labeled clips |

---

## 🖥️ Dashboard Tour

A walkthrough of the Safety Command Center, captured live from the running app (analyzing `construction_steelwork.mp4` with a user-drawn restricted zone).

**Annotated video** — live playback with worker IDs, PPE evidence, the restricted zone, an alert banner, and the risk-score HUD. The KPI row on top condenses the whole run: peak risk, compliance, event count, workers seen, most dangerous zone, top violation:

![Dashboard — annotated video](docs/images/dashboard_annotated_video.png)

**Alerts** — every incident with severity, video timestamp, type, description, and detector confidence; filterable by severity:

![Dashboard — alerts](docs/images/dashboard_alerts.png)

**Evidence viewer** — click any incident to see its screenshot proof (here, from the person-down test video — note the fall event marked critical in the table):

![Dashboard — alert evidence](docs/images/dashboard_alert_evidence.png)

**Timeline** — the 0–100 risk score over the video:

![Dashboard — timeline](docs/images/dashboard_timeline.png)

**Event histogram** — safety events over video time, colored by type:

![Dashboard — event histogram](docs/images/dashboard_event_histogram.png)

**Heatmap** — where workers actually spent their time, overlaid on the scene:

![Dashboard — heatmap](docs/images/dashboard_heatmap.png)

**Report** — one click generates the PDF incident report (summary + evidence pages) with a download button:

![Dashboard — report](docs/images/dashboard_report.png)

**Assistant** — plain-language questions answered from the incident database (exact counts from SQL, incidents from semantic retrieval):

![Dashboard — assistant](docs/images/dashboard_assistant.png)

---

## 📸 Results & Evidence

**The full alarm chain on one clip** — zone intrusion → no-vest → no-helmet → confirmed fall (compliance 0%, risk score peaks 70/100):

![Fall detected](docs/images/fall_detected.jpg)

**Proximity detection** — distance lines in real meters, critical pairs in red:

![Proximity alert](docs/images/proximity_alert.jpg)

**Re-ID identity match** — the same worker re-identified across two cameras (one global ID):

![Re-ID match](docs/images/reid_identity_match.jpg)

**Worker-position danger heatmap** — accumulated over the full video:

![Danger heatmap](docs/images/danger_heatmap.png)

**Historical observations** (reported on RTX 4050 Laptop 6 GB, 1280×720 processing). These development observations lack retained machine-readable results or independent accuracy labels.

| Metric | Value |
|--------|-------|
| End-to-end pipeline speed | **25–29 FPS** (detection + pose + tracking + all rule engines + annotation + encode) |
| Compliant-crew false alarms | **0** on the steelwork video (100% compliance, correctly) |
| Automated checks | CPU tests and a generated-video integration test ([details below](#-testing--verification)) |
| Live observations | Four development videos and their reported outcomes ([details below](#-testing--verification)) |

---

## ⚡ Edge Deployment Benchmarks

`scripts/benchmark.py` measures **single-image detector latency at 640×640**. PyTorch timings include the Ultralytics prediction pipeline. Direct ONNX Runtime timings include preprocessing and the forward pass. The manifest identifies the timed stages per variant and separates these timings from end-to-end safety pipeline speed.

Measured local CPU sample on Windows 11 with an Intel i7-11800H. Each variant used 24 sampled steelwork frames across three repetitions. The files record the base commit, changed working tree and tested script hash. These timings do not estimate event accuracy.

| Configuration | Size | p50 | p95 |
| --- | ---: | ---: | ---: |
| PyTorch FP32 CPU | 19.15 MB | 73.4 ms | 80.1 ms |
| ONNX Runtime FP32 CPU | 37.94 MB | 77.6 ms | 80.0 ms |
| ONNX Runtime INT8 CPU | 9.87 MB | 118.4 ms | 124.5 ms |

[Measured manifest](benchmarks/cpu-steelwork/manifest.json) · [Raw latency samples](benchmarks/cpu-steelwork/latency.csv) · [Latency chart](benchmarks/cpu-steelwork/latency.png)

Historical detector measurements from project development are retained below for context. Raw samples for this table were unavailable for verification.

| Configuration | Size (MB) | Mean latency | p95 | Throughput |
|---|---|---|---|---|
| PyTorch FP32 (GPU) | 19.2 | 14.5 ms | 16.0 ms | **68.9 FPS** |
| PyTorch FP16 (GPU) | 19.2 | 12.4 ms | 12.8 ms | **80.8 FPS** |
| PyTorch FP32 (CPU) | 19.2 | 63.8 ms | 69.0 ms | **15.7 FPS** |
| ONNX Runtime FP32 (CPU) | 37.9 | 67.5 ms | 71.1 ms | **14.8 FPS** |
| ONNX Runtime INT8 (CPU) | **9.9** | 115.4 ms | 125.2 ms | **8.7 FPS** |

The historical INT8 export was about one-quarter of the FP32 file size and slower on that desktop test. Edge-device speed and detection accuracy need separate measurements on the target hardware.

---

## 🏛️ System Architecture

```
Video / Camera
      │  frame by frame
      ▼
┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐
│ YOLO11        │──▶│ ByteTrack    │──▶│ Safety rule engines        │
│ detection     │   │ tracking     │   │  · PPE compliance          │
│ (+ YOLO-pose) │   │ (worker IDs) │   │  · Zone entry / dwell      │
└──────────────┘   └──────┬───────┘   │  · Fall state machine      │
                          │           │  · Proximity (meters, via  │
                   ┌──────▼───────┐   │    ground-plane homography)│
                   │ Re-ID gallery │   └─────────────┬──────────────┘
                   │ (global IDs)  │                 ▼
                   └──────────────┘   ┌─────────────────────────────┐
                                      │ Risk score (0–100, decaying)│
                                      └─────────────┬───────────────┘
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │ Event store (SQLite) + screenshots  │
                              └───────┬───────────┬─────────────────┘
                                      ▼           ▼            ▼
                         ┌──────────────┐  ┌───────────┐  ┌──────────────┐
                         │ Streamlit    │  │ PDF       │  │ RAG safety   │
                         │ dashboard    │  │ reports   │  │ assistant    │
                         └──────────────┘  └───────────┘  └──────────────┘
```

**Design principles:**

- **Model-agnostic core.** The detector translates raw model labels into a canonical taxonomy (`ObjectClass.WORKER`, `ObjectClass.NO_HELMET`, …); tracking, safety rules, storage, and reporting never see YOLO internals, so models can be swapped freely.
- **Config-driven.** Every threshold, path, and model choice lives in [`configs/config.yaml`](configs/config.yaml), parsed into frozen typed dataclasses that fail loudly on bad config — no magic numbers in code.
- **Testable safety logic.** All rule engines operate on plain dataclasses — the test suite runs in seconds with no GPU and no model weights.
- **Logging, not prints.** Rotating file + console logging throughout; every alert is traceable.

---

## 📁 Project Structure

```
VisionGuard-Safety-AI/
├── configs/
│   ├── config.yaml            # All paths, thresholds, model settings
│   ├── zones.json             # Restricted zones (drawn with the zone editor)
│   └── calibration.json       # Camera ground-plane calibration (demo values)
├── src/visionguard/
│   ├── detection/             # YOLO wrapper, pose estimation, core types
│   ├── tracking/              # ByteTrack wrapper (IDs, trajectories)
│   ├── safety/                # PPE / zones / falls / proximity / risk engines
│   ├── spatial/               # Ground-plane homography (pixels → meters)
│   ├── reid/                  # Cross-camera Re-ID (appearance gallery)
│   ├── temporal/              # Pose-sequence behavior model (GRU)
│   ├── assistant/             # RAG safety assistant (FAISS + Claude)
│   ├── storage/               # SQLite event store
│   ├── dashboard/             # Streamlit Safety Command Center
│   ├── reporting/             # PDF incident report builder
│   ├── utils/                 # Config, logging, video I/O, drawing
│   └── pipeline.py            # End-to-end orchestrator
├── scripts/
│   ├── download_assets.py     # Fetch model weights + all sample/test videos
│   ├── run_pipeline.py        # CLI analysis runner
│   ├── define_zones.py        # Interactive polygon zone editor
│   ├── calibrate_camera.py    # Ground-plane calibration tool
│   ├── reid_demo.py           # Cross-camera identity matching demo
│   ├── benchmark.py           # ONNX export + latency/FPS benchmark suite
│   └── train_temporal.py      # Train the pose-sequence behavior model
├── tests/                     # 76 pytest tests (pure-logic, GPU-free)
├── docs/images/               # README evidence images
├── data/                      # Videos & datasets (git-ignored)
├── models/                    # Model weights (git-ignored)
└── outputs/                   # Annotated videos, screenshots, heatmaps,
                               # reports, logs, events.db (git-ignored)
```

---

## 🧠 Models Used

| Model | Role | Params/Size | Source | License |
|-------|------|-------------|--------|---------|
| **YOLO11s (PPE fine-tune)** | Workers, PPE presence *and absence*, machinery, vehicles (8 mapped classes) | 19.2 MB | [yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) | AGPL-3.0 |
| **YOLO11n-pose** | 17 COCO keypoints per person (fall detection) | 6 MB | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| **CLIP ViT-B/32** | Appearance embeddings for Re-ID | ~350 MB | [sentence-transformers](https://huggingface.co/sentence-transformers/clip-ViT-B-32) | Apache-2.0 |
| **all-MiniLM-L6-v2** | Text embeddings for RAG retrieval | ~90 MB | [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 |
| **PoseSequence GRU** | Temporal walk/bend/fall classifier | ~50K params | Trained in-repo (`scripts/train_temporal.py`) | — |
| **Claude (`claude-opus-4-8`)** | RAG answer synthesis (optional — graceful fallback without a key) | API | [Anthropic](https://www.anthropic.com) | — |

---

## 📦 Installation

Use Python 3.11 or 3.12 on Windows or Linux x86-64.

```bash
git clone https://github.com/Ameen-GAMAL/VisionGuard-Safety-AI.git
cd VisionGuard-Safety-AI

python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)

python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .

python scripts/download_assets.py # model weights and the default clip
```

Choose `requirements-gpu.txt` in a separate environment for the CUDA 12.1 runtime. Install `requirements-assistant.txt` to enable semantic retrieval, Re-ID and the optional Anthropic client. `requirements-dev.txt` installs the CPU test stack without Streamlit, FAISS, Anthropic, detector weights or video assets. `requirements.txt` keeps the original full GPU installation available.

Use `python scripts/download_assets.py --group all` for the additional field clips. Each file is checked against [the size and SHA-256 manifest](configs/assets.lock.json). Use `--verify-only` to check a populated cache offline. Follow the [asset terms](THIRD_PARTY_NOTICES.md) before use or redistribution.

---

## 🚀 Usage

### Safety Command Center (the main experience)

```bash
python -m streamlit run src/visionguard/dashboard/app.py
```

Pick a video in the sidebar → **Run analysis** → explore KPIs, annotated playback, alerts with evidence photos, the risk-score timeline, the danger heatmap, one-click PDF export, and the Assistant chat tab.

### Reproduce the synthetic walkthrough

Install `requirements-core.txt` and the editable package. Run `python scripts/synthetic_demo.py` to generate the replay, an isolated event database, annotated video and a sample PDF under `outputs/synthetic-demo/`. Pass `--output-dir` for a new destination. The run needs no network, real footage or detector weights.

### Analyze a video from the CLI

```bash
python scripts/run_pipeline.py                          # config default video
python scripts/run_pipeline.py --source my_site.mp4    # any video file
python scripts/run_pipeline.py --source 0              # live webcam
```

Produces: annotated video + danger heatmap in `outputs/`, events + evidence screenshots in the SQLite store, and a console summary (FPS, compliance, event counts).

### Draw restricted zones (once per camera)

```bash
python scripts/define_zones.py --video data/videos/construction_steelwork.mp4
```

Left-click polygon corners → **N** to name the zone → **S** to save. Zones live in `configs/zones.json` (normalized coordinates — resolution-independent).

### Calibrate real-world distances (once per camera)

```bash
python scripts/calibrate_camera.py
```

Click 4+ ground points whose real-world positions you know (e.g. corners of a slab you measured); enter each position in meters. Enables metric proximity detection.

### Cross-camera Re-ID demo

```bash
python scripts/reid_demo.py --videos camA.mp4 camB.mp4   # two cameras
python scripts/reid_demo.py                              # one video split in half
```

Prints the identity-matching table and saves montage proof sheets to `outputs/reid/`.

### Run the benchmarks on your machine

```bash
python scripts/benchmark.py --frames 100 --warmup 10 --repetitions 3
```

Each run writes a machine-readable manifest, per-frame latency CSV and percentile chart to `outputs/benchmarks/`. The manifest records input and output checksums, sampled frames, timing scope, Python and backend versions, device details, Git commit and whether the working tree had changes. Use `--cpu-only` or `--pytorch-only` to restrict the run. Review files before sharing evidence involving private assets.

The manual GitHub workflow is intended for a trusted self-hosted runner labelled `gpu`. It runs the pinned public sample and can attest the exported measurements. See [field evaluation protocol](docs/FIELD_VALIDATION.md) for validating workplace detection and end-to-end timing separately.

### Retrain the temporal behavior model

```bash
python scripts/train_temporal.py    # ~30 s; prints per-class held-out accuracy
```

### Enable full AI assistant answers (optional)

The assistant answers locally by default. To allow optional cloud synthesis, first review [the outbound data and privacy settings](PRIVACY.md), install `requirements-assistant.txt`, set `assistant.cloud_enabled: true` and set `ANTHROPIC_API_KEY` in the process environment. In the Assistant tab, check the consent box for the run before submitting a question. Cloud failures fall back to a local answer.

---

## ⚙️ Configuration

Everything tunable lives in [`configs/config.yaml`](configs/config.yaml) — no magic numbers in code. Highlights:

| Section | What you control |
|---|---|
| `video` | Source, frame skip, processing resolution |
| `detection` | Model path, confidence/IoU thresholds, device (auto/cpu/cuda) |
| `tracking` | Track activation, occlusion buffer, matching strictness |
| `ppe` | Required equipment, smoothing window, violation ratio, cooldowns |
| `zones` | Zone file, dwell threshold, re-entry cooldown |
| `fall` | Torso-angle threshold, stay-down confirmation time, cooldowns |
| `proximity` | Calibration file, warning/critical distances (meters), cooldowns |
| `risk_score` | Rolling window, per-event-type weights |
| `reid` | Embedding model, similarity threshold (site-dependent — see field notes) |
| `assistant` | Cloud opt-in, Claude model, local embedding model and retrieval top-k |
| `events` | Local event database and optional screenshots for new runs |

The loader parses YAML into **frozen typed dataclasses** — a typo or missing key fails loudly at startup, not silently mid-video.

---

## ✅ Testing & Verification

### CPU tests and generated-video smoke test

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

| Area | What's covered |
|---|---|
| Geometry | IoU, ground points, containment, homography math (incl. perspective un-warping: a 280 px far edge and a 1080 px near edge both measure 10 m) |
| PPE engine | Debounce windows, anatomical association, flicker immunity, overlapping workers, fail-fast config |
| Zones | Entry/dwell/re-entry, boundary-flicker debouncing (regression test from real footage), per-class applicability |
| Falls | Torso-angle math, stay-down confirmation, stumble rejection, duplicate suppression, keypoint-free fallback |
| Proximity | Risk tiers, escalation, cooldowns, retreat/re-approach, multi-worker independence |
| Risk score | Decay, clamping, weight escalation, peak memory, band labels |
| Re-ID gallery | Cross-camera identity sharing, sticky bindings, threshold boundaries, centroid learning |
| Temporal model | Feature invariances (translation/scale), synthetic-data sanity, save/load, *does-it-actually-learn* check |
| Storage & reports | Round-trips, ordering, isolation, PDF generation, missing-evidence resilience |
| Assistant | Fact extraction, retrieval-failure resilience, and default-off cloud behavior even when a key is present |
| Generated video | Deterministic model fixtures feed real tracking, PPE rules, SQLite, annotated output and PDF creation without external files. Install `requirements-dashboard.txt` to verify the same run in Streamlit. |

### Live-footage verification suite

These four clips were used in development. Their observed results illustrate individual scenarios. The project has no labeled field dataset or independent accuracy study for these examples. The field clips can be acquired via `scripts/download_assets.py --group all`.

| Test video | Scenario exercised | Historical observation |
|---|---|---|
| `construction_steelwork.mp4` | No false alarms on a *compliant* crew; proximity vs. crane machinery | 100% compliance, 0 false PPE alerts, 2 near-misses flagged |
| `test_person_down.mp4` | **The full alarm chain in one clip** | Zone intrusion → no-vest → no-helmet → **FALL (critical)**; compliance 0%; risk peaks 70/100 |
| `test_ppe_digging.mp4` | Zone alerts + boundary-flicker debouncing | 2 clean intrusion alerts (this video *found* the alert-spam bug) |
| `test_forklift.mp4` | Vehicle/machinery detection + proximity mechanics indoors | Forklifts detected, proximity events raised |

---

## 🔍 Engineering Field Notes

Honest findings from building and verifying on real footage — the kind of detail that separates a demo from a system:

- **The verification suite caught a real bug.** A worker hovering at a zone boundary fired 7 intrusion alerts in 4 seconds (one per flicker re-entry). Fixed with a config-driven re-entry cooldown + a regression test. Alarm fatigue is the #1 killer of real safety systems.
- **Model docs lied about label spelling.** The PPE model's documentation lists `Machinery`/`Vehicle`; the shipped weights embed `machinery`/`vehicle`. Vehicle detection was silently dead until a log line (`ignored: [...]`) exposed it — the class map now covers both spellings, verified against the loaded model.
- **Uniformed crews are the canonical Re-ID hard case.** Identical vests + helmets compress everyone into a tight embedding cluster (measured: different-person similarity reaches 0.93). The matching threshold is site-dependent config, tuned here from a measured similarity distribution.
- **INT8 performance depends on the execution provider.** The historical desktop run recorded a smaller file and higher latency. ARM timing remains unmeasured here.
- **Zones and calibration are per-camera config.** A zone or homography drawn on one camera view is meaningless on another — which is exactly why the interactive `define_zones.py` / `calibrate_camera.py` tools exist.
- **cu126 PyTorch + a 537-series NVIDIA driver = `CUDA error: device busy`.** Enumeration succeeds, allocation fails. The cu121 build matches the driver natively; pinned + documented in `requirements.txt`.

---

## 🛠️ Technologies Used

- **Python 3.11** — the entire codebase, typed and documented
- **PyTorch 2.5 (CUDA 12.1)** — deep-learning runtime for all models
- **Ultralytics YOLO11** — object detection (PPE fine-tune) and pose estimation
- **supervision (ByteTrack)** — multi-object tracking with occlusion tolerance
- **OpenCV** — video I/O, homography (`findHomography` / `perspectiveTransform`), annotation, interactive tools
- **sentence-transformers** — CLIP ViT-B/32 appearance embeddings (Re-ID) + MiniLM text embeddings (RAG)
- **FAISS** — vector similarity search over the incident history
- **Anthropic Claude API** — grounded natural-language answer synthesis for the assistant
- **Streamlit + Plotly** — the Safety Command Center dashboard and charts
- **SQLite** — zero-config event store (schema designed for a later PostgreSQL swap)
- **ReportLab** — PDF incident report generation
- **ONNX / ONNX Runtime** — model export, INT8 dynamic quantization, edge benchmarking
- **pytest** — component and synthetic pipeline checks; **Streamlit AppTest** — dashboard smoke test
- **Typed frozen-dataclass config, rotating-file logging, pinned requirements** — engineering hygiene throughout

---

## 🙏 Attribution

| Asset | Source | License |
|-------|--------|---------|
| PPE detection model | [yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) | AGPL-3.0 |
| YOLO11 / YOLO11-pose | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| Steelwork video | [Pexels — manas patra](https://www.pexels.com/video/workers-on-construction-11798561/) | Pexels License |
| Test videos (person down / digging / forklift) | Pexels — Ron Lach, K., kelly | Pexels License |

For upstream terms and source attribution, see [third-party notices](THIRD_PARTY_NOTICES.md) and [contributing](CONTRIBUTING.md).
