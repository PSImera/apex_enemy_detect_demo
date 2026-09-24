# Apex Legends Enemy Detector

> [Русская версия](README-RU.md)

Demo app for reviewing YOLOv8 enemy detection results on Apex Legends gameplay videos. Upload a video, choose a model, and get a processed output with bounding boxes and optional stretched resolution fix.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-FF4B4B?logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-required-76B900?logo=nvidia&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-optional-76B900?logo=nvidia&logoColor=white)

> Final project for the [Deep Learning School course](https://stepik.org/250969) on Stepik.

![Detection Demo](docs/demo.gif)

---

> **Disclaimer:** This project is a purely educational tool built as a final assignment for a Deep Learning course. It processes **pre-recorded video files** and has no interaction with the game process, memory, or network. It cannot be used as a real-time cheat or aimbot. It does not violate any game's anti-cheat systems. The goal is to demonstrate object detection techniques using YOLOv8, not to gain any in-game advantage.

---

## Settings Interface

![interface](docs/interface.png)

---

## Features

- **Smart Detection:** Center-focused search area for better performance.
- **Stretched Res Support:** Corrects stretched video to 1:1 proportions for accurate AI inference.
- **Audio Sync Fix:** Handles variable FPS and missing frames to keep audio in sync.
- **Task Queue:** Asynchronous processing using FastAPI and Worker.
- **Selectable Inference Backend:** Run on PyTorch, or accelerate with TensorRT (~9x faster).

## Models

Two custom YOLOv8 models hosted on HuggingFace: [PSImera/apex_enemy_detect](https://huggingface.co/PSImera/apex_enemy_detect).

| | **YOLOv8n (nano)** | **YOLOv8m (medium)** |
|---|---|---|
| File | `apex_detect_v8n_v2.1.pt` | `apex_detect_v8m_v2.1.pt` |
| Parameters | ~3.2M | ~25.9M |
| mAP@50 | 0.930 | 0.938 |
| mAP@50-95 | 0.756 | 0.796 |
| Better for | Speed / low-end GPU | Accuracy |

Training details, metrics, and curves are available on the [model page](https://huggingface.co/PSImera/apex_enemy_detect).

---

## Inference Backends

The backend is chosen in the interface, above the upload area.

| | **PyTorch** | **TensorRT** |
|---|---|---|
| Preparation | none | builds an engine on first use |
| Pipeline | Ultralytics | fully GPU-resident |
| Precision | FP32 | FP16 (default) or FP32 |

Speed at 640x640:

| Model | PyTorch FP32 | TensorRT FP32 | TensorRT FP16 |
|---|---|---|---|
| YOLOv8n (nano) | 59 FPS | 380 FPS | **526 FPS** |
| YOLOv8m (medium) | 50 FPS | 121 FPS | **278 FPS** |

> Measured on an RTX 3070 Ti over real 1920x1080 @ 60 gameplay — 1200 frames,
> median. Your numbers will differ.

Unlike the PyTorch path, FP16 is a real win on TensorRT: 1.4x on YOLOv8n and
2.3x on YOLOv8m. Box counts matched FP32 frame for frame, so the default costs
nothing in accuracy. Why FP16 helps here but not on PyTorch is covered below.

### What the FPS counter means

The number burned into the result video is **real-time speed**: what this model
would sustain driving a live source on this GPU. Reading and writing the video
file belong to this demo, not to the model, so they are excluded. Saving the
result therefore takes longer than the counter suggests — that is expected.

That overhead is still kept small. Frames are piped into ffmpeg and encoded with
**NVENC** (`h264_nvenc`, falling back to `libx264` if the encoder is
unavailable), so the audio merge that follows copies the video stream instead of
re-encoding it — previously every frame went through a software encoder twice.
The Search Area frame is identical on every frame, so it is rasterised once and
composited from a cached layer (6.1 → 0.69 ms per frame) rather than redrawn.

### Why TensorRT is so much faster

At one frame at a time, YOLO inference is **launch-bound**: the GPU finishes
each kernel faster than Python can queue the next one. Measured on YOLOv8n, the
kernel-launch time alone (11.37 ms) accounted for the entire forward pass
(11.30 ms). That is also why FP16 buys nothing on the PyTorch path — there is no
arithmetic bottleneck to relieve — while the same GPU shows a clean 3x FP16 gain
on a large matmul.

This is exactly why FP16 pays off once TensorRT is in play. With the launch
bottleneck gone, arithmetic is what is left to speed up, so the same switch that
did nothing on PyTorch is worth 1.4–2.3x here. The heavier the model, the more
it matters: YOLOv8m gains more from FP16 than YOLOv8n does.

The TensorRT backend removes that bottleneck instead of the arithmetic:

| Change | Effect |
|---|---|
| CUDA graph replaces per-kernel launches | engine 1.84 → 0.80 ms |
| Preprocessing on GPU, frame never leaves it | 1.20 → 0.17 ms |
| Box drawing on GPU instead of OpenCV | ~6 → 0.21 ms |

Separately, BoT-SORT's global motion compensation was disabled for every
backend. Profiling showed `calcOpticalFlowPyrLK` taking 75% of the whole
`track()` call — ~44 ms per frame, eight times the inference itself. Turning it
off took tracking from ~20 to ~165 FPS with no visible loss on gameplay
footage; the override lives in `backend/tracking.py`.

Total: **~1.9 ms per frame** on YOLOv8n at FP16.

Tracking stays on the stock BoT-SORT. A hand-written GPU tracker was tried and
dropped: it drew boxes on scenery the model had not detected, and it was
*slower* — 2.03 ms against 0.49 ms — because associating on the GPU forces a
sync every frame, while BoT-SORT works on a handful of numpy rows.

### How the engine cache works

TensorRT compiles a model into an engine optimized for one **fixed** input shape,
so an engine is tied to a specific combination of settings:

```
models/engines/apex_detect_v8n_v2.1_640x640_fp16.engine
              └─────model─────────┘└─size─┘└ precision
```

- The engine is built automatically on the first run with a given combination, which
  takes several minutes (~5 min for YOLOv8n), and is then **cached in the project**
  under `models/engines/`.
- Every later run with the same settings reuses the cached engine and starts instantly.
- Changing the model, the Search Area size, or the precision produces a new cache
  entry and triggers a one-off rebuild for it. Previously built engines stay cached.
- The interface tells you before you start whether an engine is already cached or
  still has to be built.

Engine settings (precision, workspace, force rebuild) appear in the sidebar when the
TensorRT backend is selected. `models/engines/` is gitignored — engines are tied to
your specific GPU and driver, so they are always built locally and never committed.

> **Force rebuild** is only needed after a GPU or driver change.

---

## Requirements

- **NVIDIA GPU with CUDA** — required; there is no CPU fallback. Both backends
  move the model to `cuda`, and the app checks for a device on startup and
  refuses to process video without one.
- **CUDA 12.8** (or adjust the torch install URL for your version)
- **Python 3.10+**
- **ffmpeg** installed and available in `PATH`
- **TensorRT** *(optional)* — only for the accelerated backend; installed via `requirements.txt`

---

## Installation

### 1. Clone the repo with submodules:
```bash
git clone --recurse-submodules https://github.com/PSImera/apex_enemy_detect_demo.git
cd apex_enemy_detect_demo
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> Change the torch index URL if you use a different CUDA version than 12.8.

The TensorRT dependencies at the bottom of `requirements.txt` are optional — skip them
if you only intend to use the PyTorch backend. They are pinned to `tensorrt-cu12` for
CUDA 12.x; on CUDA 11 use `tensorrt-cu11` instead. TensorRT 11 is not supported, as it
removed APIs Ultralytics still relies on.

**Linux (Debian/Ubuntu) — system libraries:**
```bash
sudo apt update
sudo apt install ffmpeg freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev
```

**Windows — ffmpeg:**

Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add `ffmpeg/bin` to your `PATH`.

---

## Running the App

Start the backend:
```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Start the frontend:
```bash
streamlit run frontend/app.py
```

The page `http://localhost:8501` will open automatically in your browser.
