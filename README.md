# Apex Legends Enemy Detector

> [Русская версия](README-RU.md)

Demo app for reviewing YOLOv8 enemy detection results on Apex Legends gameplay videos. Upload a video, choose a model, and get a processed output with bounding boxes and optional stretched resolution fix.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-FF4B4B?logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-required-76B900?logo=nvidia&logoColor=white)

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

## Requirements

- **GPU with CUDA support** — required for inference (NVIDIA recommended)
- **CUDA 12.8** (or adjust the torch install URL for your version)
- **Python 3.10+**
- **ffmpeg** installed and available in `PATH`

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> Change the torch index URL if you use a different CUDA version than 12.8.

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
