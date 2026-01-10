# Apex Legends Enemy Detector 🎯

AI-powered tool to analyze gameplay videos, detect enemies using YOLOv11, and fix stretched resolution issues.

## ✨ Features
- **Smart Detection:** Center-focused search area for better performance.
- **Stretched Res Support:** Corrects 4:3 stretched video to 16:9 for accurate AI inference.
- **Audio Sync Fix:** Handles variable FPS and missing frames to keep audio in sync.
- **Task Queue:** Asynchronous processing using FastAPI and Worker.

## 🛠️ Installation

### 1. **Clone the repo:**
```bash
git clone https://github.com/PSImera/apex_enemy_detect_demo.git
cd your-repo-name
```

### 2. **Create and activate virtual evbiorenment:**

```bash
python -m venv .venv
```

#### Linux / macOS (bash/zsh)
```bash
source .venv/Scripts/activate 
```

#### Windows (cmd)
```cmd
.venv\Scripts\activate
```

#### Windows (PowerShell)
```PowerShell
.venv\Scripts\Activate.ps1
```

### 3. **Install dependencies:**

```bash
python.exe -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> change torch if you use other cuda version then 12.8

#### For the project to work, system libraries are required.:

- **ffmpeg** — For work with video and audio
- **freeglut3-dev**, **libgl1-mesa-dev**, **libglu1-mesa-dev** — for OpenGL (Linux)

#### Linux installation (Debian/Ubuntu):
```bash
sudo apt update
sudo apt install ffmpeg freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev
```

#### Windows installation:

Download ffmpeg from the official website: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

Add `ffmpeg/bin` to your `PATH` environment variable.

### 4. **Run the App:**

Start Backend:
```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Start Frontend:
```bash
streamlit run frontend/app.py
```

The page `http://localhost:8501` will open automatically in your browser. Follow the on-site instructions to use the application.