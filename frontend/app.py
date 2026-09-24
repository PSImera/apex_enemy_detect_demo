import streamlit as st
import requests
import time

BACKEND_URL = "http://localhost:8000"

# Init state
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "task_id" not in st.session_state:
    st.session_state.task_id = None

st.set_page_config(
    page_title="Apex enemy detector",
    layout="centered",
)

st.title("Apex enemy detector [Demo]")
st.write("Upload a gameplay video and choose detection model")

try:
    gpu_status = requests.get(f"{BACKEND_URL}/cuda_status", timeout=5).json()
except Exception:
    gpu_status = {"available": True, "message": "", "device": None}
    st.caption("Could not reach backend to check for a GPU.")

gpu_ready = bool(gpu_status.get("available"))
if not gpu_ready:
    st.error(
        f"🚫 {gpu_status.get('message', 'No CUDA device found.')} "
        "An NVIDIA GPU with CUDA is required. You can still browse the settings, "
        "but processing is disabled."
    )

# ---------- MODEL CHOICE ----------
MODELS = {
    "accurate": "Accurate (YOLOv8m) — better quality",
    "fast": "Fast (YOLOv8n) — high FPS",
}

model_choice = st.selectbox(
    "Choose model",
    options=list(MODELS.keys()),
    format_func=lambda x: MODELS[x],
)

# ---------- INFERENCE BACKEND ----------
BACKENDS = {
    "torch": "PyTorch",
    "tensorrt": "TensorRT",
}

backend = st.radio(
    "Inference backend",
    options=list(BACKENDS.keys()),
    format_func=lambda x: BACKENDS[x],
    horizontal=True,
    help="""
**PyTorch** runs the `.pt` checkpoint directly. No preparation, ~36 FPS.

**TensorRT** runs a compiled engine with the frame kept on the GPU, ~290 FPS.
The engine is built on first use (minutes) and cached in `models/engines/`;
changing the model, Search Area size or precision triggers a rebuild. Engines
are tied to your GPU and driver.

The FPS counter reports real-time speed and excludes reading and writing the
video, so saving the file takes longer than it suggests.
""",
)


# ---------- SIDEBAR SETTINGS ----------
st.sidebar.header("Settings")

# --- TensorRT Engine ---
trt_half = True
trt_workspace = 4.0
trt_force_rebuild = False

if backend == "tensorrt":
    st.sidebar.subheader(
        "TensorRT Engine",
        help="""
**Engine build settings.**

The engine is compiled once per *model + search area + precision* and cached in
`models/engines/`. Reusing the same settings reuses the cached engine; changing
any of them triggers a one-off rebuild for that new combination.
""",
    )

    trt_precision = st.sidebar.radio(
        "Precision",
        options=["fp16", "fp32"],
        horizontal=True,
        help="**FP16** is roughly twice as fast and is the recommended default. "
        "**FP32** is slightly more accurate but noticeably slower.",
    )
    trt_half = trt_precision == "fp16"

    trt_workspace = st.sidebar.slider(
        "Workspace (GiB)",
        min_value=1.0,
        max_value=8.0,
        value=4.0,
        step=1.0,
        help="Memory TensorRT may use while optimizing the engine. "
        "Larger values can find faster kernels but need more free VRAM. "
        "Only affects build time, not inference.",
    )

    trt_force_rebuild = st.sidebar.checkbox(
        "Force rebuild",
        value=False,
        help="Rebuild the engine even if a cached one exists. "
        "Use after a GPU or driver change.",
    )

    st.sidebar.markdown("---")

# --- Model Search Area ---
st.sidebar.subheader(
    "Search Area Settings",
    help="""
* **Resolution:** Model work area. YOLOv8 is optimized for 640x640.
* **Constraint:** Must be a multiple of 32 (architecture requirement).
* **Trade-off:** Higher values (800, 960) cover more screen but increase VRAM and latency.
* **Accuracy:** Values much larger than 640px may reduce detection accuracy.
* **Bounds:** Area is automatically limited by the video dimensions.
""",
)

szw_col, szh_col, rad_col, clr_col = st.sidebar.columns(4)

with szw_col:
    imgsz_w = st.number_input(
        "Width",
        min_value=64,
        value=640,
        placeholder="640",
    )
with szh_col:
    imgsz_h = st.number_input(
        "Height",
        min_value=64,
        value=640,
        placeholder="640",
    )
with rad_col:
    radius = st.number_input(
        "Rounding",
        min_value=0,
        value=None,
        placeholder="e.g. 15",
        help="""
**Visual style of the search area overlay.**

* **Default:** If empty, it creates a circle or oval based on dimensions.
* **Square:** Set to **0** for sharp corners.
* **Note:** This setting is purely cosmetic and does not affect model detection accuracy.
""",
    )
with clr_col:
    search_area_color_hex = st.color_picker(
        "Color",
        value="#FF00FF",
        help="""
Color of area **boarder** and **fill**
""",
    )
st.sidebar.markdown("---")

# --- Game Resolution ---
st.sidebar.subheader(
    "Game Resolution (Optional)",
    help="""
**Required for stretched resolutions (e.g., 4:3 stretched to 16:9).**

* **Why?** If you play at 1440x1080 but record at 1920x1080, the image is distorted.
* **Accuracy:** Setting this resolution allows the model to "reverse" the stretch before detection.
* **Result:** Objects return to their natural proportions, significantly improving detection accuracy.
""",
)

w_col, h_col = st.sidebar.columns(2)

with w_col:
    game_width = st.number_input(
        "Width",
        min_value=100,
        value=None,
        placeholder="e.g. 1440",
    )
with h_col:
    game_height = st.number_input(
        "Height",
        min_value=100,
        value=None,
        placeholder="e.g. 1080",
    )
real_game_resolution = (
    f"{int(game_width)}x{int(game_height)}" if game_width and game_height else None
)

st.sidebar.subheader(
    "Detection",
    help="""
Thresholds applied to every backend.

* **Confidence:** minimum score for a detection to count. Lower values catch
  more distant enemies but start picking up trees and rocks.
* **NMS IoU:** how much two boxes may overlap before the weaker one is dropped.
""",
)

conf_col, iou_col = st.sidebar.columns(2)
with conf_col:
    conf = st.number_input(
        "Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )
with iou_col:
    iou = st.number_input(
        "NMS IoU",
        min_value=0.1,
        max_value=0.9,
        value=0.40,
        step=0.05,
    )


st.sidebar.markdown("---")
fix_sync = st.sidebar.checkbox(
    "Fix Audio Sync",
    value=False,
    help="Enable this if your audio goes out of sync with video. "
    "Adds extra processing time at the start.",
)

# Warn if the chosen TensorRT combo still needs a build
if backend == "tensorrt":
    try:
        engine_info = requests.get(
            f"{BACKEND_URL}/engine_status",
            params={
                "model_choice": model_choice,
                "imgsz_w": int(imgsz_w),
                "imgsz_h": int(imgsz_h),
                "trt_half": trt_half,
            },
            timeout=5,
        ).json()
        if trt_force_rebuild:
            st.warning(
                "♻️ Force rebuild is on — the engine will be "
                "rebuilt even though a cached one may exist."
            )
        elif engine_info.get("exists"):
            st.success(
                f"⚡ Cached TensorRT engine ready "
                f"({model_choice}, {imgsz_w}x{imgsz_h}, {trt_precision}) — "
                "inference will reuse it."
            )
        else:
            st.info(
                f"⚙️ No cached engine for {model_choice} at "
                f"{imgsz_w}x{imgsz_h} ({trt_precision}) yet. It will be built "
                "on the first run (several minutes), then reused."
            )
    except Exception:
        st.caption("Could not reach backend to check engine cache.")

# Upload and show video
uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file:
    st.subheader("Original Video")
    st.video(uploaded_file)

# Start detection button
if st.button("Detect", disabled=uploaded_file is None or not gpu_ready):
    st.session_state.processed_video = None
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {
        "model_choice": model_choice,
        "backend": backend,
        "trt_half": trt_half,
        "trt_workspace": trt_workspace,
        "trt_force_rebuild": trt_force_rebuild,
        "conf": conf,
        "iou": iou,
        "imgsz_w": imgsz_w,
        "imgsz_h": imgsz_h,
        "real_game_resolution": real_game_resolution,
        "search_area_color_hex": search_area_color_hex,
        "search_area_radius": radius,
        "fix_sync": fix_sync,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload", files=files, data=data, timeout=60
        )

        if response.status_code == 200:
            task_id = response.json()["task_id"]
            st.session_state.task_id = task_id

            progress_container = st.container()
            with progress_container:
                status_text = st.empty()
                progress_bar = st.progress(0)

                finished = False
                while not finished:
                    try:
                        status_res = requests.get(
                            f"{BACKEND_URL}/status/{task_id}"
                        ).json()
                        state = status_res.get("status")
                        progress = status_res.get("progress", 0)
                        message = status_res.get("message", "Waiting...")

                        if state == "building_engine":
                            status_text.warning(f"⚙️ {message}")
                            progress_bar.progress(progress)
                        elif state == "repairing":
                            status_text.warning(f"🛠️ **Repairing:** {message}")
                            progress_bar.progress(progress)
                        elif state == "processing":
                            status_text.info(f"🔍 **Processing:** {message}")
                            progress_bar.progress(progress)
                        elif state == "merging":
                            status_text.info(f"🎵 **Merging Audio:** {message}")
                            progress_bar.progress(99)
                        elif state == "done":
                            status_text.success("✅ **Success!** Video is ready.")
                            progress_bar.progress(100)
                            finished = True
                        elif state == "failed":
                            st.error(f"❌ Error: {status_res.get('message')}")
                            finished = True

                        if not finished:
                            time.sleep(1)
                    except Exception as poll_error:
                        st.warning(f"Connection lost, retrying... ({poll_error})")
                        time.sleep(2)

        else:
            st.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

# Progress bar status
if st.session_state.task_id and st.session_state.processed_video is None:
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    stop_polling = False
    while not stop_polling:
        try:
            r = requests.get(f"{BACKEND_URL}/status/{st.session_state.task_id}").json()
            status = r.get("status")
            progress = r.get("progress", 0)

            progress_bar.progress(progress)
            status_placeholder.info(f"Status: {status} ({progress}%)")

            if status == "done":
                res = requests.get(f"{BACKEND_URL}/result/{st.session_state.task_id}")
                if res.status_code == 200:
                    st.session_state.processed_video = res.content
                    st.rerun()
                else:
                    st.error(f"Error getting result: {res.text}")
                stop_polling = True
            elif status == "error":
                st.error("Backend processing failed")
                stop_polling = True
        except Exception as e:
            st.error(f"Connection lost: {e}")
            break
        time.sleep(1)

# Show result
if st.session_state.processed_video:
    st.markdown("---")
    st.subheader("Result Video")
    st.video(st.session_state.processed_video)
    st.caption(
        "ℹ️ **The FPS counter shows real-time speed** — detection and "
        "tracking only. Reading and writing the video file belong to this demo, "
        "not to the model, so they are excluded."
    )
    st.download_button(
        label="⬇️ Download result",
        data=st.session_state.processed_video,
        file_name="processed_apex.mp4",
        mime="video/mp4",
    )
