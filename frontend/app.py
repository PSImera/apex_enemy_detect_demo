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


# ---------- SIDEBAR SETTINGS ----------
st.sidebar.header("Settings")

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

st.sidebar.markdown("---")
fix_sync = st.sidebar.checkbox(
    "Fix Audio Sync",
    value=False,
    help="Enable this if your audio goes out of sync with video. "
    "Adds extra processing time at the start.",
)

# Upload and show video
uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file:
    st.subheader("Original Video")
    st.video(uploaded_file)

# Start detection button
if st.button("Detect", disabled=uploaded_file is None):
    st.session_state.processed_video = None
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {
        "model_choice": model_choice,
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

                        if state == "repairing":
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
    st.download_button(
        label="⬇️ Download result",
        data=st.session_state.processed_video,
        file_name="processed_apex.mp4",
        mime="video/mp4",
    )
