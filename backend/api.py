from pathlib import Path
import shutil
import uuid
import time
import queue
import threading
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

from backend.worker import process_video_with_tracking

app = FastAPI(title="Apex Enemy Detector API")


TASKS_DIR = Path("tasks")
TASKS_DIR.mkdir(exist_ok=True)

task_queue = queue.Queue()
tasks_status = {}


def worker_loop():
    """main worker queue loop"""
    while True:
        task_data = task_queue.get()
        if task_data is None:
            break

        task_id = task_data["task_id"]
        try:
            process_video_with_tracking(**task_data["params"])
        except Exception as e:
            print(f"Error in worker: {e}")
            tasks_status[task_id]["status"] = "error"
        finally:
            task_queue.task_done()


threading.Thread(target=worker_loop, daemon=True).start()


def cleanup_old_tasks():
    """Cleanup old tsaks (older then 6 hours) every 1h"""
    while True:
        now = time.time()
        tasks_path = Path("tasks")
        if tasks_path.exists():
            for folder in tasks_path.iterdir():
                if folder.is_dir():
                    folder_age = now - folder.stat().st_mtime
                    if folder_age > 6 * 3600:
                        shutil.rmtree(folder)
        time.sleep(3600)


threading.Thread(target=cleanup_old_tasks, daemon=True).start()


def parse_resolution(value):
    if not value:
        return None
    w, h = value.lower().split("x")
    return int(w), int(h)


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


@app.post("/upload")
async def upload_video(
    file: UploadFile,
    model_choice: str = Form(...),
    imgsz_w: str = Form("640"),
    imgsz_h: str = Form("640"),
    real_game_resolution: str = Form(None),
    search_area_color_hex: str = Form("#FF00FF"),
    search_area_radius: str = Form(None),
    fix_sync: str = Form("false"),
):
    task_id = str(uuid.uuid4())
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    input_path = task_dir / file.filename
    output_path = task_dir / "result.mp4"

    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    tasks_status[task_id] = {
        "status": "queued",
        "result": None,
        "progress": 0,
    }

    task_params = {
        "task_id": task_id,
        "params": {
            "model_choice": model_choice,
            "input_path": input_path,
            "output_path": output_path,
            "task_id": task_id,
            "tasks_status": tasks_status,
            "imgsz_w": int(imgsz_w),
            "imgsz_h": int(imgsz_h),
            "real_game_resolution": parse_resolution(real_game_resolution),
            "search_area_color": hex_to_bgr(search_area_color_hex),
            "search_area_radius": search_area_radius,
            "fix_sync": fix_sync.lower() == "true",
        },
    }
    task_queue.put(task_params)

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks_status:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return tasks_status[task_id]


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = tasks_status.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    if task["status"] != "done":
        return JSONResponse(
            status_code=400, content={"error": f"Task is {task['status']}"}
        )

    file_path = Path(task["result"])
    if not file_path.exists():
        return JSONResponse(
            status_code=404, content={"error": "File not found on disk"}
        )

    return FileResponse(path=file_path, media_type="video/mp4", filename="result.mp4")
