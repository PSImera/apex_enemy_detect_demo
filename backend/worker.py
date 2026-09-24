import cv2
import time
import threading
import torch
import subprocess
import os
from pathlib import Path
import statistics
from collections import deque

from backend.drawing import (
    SearchAreaOverlay,
    class_color,
    draw_boxes_gpu,
    draw_fps,
    draw_labels,
)
from backend.models import (
    ENGINE_BUILD_ESTIMATE_S,
    engine_is_cached,
    get_model,
    prepare_engine,
)
from backend.tracking import tracker_yaml
from backend.video import FrameWriter


def run_with_elapsed_progress(
    fn, task_id, tasks_status, estimate_s, label, poll_s=1.0
):
    result = {}

    def _run():
        try:
            result["value"] = fn()
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    started = time.perf_counter()
    thread.start()

    while thread.is_alive():
        elapsed = time.perf_counter() - started
        pct = min(int(elapsed / estimate_s * 100), 99)
        remaining = max(estimate_s - elapsed, 0)
        tasks_status[task_id].update(
            {
                "status": "building_engine",
                "progress": pct,
                "message": (
                    f"{label} — {int(elapsed)}s elapsed, "
                    f"~{int(remaining)}s left (one-off, then cached)"
                ),
            }
        )
        thread.join(timeout=poll_s)

    if "error" in result:
        raise result["error"]
    return result["value"]


def process_video_with_tracking(
    model_choice,
    input_path,
    output_path,
    task_id,
    tasks_status,
    backend="torch",
    trt_half=True,
    trt_workspace=4.0,
    trt_force_rebuild=False,
    conf=0.5,
    iou=0.4,
    imgsz_w=640,
    imgsz_h=640,
    real_game_resolution=None,
    search_area_color=(255, 0, 255),
    search_area_radius=None,
    show_video=False,
    fix_sync=False,
):
    # temp path for save sound
    abs_input = str(Path(input_path).resolve())
    abs_output = str(Path(output_path).resolve())
    temp_output = abs_output.replace(".mp4", "_nosound.mp4")

    tasks_status[task_id].update(
        {"status": "starting", "result": abs_output, "progress": 0}
    )

    cap = None
    out = None

    try:
        infer_window = deque(maxlen=30)

        if fix_sync:
            tasks_status[task_id].update(
                {"status": "repairing", "message": "Fixing audio sync... Please wait."}
            )

            fixed_input = str(input_path).replace(".mp4", "_fixed.mp4")

            repair_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                abs_input,
                "-filter_complex",
                "[0:v]fps=fps=60[v]",
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "copy",
                fixed_input,
            ]

            subprocess.run(repair_cmd, check=True)
            abs_input = fixed_input

        tasks_status[task_id].update(
            {"status": "processing", "message": "Initializing model and video..."}
        )

        cap = cv2.VideoCapture(abs_input)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        def _report(message):
            tasks_status[task_id].update(
                {"status": "building_engine", "message": message}
            )

        def _load_model():
            if backend == "tensorrt":
                return prepare_engine(
                    model_choice,
                    imgsz_w=imgsz_w,
                    imgsz_h=imgsz_h,
                    half=trt_half,
                    workspace=trt_workspace,
                    force_rebuild=trt_force_rebuild,
                    progress_cb=_report,
                )
            return get_model(
                model_choice,
                backend=backend,
                imgsz_w=imgsz_w,
                imgsz_h=imgsz_h,
                half=trt_half,
                workspace=trt_workspace,
                force_rebuild=trt_force_rebuild,
                progress_cb=_report,
            )

        if backend == "tensorrt" and not engine_is_cached(
            model_choice, imgsz_w, imgsz_h, trt_half, trt_force_rebuild
        ):
            model = run_with_elapsed_progress(
                _load_model,
                task_id,
                tasks_status,
                estimate_s=ENGINE_BUILD_ESTIMATE_S.get(model_choice, 300),
                label=f"Building TensorRT engine for '{model_choice}'",
            )
        else:
            model = _load_model()

        tasks_status[task_id].update(
            {"status": "processing", "message": "Model ready, analyzing video..."}
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stretched_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stretched_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = FrameWriter(
            temp_output, stretched_width, stretched_height, video_fps
        )

        # for stretched
        if real_game_resolution is None:
            real_game_resolution = (stretched_width, stretched_height)
        scale_x = stretched_width / real_game_resolution[0]
        scale_y = stretched_height / real_game_resolution[1]

        game_crop_w = min(imgsz_w, real_game_resolution[0])
        game_crop_h = min(imgsz_h, real_game_resolution[1])

        game_cx, game_cy = real_game_resolution[0] // 2, real_game_resolution[1] // 2

        gx0 = max(0, game_cx - game_crop_w // 2)
        gy0 = max(0, game_cy - game_crop_h // 2)
        gx1 = min(real_game_resolution[0], gx0 + game_crop_w)
        gy1 = min(real_game_resolution[1], gy0 + game_crop_h)

        if gx1 == real_game_resolution[0]:
            gx0 = max(0, real_game_resolution[0] - game_crop_w)
        if gy1 == real_game_resolution[1]:
            gy0 = max(0, real_game_resolution[1] - game_crop_h)

        x0_orig = int(gx0 * scale_x)
        y0_orig = int(gy0 * scale_y)
        x1_orig = int(gx1 * scale_x)
        y1_orig = int(gy1 * scale_y)

        final_crop_w = x1_orig - x0_orig
        final_crop_h = y1_orig - y0_orig

        if backend == "tensorrt":
            from backend.engine import FastDetector

            class_names = {0: "enemy", 1: "mate"}
            fast = FastDetector(
                model,  # prepare_engine() returned the engine path
                (x0_orig, y0_orig, x1_orig, y1_orig),
                imgsz_w=imgsz_w,
                imgsz_h=imgsz_h,
                conf=conf,
                iou=iou,
                tracker_cfg=None,
                frame_rate=max(1, video_fps),
            )
            gpu_colors = {
                idx: torch.tensor(
                    class_color(name), dtype=torch.uint8, device="cuda"
                )
                for idx, name in class_names.items()
            }
            box_scale = (final_crop_w / imgsz_w, final_crop_h / imgsz_h)
            box_offset = (x0_orig, y0_orig)
        else:
            class_names = model.names

        if search_area_radius is not None:
            overlay_radius = int(search_area_radius)
        else:
            overlay_radius = int(min(final_crop_w, final_crop_h) // 2)
        search_overlay = SearchAreaOverlay(
            (stretched_height, stretched_width),
            (x0_orig, y0_orig),
            (x1_orig, y1_orig),
            search_area_color,
            radius=overlay_radius,
            label="Search Area",
        )

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if backend == "tensorrt":
                frame_gpu = torch.from_numpy(frame).cuda()
                torch.cuda.synchronize()

                infer_start = time.perf_counter()
                rt_boxes, rt_ids, rt_cls = fast(frame_gpu)
                draw_boxes_gpu(
                    frame_gpu,
                    rt_boxes,
                    rt_ids,
                    rt_cls,
                    gpu_colors,
                    box_scale,
                    box_offset,
                )
                torch.cuda.synchronize()
                infer_window.append(time.perf_counter() - infer_start)
                frame = frame_gpu.cpu().numpy()

                # Labels are CPU work, deliberately outside the timed section.
                draw_labels(
                    frame, rt_boxes, rt_ids, rt_cls, class_names,
                    box_scale, box_offset,
                )
                boxes_obj = None
            else:
                frame_cropped = frame[y0_orig:y1_orig, x0_orig:x1_orig]
                frame_for_model = cv2.resize(frame_cropped, (imgsz_w, imgsz_h))

                infer_start = time.perf_counter()
                results = model.track(
                    frame_for_model,
                    iou=iou,
                    conf=conf,
                    persist=True,
                    imgsz=(imgsz_w, imgsz_h),
                    verbose=False,
                    tracker=tracker_yaml(),
                )
                infer_window.append(time.perf_counter() - infer_start)
                boxes_obj = results[0].boxes

            if boxes_obj is not None and boxes_obj.id is not None:
                boxes = boxes_obj.xyxy.cpu().numpy().astype(float)
                ids = boxes_obj.id.cpu().numpy().astype(int)
                clss = boxes_obj.cls.cpu().numpy().astype(int)

                scale_x_box = final_crop_w / imgsz_w
                scale_y_box = final_crop_h / imgsz_h

                for box, obj_id, cls in zip(boxes, ids, clss):
                    x0 = int(box[0] * scale_x_box) + x0_orig
                    y0 = int(box[1] * scale_y_box) + y0_orig
                    x1 = int(box[2] * scale_x_box) + x0_orig
                    y1 = int(box[3] * scale_y_box) + y0_orig

                    name = class_names[cls]
                    color = class_color(name)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
                    cv2.putText(
                        frame,
                        f"{name}_{obj_id}",
                        (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

            search_overlay.apply(frame)

            infer_ms = statistics.median(infer_window) * 1000
            draw_fps(frame, 1000.0 / infer_ms if infer_ms else 0.0)

            out.write(frame)

            if show_video:
                disp_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                cv2.imshow("frame", disp_frame)

            progress = int((frame_idx / total_frames) * 100)
            tasks_status[task_id].update(
                {
                    "progress": progress,
                    "message": f"Analyzing frame {frame_idx}/{total_frames}...",
                }
            )

        cap.release()
        cap = None
        out.release()
        out = None

        # --- Merge in FFmpeg ---
        tasks_status[task_id].update(
            {"status": "merging", "message": "Merging audio and video..."}
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output,
            "-i",
            abs_input,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-shortest",
            abs_output,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # delete temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)

        tasks_status[task_id].update(
            {"status": "done", "progress": 100, "message": "Done!"}
        )

    except Exception as e:
        if tasks_status and task_id in tasks_status:
            tasks_status[task_id].update({"status": "failed", "message": str(e)})
    finally:
        if out is not None:
            try:
                out.release()
            except Exception:
                pass
        if cap is not None:
            cap.release()
