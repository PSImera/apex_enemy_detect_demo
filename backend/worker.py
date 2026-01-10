import cv2
import time
import subprocess
import os
from pathlib import Path
from collections import deque

from backend.models import get_model


def draw_rounded_rect(
    img, pt1, pt2, color, thickness=1, radius=15, alpha_fill=0.05, label=None
):
    """Рисует скруглённый прямоугольник с полупрозрачной заливкой и подписью"""
    overlay = img.copy()
    x0, y0 = pt1
    x1, y1 = pt2

    # --- скруглённая заливка ---
    if radius > 0:
        # четыре угла
        cv2.ellipse(
            overlay, (x0 + radius, y0 + radius), (radius, radius), 180, 0, 90, color, -1
        )
        cv2.ellipse(
            overlay, (x1 - radius, y0 + radius), (radius, radius), 270, 0, 90, color, -1
        )
        cv2.ellipse(
            overlay, (x0 + radius, y1 - radius), (radius, radius), 90, 0, 90, color, -1
        )
        cv2.ellipse(
            overlay, (x1 - radius, y1 - radius), (radius, radius), 0, 0, 90, color, -1
        )
        # соединяем углы линиями
        cv2.rectangle(overlay, (x0 + radius, y0), (x1 - radius, y1), color, -1)
        cv2.rectangle(overlay, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    else:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

    # применяем прозрачность
    img = cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0)

    # --- рамка с теми же скруглёнными углами ---
    if radius > 0:
        cv2.ellipse(
            img,
            (x0 + radius, y0 + radius),
            (radius, radius),
            180,
            0,
            90,
            color,
            thickness,
        )
        cv2.ellipse(
            img,
            (x1 - radius, y0 + radius),
            (radius, radius),
            270,
            0,
            90,
            color,
            thickness,
        )
        cv2.ellipse(
            img,
            (x0 + radius, y1 - radius),
            (radius, radius),
            90,
            0,
            90,
            color,
            thickness,
        )
        cv2.ellipse(
            img,
            (x1 - radius, y1 - radius),
            (radius, radius),
            0,
            0,
            90,
            color,
            thickness,
        )
        cv2.line(img, (x0 + radius, y0), (x1 - radius, y0), color, thickness)
        cv2.line(img, (x0 + radius, y1), (x1 - radius, y1), color, thickness)
        cv2.line(img, (x0, y0 + radius), (x0, y1 - radius), color, thickness)
        cv2.line(img, (x1, y0 + radius), (x1, y1 - radius), color, thickness)
    else:
        cv2.rectangle(img, pt1, pt2, color, thickness)

    # --- подпись без прямоугольника, моноширный, цвет как рамки ---
    if label:
        font_scale = 0.5
        font_thick = 1
        font = cv2.FONT_HERSHEY_SIMPLEX  # моноширный стандартный
        text_size = cv2.getTextSize(label, font, font_scale, font_thick)[0]
        text_x = x0 + (x1 - x0 - text_size[0]) // 2
        text_y = y0 - 5  # чуть выше верхней границы
        cv2.putText(img, label, (text_x, text_y), font, font_scale, color, font_thick)

    return img


def process_video_with_tracking(
    model_choice,
    input_path,
    output_path,
    task_id,
    tasks_status,
    imgsz_w=640,
    imgsz_h=640,
    real_game_resolution=None,
    search_area_color=(255, 0, 255),
    search_area_radius=None,
    show_video=False,
    fix_sync=False,
):
    # Создаем временный путь для видео без звука
    abs_input = str(Path(input_path).resolve())
    abs_output = str(Path(output_path).resolve())
    temp_output = abs_output.replace(".mp4", "_nosound.mp4")

    tasks_status[task_id]["result"] = abs_output
    tasks_status[task_id]["status"] = "processing"
    tasks_status[task_id]["progress"] = 0

    try:
        # for fps
        fps_window = deque(maxlen=30)
        prev_time = time.perf_counter()

        if fix_sync:
            tasks_status[task_id]["status"] = "repairing video..."

            fixed_input = str(input_path).replace(".mp4", "_fixed.mp4")

            repair_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                abs_input,
                "-filter_complex",
                "[0:v]fps=fps=60[v]",  # Создаем поток [v] с 60 FPS
                "-map",
                "[v]",  # Явно берем видео из фильтра
                "-map",
                "0:a?",  # Берем все аудиодорожки (если есть)
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "copy",  # Копируем аудио без перекодирования
                fixed_input,
            ]

            subprocess.run(repair_cmd, check=True)
            abs_input = fixed_input

        cap = cv2.VideoCapture(abs_input)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        model = get_model(model_choice)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stretched_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stretched_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            temp_output, fourcc, video_fps, (stretched_width, stretched_height)
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

        frame_count = 0
        while True:
            now = time.perf_counter()
            fps_window.append(1.0 / (now - prev_time))
            prev_time = now
            fps_avg = sum(fps_window) / len(fps_window)

            ret, frame = cap.read()
            if not ret:
                break

            # вырезаем область
            frame_cropped = frame[y0_orig:y1_orig, x0_orig:x1_orig]
            frame_for_model = cv2.resize(frame_cropped, (imgsz_w, imgsz_h))

            # inference
            results = model.track(
                frame_for_model,
                iou=0.4,
                conf=0.5,
                persist=True,
                imgsz=(imgsz_w, imgsz_h),
                verbose=False,
                tracker="botsort.yaml",
            )

            class_names = model.names
            boxes_obj = results[0].boxes

            if boxes_obj.id is not None:
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
                    color = (255, 255, 255)
                    if name == "enemy":
                        color = (0, 0, 255)
                    elif name == "mate":
                        color = (0, 255, 0)

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

            # область поиска
            if search_area_radius is not None:
                current_radius = int(search_area_radius)
            else:
                current_radius = int(min(final_crop_w, final_crop_h) // 2)
            frame = draw_rounded_rect(
                frame,
                (x0_orig, y0_orig),
                (x1_orig, y1_orig),
                search_area_color,
                radius=current_radius,
                label="Search Area",
            )

            # fps
            cv2.putText(
                frame,
                f"FPS: {fps_avg:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                1,
            )

            out.write(frame)
            frame_count += 1

            if show_video:
                disp_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                cv2.imshow("frame", disp_frame)

            # Обновляем прогресс каждые 10 кадров (чтобы не спамить)
            if (
                tasks_status is not None
                and task_id in tasks_status
                and frame_count % 10 == 0
            ):
                tasks_status[task_id]["progress"] = int(
                    (frame_count / total_frames) * 100
                )
                tasks_status[task_id]["status"] = "processing"

        cap.release()
        out.release()

        # --- Склейка со звуком через FFmpeg ---
        tasks_status[task_id]["status"] = "finalizing"  # Статус для пользователя

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output,  # Наше обработанное видео (без звука)
            "-i",
            abs_input,  # Оригинальное видео (отсюда берем звук)
            "-c:v",
            "libx264",  # Кодируем в H.264
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "aac",  # Кодируем звук
            "-map",
            "0:v:0",  # Видео из первого файла
            "-map",
            "1:a:0?",  # Аудио из второго
            "-async",
            "1",  # Синхронизация аудио по старту
            "-vsync",
            "cfr",  # Принудительно Constant Frame Rate (синхронизирует кадры)
            "-shortest",  # Обрезать по самому короткому потоку
            abs_output,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # Удаляем временный файл без звука
        if os.path.exists(temp_output):
            os.remove(temp_output)

        tasks_status[task_id]["status"] = "done"
        tasks_status[task_id]["progress"] = 100

    except Exception as e:
        if tasks_status and task_id in tasks_status:
            tasks_status[task_id]["status"] = "error"
        print(f"Task {task_id} failed: {e}")


if __name__ == "__main__":
    process_video_with_tracking(
        model_name="accurate",
        input_path="test_videos/test2.mp4",
        output_path="test_videos/test2_output.mp4",
        task_id="test",
        tasks_status={},
        imgsz=640,
        real_game_resolution=(1440, 1080),
        show_video=True,
        search_area_color=(255, 0, 255),
    )
