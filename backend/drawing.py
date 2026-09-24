"""Overlay drawing: detection boxes, the search-area frame and the FPS counter."""

from __future__ import annotations

import cv2
import numpy as np

BOX_COLORS_BGR = {"enemy": (0, 0, 255), "mate": (0, 255, 0)}
FPS_COLOR_BGR = (0, 255, 0)


def class_color(name: str):
    """BGR colour for a class name, white for anything unexpected."""
    return BOX_COLORS_BGR.get(name, (255, 255, 255))


def draw_fps(frame, fps: float):
    """Burn the real-time FPS counter into the top-left corner."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        FPS_COLOR_BGR,
        2,
    )
    return frame


def draw_labels(frame, boxes, ids, cls, class_names, scale, offset):
    """Write "<class>_<id>" above each box."""
    if ids is None or len(boxes) == 0:
        return frame

    sx, sy = scale
    ox, oy = offset
    for (x0, y0, _, _), obj_id, c in zip(
        np.asarray(boxes).tolist(), np.asarray(ids).tolist(), np.asarray(cls).tolist()
    ):
        name = class_names.get(int(c), str(c))
        cv2.putText(
            frame,
            f"{name}_{int(obj_id)}",
            (int(x0 * sx) + ox, int(y0 * sy) + oy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            class_color(name),
            1,
        )
    return frame


def draw_rounded_rect(
    img, pt1, pt2, color, thickness=1, radius=15, alpha_fill=0.05, label=None
):
    """Draws a rounded rectangle with a translucent fill and a caption."""
    overlay = img.copy()
    x0, y0 = pt1
    x1, y1 = pt2

    # --- fill ---
    if radius > 0:
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
        cv2.rectangle(overlay, (x0 + radius, y0), (x1 - radius, y1), color, -1)
        cv2.rectangle(overlay, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    else:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

    # search area fill transperency
    img = cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0)

    # --- boarder ---
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

    # --- caption ---
    if label:
        font_scale = 0.5
        font_thick = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, font_thick)[0]
        text_x = x0 + (x1 - x0 - text_size[0]) // 2
        text_y = y0 - 5
        cv2.putText(img, label, (text_x, text_y), font, font_scale, color, font_thick)

    return img


def draw_boxes_gpu(frame_gpu, boxes, ids, cls, class_colors, scale, offset, thickness=2):
    """Draw box edges into a GPU frame tensor (0.21 ms at 1080p)."""
    if len(boxes) == 0:
        return frame_gpu

    sx, sy = scale
    ox, oy = offset
    h, w = frame_gpu.shape[:2]

    xyxy = np.asarray(boxes, dtype=np.float32).copy()
    xyxy[:, 0] = xyxy[:, 0] * sx + ox
    xyxy[:, 1] = xyxy[:, 1] * sy + oy
    xyxy[:, 2] = xyxy[:, 2] * sx + ox
    xyxy[:, 3] = xyxy[:, 3] * sy + oy
    xyxy = np.rint(xyxy).astype(int)
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w - 1)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h - 1)

    t = max(1, int(thickness))
    for (x0, y0, x1, y1), c in zip(xyxy.tolist(), np.asarray(cls).tolist()):
        color = class_colors.get(int(c))
        if color is None:
            continue
        frame_gpu[y0 : min(y0 + t, h), x0:x1] = color
        frame_gpu[max(y1 - t, 0) : y1, x0:x1] = color
        frame_gpu[y0:y1, x0 : min(x0 + t, w)] = color
        frame_gpu[y0:y1, max(x1 - t, 0) : x1] = color

    return frame_gpu


class SearchAreaOverlay:
    def __init__(self, shape, pt1, pt2, color, thickness=1, radius=15,
                 alpha_fill=0.05, label=None):
        h, w = shape[:2]

        filled = draw_rounded_rect(
            np.zeros((h, w, 3), np.uint8), pt1, pt2, color,
            thickness, radius, alpha_fill=1.0, label=label,
        )
        strokes = draw_rounded_rect(
            np.zeros((h, w, 3), np.uint8), pt1, pt2, color,
            thickness, radius, alpha_fill=0.0, label=label,
        )

        opaque = strokes.any(axis=2)
        touched = filled.any(axis=2) | opaque

        ys, xs = np.nonzero(touched)
        if len(ys) == 0:
            self.roi = None
            return

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        self.roi = (y0, y1, x0, x1)
        self.alpha = float(alpha_fill)

        self.fill = np.ascontiguousarray(filled[y0:y1, x0:x1])
        self.strokes = np.ascontiguousarray(strokes[y0:y1, x0:x1])
        self.fill_mask = np.ascontiguousarray(
            (filled[y0:y1, x0:x1].any(axis=2) * 255).astype(np.uint8)
        )
        self.stroke_mask = np.ascontiguousarray((opaque[y0:y1, x0:x1] * 255).astype(np.uint8))
        self._buf = np.empty_like(self.fill)

    def apply(self, frame):
        if self.roi is None:
            return frame

        y0, y1, x0, x1 = self.roi
        region = frame[y0:y1, x0:x1]

        cv2.addWeighted(
            self.fill, self.alpha, region, 1.0 - self.alpha, 0, dst=self._buf
        )
        cv2.copyTo(self._buf, self.fill_mask, dst=region)
        cv2.copyTo(self.strokes, self.stroke_mask, dst=region)

        return frame
