from __future__ import annotations

import subprocess


_nvenc_cache = None


def nvenc_available() -> bool:
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    _nvenc_cache = _probe_nvenc()
    return _nvenc_cache


def _probe_nvenc() -> bool:
    try:
        listed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    if "h264_nvenc" not in listed.stdout:
        return False

    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    return True


class FrameWriter:
    def __init__(self, path: str, width: int, height: int, fps: int, use_nvenc=None):
        if use_nvenc is None:
            use_nvenc = nvenc_available()
        self.encoder = "h264_nvenc" if use_nvenc else "libx264"

        codec_args = (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
            if use_nvenc
            else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
        )

        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
                *codec_args,
                "-pix_fmt", "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame):
        self.proc.stdin.write(frame.data)

    def release(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        proc, self.proc = self.proc, None
        code = proc.wait()
        if code != 0:
            err = proc.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg writer failed ({self.encoder}): {err}")
