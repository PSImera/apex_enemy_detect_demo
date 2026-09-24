"""TensorRT engine and the GPU-resident detection path it drives.

Ultralytics re-binds tensors and rebuilds result objects on every call, which at
this speed costs more than the inference. Here the buffers are bound once, the
engine runs through a CUDA graph, and the frame stays on the GPU from crop to
detection.

Measured on an RTX 3070 Ti, YOLOv8n @ 640x640: ~1.9 ms per frame
(preprocess 0.17 + engine 0.86 + NMS 0.37 + BoT-SORT 0.49).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils.nms import non_max_suppression

from backend.tracking import DetResults, build_tracker

_EMPTY = (
    np.zeros((0, 4), dtype=np.float32),
    np.zeros(0, dtype=int),
    np.zeros(0, dtype=int),
)


class RawEngine:
    """A TensorRT engine with its input and output buffers bound once."""

    def __init__(self, path: str, device: str = "cuda"):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            # Ultralytics engines start with a metadata header (4-byte length
            # + JSON); a plain engine starts with the plan itself.
            meta_len = int.from_bytes(f.read(4), "little")
            try:
                f.read(meta_len).decode("utf-8")
            except UnicodeDecodeError:
                f.seek(0)
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(
                f"Could not load TensorRT engine: {path}. Engines are tied to the "
                "TensorRT version that built them — rebuild it if TensorRT changed."
            )

        self.engine = engine
        self.ctx = engine.create_execution_context()

        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inputs = [
            n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]

        self.buffers = {}
        for name in names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            np_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            buf = torch.empty(shape, dtype=getattr(torch, np_dtype.name), device=device)
            self.buffers[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())

        self.input = self.buffers[inputs[0]]
        self.output = self.buffers[outputs[0]]
        self._graph = None

    def capture(self):
        """Capture the engine call into a CUDA graph.

        The engine is launch-bound, so replaying one graph instead of issuing
        every kernel from Python roughly halves the call (1.84 -> 0.80 ms).
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                self.ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        self._graph = graph
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.input.copy_(x)
        if self._graph is not None:
            self._graph.replay()
        else:
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.output


class FastDetector:
    """Crop, infer, NMS and track a frame that already lives on the GPU."""

    def __init__(
        self,
        engine_file: str,
        crop_box,
        imgsz_w: int = 640,
        imgsz_h: int = 640,
        conf: float = 0.5,
        iou: float = 0.4,
        track: bool = True,
        tracker_cfg: str | None = None,
        frame_rate: int = 60,
    ):
        self.engine = RawEngine(str(engine_file)).capture()
        self.x0, self.y0, self.x1, self.y1 = crop_box
        self.imgsz = (imgsz_h, imgsz_w)
        self.conf = conf
        self.iou = iou

        self.tracker = build_tracker(frame_rate, tracker_cfg) if track else None
        # Only read for motion compensation, which the project config disables.
        self._track_img = np.zeros((imgsz_h, imgsz_w, 3), dtype=np.uint8)

        self._warmup()

    def _warmup(self, rounds: int = 12):
        """Absorb lazy allocations, so the first measured frame is not the slow one."""
        dummy = torch.zeros(
            (self.y1 + 1, self.x1 + 1, 3), dtype=torch.uint8, device="cuda"
        )
        for _ in range(rounds):
            non_max_suppression(
                self.engine(self.preprocess(dummy)), self.conf, self.iou
            )
        torch.cuda.synchronize()

    def preprocess(self, frame_gpu: torch.Tensor) -> torch.Tensor:
        """Crop and convert HWC uint8 BGR into the engine's NCHW input."""
        crop = frame_gpu[self.y0 : self.y1, self.x0 : self.x1]
        chw = crop.permute(2, 0, 1)[None].flip(1)
        chw = chw.to(self.engine.input.dtype).div_(255)
        return F.interpolate(chw, size=self.imgsz, mode="bilinear", align_corners=False)

    def __call__(self, frame_gpu: torch.Tensor):
        """Return (boxes, ids, classes) in model coordinates, as numpy arrays."""
        det = non_max_suppression(
            self.engine(self.preprocess(frame_gpu)), self.conf, self.iou
        )[0]

        if self.tracker is None:
            d = det.cpu().numpy()
            return d[:, :4], None, d[:, 5].astype(int)

        # Empty frames still go through, so lost tracks age out.
        tracks = self.tracker.update(DetResults.from_detections(det), self._track_img)
        if len(tracks) == 0:
            return _EMPTY

        # BoT-SORT returns [x1, y1, x2, y2, track_id, score, cls, idx]
        return tracks[:, :4], tracks[:, 4].astype(int), tracks[:, 6].astype(int)
