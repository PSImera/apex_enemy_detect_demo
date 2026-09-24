from ultralytics import YOLO
from pathlib import Path
import threading

MODELS_DIR = Path(__file__).parent.parent / "models" / "apex_enemy_detect"
# Engine cache lives inside the project and is gitignored.
ENGINES_DIR = Path(__file__).parent.parent / "models" / "engines"
MODELS = {
    "fast": MODELS_DIR / "apex_detect_v8n_v2.1.pt",
    "accurate": MODELS_DIR / "apex_detect_v8m_v2.1.pt",
}

BACKENDS = ("torch", "tensorrt")

ENGINE_BUILD_ESTIMATE_S = {
    "fast": 300,
    "accurate": 600,
}

_model_cache = {}
_build_lock = threading.Lock()


def engine_path(model_choice: str, imgsz_w: int, imgsz_h: int, half: bool = True):
    precision = "fp16" if half else "fp32"
    stem = MODELS[model_choice].stem
    return ENGINES_DIR / f"{stem}_{imgsz_w}x{imgsz_h}_{precision}.engine"


def engine_is_cached(
    model_choice: str,
    imgsz_w: int,
    imgsz_h: int,
    half: bool = True,
    force_rebuild: bool = False,
):
    if force_rebuild:
        return False
    return engine_path(model_choice, imgsz_w, imgsz_h, half).exists()


def build_engine(
    model_choice: str,
    imgsz_w: int = 640,
    imgsz_h: int = 640,
    half: bool = True,
    workspace: float = 4.0,
    force: bool = False,
    progress_cb=None,
):
    if model_choice not in MODELS:
        raise ValueError(f"Unknown model: {model_choice}")

    target = engine_path(model_choice, imgsz_w, imgsz_h, half)

    with _build_lock:
        if target.exists() and not force:
            if progress_cb:
                progress_cb(f"Using cached engine {target.name}")
            return target

        if progress_cb:
            progress_cb(f"Building TensorRT engine {target.name} (may take minutes)...")

        ENGINES_DIR.mkdir(parents=True, exist_ok=True)

        model = YOLO(MODELS[model_choice])
        exported = Path(
            model.export(
                format="engine",
                imgsz=(imgsz_h, imgsz_w),
                half=half,
                device=0,
                workspace=workspace,
                simplify=True,
                verbose=False,
            )
        )

        if exported.resolve() != target.resolve():
            exported.replace(target)
        leftover = exported.with_suffix(".onnx")
        if leftover.exists():
            leftover.unlink()

        if progress_cb:
            progress_cb(f"Engine ready: {target.name}")

        return target


def prepare_engine(
    model_choice: str,
    imgsz_w: int = 640,
    imgsz_h: int = 640,
    half: bool = True,
    workspace: float = 4.0,
    force_rebuild: bool = False,
    progress_cb=None,
):
    return build_engine(
        model_choice,
        imgsz_w=imgsz_w,
        imgsz_h=imgsz_h,
        half=half,
        workspace=workspace,
        force=force_rebuild,
        progress_cb=progress_cb,
    )


def get_model(
    model_choice: str,
    backend: str = "torch",
    imgsz_w: int = 640,
    imgsz_h: int = 640,
    half: bool = True,
    workspace: float = 4.0,
    force_rebuild: bool = False,
    progress_cb=None,
):
    if model_choice not in MODELS:
        raise ValueError(f"Unknown model: {model_choice}")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")

    if backend == "torch":
        cache_key = ("torch", model_choice)
    else:
        cache_key = ("tensorrt", model_choice, imgsz_w, imgsz_h, half)

    if cache_key in _model_cache and not force_rebuild:
        return _model_cache[cache_key]

    if backend == "torch":
        model = YOLO(MODELS[model_choice])
        model.to("cuda")
        model.fuse()
    else:
        path = build_engine(
            model_choice,
            imgsz_w=imgsz_w,
            imgsz_h=imgsz_h,
            half=half,
            workspace=workspace,
            force=force_rebuild,
            progress_cb=progress_cb,
        )
        model = YOLO(path, task="detect")

    _model_cache[cache_key] = model
    return model
