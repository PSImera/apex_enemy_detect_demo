from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS = {
    "fast": BASE_DIR / "models" / "apex_detect_v8n_v2.1.pt",
    "accurate": BASE_DIR / "models" / "apex_detect_v8m_v2.1.pt",
}

_model_cache = {}


def get_model(model_choice: str):
    if model_choice not in MODELS:
        raise ValueError("Unknown model")

    if model_choice not in _model_cache:
        model = YOLO(MODELS[model_choice])
        model.to("cuda")
        model.fuse()
        model.model.half()
        _model_cache[model_choice] = model

    return _model_cache[model_choice]
