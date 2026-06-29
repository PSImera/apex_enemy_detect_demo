from ultralytics import YOLO
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models" / "apex_enemy_detect"
MODELS = {
    "fast": MODELS_DIR / "apex_detect_v8n_v2.1.pt",
    "accurate": MODELS_DIR / "apex_detect_v8m_v2.1.pt",
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
