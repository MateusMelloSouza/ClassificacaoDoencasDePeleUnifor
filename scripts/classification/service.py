from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np


CLASS_NAMES_PT = ["Benigno", "Maligno", "Pré-Maligno"]

SPECIALIST_CLASS_ORDER = {
    "benignos_vs_malignos": ["Benigno", "Maligno"],
    "malignos_vs_premalignos": ["Maligno", "Pré-Maligno"],
    "premalignos_vs_benignos": ["Pré-Maligno", "Benigno"],
}

SPECIALIST_PAIRS = {
    ("Benigno", "Maligno"): "benignos_vs_malignos",
    ("Maligno", "Benigno"): "benignos_vs_malignos",
    ("Maligno", "Pré-Maligno"): "malignos_vs_premalignos",
    ("Pré-Maligno", "Maligno"): "malignos_vs_premalignos",
    ("Benigno", "Pré-Maligno"): "premalignos_vs_benignos",
    ("Pré-Maligno", "Benigno"): "premalignos_vs_benignos",
}

ENSEMBLE_GENERAL_WEIGHT = 0.6280209863015969
SPECIALIST_WEIGHTS = {
    "benignos_vs_malignos": 0.2072561496520236,
    "malignos_vs_premalignos": 0.1646252330146009,
    "premalignos_vs_benignos": 9.763103177869513e-05,
}

CASCADE_GENERAL_WEIGHT = 0.4
CASCADE_SPECIALIST_WEIGHT = 0.6
CASCADE_MIN_TOP_CONFIDENCE = 0.65
CASCADE_CONFIDENCE_MARGIN = 0.15
CASCADE_WEIGHT_OVERRIDES = {
    "benignos_vs_malignos": {"general": 0.2, "specialist": 0.8},
}

DIRECT_GENERAL_WEIGHT = 0.3
DIRECT_SPECIALIST_WEIGHT = 0.7

VALID_INFERENCE_MODES = {"ensemble", "cascade", "direct-specialist"}
_INFERENCE_MODE = "ensemble"


def set_inference_mode(mode: str) -> None:
    """Define qual estratégia de inferência será utilizada."""
    global _INFERENCE_MODE
    normalized = mode.strip().lower()
    if normalized not in VALID_INFERENCE_MODES:
        raise ValueError(f"Modo inválido: {mode}. Opções: {sorted(VALID_INFERENCE_MODES)}")
    _INFERENCE_MODE = normalized


def set_ensemble_weights(general_weight: float, specialist_weights: dict[str, float]) -> None:
    global ENSEMBLE_GENERAL_WEIGHT, SPECIALIST_WEIGHTS
    ENSEMBLE_GENERAL_WEIGHT = float(general_weight)
    SPECIALIST_WEIGHTS = {
        key: float(specialist_weights.get(key, SPECIALIST_WEIGHTS.get(key, 0.0)))
        for key in SPECIALIST_CLASS_ORDER
    }


def set_cascade_weights(general_weight: float, specialist_weight: float,
                        *, min_top_confidence: float | None = None,
                        confidence_margin: float | None = None) -> None:
    global CASCADE_GENERAL_WEIGHT, CASCADE_SPECIALIST_WEIGHT
    global CASCADE_MIN_TOP_CONFIDENCE, CASCADE_CONFIDENCE_MARGIN
    total = max(general_weight + specialist_weight, 1e-6)
    CASCADE_GENERAL_WEIGHT = float(general_weight / total)
    CASCADE_SPECIALIST_WEIGHT = float(specialist_weight / total)
    if min_top_confidence is not None:
        CASCADE_MIN_TOP_CONFIDENCE = float(min_top_confidence)
    if confidence_margin is not None:
        CASCADE_CONFIDENCE_MARGIN = float(confidence_margin)


def set_direct_weights(general_weight: float, specialist_weight: float) -> None:
    global DIRECT_GENERAL_WEIGHT, DIRECT_SPECIALIST_WEIGHT
    total = max(general_weight + specialist_weight, 1e-6)
    DIRECT_GENERAL_WEIGHT = float(general_weight / total)
    DIRECT_SPECIALIST_WEIGHT = float(specialist_weight / total)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def _get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _weights_path_general() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "Treinamento Modelos" / "resnetrs50" / "best_model.pth",
        root / "Treinamento Modelos" / "resnetrs50" / "outputs" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Arquivo de pesos 'best_model.pth' não encontrado em Treinamento Modelos/resnetrs50.")


def _weights_path_specialist(pair_name: str) -> Path | None:
    root = Path(__file__).resolve().parents[2]
    path = root / "Treinamento Modelos" / "especialistas" / "resnetrs50" / pair_name / "best_model.pth"
    if path.exists():
        return path
    return None


@lru_cache(maxsize=1)
def _load_general_model():
    weights_path = _weights_path_general()
    device = _get_device()
    model = timm.create_model('resnetrs50', pretrained=False, num_classes=len(CLASS_NAMES_PT))
    state = torch.load(weights_path, map_location=device)

    new_state = {}
    for k, v in state.items():
        if k == 'fc.1.weight':
            new_state['fc.weight'] = v
        elif k == 'fc.1.bias':
            new_state['fc.bias'] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.eval()
    model.to(device)
    return model, device


@lru_cache(maxsize=3)
def _load_specialist_model(pair_name: str):
    weights_path = _weights_path_specialist(pair_name)
    if weights_path is None:
        return None, None

    device = _get_device()
    model = timm.create_model('resnetrs50', pretrained=False, num_classes=2)
    state = torch.load(weights_path, map_location=device)

    new_state = {}
    for k, v in state.items():
        if k == 'fc.1.weight':
            new_state['fc.weight'] = v
        elif k == 'fc.1.bias':
            new_state['fc.bias'] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.eval()
    model.to(device)
    return model, device


def _specialist_inference(tensor: torch.Tensor, specialist_key: str) -> tuple[np.ndarray | None, str | None]:
    specialist_model, specialist_device = _load_specialist_model(specialist_key)
    if specialist_model is None:
        return None, None

    with torch.inference_mode():
        tensor_on_device = tensor if tensor.device == specialist_device else tensor.to(specialist_device)
        logits = specialist_model(tensor_on_device)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return probs, specialist_key


def _get_top_indices(probs: np.ndarray) -> tuple[int, int]:
    top_2_indices = np.argsort(probs)[-2:][::-1]
    return int(top_2_indices[0]), int(top_2_indices[1])


def _classify_with_ensemble(general_probs: np.ndarray, tensor: torch.Tensor) -> dict:
    combined_probs = general_probs * ENSEMBLE_GENERAL_WEIGHT
    specialist_keys_used: list[str] = []

    for specialist_key, class_order in SPECIALIST_CLASS_ORDER.items():
        specialist_probs, specialist_used = _specialist_inference(tensor, specialist_key)
        if specialist_probs is None or specialist_used is None:
            continue

        weight = SPECIALIST_WEIGHTS.get(specialist_key, 0.0)
        for idx, class_name in enumerate(class_order):
            if class_name not in CLASS_NAMES_PT or idx >= len(specialist_probs):
                continue
            general_index = CLASS_NAMES_PT.index(class_name)
            combined_probs[general_index] += specialist_probs[idx] * weight

        specialist_keys_used.append(specialist_used)

    combined_sum = combined_probs.sum()
    if combined_sum > 0:
        combined_probs /= combined_sum

    final_idx = int(combined_probs.argmax())
    final_prediction = CLASS_NAMES_PT[final_idx]
    all_probs = {CLASS_NAMES_PT[i]: float(round(prob, 4)) for i, prob in enumerate(combined_probs)}

    return {
        "prediction": final_prediction,
        "probabilities": all_probs,
        "specialist_used": bool(specialist_keys_used),
        "specialists": specialist_keys_used,
    }


def _classify_with_cascade(general_probs: np.ndarray, tensor: torch.Tensor) -> dict:
    top_idx_1, top_idx_2 = _get_top_indices(general_probs)
    top_class_1 = CLASS_NAMES_PT[top_idx_1]
    top_class_2 = CLASS_NAMES_PT[top_idx_2]
    top_prob_1 = float(general_probs[top_idx_1])
    top_prob_2 = float(general_probs[top_idx_2])

    specialist_key = SPECIALIST_PAIRS.get((top_class_1, top_class_2))
    use_specialist = False
    margin = top_prob_1 - top_prob_2
    if specialist_key and (
        specialist_key == "benignos_vs_malignos"
        or top_prob_1 < CASCADE_MIN_TOP_CONFIDENCE
        or margin < CASCADE_CONFIDENCE_MARGIN
    ):
        use_specialist = True

    combined_probs = general_probs.copy()
    specialist_used: list[str] = []

    if use_specialist and specialist_key:
        specialist_probs, specialist_identifier = _specialist_inference(tensor, specialist_key)
        if specialist_probs is not None and specialist_identifier is not None:
            class_order = SPECIALIST_CLASS_ORDER.get(specialist_key, [top_class_1, top_class_2])
            weights = CASCADE_WEIGHT_OVERRIDES.get(
                specialist_key,
                {"general": CASCADE_GENERAL_WEIGHT, "specialist": CASCADE_SPECIALIST_WEIGHT},
            )
            for idx, class_name in enumerate(class_order):
                if class_name not in CLASS_NAMES_PT or idx >= len(specialist_probs):
                    continue
                general_index = CLASS_NAMES_PT.index(class_name)
                combined_probs[general_index] = (
                    general_probs[general_index] * weights["general"]
                    + specialist_probs[idx] * weights["specialist"]
                )
            specialist_used.append(specialist_identifier)

    combined_sum = combined_probs.sum()
    if combined_sum > 0:
        combined_probs /= combined_sum

    final_idx = int(combined_probs.argmax())
    final_prediction = CLASS_NAMES_PT[final_idx]
    all_probs = {CLASS_NAMES_PT[i]: float(round(prob, 4)) for i, prob in enumerate(combined_probs)}

    return {
        "prediction": final_prediction,
        "probabilities": all_probs,
        "specialist_used": bool(specialist_used),
        "specialists": specialist_used,
    }


def _classify_with_direct_specialist(general_probs: np.ndarray, tensor: torch.Tensor) -> dict:
    top_idx_1, top_idx_2 = _get_top_indices(general_probs)
    top_class_1 = CLASS_NAMES_PT[top_idx_1]
    top_class_2 = CLASS_NAMES_PT[top_idx_2]

    specialist_key = SPECIALIST_PAIRS.get((top_class_1, top_class_2))
    if specialist_key is None:
        # fallback to ensemble combination if we cannot route to a specialist pair
        return _classify_with_ensemble(general_probs, tensor)

    specialist_probs, specialist_identifier = _specialist_inference(tensor, specialist_key)
    combined_probs = general_probs.copy()
    specialist_used: list[str] = []

    if specialist_probs is not None and specialist_identifier is not None:
        class_order = SPECIALIST_CLASS_ORDER.get(specialist_key, [top_class_1, top_class_2])
        for idx, class_name in enumerate(class_order):
            if class_name not in CLASS_NAMES_PT or idx >= len(specialist_probs):
                continue
            general_index = CLASS_NAMES_PT.index(class_name)
            combined_probs[general_index] = (
                general_probs[general_index] * DIRECT_GENERAL_WEIGHT
                + specialist_probs[idx] * DIRECT_SPECIALIST_WEIGHT
            )
        specialist_used.append(specialist_identifier)

    combined_sum = combined_probs.sum()
    if combined_sum > 0:
        combined_probs /= combined_sum

    final_idx = int(combined_probs.argmax())
    final_prediction = CLASS_NAMES_PT[final_idx]
    all_probs = {CLASS_NAMES_PT[i]: float(round(prob, 4)) for i, prob in enumerate(combined_probs)}

    return {
        "prediction": final_prediction,
        "probabilities": all_probs,
        "specialist_used": bool(specialist_used),
        "specialists": specialist_used,
    }


def classify_image_bytes(image_bytes: bytes) -> dict:
    device = _get_device()

    general_model, _ = _load_general_model()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = general_model(tensor)
        general_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    if _INFERENCE_MODE == "cascade":
        return _classify_with_cascade(general_probs, tensor)
    if _INFERENCE_MODE == "direct-specialist":
        return _classify_with_direct_specialist(general_probs, tensor)
    return _classify_with_ensemble(general_probs, tensor)
