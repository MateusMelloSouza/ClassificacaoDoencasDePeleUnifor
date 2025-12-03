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
ENSEMBLE_GENERAL_WEIGHT = 0.6280209863015969
SPECIALIST_WEIGHTS = {
    "benignos_vs_malignos": 0.3072561496520236,
    "malignos_vs_premalignos": 9.763103177869513e-05,
    "premalignos_vs_benignos": 9.763103177869513e-05,
}

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


def classify_image_bytes(image_bytes: bytes) -> dict:
    device = _get_device()

    general_model, _ = _load_general_model()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = general_model(tensor)
        general_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Combina probabilidades do generalista com todos os especialistas disponíveis.
    combined_probs = general_probs * ENSEMBLE_GENERAL_WEIGHT
    specialist_keys_used: list[str] = []

    for specialist_key, class_order in SPECIALIST_CLASS_ORDER.items():
        specialist_model, specialist_device = _load_specialist_model(specialist_key)
        if specialist_model is None:
            continue

        with torch.inference_mode():
            tensor_on_device = tensor if tensor.device == specialist_device else tensor.to(specialist_device)
            specialist_logits = specialist_model(tensor_on_device)
            specialist_probs = torch.softmax(specialist_logits, dim=1)[0].cpu().numpy()

        weight = SPECIALIST_WEIGHTS.get(specialist_key, 0.0)
        for idx, class_name in enumerate(class_order):
            if class_name not in CLASS_NAMES_PT or idx >= len(specialist_probs):
                continue
            general_index = CLASS_NAMES_PT.index(class_name)
            combined_probs[general_index] += specialist_probs[idx] * weight

        specialist_keys_used.append(specialist_key)

    combined_sum = combined_probs.sum()
    if combined_sum > 0:
        combined_probs /= combined_sum

    final_idx = int(combined_probs.argmax())
    final_prediction = CLASS_NAMES_PT[final_idx]

    all_probs = {
        CLASS_NAMES_PT[i]: float(round(prob, 4)) for i, prob in enumerate(combined_probs)
    }

    return {
        "prediction": final_prediction,
        "probabilities": all_probs,
        "specialist_used": bool(specialist_keys_used),
        "specialists": specialist_keys_used,
    }
