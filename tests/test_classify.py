from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.classification.service import CLASS_NAMES_PT, classify_image_bytes


DATASET_ROOT = Path("/home/mateus/data/val")
CLASS_DIR_TO_LABEL = {
    "Benignos": "Benigno",
    "Malignos": "Maligno",
    "Pre-Malignos": "Pré-Maligno",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
RESULTS_DIR = PROJECT_ROOT / "results"
CLASS_LABELS = list(CLASS_NAMES_PT)


def iter_images(dataset_root: Path = DATASET_ROOT) -> list[tuple[Path, str]]:
    image_entries: list[tuple[Path, str]] = []
    for class_dir, label in CLASS_DIR_TO_LABEL.items():
        folder_path = dataset_root / class_dir
        if not folder_path.exists():
            print(f"⚠️ Diretório não encontrado: {folder_path}")
            continue

        for image_path in sorted(folder_path.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_entries.append((image_path, label))

    return image_entries


def load_dataset_bytes(image_entries: Sequence[tuple[Path, str]]) -> list[bytes]:
    return [entry[0].read_bytes() for entry in image_entries]


def _build_confusion_figure(
    cm: np.ndarray,
    cm_normalized: np.ndarray,
    label_order: Sequence[str],
    accuracy: float,
    correct_predictions: int,
    total_images: int,
    report: str,
    figure_path: Path,
) -> None:
    fig, (ax_matrix, ax_text) = plt.subplots(
        ncols=2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    im = ax_matrix.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax_matrix.set_title("Matriz de Confusão Normalizada")
    tick_marks = np.arange(len(label_order))
    ax_matrix.set_xticks(tick_marks, label_order, rotation=45, ha="right")
    ax_matrix.set_yticks(tick_marks, label_order)
    fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

    for i, j in np.ndindex(cm_normalized.shape):
        value = cm_normalized[i, j]
        text = f"{value:.2f}\n({cm[i, j]})"
        ax_matrix.text(j, i, text, ha="center", va="center", color="black")

    ax_text.axis("off")
    summary_lines = [
        f"Acurácia: {accuracy:.4f} ({correct_predictions}/{total_images})",
        "",
        "Relatório de Classificação:",
        report,
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def evaluate_dataset(
    dataset_root: Path = DATASET_ROOT,
    *,
    progress: bool = True,
    save_outputs: bool = True,
    verbose: bool = True,
    results_dir: Path = RESULTS_DIR,
    dataset_entries: Sequence[tuple[Path, str]] | None = None,
    image_bytes_list: Sequence[bytes] | None = None,
) -> dict[str, Any]:
    if dataset_entries is None:
        if not dataset_root.exists():
            message = f"Diretório de validação não encontrado: {dataset_root}"
            if verbose:
                print(message)
            return {"success": False, "reason": message}
        dataset_entries = iter_images(dataset_root)
    elif not dataset_entries:
        return {"success": False, "reason": "Nenhuma imagem fornecida para avaliação."}

    if not dataset_entries:
        message = "Nenhuma imagem encontrada no conjunto de validação especificado."
        if verbose:
            print(message)
        return {"success": False, "reason": message}

    total_images = len(dataset_entries)
    if image_bytes_list is not None and len(image_bytes_list) != total_images:
        raise ValueError("O tamanho de image_bytes_list deve coincidir com dataset_entries.")

    iterator: Iterable[int] = range(total_images)
    if progress:
        iterator = tqdm(iterator, desc="Classificando", unit="imagem", total=total_images)

    y_true: list[str] = []
    y_pred: list[str] = []

    for idx in iterator:
        image_path, true_label = dataset_entries[idx]
        if image_bytes_list is None:
            image_bytes = image_path.read_bytes()
        else:
            image_bytes = image_bytes_list[idx]

        try:
            result = classify_image_bytes(image_bytes)
        except Exception as exc:  # noqa: BLE001 - queremos o stack trace completo
            if verbose:
                print(f"Erro ao classificar {image_path}: {exc}")
            raise

        predicted_label = result["prediction"]
        y_true.append(true_label)
        y_pred.append(predicted_label)

    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_predictions / total_images if total_images else 0.0

    label_order = CLASS_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=label_order, normalize="true")
    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=label_order,
        zero_division=0,
    )

    if save_outputs:
        results_dir.mkdir(parents=True, exist_ok=True)
        report_path = results_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        figure_path = results_dir / "validation_confusion_report.png"
        _build_confusion_figure(cm, cm_normalized, label_order, accuracy, correct_predictions, total_images, report, figure_path)

    if verbose:
        print(f"Imagens avaliadas: {total_images}")
        print(f"Acurácia: {accuracy:.4f} ({correct_predictions}/{total_images})")
        print("\nMatriz de confusão (linhas=verdade, colunas=predição):")
        for label, row in zip(label_order, cm):
            print(f"{label:>12}: {row}")

        print("\nRelatório de classificação:\n")
        print(report)
        if save_outputs:
            print(f"\nRelatório em texto salvo em: {results_dir / 'classification_report.txt'}")
            print(f"Figura consolidada salva em: {results_dir / 'validation_confusion_report.png'}")

        misclassified_counter = Counter()
        for true, pred in zip(y_true, y_pred):
            if true != pred:
                misclassified_counter[(true, pred)] += 1

        if misclassified_counter:
            print("Erros por par (verdade -> predição):")
            for (true, pred), count in misclassified_counter.items():
                print(f"  {true:>12} -> {pred:<12}: {count}")

    return {
        "success": True,
        "accuracy": accuracy,
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "classification_report": report,
    }


def main() -> None:
    evaluate_dataset()


if __name__ == "__main__":
    main()