from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.classification.service import set_inference_mode
from tests.test_classify import (
    DATASET_ROOT,
    RESULTS_DIR,
    evaluate_dataset,
    iter_images,
    load_dataset_bytes,
)

SUPPORTED_MODES = ("ensemble", "cascade", "direct-specialist")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa avaliacao do classificador nos tres modos principais.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=SUPPORTED_MODES,
        choices=SUPPORTED_MODES,
        help="Lista de modos a serem avaliados (default: todos).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Diretorio com o conjunto de validacao (default: {DATASET_ROOT}).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_DIR,
        help="Diretorio base onde salvar cada resultado (default: results/).",
    )
    return parser.parse_args()


def _run_mode(
    mode: str,
    dataset_entries: Sequence[tuple[Path, str]],
    image_bytes: Sequence[bytes],
    results_root: Path,
) -> None:
    set_inference_mode(mode)
    output_dir = results_root / f"{mode}_test"
    print(f"\n[run] Rodando modo '{mode}' -> {output_dir}")
    evaluate_dataset(
        dataset_entries=dataset_entries,
        image_bytes_list=image_bytes,
        progress=True,
        save_outputs=True,
        verbose=True,
        results_dir=output_dir,
    )


def main() -> None:
    args = _parse_args()

    dataset_entries = iter_images(args.dataset_root)
    if not dataset_entries:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada no diretorio de validacao: {args.dataset_root}",
        )
    image_bytes = load_dataset_bytes(dataset_entries)

    for mode in args.modes:
        _run_mode(mode, dataset_entries, image_bytes, args.results_root)


if __name__ == "__main__":
    main()
