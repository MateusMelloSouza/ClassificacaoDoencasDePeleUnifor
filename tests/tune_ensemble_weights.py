from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.classification import service
from tests.test_classify import (
    CLASS_LABELS,
    DATASET_ROOT,
    RESULTS_DIR,
    evaluate_dataset,
    iter_images,
    load_dataset_bytes,
)


MODE_NAMES = ("ensemble", "cascade", "direct-specialist")
SPECIALIST_KEY_ORDER = [
    "benignos_vs_malignos",
    "malignos_vs_premalignos",
    "premalignos_vs_benignos",
]


def _normalise(mode: str, weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 1e-4, None)
    return clipped / clipped.sum()


def _random_weights(mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "ensemble":
        return rng.dirichlet(np.ones(4, dtype=float))
    return _normalise(mode, rng.random(2))


def _crossover(mode: str, parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.3, 0.7)
    child = alpha * parent_a + (1.0 - alpha) * parent_b
    return _normalise(mode, child)


def _mutate(mode: str, weights: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=scale, size=weights.shape)
    mutated = weights + noise
    return _normalise(mode, mutated)


def _format_weights(mode: str, weights: Sequence[float]) -> str:
    if mode == "ensemble":
        names = ["general", *SPECIALIST_KEY_ORDER]
    else:
        names = ["general", "specialist"]
    return ", ".join(f"{name}={value:.3f}" for name, value in zip(names, weights))


def _apply_weights(mode: str, weights: Sequence[float]) -> None:
    if mode == "ensemble":
        general = float(weights[0])
        specialist_map = {
            specialist: float(weights[idx + 1])
            for idx, specialist in enumerate(SPECIALIST_KEY_ORDER)
        }
        service.set_inference_mode("ensemble")
        service.set_ensemble_weights(general, specialist_map)
    elif mode == "cascade":
        general, specialist = map(float, weights[:2])
        service.set_inference_mode("cascade")
        service.set_cascade_weights(general, specialist)
    elif mode == "direct-specialist":
        general, specialist = map(float, weights[:2])
        service.set_inference_mode("direct-specialist")
        service.set_direct_weights(general, specialist)
    else:
        raise ValueError(f"Modo desconhecido: {mode}")


def _evaluate_weights(
    mode: str,
    weights: Sequence[float],
    dataset_entries: Sequence[Tuple[Path, str]],
    image_bytes: Sequence[bytes],
    results_dir: Path,
) -> Tuple[float, dict]:
    _apply_weights(mode, weights)
    result = evaluate_dataset(
        dataset_entries=dataset_entries,
        image_bytes_list=image_bytes,
        progress=False,
        save_outputs=False,
        verbose=False,
        results_dir=results_dir,
    )
    if not result.get("success"):
        raise RuntimeError(result.get("reason", "Falha na avaliação."))
    return float(result["accuracy"]), result


def run_ga(
    *,
    mode: str,
    population_size: int,
    generations: int,
    mutation_rate: float,
    mutation_scale: float,
    random_seed: int,
    output_dir: Path,
) -> Tuple[np.ndarray, float, List[dict]]:
    if mode not in MODE_NAMES:
        raise ValueError(f"Modo inválido: {mode}")

    rng_np = np.random.default_rng(random_seed)
    rng_py = random.Random(random_seed)

    dataset_entries = iter_images(DATASET_ROOT)
    if not dataset_entries:
        raise FileNotFoundError("Nenhuma imagem encontrada no diretório de validação configurado.")
    image_bytes = load_dataset_bytes(dataset_entries)

    population: List[np.ndarray] = [
        _random_weights(mode, rng_np) for _ in range(population_size)
    ]

    best_overall: Tuple[float, np.ndarray, dict] | None = None
    history: List[dict] = []

    for generation in range(generations):
        evaluation_progress = tqdm(
            population,
            desc=f"Geração {generation + 1}/{generations} ({mode})",
            unit="pesos",
            leave=False,
        )
        scored_population: List[Tuple[float, np.ndarray, dict]] = []
        for individual in evaluation_progress:
            accuracy, result = _evaluate_weights(mode, individual, dataset_entries, image_bytes, output_dir)
            evaluation_progress.set_postfix({"acc": f"{accuracy:.4f}"})
            scored_population.append((accuracy, individual, result))

        scored_population.sort(key=lambda item: item[0], reverse=True)
        best_accuracy, best_weights, best_result = scored_population[0]
        history.append({
            "generation": generation + 1,
            "accuracy": best_accuracy,
            "weights": best_weights.tolist(),
        })

        if best_overall is None or best_accuracy > best_overall[0]:
            best_overall = (best_accuracy, best_weights.copy(), best_result)

        elite_count = max(1, math.ceil(population_size * 0.2))
        elites = [weights.copy() for _, weights, _ in scored_population[:elite_count]]

        new_population: List[np.ndarray] = elites.copy()
        while len(new_population) < population_size:
            if len(elites) >= 2:
                parent_a, parent_b = rng_py.sample(elites, 2)
            else:
                parent_a = elites[0]
                parent_b = elites[0]
            child = _crossover(mode, parent_a, parent_b, rng_py)
            if rng_py.random() < mutation_rate:
                child = _mutate(mode, child, rng_np, mutation_scale)
            new_population.append(child)

        while len(new_population) < population_size:
            new_population.append(_random_weights(mode, rng_np))

        population = new_population[:population_size]

        print(
            f"Geração {generation + 1} ({mode}): melhor acurácia={best_accuracy:.4f} | "
            f"pesos=({_format_weights(mode, best_weights)})"
        )

    assert best_overall is not None
    best_accuracy, best_weights, _ = best_overall
    _apply_weights(mode, best_weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_result = evaluate_dataset(
        dataset_entries=dataset_entries,
        image_bytes_list=image_bytes,
        progress=True,
        save_outputs=True,
        verbose=True,
        results_dir=output_dir,
    )

    tuning_summary = {
        "mode": mode,
        "best_accuracy": best_accuracy,
        "best_weights": best_weights.tolist(),
        "final_result": {
            "accuracy": final_result["accuracy"],
            "classification_report": final_result["classification_report"],
        },
        "parameters": {
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "mutation_scale": mutation_scale,
            "seed": random_seed,
        },
        "class_labels": CLASS_LABELS,
        "history": history,
        "timestamp": datetime.now().isoformat(),
        "outputs": {
            "classification_report_path": str(output_dir / "classification_report.txt"),
            "confusion_matrix_figure": str(output_dir / "validation_confusion_report.png"),
        },
    }

    summary_path = output_dir / f"{mode}_tuning_summary.json"
    summary_path.write_text(json.dumps(tuning_summary, indent=2), encoding="utf-8")
    print(f"Resumo salvo em: {summary_path}")

    return best_weights, best_accuracy, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning dos pesos do classificador (ensemble/cascade/direct) usando algoritmo genético.",
    )
    parser.add_argument("--mode", choices=MODE_NAMES, default="ensemble", help="Modo de inferência a otimizar.")
    parser.add_argument("--population", type=int, default=12, help="Tamanho da população por geração (default: 12)")
    parser.add_argument("--generations", type=int, default=6, help="Número de gerações a executar (default: 6)")
    parser.add_argument("--mutation-rate", type=float, default=0.6, help="Probabilidade de mutação por indivíduo (default: 0.6)")
    parser.add_argument("--mutation-scale", type=float, default=0.08, help="Desvio padrão do ruído gaussiano para mutação (default: 0.08)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório base para salvar relatórios deste modo (default: results/<modo>_test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (RESULTS_DIR / f"{args.mode}_test")
    best_weights, best_accuracy, _ = run_ga(
        mode=args.mode,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        random_seed=args.seed,
        output_dir=output_dir,
    )
    print(
        f"\n[{args.mode}] Melhores pesos encontrados: {_format_weights(args.mode, best_weights)} | "
        f"Acurácia={best_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
