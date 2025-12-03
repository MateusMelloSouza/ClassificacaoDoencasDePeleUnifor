from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
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


@dataclass(frozen=True)
class WeightVector:
    general: float
    benignos_vs_malignos: float
    malignos_vs_premalignos: float
    premalignos_vs_benignos: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.general,
            self.benignos_vs_malignos,
            self.malignos_vs_premalignos,
            self.premalignos_vs_benignos,
        ], dtype=float)

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "WeightVector":
        return cls(*map(float, values))


def _normalise(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 1e-4, None)
    return clipped / clipped.sum()


def _random_weights(rng: np.random.Generator) -> WeightVector:
    arr = rng.dirichlet(np.ones(4, dtype=float))
    return WeightVector.from_array(arr)


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.3, 0.7)
    child = alpha * parent_a + (1.0 - alpha) * parent_b
    return _normalise(child)


def _mutate(weights: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=scale, size=weights.shape)
    mutated = weights + noise
    return _normalise(mutated)


def _format_weights(weights: Sequence[float]) -> str:
    names = ["general", "benignos_vs_malignos", "malignos_vs_premalignos", "premalignos_vs_benignos"]
    return ", ".join(f"{name}={value:.3f}" for name, value in zip(names, weights))


def _set_service_weights(weights: Sequence[float]) -> None:
    general, bm, mp, pb = map(float, weights)
    service.ENSEMBLE_GENERAL_WEIGHT = general
    service.SPECIALIST_WEIGHTS.update({
        "benignos_vs_malignos": bm,
        "malignos_vs_premalignos": mp,
        "premalignos_vs_benignos": pb,
    })


def _evaluate_weights(
    weights: Sequence[float],
    dataset_entries: Sequence[Tuple[Path, str]],
    image_bytes: Sequence[bytes],
) -> Tuple[float, dict]:
    _set_service_weights(weights)
    result = evaluate_dataset(
        dataset_entries=dataset_entries,
        image_bytes_list=image_bytes,
        progress=False,
        save_outputs=False,
        verbose=False,
    )
    if not result.get("success"):
        raise RuntimeError(result.get("reason", "Falha na avaliação."))
    return float(result["accuracy"]), result


def run_ga(
    *,
    population_size: int,
    generations: int,
    mutation_rate: float,
    mutation_scale: float,
    random_seed: int,
) -> Tuple[WeightVector, float, List[dict]]:
    rng_np = np.random.default_rng(random_seed)
    rng_py = random.Random(random_seed)

    dataset_entries = iter_images(DATASET_ROOT)
    if not dataset_entries:
        raise FileNotFoundError("Nenhuma imagem encontrada no diretório de validação configurado.")
    image_bytes = load_dataset_bytes(dataset_entries)

    population: List[np.ndarray] = [
        _random_weights(rng_np).as_array() for _ in range(population_size)
    ]

    best_overall: Tuple[float, np.ndarray, dict] | None = None
    history: List[dict] = []

    for generation in range(generations):
        evaluation_progress = tqdm(
            population,
            desc=f"Geração {generation + 1}/{generations}",
            unit="pesos",
            leave=False,
        )
        scored_population: List[Tuple[float, np.ndarray, dict]] = []
        for individual in evaluation_progress:
            accuracy, result = _evaluate_weights(individual, dataset_entries, image_bytes)
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
            child = _crossover(parent_a, parent_b, rng_py)
            if rng_py.random() < mutation_rate:
                child = _mutate(child, rng_np, mutation_scale)
            new_population.append(child)

        while len(new_population) < population_size:
            new_population.append(_random_weights(rng_np).as_array())

        population = new_population[:population_size]

        print(
            f"Geração {generation + 1}: melhor acurácia={best_accuracy:.4f} | "
            f"pesos=({_format_weights(best_weights)})"
        )

    assert best_overall is not None
    best_accuracy, best_weights, best_result = best_overall
    _set_service_weights(best_weights)

    final_result = evaluate_dataset(
        dataset_entries=dataset_entries,
        image_bytes_list=image_bytes,
        progress=True,
        save_outputs=True,
        verbose=True,
    )

    tuning_summary = {
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
            "classification_report_path": str(RESULTS_DIR / "classification_report.txt"),
            "confusion_matrix_figure": str(RESULTS_DIR / "validation_confusion_report.png"),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "ensemble_tuning_summary.json"
    summary_path.write_text(json.dumps(tuning_summary, indent=2), encoding="utf-8")
    print(f"Resumo salvo em: {summary_path}")

    return WeightVector.from_array(best_weights), best_accuracy, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning dos pesos do ensemble (generalista + especialistas) usando algoritmo genético.",
    )
    parser.add_argument("--population", type=int, default=12, help="Tamanho da população por geração (default: 12)")
    parser.add_argument("--generations", type=int, default=6, help="Número de gerações a executar (default: 6)")
    parser.add_argument("--mutation-rate", type=float, default=0.6, help="Probabilidade de mutação por indivíduo (default: 0.6)")
    parser.add_argument("--mutation-scale", type=float, default=0.08, help="Desvio padrão do ruído gaussiano para mutação (default: 0.08)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_weights, best_accuracy, history = run_ga(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        random_seed=args.seed,
    )
    print(
        f"\nMelhores pesos encontrados: {_format_weights(best_weights.as_array())} | "
        f"Acurácia={best_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
