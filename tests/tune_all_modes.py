from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tests.tune_ensemble_weights import MODE_NAMES, RESULTS_DIR, run_ga

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o algoritmo genetico para todos os modos suportados.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=MODE_NAMES,
        default=MODE_NAMES,
        help="Lista de modos para otimizar (default: todos).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=10,
        help="Tamanho da populacao por geracao (default: 15).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Numero de geracoes (default: 20).",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.8,
        help="Probabilidade de mutacao (default: 0.8).",
    )
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.1,
        help="Desvio padrao do ruido da mutacao (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default= np.random.randint(0, 100),
        help="Semente aleatoria (default: 42).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_DIR / "ga_runs",
        help="Diretorio base para salvar os resultados de tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for mode in args.modes:
        print(f"\n===== Tuning para modo '{mode}' =====")
        output_dir = args.results_root / f"{mode}_ga"
        run_ga(
            mode=mode,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            random_seed=args.seed,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
