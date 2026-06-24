"""Executa testes estatísticos relacionados às hipóteses do trabalho.

Os testes complementam a avaliação preditiva feita em `modelagem.py`. Aqui as
hipóteses são avaliadas por associação estatística entre grupos de atributos e o
resultado das partidas da Copa.

Testes utilizados:

- qui-quadrado de independência entre faixas de atributos e classe do resultado;
- correlação de Spearman entre atributos numéricos e resultado direcional.

O resultado direcional usa 1 para vitória da Seleção A, 0 para empate e -1 para
vitória da Seleção B. Os testes indicam associação, não causalidade.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Callable

from scipy.stats import chi2_contingency, spearmanr


DEFAULT_INPUT = Path("dados_processados/dataset_partidas_copa.csv")
DEFAULT_OUTPUT_DIR = Path("testes_hipoteses")

RESULT_LABELS = {
    "0": "Vitória A",
    "1": "Empate",
    "2": "Vitória B",
}


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando.

    Returns:
        Namespace com caminho de entrada e pasta de saída.
    """
    parser = argparse.ArgumentParser(
        description="Executa testes estatísticos para as hipóteses do trabalho."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Caminho do dataset processado.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta onde os resultados serão salvos.",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> list[dict[str, str]]:
    """Carrega o dataset processado.

    Args:
        path: Caminho do CSV.

    Returns:
        Lista de linhas como dicionários.
    """
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def ensure_dirs(output_dir: Path) -> Path:
    """Cria a pasta de tabelas dos testes.

    Args:
        output_dir: Pasta principal.

    Returns:
        Pasta de tabelas.
    """
    tables_dir = output_dir / "tabelas"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Salva uma tabela em CSV.

    Args:
        path: Caminho de saída.
        fieldnames: Colunas da tabela.
        rows: Linhas da tabela.
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float:
    """Converte texto para float.

    Args:
        value: Valor textual.

    Returns:
        Valor numérico.
    """
    return float(value)


def directional_result(row: dict[str, str]) -> int:
    """Converte a classe do resultado para escala direcional.

    Args:
        row: Linha do dataset.

    Returns:
        1 para vitória A, 0 para empate, -1 para vitória B.
    """
    if row["result"] == "0":
        return 1
    if row["result"] == "2":
        return -1
    return 0


def rank_bucket(row: dict[str, str]) -> str:
    """Categoriza a diferença de ranking.

    Args:
        row: Linha do dataset.

    Returns:
        Faixa textual de diferença de ranking.
    """
    value = to_float(row["diff_rank"])
    if value < -30:
        return "A muito melhor"
    if value < -10:
        return "A melhor"
    if value < 10:
        return "Equilibrado"
    if value < 30:
        return "B melhor"
    return "B muito melhor"


def sign_bucket(row: dict[str, str], column: str) -> str:
    """Categoriza uma diferença numérica em vantagem de A, equilíbrio ou B.

    Args:
        row: Linha do dataset.
        column: Coluna de diferença.

    Returns:
        Categoria textual.
    """
    value = to_float(row[column])
    if value > 0:
        return "A maior"
    if value < 0:
        return "B maior"
    return "Equilibrado"


def median_bucket(rows: list[dict[str, str]], column: str) -> Callable[[dict[str, str]], str]:
    """Cria uma categorização por mediana para uma diferença absoluta.

    Args:
        rows: Linhas do dataset.
        column: Coluna numérica.

    Returns:
        Função que classifica uma linha como baixo ou alto.
    """
    values = [abs(to_float(row[column])) for row in rows]
    cutoff = median(values)

    def categorize(row: dict[str, str]) -> str:
        value = abs(to_float(row[column]))
        if value >= cutoff:
            return f"diferença alta (>= {cutoff:.3f})"
        return f"diferença baixa (< {cutoff:.3f})"

    return categorize


def build_contingency(
    rows: list[dict[str, str]],
    categorizer: Callable[[dict[str, str]], str],
) -> tuple[list[str], list[list[int]]]:
    """Monta tabela de contingência categoria x resultado.

    Args:
        rows: Linhas do dataset.
        categorizer: Função que gera a categoria de uma linha.

    Returns:
        Tupla com nomes das categorias e matriz de contagens.
    """
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[categorizer(row)][row["result"]] += 1

    categories = sorted(grouped)
    matrix = []
    for category in categories:
        matrix.append([grouped[category][label] for label in ["0", "1", "2"]])
    return categories, matrix


def run_chi_square_test(
    rows: list[dict[str, str]],
    hypothesis: str,
    variable: str,
    categorizer: Callable[[dict[str, str]], str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Executa qui-quadrado de independência.

    Args:
        rows: Linhas do dataset.
        hypothesis: Nome da hipótese.
        variable: Variável testada.
        categorizer: Função de categorização.

    Returns:
        Resumo do teste e tabela de contingência em formato longo.
    """
    categories, matrix = build_contingency(rows, categorizer)
    chi2, p_value, degrees_freedom, expected = chi2_contingency(matrix)
    total = sum(sum(row) for row in matrix)
    rows_count = len(matrix)
    cols_count = len(matrix[0]) if matrix else 0
    cramers_v = math.sqrt(chi2 / (total * max(1, min(rows_count - 1, cols_count - 1))))
    significant = p_value < 0.05

    summary = {
        "hipotese": hypothesis,
        "variavel": variable,
        "teste": "qui-quadrado",
        "estatistica": round(float(chi2), 6),
        "p_valor": round(float(p_value), 6),
        "graus_liberdade": int(degrees_freedom),
        "tamanho_efeito": round(float(cramers_v), 6),
        "medida_efeito": "V de Cramer",
        "significativo_5pct": int(significant),
        "interpretacao": interpret_chi_square(hypothesis, significant, cramers_v),
    }

    contingency_rows = []
    for category_index, category in enumerate(categories):
        category_total = sum(matrix[category_index])
        for result_index, result in enumerate(["0", "1", "2"]):
            count = matrix[category_index][result_index]
            contingency_rows.append(
                {
                    "hipotese": hypothesis,
                    "variavel": variable,
                    "categoria": category,
                    "classe_resultado": result,
                    "rotulo_resultado": RESULT_LABELS[result],
                    "quantidade": count,
                    "percentual_na_categoria": round(
                        100 * count / category_total if category_total else 0,
                        2,
                    ),
                    "esperado": round(float(expected[category_index][result_index]), 3),
                }
            )

    return summary, contingency_rows


def interpret_chi_square(
    hypothesis: str,
    significant: bool,
    effect_size: float,
) -> str:
    """Gera interpretação textual para o qui-quadrado.

    Args:
        hypothesis: Nome da hipótese.
        significant: Indica p-valor menor que 0,05.
        effect_size: V de Cramer.

    Returns:
        Interpretação resumida.
    """
    if not significant:
        return f"{hypothesis}: não houve evidência estatística forte de associação."
    if effect_size < 0.1:
        strength = "efeito fraco"
    elif effect_size < 0.3:
        strength = "efeito moderado"
    else:
        strength = "efeito forte"
    return f"{hypothesis}: associação estatisticamente significativa com {strength}."


def run_spearman_test(
    rows: list[dict[str, str]],
    hypothesis: str,
    variable: str,
) -> dict[str, object]:
    """Executa correlação de Spearman entre atributo e resultado direcional.

    Args:
        rows: Linhas do dataset.
        hypothesis: Nome da hipótese.
        variable: Variável numérica.

    Returns:
        Resumo do teste.
    """
    x_values = [to_float(row[variable]) for row in rows]
    y_values = [directional_result(row) for row in rows]
    statistic, p_value = spearmanr(x_values, y_values)
    significant = p_value < 0.05
    return {
        "hipotese": hypothesis,
        "variavel": variable,
        "teste": "Spearman",
        "estatistica": round(float(statistic), 6),
        "p_valor": round(float(p_value), 6),
        "graus_liberdade": "",
        "tamanho_efeito": round(abs(float(statistic)), 6),
        "medida_efeito": "|rho|",
        "significativo_5pct": int(significant),
        "interpretacao": interpret_spearman(hypothesis, significant, statistic),
    }


def interpret_spearman(
    hypothesis: str,
    significant: bool,
    statistic: float,
) -> str:
    """Gera interpretação textual para correlação de Spearman.

    Args:
        hypothesis: Nome da hipótese.
        significant: Indica p-valor menor que 0,05.
        statistic: Coeficiente de Spearman.

    Returns:
        Interpretação resumida.
    """
    if not significant:
        return f"{hypothesis}: correlação sem evidência estatística forte."
    direction = "positiva" if statistic > 0 else "negativa"
    return f"{hypothesis}: correlação {direction} estatisticamente significativa."


def run_tests(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Executa todos os testes estatísticos.

    Args:
        rows: Linhas do dataset.
        output_dir: Pasta de saída.
    """
    tables_dir = ensure_dirs(output_dir)
    summary_rows = []
    contingency_rows = []

    chi_tests = [
        ("H1 - Ranking FIFA", "diff_rank_faixa", rank_bucket),
        (
            "H2 - Forma recente",
            "diff_last10_points_per_game_sinal",
            lambda row: sign_bucket(row, "diff_last10_points_per_game"),
        ),
        (
            "H3 - Ciclo pré-Copa",
            "diff_cycle_points_per_game_sinal",
            lambda row: sign_bucket(row, "diff_cycle_points_per_game"),
        ),
        (
            "H4 - Força dos adversários",
            "diff_last10_opponent_points_mean_abs",
            median_bucket(rows, "diff_last10_opponent_points_mean"),
        ),
    ]

    for hypothesis, variable, categorizer in chi_tests:
        summary, contingency = run_chi_square_test(
            rows,
            hypothesis,
            variable,
            categorizer,
        )
        summary_rows.append(summary)
        contingency_rows.extend(contingency)

    spearman_tests = [
        ("H1 - Ranking FIFA", "diff_rank"),
        ("H1 - Pontos FIFA", "diff_fifa_points"),
        ("H2 - Forma recente", "diff_last10_points_per_game"),
        ("H3 - Ciclo pré-Copa", "diff_cycle_points_per_game"),
        ("H3 - Ataque no ciclo", "diff_cycle_goals_for_per_game"),
        ("H3 - Defesa no ciclo", "diff_cycle_goals_against_per_game"),
        ("H4 - Força dos adversários", "diff_last10_opponent_points_mean"),
    ]

    for hypothesis, variable in spearman_tests:
        summary_rows.append(run_spearman_test(rows, hypothesis, variable))

    write_csv(
        tables_dir / "resumo_testes_hipoteses.csv",
        [
            "hipotese",
            "variavel",
            "teste",
            "estatistica",
            "p_valor",
            "graus_liberdade",
            "tamanho_efeito",
            "medida_efeito",
            "significativo_5pct",
            "interpretacao",
        ],
        summary_rows,
    )
    write_csv(
        tables_dir / "tabelas_contingencia.csv",
        [
            "hipotese",
            "variavel",
            "categoria",
            "classe_resultado",
            "rotulo_resultado",
            "quantidade",
            "percentual_na_categoria",
            "esperado",
        ],
        contingency_rows,
    )
    write_markdown_summary(output_dir, summary_rows)

    print(f"Testes de hipóteses concluídos em: {output_dir}")
    print(f"Tabelas geradas em: {tables_dir}")


def write_markdown_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Gera resumo em Markdown dos testes.

    Args:
        output_dir: Pasta de saída.
        rows: Resumos dos testes.
    """
    lines = [
        "# Testes Estatísticos das Hipóteses",
        "",
        "Os testes complementam a modelagem preditiva. Eles avaliam associação",
        "estatística entre atributos e resultado, sem afirmar causalidade.",
        "",
        "| Hipótese | Variável | Teste | Estatística | p-valor | Efeito | Significativo 5% |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['hipotese']} | {row['variavel']} | {row['teste']} | "
            f"{row['estatistica']} | {row['p_valor']} | {row['tamanho_efeito']} | "
            f"{row['significativo_5pct']} |"
        )

    lines.extend(["", "## Interpretação resumida", ""])
    for row in rows:
        lines.append(f"- {row['interpretacao']}")

    (output_dir / "resumo_testes_hipoteses.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Ponto de entrada do script."""
    args = parse_args()
    rows = read_dataset(args.input)
    if not rows:
        raise ValueError("O dataset está vazio.")
    run_tests(rows, args.output_dir)


if __name__ == "__main__":
    main()
