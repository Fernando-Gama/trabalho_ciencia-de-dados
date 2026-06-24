"""Executa a análise exploratória do dataset derivado da Copa do Mundo.

O script lê o arquivo processado com uma linha por partida de Copa e gera
artefatos úteis para o relatório: um resumo em Markdown, tabelas CSV e, quando
o pacote matplotlib está instalado, gráficos em PNG.

As análises são descritivas e não treinam modelos. O objetivo é entender a base,
verificar a distribuição do alvo e observar relações iniciais entre ranking,
forma pré-Copa, desempenho no ciclo e resultado das partidas.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable


DEFAULT_INPUT = Path("dados_processados/dataset_partidas_copa.csv")
DEFAULT_OUTPUT_DIR = Path("analises")

RESULT_LABELS = {
    "0": "Vitória da Seleção A",
    "1": "Empate",
    "2": "Vitória da Seleção B",
}

RESULT_SHORT_LABELS = {
    "0": "Vitoria A",
    "1": "Empate",
    "2": "Vitoria B",
}

METADATA_COLUMNS = {
    "match_id",
    "world_cup_year",
    "match_date",
    "cycle_start",
    "feature_cutoff",
    "ranking_year",
    "ranking_semester",
    "team_a",
    "team_b",
    "country",
}

TARGET_COLUMNS = {"score_a", "score_b", "result"}


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando.

    Returns:
        Namespace com caminho de entrada e pasta de saída.
    """
    parser = argparse.ArgumentParser(
        description="Gera análise exploratória do dataset de partidas da Copa."
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
        help="Pasta onde os artefatos da análise serão salvos.",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> list[dict[str, str]]:
    """Carrega o CSV processado.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        Lista de linhas representadas como dicionários.

    Raises:
        FileNotFoundError: Quando o dataset não existe.
    """
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Cria as pastas de saída da análise.

    Args:
        output_dir: Pasta principal dos artefatos.

    Returns:
        Tupla com os caminhos das pastas de tabelas e gráficos.
    """
    tables_dir = output_dir / "tabelas"
    charts_dir = output_dir / "graficos"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, charts_dir


def to_float(value: str) -> float | None:
    """Converte um valor textual para float quando possível.

    Args:
        value: Valor lido do CSV.

    Returns:
        Valor numérico ou None quando a conversão não é possível.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def numeric_values(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    """Extrai valores numéricos válidos de uma coluna.

    Args:
        rows: Linhas do dataset.
        column: Nome da coluna.

    Returns:
        Lista de valores numéricos.
    """
    values = []
    for row in rows:
        value = to_float(row.get(column, ""))
        if value is not None and not math.isnan(value):
            values.append(value)
    return values


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Verifica se uma coluna pode ser tratada como numérica.

    Args:
        rows: Linhas do dataset.
        column: Nome da coluna.

    Returns:
        True quando todos os valores preenchidos são numéricos.
    """
    filled = [row.get(column, "") for row in rows if row.get(column, "") != ""]
    if not filled:
        return False
    return all(to_float(value) is not None for value in filled)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Salva uma tabela em CSV.

    Args:
        path: Caminho do CSV de saída.
        fieldnames: Ordem das colunas.
        rows: Linhas da tabela.
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentage(part: int, total: int) -> float:
    """Calcula percentual com proteção contra divisão por zero.

    Args:
        part: Numerador.
        total: Denominador.

    Returns:
        Percentual entre 0 e 100.
    """
    if total == 0:
        return 0.0
    return 100.0 * part / total


def signed_result(result: str) -> int:
    """Converte o alvo para escala direcional.

    Args:
        result: Classe original do alvo.

    Returns:
        1 para vitória de A, 0 para empate e -1 para vitória de B.
    """
    if result == "0":
        return 1
    if result == "2":
        return -1
    return 0


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    """Calcula correlação de Pearson entre duas listas.

    Args:
        x_values: Primeira variável.
        y_values: Segunda variável.

    Returns:
        Coeficiente de correlação ou None quando não há variação suficiente.
    """
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    denominator = x_denominator * y_denominator
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_dataset(rows: list[dict[str, str]]) -> dict[str, object]:
    """Calcula informações gerais do dataset.

    Args:
        rows: Linhas do dataset.

    Returns:
        Dicionário com dimensões e informações gerais.
    """
    columns = list(rows[0].keys()) if rows else []
    world_cups = sorted({row["world_cup_year"] for row in rows})
    teams = sorted({row["team_a"] for row in rows} | {row["team_b"] for row in rows})
    return {
        "rows": len(rows),
        "columns": len(columns),
        "world_cups": world_cups,
        "teams": teams,
        "feature_columns": [
            column
            for column in columns
            if column not in METADATA_COLUMNS and column not in TARGET_COLUMNS
        ],
    }


def build_missing_table(rows: list[dict[str, str]], columns: list[str]) -> list[dict[str, object]]:
    """Monta tabela de valores ausentes por coluna.

    Args:
        rows: Linhas do dataset.
        columns: Colunas do dataset.

    Returns:
        Tabela com contagem e percentual de ausentes.
    """
    total = len(rows)
    table = []
    for column in columns:
        missing = sum(1 for row in rows if row.get(column, "") == "")
        table.append(
            {
                "coluna": column,
                "ausentes": missing,
                "percentual_ausente": round(percentage(missing, total), 2),
            }
        )
    return table


def build_result_distribution(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Monta a distribuição geral do alvo.

    Args:
        rows: Linhas do dataset.

    Returns:
        Tabela com frequência de cada classe.
    """
    total = len(rows)
    counts = Counter(row["result"] for row in rows)
    return [
        {
            "classe": result,
            "rotulo": RESULT_LABELS[result],
            "quantidade": counts[result],
            "percentual": round(percentage(counts[result], total), 2),
        }
        for result in sorted(RESULT_LABELS)
    ]


def build_result_by_cup(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Calcula a distribuição do alvo por edição da Copa.

    Args:
        rows: Linhas do dataset.

    Returns:
        Tabela com contagens por Copa e classe.
    """
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["world_cup_year"]][row["result"]] += 1

    table = []
    for year in sorted(grouped):
        total = sum(grouped[year].values())
        table.append(
            {
                "world_cup_year": year,
                "vitorias_a": grouped[year]["0"],
                "empates": grouped[year]["1"],
                "vitorias_b": grouped[year]["2"],
                "total": total,
                "percentual_empates": round(percentage(grouped[year]["1"], total), 2),
            }
        )
    return table


def build_numeric_summary(
    rows: list[dict[str, str]], columns: list[str]
) -> list[dict[str, object]]:
    """Calcula estatísticas descritivas das colunas numéricas.

    Args:
        rows: Linhas do dataset.
        columns: Colunas do dataset.

    Returns:
        Tabela com média, desvio, mínimo, mediana e máximo.
    """
    table = []
    for column in columns:
        values = numeric_values(rows, column)
        if not values:
            continue
        table.append(
            {
                "coluna": column,
                "quantidade": len(values),
                "media": round(mean(values), 4),
                "desvio_padrao": round(pstdev(values), 4),
                "minimo": round(min(values), 4),
                "mediana": round(median(values), 4),
                "maximo": round(max(values), 4),
            }
        )
    return table


def build_team_performance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Agrega o desempenho das seleções nas partidas da Copa.

    Args:
        rows: Linhas do dataset.

    Returns:
        Tabela com jogos, vitórias, empates, derrotas e gols.
    """
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        team_a = row["team_a"]
        team_b = row["team_b"]
        score_a = int(float(row["score_a"]))
        score_b = int(float(row["score_b"]))

        stats[team_a]["jogos"] += 1
        stats[team_b]["jogos"] += 1
        stats[team_a]["gols_pro"] += score_a
        stats[team_a]["gols_contra"] += score_b
        stats[team_b]["gols_pro"] += score_b
        stats[team_b]["gols_contra"] += score_a

        if score_a > score_b:
            stats[team_a]["vitorias"] += 1
            stats[team_b]["derrotas"] += 1
        elif score_a < score_b:
            stats[team_b]["vitorias"] += 1
            stats[team_a]["derrotas"] += 1
        else:
            stats[team_a]["empates"] += 1
            stats[team_b]["empates"] += 1

    table = []
    for team, counter in stats.items():
        games = counter["jogos"]
        points = 3 * counter["vitorias"] + counter["empates"]
        table.append(
            {
                "selecao": team,
                "jogos": games,
                "vitorias": counter["vitorias"],
                "empates": counter["empates"],
                "derrotas": counter["derrotas"],
                "gols_pro": counter["gols_pro"],
                "gols_contra": counter["gols_contra"],
                "saldo_gols": counter["gols_pro"] - counter["gols_contra"],
                "pontos": points,
                "pontos_por_jogo": round(points / games, 3),
            }
        )

    return sorted(
        table,
        key=lambda row: (
            -float(row["pontos"]),
            -float(row["saldo_gols"]),
            -float(row["gols_pro"]),
            str(row["selecao"]),
        ),
    )


def build_diff_rank_buckets(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Agrupa resultados por faixas de diferença de ranking.

    Args:
        rows: Linhas do dataset.

    Returns:
        Tabela com distribuição do alvo por faixa de diff_rank.
    """
    buckets = [
        ("A muito melhor", float("-inf"), -30),
        ("A melhor", -30, -10),
        ("Equilibrado", -10, 10),
        ("B melhor", 10, 30),
        ("B muito melhor", 30, float("inf")),
    ]
    grouped: dict[str, Counter[str]] = {label: Counter() for label, _, _ in buckets}

    for row in rows:
        diff_rank = to_float(row["diff_rank"])
        if diff_rank is None:
            continue
        for label, lower, upper in buckets:
            if lower <= diff_rank < upper:
                grouped[label][row["result"]] += 1
                break

    table = []
    for label, _, _ in buckets:
        counter = grouped[label]
        total = sum(counter.values())
        table.append(
            {
                "faixa_diff_rank": label,
                "vitorias_a": counter["0"],
                "empates": counter["1"],
                "vitorias_b": counter["2"],
                "total": total,
                "percentual_vitoria_a": round(percentage(counter["0"], total), 2),
                "percentual_empate": round(percentage(counter["1"], total), 2),
                "percentual_vitoria_b": round(percentage(counter["2"], total), 2),
            }
        )
    return table


def build_feature_correlations(
    rows: list[dict[str, str]], columns: list[str]
) -> list[dict[str, object]]:
    """Calcula correlação entre atributos numéricos e resultado direcional.

    Args:
        rows: Linhas do dataset.
        columns: Colunas numéricas candidatas.

    Returns:
        Tabela ordenada pela maior correlação absoluta.
    """
    table = []
    for column in columns:
        x_values = []
        y_values = []
        for row in rows:
            value = to_float(row.get(column, ""))
            if value is None:
                continue
            x_values.append(value)
            y_values.append(float(signed_result(row["result"])))
        correlation = pearson_correlation(x_values, y_values)
        if correlation is None:
            continue
        table.append(
            {
                "coluna": column,
                "correlacao_resultado_direcional": round(correlation, 4),
                "correlacao_absoluta": round(abs(correlation), 4),
            }
        )
    return sorted(table, key=lambda row: -float(row["correlacao_absoluta"]))


def build_host_table(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Resume o desempenho quando a Seleção A é anfitriã.

    Args:
        rows: Linhas do dataset.

    Returns:
        Tabela comparando jogos com e sem anfitrião na posição A.
    """
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = "A anfitria" if row["team_a_is_host"] == "1" else "A nao anfitria"
        grouped[key][row["result"]] += 1

    table = []
    for key in ["A anfitria", "A nao anfitria"]:
        counter = grouped[key]
        total = sum(counter.values())
        table.append(
            {
                "grupo": key,
                "vitorias_a": counter["0"],
                "empates": counter["1"],
                "vitorias_b": counter["2"],
                "total": total,
                "percentual_vitoria_a": round(percentage(counter["0"], total), 2),
            }
        )
    return table


def try_import_chart_libraries():
    """Importa bibliotecas de gráficos quando disponíveis.

    Returns:
        Tupla com pyplot e seaborn. O segundo item pode ser None quando seaborn
        não está instalado.
    """
    cache_dir = Path(".matplotlib-cache")
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return None, None

    try:
        import seaborn as sns  # type: ignore
    except ImportError:
        sns = None

    return plt, sns


def annotate_line_points(axis, x_labels: list[str], series: list[list[float]]) -> None:
    """Adiciona valores percentuais nos pontos de um gráfico de linhas.

    Args:
        axis: Eixo do matplotlib.
        x_labels: Rótulos do eixo x.
        series: Listas com os valores de cada linha.
    """
    offsets = [8, -14, 8]
    for series_index, values in enumerate(series):
        offset = offsets[series_index % len(offsets)]
        vertical_alignment = "bottom" if offset > 0 else "top"
        for x_index, value in enumerate(values):
            axis.annotate(
                f"{value:.1f}%",
                (x_index, value),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                va=vertical_alignment,
                fontsize=8,
            )
    axis.set_ylim(0, 80)


def annotate_horizontal_bars(axis, values: list[float], decimal_places: int = 0) -> None:
    """Adiciona rótulos ao final de barras horizontais.

    Args:
        axis: Eixo do matplotlib.
        values: Valores representados pelas barras.
        decimal_places: Quantidade de casas decimais exibidas.
    """
    if not values:
        return

    max_abs_value = max(abs(value) for value in values) or 1
    offset = max_abs_value * 0.015
    for patch, value in zip(axis.patches, values):
        x_position = value + offset if value >= 0 else value - offset
        horizontal_alignment = "left" if value >= 0 else "right"
        axis.text(
            x_position,
            patch.get_y() + patch.get_height() / 2,
            f"{value:.{decimal_places}f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
        )

    axis.set_xlim(
        min(0, min(values)) - max_abs_value * 0.18,
        max(0, max(values)) + max_abs_value * 0.18,
    )


def save_charts(
    charts_dir: Path,
    result_distribution: list[dict[str, object]],
    result_by_cup: list[dict[str, object]],
    rank_buckets: list[dict[str, object]],
    correlations: list[dict[str, object]],
    team_performance: list[dict[str, object]],
) -> list[str]:
    """Gera gráficos quando matplotlib está disponível.

    Args:
        charts_dir: Pasta de saída dos gráficos.
        result_distribution: Distribuição geral do alvo.
        result_by_cup: Distribuição por edição.
        rank_buckets: Distribuição por faixas de ranking.
        correlations: Correlações com o resultado direcional.
        team_performance: Desempenho agregado das seleções.

    Returns:
        Lista com nomes dos arquivos gerados.
    """
    plt, sns = try_import_chart_libraries()
    if plt is None:
        return []

    generated = []
    palette = ["#2E86AB", "#F6C85F", "#6F4E7C"]
    single_color = "#2E86AB"
    diverging_color = "#C44E52"

    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            palette=palette,
            font_scale=1.05,
        )
    else:
        plt.style.use("ggplot")

    labels = [str(row["rotulo"]) for row in result_distribution]
    values = [int(row["quantidade"]) for row in result_distribution]
    plt.figure(figsize=(8, 5))
    if sns is not None:
        axis = sns.barplot(x=labels, y=values, hue=labels, palette=palette, legend=False)
        for container in axis.containers:
            axis.bar_label(
                container,
                label_type="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
    else:
        bars = plt.bar(labels, values, color=palette)
        plt.bar_label(
            bars,
            label_type="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("Distribuição do Resultado")
    plt.ylabel("Quantidade de partidas")
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = charts_dir / "distribuicao_resultado.png"
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path.name)

    years = [str(row["world_cup_year"]) for row in result_by_cup]
    wins_a = [int(row["vitorias_a"]) for row in result_by_cup]
    draws = [int(row["empates"]) for row in result_by_cup]
    wins_b = [int(row["vitorias_b"]) for row in result_by_cup]
    plt.figure(figsize=(9, 5))
    bars_a = plt.bar(years, wins_a, label="Vitoria A", color=palette[0])
    bars_draw = plt.bar(years, draws, bottom=wins_a, label="Empate", color=palette[1])
    bottom_b = [wins_a[index] + draws[index] for index in range(len(years))]
    bars_b = plt.bar(years, wins_b, bottom=bottom_b, label="Vitoria B", color=palette[2])
    for bars in (bars_a, bars_draw, bars_b):
        plt.bar_label(
            bars,
            label_type="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    plt.title("Resultado por Copa")
    plt.ylabel("Quantidade de partidas")
    plt.xlabel("Ano da Copa")
    plt.legend(framealpha=0.65, facecolor="white", edgecolor="#CCCCCC")
    plt.tight_layout()
    path = charts_dir / "resultado_por_copa.png"
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path.name)

    bucket_labels = [str(row["faixa_diff_rank"]) for row in rank_buckets]
    bucket_win_a = [float(row["percentual_vitoria_a"]) for row in rank_buckets]
    bucket_draw = [float(row["percentual_empate"]) for row in rank_buckets]
    bucket_win_b = [float(row["percentual_vitoria_b"]) for row in rank_buckets]
    plt.figure(figsize=(10, 5))
    if sns is not None:
        axis = sns.lineplot(x=bucket_labels, y=bucket_win_a, marker="o", label="Vitoria A")
        sns.lineplot(x=bucket_labels, y=bucket_draw, marker="o", label="Empate")
        sns.lineplot(x=bucket_labels, y=bucket_win_b, marker="o", label="Vitoria B")
    else:
        plt.plot(bucket_labels, bucket_win_a, marker="o", label="Vitoria A")
        plt.plot(bucket_labels, bucket_draw, marker="o", label="Empate")
        plt.plot(bucket_labels, bucket_win_b, marker="o", label="Vitoria B")
        axis = plt.gca()
    annotate_line_points(axis, bucket_labels, [bucket_win_a, bucket_draw, bucket_win_b])
    plt.title("Resultado por Faixa de Diferença de Ranking")
    plt.ylabel("Percentual")
    plt.xlabel("Faixa de diferença de ranking")
    plt.xticks(rotation=15, ha="right")
    plt.legend(framealpha=0.65, facecolor="white", edgecolor="#CCCCCC")
    plt.figtext(
        0.5,
        0.01,
        "diff_rank = ranking da Seleção A - ranking da Seleção B. "
        "A muito melhor: < -30; A melhor: -30 a -10; Equilibrado: -10 a 10; "
        "B melhor: 10 a 30; B muito melhor: > 30.",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    path = charts_dir / "resultado_por_faixa_ranking.png"
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path.name)

    top_correlations = correlations[:12]
    corr_labels = [str(row["coluna"]) for row in reversed(top_correlations)]
    corr_values = [
        float(row["correlacao_resultado_direcional"])
        for row in reversed(top_correlations)
    ]
    plt.figure(figsize=(9, 6))
    colors = [single_color if value >= 0 else diverging_color for value in corr_values]
    if sns is not None:
        axis = sns.barplot(
            x=corr_values,
            y=corr_labels,
            hue=corr_labels,
            palette=colors,
            legend=False,
        )
    else:
        plt.barh(corr_labels, corr_values, color=colors)
        axis = plt.gca()
    annotate_horizontal_bars(axis, corr_values, decimal_places=3)
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.title("Maiores Correlações com o Resultado Direcional")
    plt.xlabel("Correlação de Pearson")
    plt.ylabel("")
    plt.tight_layout()
    path = charts_dir / "correlacoes_resultado.png"
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path.name)

    top_teams = team_performance[:12]
    team_labels = [str(row["selecao"]) for row in reversed(top_teams)]
    team_points = [float(row["pontos"]) for row in reversed(top_teams)]
    plt.figure(figsize=(9, 6))
    if sns is not None:
        axis = sns.barplot(x=team_points, y=team_labels, color=single_color)
    else:
        plt.barh(team_labels, team_points, color=single_color)
        axis = plt.gca()
    annotate_horizontal_bars(axis, team_points, decimal_places=0)
    plt.title("Seleções com Mais Pontos nas Copas do Dataset")
    plt.xlabel("Pontos")
    plt.ylabel("")
    plt.tight_layout()
    path = charts_dir / "top_selecoes_pontos.png"
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path.name)

    return generated


def write_markdown_summary(
    path: Path,
    dataset_summary: dict[str, object],
    result_distribution: list[dict[str, object]],
    result_by_cup: list[dict[str, object]],
    rank_buckets: list[dict[str, object]],
    correlations: list[dict[str, object]],
    team_performance: list[dict[str, object]],
    generated_charts: list[str],
) -> None:
    """Salva o resumo interpretativo da análise exploratória.

    Args:
        path: Caminho do Markdown de saída.
        dataset_summary: Informações gerais da base.
        result_distribution: Distribuição geral do alvo.
        result_by_cup: Distribuição por Copa.
        rank_buckets: Resultado por faixas de ranking.
        correlations: Correlações de atributos.
        team_performance: Desempenho das seleções.
        generated_charts: Nomes dos gráficos gerados.
    """
    top_correlations = correlations[:8]
    top_teams = team_performance[:10]
    world_cups = ", ".join(str(year) for year in dataset_summary["world_cups"])

    lines = [
        "# Análise Exploratória do Dataset",
        "",
        "## Visão geral",
        "",
        f"- Linhas: `{dataset_summary['rows']}`",
        f"- Colunas: `{dataset_summary['columns']}`",
        f"- Copas analisadas: `{world_cups}`",
        f"- Seleções presentes: `{len(dataset_summary['teams'])}`",
        f"- Variáveis explicativas candidatas: `{len(dataset_summary['feature_columns'])}`",
        "",
        "Cada linha representa uma partida da Copa do Mundo. As variáveis explicativas",
        "foram calculadas com dados anteriores ao início da Copa correspondente, e o",
        "alvo `result` indica vitória da Seleção A, empate ou vitória da Seleção B.",
        "",
        "## Distribuição do alvo",
        "",
        "| Classe | Rótulo | Quantidade | Percentual |",
        "|---|---|---:|---:|",
    ]

    for row in result_distribution:
        lines.append(
            f"| {row['classe']} | {row['rotulo']} | {row['quantidade']} | "
            f"{row['percentual']}% |"
        )

    lines.extend(
        [
            "",
            "A classe de empate tende a ser menos frequente que as vitórias, por isso a",
            "avaliação dos modelos deve usar macro-F1 além da accuracy.",
            "",
            "## Partidas por Copa",
            "",
            "| Copa | Vitória A | Empate | Vitória B | Total | % Empates |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for row in result_by_cup:
        lines.append(
            f"| {row['world_cup_year']} | {row['vitorias_a']} | {row['empates']} | "
            f"{row['vitorias_b']} | {row['total']} | {row['percentual_empates']}% |"
        )

    lines.extend(
        [
            "",
            "## Ranking FIFA e resultado",
            "",
            "`diff_rank` é calculado como ranking da Seleção A menos ranking da Seleção B.",
            "Valores negativos indicam que A estava melhor posicionada no ranking.",
            "",
            "| Faixa de ranking | Vitória A | Empate | Vitória B | Total | % Vitória A | % Vitória B |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in rank_buckets:
        lines.append(
            f"| {row['faixa_diff_rank']} | {row['vitorias_a']} | {row['empates']} | "
            f"{row['vitorias_b']} | {row['total']} | {row['percentual_vitoria_a']}% | "
            f"{row['percentual_vitoria_b']}% |"
        )

    lines.extend(
        [
            "",
            "## Atributos mais associados ao resultado",
            "",
            "A correlação abaixo usa uma versão direcional do resultado: `1` para vitória",
            "da Seleção A, `0` para empate e `-1` para vitória da Seleção B. Ela não prova",
            "causalidade, mas ajuda a identificar atributos promissores para a modelagem.",
            "",
            "| Atributo | Correlação |",
            "|---|---:|",
        ]
    )

    for row in top_correlations:
        lines.append(
            f"| {row['coluna']} | {row['correlacao_resultado_direcional']} |"
        )

    lines.extend(
        [
            "",
            "## Seleções com melhor desempenho no recorte",
            "",
            "| Seleção | Jogos | Vitórias | Empates | Derrotas | Pontos | Pontos/Jogo | Saldo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in top_teams:
        lines.append(
            f"| {row['selecao']} | {row['jogos']} | {row['vitorias']} | "
            f"{row['empates']} | {row['derrotas']} | {row['pontos']} | "
            f"{row['pontos_por_jogo']} | {row['saldo_gols']} |"
        )

    lines.extend(
        [
            "",
            "## Arquivos gerados",
            "",
            "As tabelas completas foram salvas em `analises/tabelas/`.",
        ]
    )

    if generated_charts:
        lines.append("Os gráficos foram salvos em `analises/graficos/`:")
        for chart in generated_charts:
            lines.append(f"- `{chart}`")
    else:
        lines.extend(
            [
                "Nenhum gráfico PNG foi gerado porque `matplotlib` não está instalado no",
                "ambiente atual. As tabelas e o resumo em Markdown foram gerados normalmente.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(input_path: Path, output_dir: Path) -> None:
    """Executa todo o fluxo de análise exploratória.

    Args:
        input_path: Caminho do dataset processado.
        output_dir: Pasta onde serão salvos os resultados.
    """
    rows = read_dataset(input_path)
    if not rows:
        raise ValueError("O dataset está vazio.")

    tables_dir, charts_dir = ensure_output_dirs(output_dir)
    columns = list(rows[0].keys())
    numeric_columns = [column for column in columns if is_numeric_column(rows, column)]
    feature_numeric_columns = [
        column
        for column in numeric_columns
        if column not in METADATA_COLUMNS and column not in TARGET_COLUMNS
    ]

    dataset_summary = summarize_dataset(rows)
    missing_table = build_missing_table(rows, columns)
    result_distribution = build_result_distribution(rows)
    result_by_cup = build_result_by_cup(rows)
    numeric_summary = build_numeric_summary(rows, numeric_columns)
    team_performance = build_team_performance(rows)
    rank_buckets = build_diff_rank_buckets(rows)
    feature_correlations = build_feature_correlations(rows, feature_numeric_columns)
    host_table = build_host_table(rows)

    write_csv(
        tables_dir / "valores_ausentes.csv",
        ["coluna", "ausentes", "percentual_ausente"],
        missing_table,
    )
    write_csv(
        tables_dir / "distribuicao_resultado.csv",
        ["classe", "rotulo", "quantidade", "percentual"],
        result_distribution,
    )
    write_csv(
        tables_dir / "resultado_por_copa.csv",
        [
            "world_cup_year",
            "vitorias_a",
            "empates",
            "vitorias_b",
            "total",
            "percentual_empates",
        ],
        result_by_cup,
    )
    write_csv(
        tables_dir / "estatisticas_numericas.csv",
        ["coluna", "quantidade", "media", "desvio_padrao", "minimo", "mediana", "maximo"],
        numeric_summary,
    )
    write_csv(
        tables_dir / "desempenho_selecoes.csv",
        [
            "selecao",
            "jogos",
            "vitorias",
            "empates",
            "derrotas",
            "gols_pro",
            "gols_contra",
            "saldo_gols",
            "pontos",
            "pontos_por_jogo",
        ],
        team_performance,
    )
    write_csv(
        tables_dir / "resultado_por_faixa_ranking.csv",
        [
            "faixa_diff_rank",
            "vitorias_a",
            "empates",
            "vitorias_b",
            "total",
            "percentual_vitoria_a",
            "percentual_empate",
            "percentual_vitoria_b",
        ],
        rank_buckets,
    )
    write_csv(
        tables_dir / "correlacoes_resultado.csv",
        ["coluna", "correlacao_resultado_direcional", "correlacao_absoluta"],
        feature_correlations,
    )
    write_csv(
        tables_dir / "desempenho_anfitriao.csv",
        ["grupo", "vitorias_a", "empates", "vitorias_b", "total", "percentual_vitoria_a"],
        host_table,
    )

    generated_charts = save_charts(
        charts_dir,
        result_distribution,
        result_by_cup,
        rank_buckets,
        feature_correlations,
        team_performance,
    )
    write_markdown_summary(
        output_dir / "resumo_analise_exploratoria.md",
        dataset_summary,
        result_distribution,
        result_by_cup,
        rank_buckets,
        feature_correlations,
        team_performance,
        generated_charts,
    )

    print(f"Análise exploratória concluída em: {output_dir}")
    print(f"Tabelas geradas em: {tables_dir}")
    if generated_charts:
        print(f"Gráficos gerados em: {charts_dir}")
    else:
        print("Gráficos não gerados: matplotlib não está instalado.")


def main() -> None:
    """Ponto de entrada do script."""
    args = parse_args()
    run_eda(args.input, args.output_dir)


if __name__ == "__main__":
    main()
