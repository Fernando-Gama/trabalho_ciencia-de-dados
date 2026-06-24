"""Treina e avalia modelos para prever resultados de partidas da Copa.

O script usa o dataset derivado com uma linha por partida da Copa do Mundo e
realiza uma tarefa de classificação multiclasse:

- 0: vitória da Seleção A;
- 1: empate;
- 2: vitória da Seleção B.

A avaliação é temporal. As Copas de 1998 a 2010 são usadas para treinamento,
2014 para validação, 2018 para teste final e 2022 como avaliação adicional.
Essa estratégia evita que partidas de Copas futuras influenciem o treinamento.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("dados_processados/dataset_partidas_copa.csv")
DEFAULT_OUTPUT_DIR = Path("modelagem")
RANDOM_STATE = 42

RESULT_LABELS = {
    0: "Vitória A",
    1: "Empate",
    2: "Vitória B",
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
        description="Treina e avalia modelos de classificação para a Copa."
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
        help="Pasta onde métricas, tabelas e gráficos serão salvos.",
    )
    return parser.parse_args()


def import_sklearn() -> dict[str, Any]:
    """Importa classes do scikit-learn com mensagem clara quando faltar.

    Returns:
        Dicionário com classes e funções necessárias.

    Raises:
        SystemExit: Quando scikit-learn não está instalado.
    """
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as error:
        raise SystemExit(
            "Dependência ausente: scikit-learn.\n"
            "Instale com:\n\n"
            "./venv/bin/python -m pip install scikit-learn\n"
        ) from error

    return {
        "DummyClassifier": DummyClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "make_pipeline": make_pipeline,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "StandardScaler": StandardScaler,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def read_dataset(path: Path) -> list[dict[str, str]]:
    """Carrega o dataset processado.

    Args:
        path: Caminho do CSV.

    Returns:
        Lista de linhas como dicionários.
    """
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Cria as pastas de saída da modelagem.

    Args:
        output_dir: Pasta principal da modelagem.

    Returns:
        Tupla com pastas de tabelas e gráficos.
    """
    tables_dir = output_dir / "tabelas"
    charts_dir = output_dir / "graficos"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, charts_dir


def to_float(value: str) -> float | None:
    """Converte texto para float quando possível.

    Args:
        value: Valor textual do CSV.

    Returns:
        Valor numérico ou None.
    """
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except ValueError:
        return None
    if math.isnan(converted):
        return None
    return converted


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Verifica se uma coluna pode ser usada como numérica.

    Args:
        rows: Linhas do dataset.
        column: Nome da coluna.

    Returns:
        True quando todos os valores preenchidos são numéricos.
    """
    filled = [row[column] for row in rows if row.get(column, "") != ""]
    return bool(filled) and all(to_float(value) is not None for value in filled)


def select_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    """Seleciona variáveis explicativas numéricas sem vazamento direto.

    Args:
        rows: Linhas do dataset.

    Returns:
        Lista de colunas usadas pelos modelos.
    """
    columns = list(rows[0].keys())
    blocked = METADATA_COLUMNS | TARGET_COLUMNS
    return [
        column
        for column in columns
        if column not in blocked and is_numeric_column(rows, column)
    ]


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Divide as partidas por edição da Copa.

    Args:
        rows: Linhas do dataset.

    Returns:
        Dicionário com treino, validação, teste e avaliação adicional.
    """
    splits = {
        "treino_1998_2010": [],
        "validacao_2014": [],
        "teste_2018": [],
        "avaliacao_2022": [],
    }
    for row in rows:
        year = int(row["world_cup_year"])
        if 1998 <= year <= 2010:
            splits["treino_1998_2010"].append(row)
        elif year == 2014:
            splits["validacao_2014"].append(row)
        elif year == 2018:
            splits["teste_2018"].append(row)
        elif year == 2022:
            splits["avaliacao_2022"].append(row)
    return splits


def build_matrix(
    rows: list[dict[str, str]], feature_columns: list[str]
) -> tuple[list[list[float]], list[int]]:
    """Monta matriz de atributos e vetor alvo.

    Args:
        rows: Linhas do dataset.
        feature_columns: Colunas explicativas.

    Returns:
        Tupla com X e y.
    """
    x_matrix = []
    y_values = []
    for row in rows:
        x_matrix.append([float(row[column]) for column in feature_columns])
        y_values.append(int(float(row["result"])))
    return x_matrix, y_values


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Salva uma tabela CSV.

    Args:
        path: Caminho do CSV.
        fieldnames: Colunas da tabela.
        rows: Linhas a serem salvas.
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_models(sklearn: dict[str, Any]) -> dict[str, Any]:
    """Cria os modelos avaliados no trabalho.

    Args:
        sklearn: Dicionário com classes importadas do scikit-learn.

    Returns:
        Dicionário nome -> estimador.
    """
    return {
        "baseline_classe_majoritaria": sklearn["DummyClassifier"](
            strategy="most_frequent"
        ),
        "regressao_logistica": sklearn["make_pipeline"](
            sklearn["StandardScaler"](),
            sklearn["LogisticRegression"](
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "arvore_decisao": sklearn["DecisionTreeClassifier"](
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": sklearn["RandomForestClassifier"](
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_predictions(
    sklearn: dict[str, Any],
    model_name: str,
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, object]:
    """Calcula métricas gerais para um modelo.

    Args:
        sklearn: Funções de métricas do scikit-learn.
        model_name: Nome do modelo.
        split_name: Nome da divisão avaliada.
        y_true: Alvos reais.
        y_pred: Predições do modelo.

    Returns:
        Linha com métricas agregadas.
    """
    return {
        "modelo": model_name,
        "conjunto": split_name,
        "accuracy": round(sklearn["accuracy_score"](y_true, y_pred), 4),
        "macro_f1": round(sklearn["f1_score"](y_true, y_pred, average="macro"), 4),
        "quantidade_partidas": len(y_true),
    }


def build_class_metrics(
    sklearn: dict[str, Any],
    model_name: str,
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
) -> list[dict[str, object]]:
    """Calcula precision, recall e F1 por classe.

    Args:
        sklearn: Funções de métricas do scikit-learn.
        model_name: Nome do modelo.
        split_name: Nome da divisão avaliada.
        y_true: Alvos reais.
        y_pred: Predições do modelo.

    Returns:
        Tabela de métricas por classe.
    """
    precision, recall, f1, support = sklearn["precision_recall_fscore_support"](
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    rows = []
    for index, label in enumerate([0, 1, 2]):
        rows.append(
            {
                "modelo": model_name,
                "conjunto": split_name,
                "classe": label,
                "rotulo": RESULT_LABELS[label],
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
        )
    return rows


def build_confusion_rows(
    sklearn: dict[str, Any],
    model_name: str,
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
) -> list[dict[str, object]]:
    """Cria tabela longa da matriz de confusão.

    Args:
        sklearn: Funções de métricas do scikit-learn.
        model_name: Nome do modelo.
        split_name: Nome da divisão avaliada.
        y_true: Alvos reais.
        y_pred: Predições do modelo.

    Returns:
        Matriz de confusão em formato longo.
    """
    matrix = sklearn["confusion_matrix"](y_true, y_pred, labels=[0, 1, 2])
    rows = []
    for true_index, true_label in enumerate([0, 1, 2]):
        for pred_index, pred_label in enumerate([0, 1, 2]):
            rows.append(
                {
                    "modelo": model_name,
                    "conjunto": split_name,
                    "classe_real": true_label,
                    "rotulo_real": RESULT_LABELS[true_label],
                    "classe_prevista": pred_label,
                    "rotulo_previsto": RESULT_LABELS[pred_label],
                    "quantidade": int(matrix[true_index][pred_index]),
                }
            )
    return rows


def get_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Define grupos de atributos para avaliar as hipóteses.

    Args:
        feature_columns: Todas as colunas candidatas.

    Returns:
        Dicionário nome do grupo -> colunas.
    """
    ranking = [
        column
        for column in feature_columns
        if "rank" in column or "fifa_points" in column
    ]
    forma_recente = [
        column for column in feature_columns if "last5" in column or "last10" in column
    ]
    ciclo = [
        column
        for column in feature_columns
        if "cycle" in column or "official" in column or "goals" in column
    ]
    adversarios = [
        column for column in feature_columns if "opponent_points_mean" in column
    ]
    without_adversaries = [
        column for column in feature_columns if column not in adversarios
    ]

    return {
        "h1_ranking": ranking,
        "h2_ranking_forma_recente": sorted(set(ranking + forma_recente)),
        "h3_ranking_forma_ciclo": sorted(set(ranking + forma_recente + ciclo)),
        "h4_completo_sem_forca_adversarios": without_adversaries,
        "modelo_completo": feature_columns,
    }


def fit_and_evaluate_models(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    output_dir: Path,
    tables_dir: Path,
    charts_dir: Path,
    sklearn: dict[str, Any],
) -> dict[str, object]:
    """Treina os modelos principais e salva resultados.

    Args:
        rows: Linhas do dataset.
        feature_columns: Colunas usadas pelos modelos.
        output_dir: Pasta principal da modelagem.
        tables_dir: Pasta de tabelas.
        charts_dir: Pasta de gráficos.
        sklearn: Classes e funções do scikit-learn.

    Returns:
        Dicionário com resumo da modelagem.
    """
    splits = split_rows(rows)
    train_rows = splits["treino_1998_2010"]
    x_train, y_train = build_matrix(train_rows, feature_columns)
    models = build_models(sklearn)
    metric_rows = []
    class_metric_rows = []
    confusion_rows = []
    classification_reports: dict[str, str] = {}
    predictions_rows = []
    fitted_models = {}

    evaluation_splits = {
        "validacao_2014": splits["validacao_2014"],
        "teste_2018": splits["teste_2018"],
        "avaliacao_2022": splits["avaliacao_2022"],
    }

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[model_name] = model
        for split_name, split_data in evaluation_splits.items():
            x_eval, y_eval = build_matrix(split_data, feature_columns)
            y_pred = list(model.predict(x_eval))
            metric_rows.append(
                evaluate_predictions(sklearn, model_name, split_name, y_eval, y_pred)
            )
            class_metric_rows.extend(
                build_class_metrics(sklearn, model_name, split_name, y_eval, y_pred)
            )
            confusion_rows.extend(
                build_confusion_rows(sklearn, model_name, split_name, y_eval, y_pred)
            )
            classification_reports[f"{model_name}_{split_name}"] = sklearn[
                "classification_report"
            ](
                y_eval,
                y_pred,
                labels=[0, 1, 2],
                target_names=[RESULT_LABELS[label] for label in [0, 1, 2]],
                zero_division=0,
            )
            for row, true_value, predicted_value in zip(split_data, y_eval, y_pred):
                predictions_rows.append(
                    {
                        "modelo": model_name,
                        "conjunto": split_name,
                        "world_cup_year": row["world_cup_year"],
                        "match_id": row["match_id"],
                        "team_a": row["team_a"],
                        "team_b": row["team_b"],
                        "result_real": true_value,
                        "rotulo_real": RESULT_LABELS[true_value],
                        "result_previsto": int(predicted_value),
                        "rotulo_previsto": RESULT_LABELS[int(predicted_value)],
                        "acertou": int(true_value == int(predicted_value)),
                    }
                )

    write_csv(
        tables_dir / "metricas_modelos.csv",
        ["modelo", "conjunto", "accuracy", "macro_f1", "quantidade_partidas"],
        metric_rows,
    )
    write_csv(
        tables_dir / "metricas_por_classe.csv",
        [
            "modelo",
            "conjunto",
            "classe",
            "rotulo",
            "precision",
            "recall",
            "f1",
            "support",
        ],
        class_metric_rows,
    )
    write_csv(
        tables_dir / "matrizes_confusao.csv",
        [
            "modelo",
            "conjunto",
            "classe_real",
            "rotulo_real",
            "classe_prevista",
            "rotulo_previsto",
            "quantidade",
        ],
        confusion_rows,
    )
    write_csv(
        tables_dir / "predicoes_modelos.csv",
        [
            "modelo",
            "conjunto",
            "world_cup_year",
            "match_id",
            "team_a",
            "team_b",
            "result_real",
            "rotulo_real",
            "result_previsto",
            "rotulo_previsto",
            "acertou",
        ],
        predictions_rows,
    )

    report_lines = []
    for report_name, report_text in classification_reports.items():
        report_lines.extend([f"## {report_name}", "", "```text", report_text, "```", ""])
    (output_dir / "relatorios_classificacao.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    feature_importance_rows = build_feature_importance_rows(
        fitted_models,
        feature_columns,
    )
    if feature_importance_rows:
        write_csv(
            tables_dir / "importancia_variaveis.csv",
            ["modelo", "variavel", "importancia"],
            feature_importance_rows,
        )

    save_model_charts(
        charts_dir,
        metric_rows,
        confusion_rows,
        feature_importance_rows,
    )

    best_validation = max(
        [row for row in metric_rows if row["conjunto"] == "validacao_2014"],
        key=lambda row: float(row["macro_f1"]),
    )
    return {
        "splits": splits,
        "metric_rows": metric_rows,
        "feature_importance_rows": feature_importance_rows,
        "best_validation": best_validation,
    }


def build_feature_importance_rows(
    fitted_models: dict[str, Any], feature_columns: list[str]
) -> list[dict[str, object]]:
    """Extrai importância de variáveis de modelos baseados em árvores.

    Args:
        fitted_models: Modelos já treinados.
        feature_columns: Colunas usadas no treino.

    Returns:
        Tabela com importâncias ordenadas.
    """
    rows = []
    for model_name in ["arvore_decisao", "random_forest"]:
        model = fitted_models.get(model_name)
        if model is None or not hasattr(model, "feature_importances_"):
            continue
        for feature, importance in zip(feature_columns, model.feature_importances_):
            rows.append(
                {
                    "modelo": model_name,
                    "variavel": feature,
                    "importancia": round(float(importance), 6),
                }
            )
    return sorted(rows, key=lambda row: (str(row["modelo"]), -float(row["importancia"])))


def evaluate_hypotheses(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    tables_dir: Path,
    sklearn: dict[str, Any],
) -> list[dict[str, object]]:
    """Avalia grupos de atributos ligados às hipóteses do trabalho.

    Args:
        rows: Linhas do dataset.
        feature_columns: Colunas candidatas.
        tables_dir: Pasta de tabelas.
        sklearn: Classes e funções do scikit-learn.

    Returns:
        Tabela de métricas por grupo de atributos.
    """
    splits = split_rows(rows)
    groups = get_feature_groups(feature_columns)
    metric_rows = []

    for group_name, columns in groups.items():
        if not columns:
            continue
        x_train, y_train = build_matrix(splits["treino_1998_2010"], columns)
        model = sklearn["make_pipeline"](
            sklearn["StandardScaler"](),
            sklearn["LogisticRegression"](
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(x_train, y_train)
        for split_name in ["validacao_2014", "teste_2018", "avaliacao_2022"]:
            x_eval, y_eval = build_matrix(splits[split_name], columns)
            y_pred = list(model.predict(x_eval))
            metric = evaluate_predictions(
                sklearn,
                group_name,
                split_name,
                y_eval,
                y_pred,
            )
            metric["quantidade_variaveis"] = len(columns)
            metric_rows.append(metric)

    write_csv(
        tables_dir / "avaliacao_hipoteses.csv",
        [
            "modelo",
            "conjunto",
            "accuracy",
            "macro_f1",
            "quantidade_partidas",
            "quantidade_variaveis",
        ],
        metric_rows,
    )
    return metric_rows


def try_import_charts():
    """Importa matplotlib e seaborn quando disponíveis.

    Returns:
        Tupla com pyplot e seaborn. O seaborn pode ser None.
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


def save_model_charts(
    charts_dir: Path,
    metric_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    feature_importance_rows: list[dict[str, object]],
) -> list[str]:
    """Salva gráficos da avaliação dos modelos.

    Args:
        charts_dir: Pasta de gráficos.
        metric_rows: Métricas agregadas.
        confusion_rows: Matrizes de confusão em formato longo.
        feature_importance_rows: Importâncias de variáveis.

    Returns:
        Lista com nomes dos gráficos gerados.
    """
    plt, sns = try_import_charts()
    if plt is None:
        return []

    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    else:
        plt.style.use("ggplot")

    generated = []
    model_order = [
        "baseline_classe_majoritaria",
        "regressao_logistica",
        "arvore_decisao",
        "random_forest",
    ]

    for metric_name in ["accuracy", "macro_f1"]:
        selected = [
            row
            for row in metric_rows
            if row["conjunto"] in {"validacao_2014", "teste_2018"}
        ]
        labels = [f"{row['modelo']}\n{row['conjunto']}" for row in selected]
        values = [float(row[metric_name]) for row in selected]
        plt.figure(figsize=(12, 5))
        if sns is not None:
            axis = sns.barplot(x=labels, y=values, color="#2E86AB")
        else:
            bars = plt.bar(labels, values)
            axis = plt.gca()
            axis.bar_label(bars, padding=3, fmt="%.3f")
        if sns is not None:
            axis.bar_label(axis.containers[0], padding=3, fmt="%.3f")
        plt.title(f"Comparação dos Modelos - {metric_name}")
        plt.ylabel(metric_name)
        plt.xlabel("")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        path = charts_dir / f"comparacao_modelos_{metric_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(path.name)

    best_test_model = max(
        [row for row in metric_rows if row["conjunto"] == "teste_2018"],
        key=lambda row: float(row["macro_f1"]),
    )["modelo"]
    matrix_values = [
        [0 for _ in RESULT_LABELS]
        for _ in RESULT_LABELS
    ]
    for row in confusion_rows:
        if row["modelo"] == best_test_model and row["conjunto"] == "teste_2018":
            matrix_values[int(row["classe_real"])][int(row["classe_prevista"])] = int(
                row["quantidade"]
            )

    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            matrix_values,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[RESULT_LABELS[label] for label in [0, 1, 2]],
            yticklabels=[RESULT_LABELS[label] for label in [0, 1, 2]],
        )
    else:
        axis = plt.imshow(matrix_values, cmap="Blues")
        plt.colorbar(axis)
        plt.xticks(range(3), [RESULT_LABELS[label] for label in [0, 1, 2]])
        plt.yticks(range(3), [RESULT_LABELS[label] for label in [0, 1, 2]])
        for i, row in enumerate(matrix_values):
            for j, value in enumerate(row):
                plt.text(j, i, str(value), ha="center", va="center")
    plt.title(f"Matriz de Confusão - {best_test_model} - Teste 2018")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.tight_layout()
    path = charts_dir / "matriz_confusao_melhor_modelo_2018.png"
    plt.savefig(path, dpi=150)
    plt.close()
    generated.append(path.name)

    rf_importances = [
        row
        for row in feature_importance_rows
        if row["modelo"] == "random_forest"
    ][:12]
    if rf_importances:
        labels = [str(row["variavel"]) for row in reversed(rf_importances)]
        values = [float(row["importancia"]) for row in reversed(rf_importances)]
        plt.figure(figsize=(9, 6))
        if sns is not None:
            axis = sns.barplot(x=values, y=labels, color="#2E86AB")
        else:
            plt.barh(labels, values, color="#2E86AB")
            axis = plt.gca()
        for patch, value in zip(axis.patches, values):
            axis.text(
                value + max(values) * 0.015,
                patch.get_y() + patch.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8,
            )
        axis.set_xlim(0, max(values) * 1.18)
        plt.title("Principais Variáveis - Random Forest")
        plt.xlabel("Importância")
        plt.ylabel("")
        plt.tight_layout()
        path = charts_dir / "importancia_variaveis_random_forest.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated.append(path.name)

    return generated


def write_markdown_summary(
    output_dir: Path,
    feature_columns: list[str],
    model_summary: dict[str, object],
    hypothesis_rows: list[dict[str, object]],
) -> None:
    """Gera resumo em Markdown da modelagem.

    Args:
        output_dir: Pasta principal da modelagem.
        feature_columns: Colunas usadas.
        model_summary: Resumo retornado pela modelagem.
        hypothesis_rows: Métricas dos grupos de hipóteses.
    """
    splits = model_summary["splits"]
    metric_rows = model_summary["metric_rows"]
    best_validation = model_summary["best_validation"]
    best_test = max(
        [row for row in metric_rows if row["conjunto"] == "teste_2018"],
        key=lambda row: float(row["macro_f1"]),
    )
    best_extra = max(
        [row for row in metric_rows if row["conjunto"] == "avaliacao_2022"],
        key=lambda row: float(row["macro_f1"]),
    )
    validation_hypotheses = [
        row for row in hypothesis_rows if row["conjunto"] == "validacao_2014"
    ]

    lines = [
        "# Modelagem e Avaliação",
        "",
        "## Configuração",
        "",
        f"- Variáveis explicativas usadas no modelo completo: `{len(feature_columns)}`",
        f"- Treino: `{len(splits['treino_1998_2010'])}` partidas, Copas 1998-2010",
        f"- Validação: `{len(splits['validacao_2014'])}` partidas, Copa 2014",
        f"- Teste final: `{len(splits['teste_2018'])}` partidas, Copa 2018",
        f"- Avaliação adicional: `{len(splits['avaliacao_2022'])}` partidas, Copa 2022",
        "",
        "Foram removidos do treino os placares `score_a` e `score_b`, o alvo `result`",
        "e metadados como nomes das seleções, datas e identificadores de partidas.",
        "",
        "## Melhor desempenho",
        "",
        "| Conjunto | Modelo | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
        f"| Validação 2014 | {best_validation['modelo']} | {best_validation['accuracy']} | {best_validation['macro_f1']} |",
        f"| Teste 2018 | {best_test['modelo']} | {best_test['accuracy']} | {best_test['macro_f1']} |",
        f"| Avaliação 2022 | {best_extra['modelo']} | {best_extra['accuracy']} | {best_extra['macro_f1']} |",
        "",
        "A métrica principal recomendada é o macro-F1, pois considera o desempenho",
        "médio entre vitória da Seleção A, empate e vitória da Seleção B.",
        "",
        "## Avaliação das hipóteses",
        "",
        "| Grupo de atributos | Accuracy validação | Macro-F1 validação | Variáveis |",
        "|---|---:|---:|---:|",
    ]

    for row in validation_hypotheses:
        lines.append(
            f"| {row['modelo']} | {row['accuracy']} | {row['macro_f1']} | "
            f"{row['quantidade_variaveis']} |"
        )

    lines.extend(
        [
            "",
            "## Arquivos gerados",
            "",
            "- `tabelas/metricas_modelos.csv`: comparação dos modelos;",
            "- `tabelas/metricas_por_classe.csv`: precision, recall e F1 por classe;",
            "- `tabelas/matrizes_confusao.csv`: matrizes de confusão em formato longo;",
            "- `tabelas/predicoes_modelos.csv`: predições por partida;",
            "- `tabelas/avaliacao_hipoteses.csv`: avaliação dos grupos de atributos;",
            "- `tabelas/importancia_variaveis.csv`: importância das variáveis em árvores;",
            "- `graficos/`: gráficos da avaliação, quando matplotlib está instalado.",
        ]
    )

    (output_dir / "resumo_modelagem.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_feature_list(tables_dir: Path, feature_columns: list[str]) -> None:
    """Salva a lista de variáveis usadas nos modelos.

    Args:
        tables_dir: Pasta de tabelas.
        feature_columns: Colunas explicativas.
    """
    rows = [{"variavel": column} for column in feature_columns]
    write_csv(tables_dir / "variaveis_modelo.csv", ["variavel"], rows)


def write_class_distribution(tables_dir: Path, rows: list[dict[str, str]]) -> None:
    """Salva a distribuição do alvo por divisão temporal.

    Args:
        tables_dir: Pasta de tabelas.
        rows: Linhas do dataset.
    """
    split_data = split_rows(rows)
    output_rows = []
    for split_name, split_rows_data in split_data.items():
        counts = Counter(int(row["result"]) for row in split_rows_data)
        total = sum(counts.values())
        for label in [0, 1, 2]:
            output_rows.append(
                {
                    "conjunto": split_name,
                    "classe": label,
                    "rotulo": RESULT_LABELS[label],
                    "quantidade": counts[label],
                    "percentual": round(100 * counts[label] / total, 2),
                }
            )
    write_csv(
        tables_dir / "distribuicao_alvo_por_conjunto.csv",
        ["conjunto", "classe", "rotulo", "quantidade", "percentual"],
        output_rows,
    )


def run_modeling(input_path: Path, output_dir: Path) -> None:
    """Executa o fluxo completo de modelagem.

    Args:
        input_path: Caminho do dataset processado.
        output_dir: Pasta de saída.
    """
    sklearn = import_sklearn()
    rows = read_dataset(input_path)
    if not rows:
        raise ValueError("O dataset está vazio.")

    tables_dir, charts_dir = ensure_output_dirs(output_dir)
    feature_columns = select_feature_columns(rows)

    write_feature_list(tables_dir, feature_columns)
    write_class_distribution(tables_dir, rows)

    model_summary = fit_and_evaluate_models(
        rows,
        feature_columns,
        output_dir,
        tables_dir,
        charts_dir,
        sklearn,
    )
    hypothesis_rows = evaluate_hypotheses(rows, feature_columns, tables_dir, sklearn)
    write_markdown_summary(output_dir, feature_columns, model_summary, hypothesis_rows)

    print(f"Modelagem concluída em: {output_dir}")
    print(f"Tabelas geradas em: {tables_dir}")
    print(f"Gráficos gerados em: {charts_dir}")


def main() -> None:
    """Ponto de entrada do script."""
    args = parse_args()
    run_modeling(args.input, args.output_dir)


if __name__ == "__main__":
    main()
