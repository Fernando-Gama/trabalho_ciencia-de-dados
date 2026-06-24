"""Constrói o dataset de partidas de Copa do Mundo a partir dos dados originais.

O script combina resultados internacionais e rankings semestrais da FIFA. Cada
linha produzida representa uma partida de Copa do Mundo. As variáveis explicativas
são calculadas somente com informações disponíveis antes do início da respectiva
Copa, evitando vazamento temporal.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_RESULTS = Path("dados_originais/resultados_1972_2026/results.csv")
DEFAULT_RANKING = Path(
    "dados_originais/ranking_semestre1992_ate_2024/fifa_mens_rank.csv"
)
DEFAULT_OUTPUT = Path("dados_processados/dataset_partidas_copa.csv")
WORLD_CUP = "FIFA World Cup"
FRIENDLY = "Friendly"

COMPACT_TEAM_FEATURES = (
    "rank",
    "fifa_points",
    "days_since_last_match",
    "cycle_games",
    "cycle_points_per_game",
    "cycle_goals_for_per_game",
    "cycle_goals_against_per_game",
    "official_points_per_game",
    "last5_points_per_game",
    "last10_points_per_game",
    "last10_opponent_points_mean",
)

COMPACT_DIFFERENCES = (
    "diff_rank",
    "diff_fifa_points",
    "diff_cycle_points_per_game",
    "diff_cycle_goals_for_per_game",
    "diff_cycle_goals_against_per_game",
    "diff_last5_points_per_game",
    "diff_last10_points_per_game",
    "diff_last10_opponent_points_mean",
)

# Os resultados usam nomes diferentes daqueles adotados no ranking da FIFA.
# A lista e a ordem dos candidatos são explícitas para impedir associações fuzzy.
RANKING_NAME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Brunei": ("Brunei Darussalam",),
    "Cape Verde": ("Cabo Verde", "Cape Verde Islands"),
    "China": ("China PR",),
    "DR Congo": ("Congo DR",),
    "Iran": ("IR Iran",),
    "Ivory Coast": ("Côte d'Ivoire",),
    "Kyrgyzstan": ("Kyrgyz Republic",),
    "North Korea": ("Korea DPR",),
    "Serbia": ("Yugoslavia", "Serbia and Montenegro"),
    "South Korea": ("Korea Republic",),
    "Taiwan": ("Chinese Taipei",),
    "United States": ("USA",),
    "United States Virgin Islands": ("US Virgin Islands",),
}

RESULT_NAME_ALIASES: dict[str, str] = {
    "China PR": "China",
    "Congo DR": "DR Congo",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "USA": "United States",
}


@dataclass(frozen=True)
class Match:
    """Representa uma partida internacional.

    Attributes:
        match_date: Data da partida.
        home_team: Seleção registrada como mandante.
        away_team: Seleção registrada como visitante.
        home_score: Gols do mandante, ou ``None`` se ainda não houver placar.
        away_score: Gols do visitante, ou ``None`` se ainda não houver placar.
        tournament: Nome da competição.
        country: País em que a partida foi disputada.
        neutral: Indica campo neutro.
    """

    match_date: date
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    tournament: str
    country: str
    neutral: bool


@dataclass(frozen=True)
class RankingEntry:
    """Contém uma observação semestral do ranking FIFA.

    Attributes:
        rank: Posição da seleção no período.
        points: Pontos FIFA no período.
    """

    rank: int
    points: float


class RankingIndex:
    """Indexa e consulta rankings usando sempre o semestre anterior."""

    def __init__(self, rows: Iterable[dict[str, str]]) -> None:
        """Cria o índice e ignora duplicações exatas do arquivo de ranking.

        Args:
            rows: Linhas lidas do CSV de ranking.

        Raises:
            ValueError: Se duas linhas diferentes definirem a mesma seleção e
                período.
        """
        self._entries: dict[tuple[int, int, str], RankingEntry] = {}
        for row in rows:
            key = (int(row["date"]), int(row["semester"]), row["team"])
            entry = RankingEntry(
                rank=int(row["rank"]), points=float(row["total.points"])
            )
            previous = self._entries.get(key)
            if previous is not None and previous != entry:
                raise ValueError(f"Ranking conflitante para {key}: {previous} e {entry}")
            self._entries[key] = entry

    @staticmethod
    def previous_period(reference_date: date) -> tuple[int, int]:
        """Retorna o semestre imediatamente anterior ao da data informada.

        Args:
            reference_date: Data que limita a disponibilidade das informações.

        Returns:
            Tupla ``(ano, semestre)`` do período anterior.
        """
        if reference_date.month <= 6:
            return reference_date.year - 1, 2
        return reference_date.year, 1

    def get_for_period(
        self, team: str, year: int, semester: int
    ) -> RankingEntry | None:
        """Busca o ranking de uma seleção em um período específico.

        Args:
            team: Nome usado no arquivo de resultados.
            year: Ano do ranking.
            semester: Semestre do ranking.

        Returns:
            Ranking encontrado ou ``None`` quando não houver correspondência.
        """
        names = (team, *RANKING_NAME_CANDIDATES.get(team, ()))
        for name in names:
            entry = self._entries.get((year, semester, name))
            if entry is not None:
                return entry
        return None

    def get_previous(self, team: str, reference_date: date) -> RankingEntry | None:
        """Busca o ranking do semestre anterior a uma data.

        Args:
            team: Nome usado no arquivo de resultados.
            reference_date: Data da partida ou início do torneio.

        Returns:
            Ranking anterior ou ``None`` quando indisponível.
        """
        year, semester = self.previous_period(reference_date)
        return self.get_for_period(team, year, semester)


def parse_score(value: str) -> int | None:
    """Converte um placar textual, preservando partidas ainda não realizadas.

    Args:
        value: Valor encontrado no CSV.

    Returns:
        Placar inteiro ou ``None`` para valores ausentes.
    """
    return None if value.strip().upper() in {"", "NA"} else int(value)


def canonical_result_name(name: str) -> str:
    """Padroniza aliases conhecidos usados eventualmente nos resultados.

    Args:
        name: Nome original da seleção.

    Returns:
        Nome canônico adotado para o histórico de partidas.
    """
    return RESULT_NAME_ALIASES.get(name, name)


def load_matches(path: Path) -> list[Match]:
    """Carrega e ordena as partidas internacionais.

    Args:
        path: Caminho do arquivo ``results.csv``.

    Returns:
        Partidas ordenadas cronologicamente.
    """
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        matches = [
            Match(
                match_date=date.fromisoformat(row["date"]),
                home_team=canonical_result_name(row["home_team"]),
                away_team=canonical_result_name(row["away_team"]),
                home_score=parse_score(row["home_score"]),
                away_score=parse_score(row["away_score"]),
                tournament=row["tournament"],
                country=canonical_result_name(row["country"]),
                neutral=row["neutral"].upper() == "TRUE",
            )
            for row in rows
        ]
    return sorted(matches, key=lambda match: match.match_date)


def load_ranking(path: Path) -> RankingIndex:
    """Carrega o histórico semestral do ranking FIFA.

    Args:
        path: Caminho do CSV de ranking.

    Returns:
        Índice preparado para consultas temporais.
    """
    with path.open(encoding="utf-8-sig", newline="") as source:
        return RankingIndex(csv.DictReader(source))


def result_for_team(match: Match, team: str) -> tuple[int, int, str, str]:
    """Obtém placar, resultado e adversário sob a perspectiva de uma seleção.

    Args:
        match: Partida concluída.
        team: Seleção cuja perspectiva será utilizada.

    Returns:
        Gols a favor, gols contra, resultado ``W/D/L`` e adversário.

    Raises:
        ValueError: Se a partida não possuir placar ou a seleção não participar.
    """
    if match.home_score is None or match.away_score is None:
        raise ValueError("Uma partida sem placar não pode compor o histórico.")
    if team == match.home_team:
        goals_for, goals_against, opponent = (
            match.home_score,
            match.away_score,
            match.away_team,
        )
    elif team == match.away_team:
        goals_for, goals_against, opponent = (
            match.away_score,
            match.home_score,
            match.home_team,
        )
    else:
        raise ValueError(f"{team} não participa da partida {match}.")
    outcome = "W" if goals_for > goals_against else "L" if goals_for < goals_against else "D"
    return goals_for, goals_against, outcome, opponent


def summarize_matches(
    matches: Sequence[Match], team: str, ranking: RankingIndex
) -> dict[str, int | float | None]:
    """Resume um conjunto cronológico de partidas de uma seleção.

    As médias de força dos adversários usam, para cada confronto histórico, o
    ranking do semestre anterior àquele confronto.

    Args:
        matches: Partidas anteriores já filtradas para a seleção.
        team: Seleção analisada.
        ranking: Índice temporal do ranking FIFA.

    Returns:
        Estatísticas de resultados, gols, contexto e força dos adversários.
    """
    games = len(matches)
    if games == 0:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points_per_game": None,
            "goals_for_per_game": None,
            "goals_against_per_game": None,
            "goal_difference_per_game": None,
            "opponent_rank_mean": None,
            "opponent_points_mean": None,
            "ranked_opponents": 0,
        }

    wins = draws = losses = goals_for = goals_against = 0
    opponent_ranks: list[int] = []
    opponent_points: list[float] = []
    for match in matches:
        scored, conceded, outcome, opponent = result_for_team(match, team)
        goals_for += scored
        goals_against += conceded
        wins += outcome == "W"
        draws += outcome == "D"
        losses += outcome == "L"
        opponent_ranking = ranking.get_previous(opponent, match.match_date)
        if opponent_ranking is not None:
            opponent_ranks.append(opponent_ranking.rank)
            opponent_points.append(opponent_ranking.points)

    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points_per_game": (3 * wins + draws) / games,
        "goals_for_per_game": goals_for / games,
        "goals_against_per_game": goals_against / games,
        "goal_difference_per_game": (goals_for - goals_against) / games,
        "opponent_rank_mean": mean_or_none(opponent_ranks),
        "opponent_points_mean": mean_or_none(opponent_points),
        "ranked_opponents": len(opponent_ranks),
    }


def mean_or_none(values: Sequence[int | float]) -> float | None:
    """Calcula uma média sem inventar valor para uma sequência vazia.

    Args:
        values: Valores numéricos.

    Returns:
        Média aritmética ou ``None``.
    """
    return sum(values) / len(values) if values else None


def prefixed(summary: dict[str, object], prefix: str) -> dict[str, object]:
    """Adiciona um prefixo aos nomes de um conjunto de atributos.

    Args:
        summary: Atributos calculados.
        prefix: Prefixo identificador da equipe e da janela.

    Returns:
        Novo dicionário com nomes prefixados.
    """
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def team_features(
    team: str,
    history: Sequence[Match],
    tournament_start: date,
    ranking: RankingIndex,
) -> dict[str, object]:
    """Calcula os atributos pré-Copa de uma seleção.

    Args:
        team: Seleção analisada.
        history: Partidas concluídas do ciclo, anteriores à Copa.
        tournament_start: Data de abertura da Copa.
        ranking: Índice temporal do ranking FIFA.

    Returns:
        Ranking, descanso, estatísticas do ciclo e janelas recentes.
    """
    team_history = [
        match
        for match in history
        if team in {match.home_team, match.away_team}
        and match.home_score is not None
        and match.away_score is not None
    ]
    official = [match for match in team_history if match.tournament != FRIENDLY]
    friendly = [match for match in team_history if match.tournament == FRIENDLY]
    neutral = [match for match in team_history if match.neutral]
    current_ranking = ranking.get_previous(team, tournament_start)
    rank_year, rank_semester = ranking.previous_period(tournament_start)

    features: dict[str, object] = {
        "rank": current_ranking.rank if current_ranking else None,
        "fifa_points": current_ranking.points if current_ranking else None,
        "ranking_missing": int(current_ranking is None),
        "ranking_year": rank_year,
        "ranking_semester": rank_semester,
        "days_since_last_match": (
            (tournament_start - team_history[-1].match_date).days
            if team_history
            else None
        ),
    }
    features.update(prefixed(summarize_matches(team_history, team, ranking), "cycle"))
    features.update(prefixed(summarize_matches(official, team, ranking), "official"))
    features.update(prefixed(summarize_matches(friendly, team, ranking), "friendly"))
    features.update(prefixed(summarize_matches(neutral, team, ranking), "neutral"))
    features.update(prefixed(summarize_matches(team_history[-5:], team, ranking), "last5"))
    features.update(prefixed(summarize_matches(team_history[-10:], team, ranking), "last10"))
    return features


def previous_world_cup_end(matches: Sequence[Match], year: int) -> date:
    """Localiza o fim da Copa imediatamente anterior.

    Args:
        matches: Todas as partidas disponíveis.
        year: Ano da Copa que será prevista.

    Returns:
        Data da última partida da Copa anterior.

    Raises:
        ValueError: Se não existir Copa anterior no arquivo.
    """
    previous = [
        match.match_date
        for match in matches
        if match.tournament == WORLD_CUP and match.match_date.year == year - 4
    ]
    if not previous:
        raise ValueError(f"Não foi encontrada a Copa anterior à edição de {year}.")
    return max(previous)


def target_class(match: Match) -> int:
    """Codifica o resultado antes de uma eventual disputa de pênaltis.

    Args:
        match: Partida de Copa concluída.

    Returns:
        ``0`` para vitória do mandante, ``1`` para empate e ``2`` para vitória
        do visitante.

    Raises:
        ValueError: Se a partida não possuir placar.
    """
    if match.home_score is None or match.away_score is None:
        raise ValueError("Não é possível criar o alvo de uma partida sem placar.")
    if match.home_score > match.away_score:
        return 0
    if match.home_score == match.away_score:
        return 1
    return 2


def difference_features(
    team_a: dict[str, object], team_b: dict[str, object]
) -> dict[str, float | None]:
    """Calcula diferenças A menos B para os principais indicadores.

    Args:
        team_a: Atributos da seleção A.
        team_b: Atributos da seleção B.

    Returns:
        Diferenças, mantendo ``None`` quando algum componente estiver ausente.
    """
    keys = (
        "rank",
        "fifa_points",
        "cycle_points_per_game",
        "cycle_goals_for_per_game",
        "cycle_goals_against_per_game",
        "last5_points_per_game",
        "last10_points_per_game",
        "last10_opponent_points_mean",
    )
    differences: dict[str, float | None] = {}
    for key in keys:
        value_a = team_a.get(key)
        value_b = team_b.get(key)
        differences[f"diff_{key}"] = (
            float(value_a) - float(value_b)
            if value_a is not None and value_b is not None
            else None
        )
    return differences


def build_dataset(
    matches: Sequence[Match],
    ranking: RankingIndex,
    start_year: int,
    end_year: int,
) -> list[dict[str, object]]:
    """Constrói as linhas de partidas das Copas selecionadas.

    Args:
        matches: Histórico completo de partidas.
        ranking: Índice semestral do ranking FIFA.
        start_year: Primeira edição incluída.
        end_year: Última edição incluída.

    Returns:
        Linhas prontas para gravação em CSV.

    Raises:
        ValueError: Se uma edição não tiver partidas concluídas.
    """
    dataset: list[dict[str, object]] = []
    match_number = 0
    for year in range(start_year, end_year + 1, 4):
        cup_matches = [
            match
            for match in matches
            if match.tournament == WORLD_CUP
            and match.match_date.year == year
            and match.home_score is not None
            and match.away_score is not None
        ]
        if not cup_matches:
            raise ValueError(f"Não foram encontradas partidas concluídas da Copa de {year}.")

        tournament_start = min(match.match_date for match in cup_matches)
        cycle_start = previous_world_cup_end(matches, year)
        history = [
            match
            for match in matches
            if cycle_start < match.match_date < tournament_start
            and match.home_score is not None
            and match.away_score is not None
        ]
        teams = {team for match in cup_matches for team in (match.home_team, match.away_team)}
        feature_cache = {
            team: team_features(team, history, tournament_start, ranking) for team in teams
        }

        for match in cup_matches:
            match_number += 1
            features_a = feature_cache[match.home_team]
            features_b = feature_cache[match.away_team]
            row: dict[str, object] = {
                "match_id": f"WC{year}_{match_number:04d}",
                "world_cup_year": year,
                "match_date": match.match_date.isoformat(),
                "cycle_start": cycle_start.isoformat(),
                "feature_cutoff": tournament_start.isoformat(),
                "team_a": match.home_team,
                "team_b": match.away_team,
                "country": match.country,
                "neutral": int(match.neutral),
                "team_a_is_host": int(match.home_team == match.country),
                "team_b_is_host": int(match.away_team == match.country),
            }
            row.update(prefixed(features_a, "team_a"))
            row.update(prefixed(features_b, "team_b"))
            row.update(difference_features(features_a, features_b))
            row.update(
                {
                    "score_a": match.home_score,
                    "score_b": match.away_score,
                    "result": target_class(match),
                }
            )
            dataset.append(row)
    return dataset


def compact_dataset(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Seleciona atributos não redundantes para o dataset principal.

    A versão completa continua sendo calculada internamente para centralizar as
    fórmulas. Esta seleção remove constantes, contagens deriváveis e atributos
    altamente repetidos, reduzindo o risco de sobreajuste.

    Args:
        rows: Linhas com o conjunto completo de atributos.

    Returns:
        Linhas com identificação, atributos compactos, diferenças e alvo.
    """
    compact_rows: list[dict[str, object]] = []
    for row in rows:
        compact: dict[str, object] = {
            "match_id": row["match_id"],
            "world_cup_year": row["world_cup_year"],
            "match_date": row["match_date"],
            "cycle_start": row["cycle_start"],
            "feature_cutoff": row["feature_cutoff"],
            "ranking_year": row["team_a_ranking_year"],
            "ranking_semester": row["team_a_ranking_semester"],
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "country": row["country"],
            "neutral": row["neutral"],
            "team_a_is_host": row["team_a_is_host"],
        }
        for side in ("team_a", "team_b"):
            for feature in COMPACT_TEAM_FEATURES:
                compact[f"{side}_{feature}"] = row[f"{side}_{feature}"]
        for feature in COMPACT_DIFFERENCES:
            compact[feature] = row[feature]
        compact.update(
            {
                "score_a": row["score_a"],
                "score_b": row["score_b"],
                "result": row["result"],
            }
        )
        compact_rows.append(compact)
    return compact_rows


def write_dataset(rows: Sequence[dict[str, object]], output: Path) -> None:
    """Grava o dataset resultante em formato CSV.

    Args:
        rows: Linhas produzidas pelo pipeline.
        output: Caminho do arquivo resultante.

    Raises:
        ValueError: Se não houver linhas para gravar.
    """
    if not rows:
        raise ValueError("O dataset resultante está vazio.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando.

    Returns:
        Argumentos validados pelo ``argparse``.
    """
    parser = argparse.ArgumentParser(
        description="Gera um dataset pré-Copa com uma linha por partida."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=1998)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--full-features",
        action="store_true",
        help="Mantém todas as 166 colunas em vez do conjunto compacto.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa leitura, transformação, validação e gravação do dataset."""
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("O ano inicial não pode ser posterior ao ano final.")
    if args.start_year % 4 != 2 or args.end_year % 4 != 2:
        raise ValueError("Os anos informados devem corresponder a edições da Copa.")

    matches = load_matches(args.results)
    ranking = load_ranking(args.ranking)
    full_rows = build_dataset(matches, ranking, args.start_year, args.end_year)
    rows = full_rows if args.full_features else compact_dataset(full_rows)
    write_dataset(rows, args.output)

    missing_rankings = sum(
        int(row["team_a_ranking_missing"]) + int(row["team_b_ranking_missing"])
        for row in full_rows
    )
    print(f"Dataset criado em: {args.output}")
    print(f"Partidas: {len(rows)}")
    print(f"Colunas: {len(rows[0])}")
    print(f"Rankings ausentes entre os lados das partidas: {missing_rankings}")


if __name__ == "__main__":
    main()
