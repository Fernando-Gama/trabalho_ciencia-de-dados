# Hipóteses do trabalho

## Problema investigado

O trabalho busca prever o resultado de partidas da Copa do Mundo utilizando três
classes:

- `0`: vitória da Seleção A;
- `1`: empate;
- `2`: vitória da Seleção B.

As hipóteses investigam se o ranking FIFA, a forma recente, o desempenho durante o
ciclo pré-Copa e a força dos adversários contribuem para a previsão do resultado.
Elas são hipóteses de associação e capacidade preditiva, não afirmações de
causalidade.

## H1 - Influência do ranking FIFA

**Hipótese:** seleções mais bem posicionadas no ranking FIFA possuem maior
probabilidade de vencer uma partida da Copa do Mundo.

### Variáveis relacionadas

- `team_a_rank` e `team_b_rank`;
- `team_a_fifa_points` e `team_b_fifa_points`;
- `diff_rank`;
- `diff_fifa_points`;
- `result`.

### Relação esperada

- `diff_rank < 0` indica posição melhor para A e deve estar associado a uma maior
  frequência de vitórias de A;
- `diff_fifa_points > 0` indica mais pontos para A e deve favorecer vitórias de A;
- relações opostas devem favorecer vitórias de B.

### Forma de avaliação

Comparar o baseline da classe majoritária com um modelo que utilize somente ranking
e pontos FIFA. A hipótese será apoiada se o modelo de ranking apresentar ganho de
macro-F1 e accuracy na validação temporal.

## H2 - Contribuição da forma recente

**Hipótese:** o desempenho nas últimas partidas acrescenta capacidade preditiva além
daquela fornecida pelo ranking FIFA.

### Variáveis relacionadas

- `team_a_last5_points_per_game` e `team_b_last5_points_per_game`;
- `team_a_last10_points_per_game` e `team_b_last10_points_per_game`;
- `diff_last5_points_per_game`;
- `diff_last10_points_per_game`.

### Relação esperada

Uma seleção com aproveitamento recente superior ao do adversário deve apresentar
maior probabilidade de vitória, mesmo após considerar a diferença de ranking.

### Forma de avaliação

Comparar dois conjuntos de atributos:

1. ranking e pontos FIFA;
2. ranking, pontos FIFA e forma nas últimas 5 e 10 partidas.

A hipótese será apoiada se a inclusão da forma recente aumentar o macro-F1 na
validação temporal.

## H3 - Contribuição do ciclo pré-Copa

**Hipótese:** o desempenho acumulado durante o ciclo pré-Copa contém informação útil
que não é representada apenas pelo ranking e pelas últimas partidas.

### Variáveis relacionadas

- `team_a_cycle_points_per_game` e `team_b_cycle_points_per_game`;
- `team_a_cycle_goals_for_per_game` e `team_b_cycle_goals_for_per_game`;
- `team_a_cycle_goals_against_per_game` e
  `team_b_cycle_goals_against_per_game`;
- `team_a_official_points_per_game` e `team_b_official_points_per_game`;
- diferenças equivalentes entre A e B.

### Relação esperada

Maior aproveitamento e maior produção ofensiva durante o ciclo devem favorecer a
seleção. Maior média de gols sofridos deve estar associada a resultados piores. O
desempenho em jogos oficiais pode ser mais representativo do contexto competitivo
que o desempenho geral.

### Forma de avaliação

Comparar os modelos:

1. ranking e forma recente;
2. ranking, forma recente e atributos do ciclo pré-Copa.

A hipótese será apoiada se os atributos do ciclo produzirem melhoria no macro-F1 e
reduzirem erros relevantes na matriz de confusão.

## H4 - Força dos adversários recentes

**Hipótese:** a forma recente é mais informativa quando considerada juntamente com a
força dos adversários enfrentados.

### Variáveis relacionadas

- `team_a_last10_opponent_points_mean`;
- `team_b_last10_opponent_points_mean`;
- `diff_last10_opponent_points_mean`.

### Relação esperada

Resultados semelhantes podem ter significados diferentes quando obtidos contra
adversários de níveis distintos. Uma seleção que manteve bom desempenho contra
adversários mais fortes pode estar mais preparada para a Copa.

### Forma de avaliação

Comparar o conjunto completo sem a força dos adversários com o mesmo conjunto
incluindo essas variáveis. A hipótese será apoiada se houver ganho de macro-F1 na
validação temporal.

## Estratégia experimental

Os experimentos serão incrementais:

| Experimento | Atributos | Hipótese avaliada |
|---|---|---|
| Baseline | Classe majoritária | Referência mínima |
| Experimento 1 | Ranking e pontos FIFA | H1 |
| Experimento 2 | Ranking e forma recente | H2 |
| Experimento 3 | Ranking, forma e ciclo pré-Copa | H3 |
| Experimento 4 | Todos os anteriores e força dos adversários | H4 |

## Validação e métricas

A separação dos dados deve respeitar a ordem temporal:

- treinamento: Copas de 1998 a 2010;
- validação: Copa de 2014;
- teste final: Copa de 2018;
- avaliação adicional: Copa de 2022.

As escolhas de atributos e modelos devem ser feitas com treinamento e validação. A
Copa de 2018 não deve orientar essas escolhas.

As métricas utilizadas serão:

- **macro-F1:** métrica principal, pois atribui a mesma importância às três classes;
- **accuracy:** proporção total de previsões corretas;
- **precision e recall por classe:** identificação dos tipos de erro;
- **matriz de confusão:** análise de quais resultados são confundidos pelo modelo.

Uma hipótese será considerada apoiada quando a inclusão de seu conjunto de atributos
melhorar o desempenho na validação, especialmente o macro-F1. Melhorias pequenas ou
instáveis devem ser interpretadas com cautela devido ao número limitado de partidas.
