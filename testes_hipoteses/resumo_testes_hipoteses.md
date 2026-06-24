# Testes Estatísticos das Hipóteses

Os testes complementam a modelagem preditiva. Eles avaliam associação
estatística entre atributos e resultado, sem afirmar causalidade.

| Hipótese | Variável | Teste | Estatística | p-valor | Efeito | Significativo 5% |
|---|---|---|---:|---:|---:|---:|
| H1 - Ranking FIFA | diff_rank_faixa | qui-quadrado | 57.119462 | 0.0 | 0.252486 | 1 |
| H2 - Forma recente | diff_last10_points_per_game_sinal | qui-quadrado | 24.976617 | 5.1e-05 | 0.16696 | 1 |
| H3 - Ciclo pré-Copa | diff_cycle_points_per_game_sinal | qui-quadrado | 26.925386 | 2.1e-05 | 0.173351 | 1 |
| H4 - Força dos adversários | diff_last10_opponent_points_mean_abs | qui-quadrado | 1.519441 | 0.467797 | 0.058238 | 0 |
| H1 - Ranking FIFA | diff_rank | Spearman | -0.369011 | 0.0 | 0.369011 | 1 |
| H1 - Pontos FIFA | diff_fifa_points | Spearman | 0.355522 | 0.0 | 0.355522 | 1 |
| H2 - Forma recente | diff_last10_points_per_game | Spearman | 0.297127 | 0.0 | 0.297127 | 1 |
| H3 - Ciclo pré-Copa | diff_cycle_points_per_game | Spearman | 0.290057 | 0.0 | 0.290057 | 1 |
| H3 - Ataque no ciclo | diff_cycle_goals_for_per_game | Spearman | 0.247088 | 0.0 | 0.247088 | 1 |
| H3 - Defesa no ciclo | diff_cycle_goals_against_per_game | Spearman | -0.195765 | 3e-05 | 0.195765 | 1 |
| H4 - Força dos adversários | diff_last10_opponent_points_mean | Spearman | 0.10571 | 0.025254 | 0.10571 | 1 |

## Interpretação resumida

- H1 - Ranking FIFA: associação estatisticamente significativa com efeito moderado.
- H2 - Forma recente: associação estatisticamente significativa com efeito moderado.
- H3 - Ciclo pré-Copa: associação estatisticamente significativa com efeito moderado.
- H4 - Força dos adversários: não houve evidência estatística forte de associação.
- H1 - Ranking FIFA: correlação negativa estatisticamente significativa.
- H1 - Pontos FIFA: correlação positiva estatisticamente significativa.
- H2 - Forma recente: correlação positiva estatisticamente significativa.
- H3 - Ciclo pré-Copa: correlação positiva estatisticamente significativa.
- H3 - Ataque no ciclo: correlação positiva estatisticamente significativa.
- H3 - Defesa no ciclo: correlação negativa estatisticamente significativa.
- H4 - Força dos adversários: correlação positiva estatisticamente significativa.
