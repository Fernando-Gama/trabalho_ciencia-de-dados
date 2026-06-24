# Análise Exploratória do Dataset

## Visão geral

- Linhas: `448`
- Colunas: `45`
- Copas analisadas: `1998, 2002, 2006, 2010, 2014, 2018, 2022`
- Seleções presentes: `67`
- Variáveis explicativas candidatas: `32`

Cada linha representa uma partida da Copa do Mundo. As variáveis explicativas
foram calculadas com dados anteriores ao início da Copa correspondente, e o
alvo `result` indica vitória da Seleção A, empate ou vitória da Seleção B.

## Distribuição do alvo

| Classe | Rótulo | Quantidade | Percentual |
|---|---|---:|---:|
| 0 | Vitória da Seleção A | 196 | 43.75% |
| 1 | Empate | 107 | 23.88% |
| 2 | Vitória da Seleção B | 145 | 32.37% |

A classe de empate tende a ser menos frequente que as vitórias, por isso a
avaliação dos modelos deve usar macro-F1 além da accuracy.

## Partidas por Copa

| Copa | Vitória A | Empate | Vitória B | Total | % Empates |
|---|---:|---:|---:|---:|---:|
| 1998 | 29 | 19 | 16 | 64 | 29.69% |
| 2002 | 28 | 16 | 20 | 64 | 25.0% |
| 2006 | 33 | 15 | 16 | 64 | 23.44% |
| 2010 | 24 | 16 | 24 | 64 | 25.0% |
| 2014 | 29 | 13 | 22 | 64 | 20.31% |
| 2018 | 25 | 13 | 26 | 64 | 20.31% |
| 2022 | 28 | 15 | 21 | 64 | 23.44% |

## Ranking FIFA e resultado

`diff_rank` é calculado como ranking da Seleção A menos ranking da Seleção B.
Valores negativos indicam que A estava melhor posicionada no ranking.

| Faixa de ranking | Vitória A | Empate | Vitória B | Total | % Vitória A | % Vitória B |
|---|---:|---:|---:|---:|---:|---:|
| A muito melhor | 35 | 9 | 7 | 51 | 68.63% | 13.73% |
| A melhor | 70 | 27 | 23 | 120 | 58.33% | 19.17% |
| Equilibrado | 62 | 40 | 47 | 149 | 41.61% | 31.54% |
| B melhor | 17 | 21 | 43 | 81 | 20.99% | 53.09% |
| B muito melhor | 12 | 10 | 25 | 47 | 25.53% | 53.19% |

## Atributos mais associados ao resultado

A correlação abaixo usa uma versão direcional do resultado: `1` para vitória
da Seleção A, `0` para empate e `-1` para vitória da Seleção B. Ela não prova
causalidade, mas ajuda a identificar atributos promissores para a modelagem.

| Atributo | Correlação |
|---|---:|
| diff_rank | -0.3431 |
| diff_fifa_points | 0.3189 |
| diff_last10_points_per_game | 0.3094 |
| diff_last5_points_per_game | 0.3078 |
| diff_cycle_points_per_game | 0.2869 |
| team_a_rank | -0.2782 |
| team_a_last10_points_per_game | 0.256 |
| team_a_last5_points_per_game | 0.2512 |

## Seleções com melhor desempenho no recorte

| Seleção | Jogos | Vitórias | Empates | Derrotas | Pontos | Pontos/Jogo | Saldo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brazil | 41 | 27 | 6 | 8 | 87 | 2.122 | 38 |
| Germany | 39 | 26 | 5 | 8 | 83 | 2.128 | 45 |
| France | 39 | 24 | 9 | 6 | 81 | 2.077 | 36 |
| Argentina | 36 | 21 | 8 | 7 | 71 | 1.972 | 26 |
| Netherlands | 30 | 19 | 8 | 3 | 65 | 2.167 | 30 |
| Spain | 30 | 16 | 8 | 6 | 56 | 1.867 | 24 |
| England | 33 | 14 | 10 | 9 | 52 | 1.576 | 19 |
| Croatia | 30 | 13 | 8 | 9 | 47 | 1.567 | 10 |
| Belgium | 22 | 12 | 6 | 4 | 42 | 1.909 | 11 |
| Portugal | 26 | 11 | 6 | 9 | 39 | 1.5 | 13 |

## Arquivos gerados

As tabelas completas foram salvas em `analises/tabelas/`.
Os gráficos foram salvos em `analises/graficos/`:
- `distribuicao_resultado.png`
- `resultado_por_copa.png`
- `resultado_por_faixa_ranking.png`
- `correlacoes_resultado.png`
- `top_selecoes_pontos.png`
