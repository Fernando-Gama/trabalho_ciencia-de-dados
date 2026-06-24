# Modelagem e Avaliação

## Configuração

- Variáveis explicativas usadas no modelo completo: `32`
- Treino: `256` partidas, Copas 1998-2010
- Validação: `64` partidas, Copa 2014
- Teste final: `64` partidas, Copa 2018
- Avaliação adicional: `64` partidas, Copa 2022

Foram removidos do treino os placares `score_a` e `score_b`, o alvo `result`
e metadados como nomes das seleções, datas e identificadores de partidas.

## Melhor desempenho

| Conjunto | Modelo | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Validação 2014 | random_forest | 0.4844 | 0.4127 |
| Teste 2018 | regressao_logistica | 0.5 | 0.4761 |
| Avaliação 2022 | random_forest | 0.4531 | 0.4018 |

A métrica principal recomendada é o macro-F1, pois considera o desempenho
médio entre vitória da Seleção A, empate e vitória da Seleção B.

## Avaliação das hipóteses

| Grupo de atributos | Accuracy validação | Macro-F1 validação | Variáveis |
|---|---:|---:|---:|
| h1_ranking | 0.5938 | 0.4799 | 6 |
| h2_ranking_forma_recente | 0.5312 | 0.4617 | 15 |
| h3_ranking_forma_ciclo | 0.4688 | 0.3931 | 28 |
| h4_completo_sem_forca_adversarios | 0.4375 | 0.3887 | 29 |
| modelo_completo | 0.4531 | 0.3789 | 32 |

## Arquivos gerados

- `tabelas/metricas_modelos.csv`: comparação dos modelos;
- `tabelas/metricas_por_classe.csv`: precision, recall e F1 por classe;
- `tabelas/matrizes_confusao.csv`: matrizes de confusão em formato longo;
- `tabelas/predicoes_modelos.csv`: predições por partida;
- `tabelas/avaliacao_hipoteses.csv`: avaliação dos grupos de atributos;
- `tabelas/importancia_variaveis.csv`: importância das variáveis em árvores;
- `graficos/`: gráficos da avaliação, quando matplotlib está instalado.
