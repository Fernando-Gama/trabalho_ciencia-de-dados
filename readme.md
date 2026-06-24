# Predição de resultados da Copa do Mundo

## Objetivo

Construir um dataset para classificação supervisionada em que cada linha representa
uma partida de Copa do Mundo e o alvo possui três classes:

- `0`: vitória da Seleção A;
- `1`: empate, inclusive quando houver vencedor posterior nos pênaltis;
- `2`: vitória da Seleção B.

## Bases utilizadas

- `dados_originais/resultados_1972_2026/results.csv`: partidas internacionais;
- `dados_originais/ranking_semestre1992_ate_2024/fifa_mens_rank.csv`: ranking FIFA.

Os arquivos originais não são alterados. O script usa somente a biblioteca padrão do
Python e gera `dados_processados/dataset_partidas_copa.csv`.

Por padrão, a saída usa um conjunto compacto de atributos para reduzir redundância e
risco de sobreajuste. A versão detalhada pode ser gerada separadamente com:

```bash
./venv/bin/python main.py --full-features \
  --output dados_processados/dataset_partidas_copa_completo.csv
```

## Execução

```bash
./venv/bin/python main.py
```

Para gerar apenas algumas edições ou mudar a saída:

```bash
./venv/bin/python main.py --start-year 2002 --end-year 2018 \
  --output dados_processados/dataset_copas_2002_2018.csv
```

## Análise exploratória

Depois de gerar o dataset processado, execute:

```bash
./venv/bin/python analise_exploratoria.py
```

Para gerar gráficos com visual mais adequado para o relatório, instale as
dependências opcionais:

```bash
./venv/bin/python -m pip install matplotlib seaborn
```

O script cria a pasta `analises/` com:

- `resumo_analise_exploratoria.md`: resumo textual com os principais achados;
- `tabelas/`: tabelas CSV com distribuição do alvo, estatísticas numéricas,
  resultados por Copa, desempenho por seleção e correlações;
- `graficos/`: gráficos PNG, caso `matplotlib` esteja instalado no ambiente.

Quando `seaborn` está instalado, o script usa automaticamente um estilo visual mais
limpo para os gráficos. Se `matplotlib` não estiver instalado, as tabelas e o resumo
em Markdown são gerados normalmente, apenas os gráficos são ignorados.

## Modelagem e avaliação

Para instalar as dependências principais do projeto:

```bash
./venv/bin/python -m pip install -r requirements.txt
```

Depois da geração do dataset, execute:

```bash
./venv/bin/python modelagem.py
```

O script treina e compara:

- baseline da classe majoritária;
- regressão logística;
- árvore de decisão;
- random forest.

A divisão temporal adotada é:

- treinamento: Copas de 1998 a 2010;
- validação: Copa de 2014;
- teste final: Copa de 2018;
- avaliação adicional: Copa de 2022.

As saídas são salvas em `modelagem/`:

- `resumo_modelagem.md`: resumo dos resultados principais;
- `tabelas/metricas_modelos.csv`: accuracy e macro-F1 por modelo;
- `tabelas/metricas_por_classe.csv`: precision, recall e F1 por classe;
- `tabelas/matrizes_confusao.csv`: matrizes de confusão;
- `tabelas/avaliacao_hipoteses.csv`: comparação dos grupos de atributos;
- `graficos/`: comparação dos modelos, matriz de confusão e importância de variáveis.

## Testes estatísticos das hipóteses

Além da avaliação preditiva, o projeto possui testes estatísticos de associação:

```bash
./venv/bin/python testes_hipoteses.py
```

As saídas são salvas em `testes_hipoteses/`:

- `resumo_testes_hipoteses.md`: resumo interpretativo dos testes;
- `tabelas/resumo_testes_hipoteses.csv`: estatística, p-valor e tamanho de efeito;
- `tabelas/tabelas_contingencia.csv`: tabelas de contingência usadas no qui-quadrado.

Foram usados qui-quadrado de independência e correlação de Spearman. Esses testes
avaliam associação estatística entre atributos e resultado, sem afirmar causalidade.

## Metodologia

Para cada edição entre 1998 e 2022, o ciclo começa depois da última partida da Copa
anterior e termina antes da abertura da Copa analisada. Amistosos e partidas oficiais
são usados no histórico, mas também são resumidos separadamente.

Os atributos são congelados na abertura da Copa. Portanto, partidas da própria
competição não modificam a forma recente das seleções. São calculados:

- desempenho no ciclo, nas últimas 5 e nas últimas 10 partidas;
- desempenho separado em jogos oficiais, amistosos e campo neutro;
- vitórias, empates, derrotas e aproveitamento por jogo;
- médias de gols marcados, sofridos e saldo;
- dias desde a última partida;
- média do ranking e dos pontos FIFA dos adversários recentes;
- diferenças entre os principais atributos das seleções A e B.

O arquivo compacto mantém uma medida representativa de cada conceito: ranking,
pontos FIFA, descanso, quantidade de jogos no ciclo, aproveitamento, ataque, defesa,
desempenho oficial, forma nas últimas 5 e 10 partidas e força dos últimos
adversários. Contagens redundantes e colunas constantes permanecem disponíveis
somente no modo `--full-features`.

As contagens são acompanhadas por médias porque as seleções disputam quantidades
diferentes de partidas durante cada ciclo.

## Prevenção de vazamento temporal

O ranking usado como força pré-Copa é sempre o semestre anterior ao semestre em que
a Copa começou. Por exemplo, a Copa de 2018 usa `2017/2`, enquanto a Copa de 2022,
disputada no segundo semestre, usa `2022/1`.

Ao medir a força de um adversário em uma partida histórica, aplica-se a mesma regra:
somente o ranking do semestre anterior àquela partida pode ser consultado. Nenhuma
janela inclui a própria partida prevista, partidas posteriores ou jogos anteriores da
mesma Copa.

`score_a` e `score_b` são mantidos exclusivamente para auditoria e criação do alvo.
Eles não devem ser fornecidos ao modelo como variáveis explicativas.

## Padronização de seleções

As bases usam convenções diferentes, como `United States`/`USA`, `Iran`/`IR Iran` e
`South Korea`/`Korea Republic`. O script possui um mapeamento explícito dessas
diferenças. Correspondência aproximada de texto não é utilizada, pois poderia unir
seleções distintas.

## Divisão temporal sugerida

- treinamento: Copas de 1998 a 2010;
- validação: Copa de 2014;
- teste final: Copa de 2018.

A Copa de 2022 pode ser reservada para uma avaliação adicional posterior. Todas as
transformações aprendidas pelo modelo, como imputação e normalização, também devem ser
ajustadas somente no conjunto de treinamento.
