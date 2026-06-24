"""Gera uma versão em PDF do relatório do projeto.

Este script é um gerador alternativo para ambientes sem compilador LaTeX. O
arquivo `relatorio.tex` continua sendo a versão LaTeX do relatório, mas este
script permite gerar `relatorio.pdf` diretamente com Python usando ReportLab.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("relatorio.pdf")


def build_styles() -> dict[str, ParagraphStyle]:
    """Cria os estilos usados no PDF.

    Returns:
        Dicionário com estilos nomeados.
    """
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubtitleCenter",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=11,
            leading=14,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=10.5,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FigureCaption",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#444444"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            backColor=colors.HexColor("#F2F2F2"),
            borderColor=colors.HexColor("#CCCCCC"),
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Reference",
            parent=styles["BodyText"],
            alignment=TA_LEFT,
            fontSize=9,
            leading=12,
            spaceAfter=8,
        )
    )
    return styles


def paragraph(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    """Cria um parágrafo justificado.

    Args:
        text: Texto do parágrafo.
        styles: Estilos do documento.

    Returns:
        Objeto Paragraph.
    """
    return Paragraph(text, styles["BodyJustify"])


def section(title: str, styles: dict[str, ParagraphStyle]) -> list:
    """Cria o título de uma seção.

    Args:
        title: Título da seção.
        styles: Estilos do documento.

    Returns:
        Lista de elementos ReportLab.
    """
    return [Spacer(1, 0.15 * cm), Paragraph(title, styles["Heading1"])]


def subsection(title: str, styles: dict[str, ParagraphStyle]) -> list:
    """Cria o título de uma subseção.

    Args:
        title: Título da subseção.
        styles: Estilos do documento.

    Returns:
        Lista de elementos ReportLab.
    """
    return [Spacer(1, 0.1 * cm), Paragraph(title, styles["Heading2"])]


def code_block(code: str, styles: dict[str, ParagraphStyle]) -> Preformatted:
    """Cria um bloco de código monoespaçado.

    Args:
        code: Código exibido.
        styles: Estilos do documento.

    Returns:
        Bloco Preformatted.
    """
    return Preformatted(code.strip(), styles["CodeBlock"])


def reference(text: str, url: str, styles: dict[str, ParagraphStyle]) -> list:
    """Cria uma referência com URL em linha própria.

    Args:
        text: Descrição da referência.
        url: Endereço da fonte.
        styles: Estilos do documento.

    Returns:
        Elementos da referência.
    """
    return [
        Paragraph(text, styles["Reference"]),
        Paragraph(f"<font name='Courier' size='7'>{url}</font>", styles["Reference"]),
    ]


def figure(path: str, caption: str, width: float = 15.5 * cm) -> KeepTogether:
    """Cria uma figura com legenda.

    Args:
        path: Caminho da imagem.
        caption: Legenda da figura.
        width: Largura desejada.

    Returns:
        Bloco de figura e legenda.
    """
    image_path = Path(path)
    image = Image(str(image_path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    styles = build_styles()
    return KeepTogether(
        [
            image,
            Paragraph(caption, styles["FigureCaption"]),
        ]
    )


def styled_table(data: list[list[str]], column_widths: list[float] | None = None) -> Table:
    """Cria uma tabela formatada.

    Args:
        data: Dados da tabela.
        column_widths: Larguras das colunas.

    Returns:
        Objeto Table.
    """
    table = Table(data, colWidths=column_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BBBBBB")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ]
        )
    )
    return table


def add_header_footer(canvas, doc) -> None:
    """Adiciona número de página ao PDF.

    Args:
        canvas: Canvas do ReportLab.
        doc: Documento em construção.
    """
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(19 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


def build_document() -> list:
    """Monta os elementos do relatório.

    Returns:
        Lista de elementos ReportLab.
    """
    styles = build_styles()
    story = [
        Paragraph(
            "Predição de Resultados de Partidas da Copa do Mundo",
            styles["TitleCenter"],
        ),
        Paragraph(
            "Uma abordagem com ranking FIFA e desempenho pré-Copa<br/>"
            "Integrante 1 · Integrante 2 · Integrante 3 · Integrante 4",
            styles["SubtitleCenter"],
        ),
    ]

    story.extend(section("1. Contexto, Motivação e Objetivo", styles))
    story.append(
        paragraph(
            "A Copa do Mundo FIFA é uma das competições esportivas mais acompanhadas "
            "do mundo e oferece um cenário rico para análise de dados. Este trabalho "
            "investiga se informações disponíveis antes do início de cada torneio, "
            "como ranking FIFA e desempenho no ciclo pré-Copa, ajudam a prever o "
            "resultado das partidas.",
            styles,
        )
    )
    story.append(
        paragraph(
            "O objetivo é construir e avaliar modelos de classificação supervisionada "
            "para prever três classes: vitória da Seleção A, empate e vitória da "
            "Seleção B.",
            styles,
        )
    )
    story.append(
        paragraph(
            "O dataset FIFA World Cup Dataset (2002-2026), indicado nas orientações "
            "do trabalho, foi considerado como referência temática. Para adequar o "
            "problema à previsão de partidas, foi construído um dataset derivado que "
            "combina resultados internacionais e rankings semestrais da FIFA. Essa "
            "decisão permitiu calcular atributos anteriores à Copa e prever o resultado "
            "de cada jogo do torneio. A edição de 2026 não foi usada na modelagem "
            "porque ainda não possui resultados finais completos, ou seja, não há "
            "rótulos reais para treinamento ou avaliação.",
            styles,
        )
    )
    story.append(
        paragraph(
            "As bases usadas foram: resultados internacionais de seleções, disponíveis "
            "em https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017; "
            "ranking masculino FIFA, obtido no Kaggle em "
            "https://www.kaggle.com/datasets/lucasyukioimafuko/fifa-mens-world-ranking; "
            "e o dataset FIFA World "
            "Cup Dataset (2002-2026), disponibilizado como referência da disciplina.",
            styles,
        )
    )

    story.extend(section("2. Enquadramento Analítico", styles))
    story.append(
        paragraph(
            "O problema foi tratado como classificação multiclasse. Foram avaliados "
            "baseline da classe majoritária, regressão logística, árvore de decisão "
            "e random forest. As métricas principais foram acurácia e macro-F1, sendo "
            "a macro-F1 especialmente importante por equilibrar a avaliação entre as "
            "três classes.",
            styles,
        )
    )
    story.append(
        paragraph(
            "A abordagem segue ideias comuns em sports analytics: usar medidas de "
            "força histórica, forma recente e comparação relativa entre adversários. "
            "No futebol, rankings, desempenho ofensivo, desempenho defensivo e forma "
            "prévia são frequentemente usados como aproximações da qualidade técnica "
            "e do momento competitivo das equipes.",
            styles,
        )
    )

    story.extend(section("3. Hipóteses Sobre o Fenômeno", styles))
    hypotheses = [
        "H1: seleções melhor posicionadas no ranking FIFA possuem maior probabilidade de vencer.",
        "H2: a forma recente acrescenta informação além do ranking.",
        "H3: o desempenho acumulado no ciclo pré-Copa contém informação útil.",
        "H4: a força dos adversários recentes melhora a interpretação da forma recente.",
    ]
    for item in hypotheses:
        story.append(paragraph(f"• {item}", styles))

    story.extend(section("4. Metodologia e Modelagem", styles))
    story.append(
        paragraph(
            "O dataset derivado foi construído a partir de uma base de resultados "
            "internacionais e uma base de rankings semestrais da FIFA. Cada linha "
            "representa uma partida de Copa do Mundo entre 1998 e 2022. Para evitar "
            "vazamento de informação, os atributos foram calculados apenas com dados "
            "anteriores ao início da Copa correspondente.",
            styles,
        )
    )
    story.append(
        paragraph(
            "A divisão temporal foi: treinamento nas Copas de 1998 a 2010, validação "
            "em 2014, teste final em 2018 e avaliação adicional em 2022. Foram "
            "removidos do treino os placares reais, o alvo e metadados como nomes "
            "das seleções e datas.",
            styles,
        )
    )
    story.append(
        paragraph(
            "O pré-processamento incluiu padronização de nomes de seleções entre as "
            "bases, seleção das partidas de Copa, definição do ciclo pré-Copa e "
            "cálculo de estatísticas agregadas. Também foram criadas variáveis de "
            "diferença entre Seleção A e Seleção B, como diferença de ranking, pontos "
            "FIFA, aproveitamento recente, gols marcados e gols sofridos.",
            styles,
        )
    )
    story.append(
        paragraph(
            "As ferramentas utilizadas foram Python, scikit-learn para modelagem, "
            "matplotlib e seaborn para visualizações, além de scripts próprios para "
            "geração do dataset, análise exploratória e avaliação dos modelos.",
            styles,
        )
    )
    story.extend(subsection("Trechos relevantes do código", styles))
    story.append(
        paragraph(
            "O primeiro trecho mostra a prevenção de vazamento temporal: o ranking de "
            "cada seleção é buscado no semestre anterior ao início da Copa.",
            styles,
        )
    )
    story.append(
        code_block(
            """
current_ranking = ranking.get_previous(team, tournament_start)
rank_year, rank_semester = ranking.previous_period(tournament_start)
""",
            styles,
        )
    )
    story.append(
        paragraph(
            "O alvo da classificação foi gerado a partir do placar: vitória da Seleção "
            "A, empate ou vitória da Seleção B. Os placares foram removidos das "
            "variáveis explicativas usadas pelos modelos.",
            styles,
        )
    )
    story.append(
        code_block(
            """
if score_a > score_b:
    result = 0
elif score_a == score_b:
    result = 1
else:
    result = 2
""",
            styles,
        )
    )
    story.append(
        paragraph(
            "A avaliação foi organizada por tempo, usando Copas anteriores para treino "
            "e Copas posteriores para validação e teste.",
            styles,
        )
    )
    story.append(
        code_block(
            """
if 1998 <= year <= 2010:
    treino.append(row)
elif year == 2014:
    validacao.append(row)
elif year == 2018:
    teste.append(row)
elif year == 2022:
    avaliacao_extra.append(row)
""",
            styles,
        )
    )

    story.extend(section("5. Dados e Resultados", styles))
    story.append(
        paragraph(
            "O dataset final possui 448 partidas, 45 colunas e 67 seleções. Para a "
            "modelagem foram usadas 32 variáveis explicativas. A classe mais frequente "
            "é vitória da Seleção A, seguida por vitória da Seleção B e empate.",
            styles,
        )
    )
    story.extend(subsection("Análise Exploratória", styles))
    story.append(
        paragraph(
            "A análise exploratória teve como objetivo compreender a composição do "
            "dataset antes da modelagem, verificar o equilíbrio entre as classes e "
            "observar relações iniciais entre força das seleções e resultado das "
            "partidas.",
            styles,
        )
    )
    story.append(figure("analises/graficos/distribuicao_resultado.png", "Figura 1: distribuição geral do alvo."))
    story.append(
        paragraph(
            "O gráfico de distribuição do alvo mostra que o empate é a classe menos "
            "frequente, enquanto as vitórias da Seleção A são a classe mais comum. "
            "Isso indica que o problema não é perfeitamente balanceado. Por esse "
            "motivo, além da acurácia, foi usada a métrica macro-F1, que dá o mesmo "
            "peso para as três classes.",
            styles,
        )
    )
    story.append(figure("analises/graficos/resultado_por_copa.png", "Figura 2: distribuição dos resultados por Copa."))
    story.append(
        paragraph(
            "O gráfico por Copa mostra que todas as edições possuem 64 partidas, mas "
            "a distribuição entre vitória da Seleção A, empate e vitória da Seleção B "
            "muda de um ano para outro. Essa variação reforça a escolha de validação "
            "temporal, pois o modelo precisa ser avaliado em Copas futuras, não em "
            "partidas misturadas aleatoriamente.",
            styles,
        )
    )
    story.append(
        paragraph(
            "A análise exploratória mostrou relação clara entre diferença de ranking "
            "e resultado. Quando a Seleção A estava muito melhor ranqueada, a frequência "
            "de vitória de A foi maior; quando a Seleção B estava melhor ranqueada, "
            "a frequência de vitória de B aumentou.",
            styles,
        )
    )
    story.append(
        paragraph(
            "As faixas do gráfico usam diff_rank = ranking da Seleção A - ranking da "
            "Seleção B. Assim, A muito melhor indica diferença menor que -30 posições; "
            "A melhor, de -30 a -10; Equilibrado, de -10 a 10; B melhor, de 10 a 30; "
            "e B muito melhor, acima de 30 posições.",
            styles,
        )
    )
    story.append(figure("analises/graficos/resultado_por_faixa_ranking.png", "Figura 3: resultado por faixa de diferença de ranking FIFA."))
    story.append(
        paragraph(
            "O gráfico de diferença de ranking é o principal indício exploratório a "
            "favor da hipótese H1. A frequência de vitórias da Seleção A cai conforme "
            "a vantagem de ranking passa para a Seleção B, enquanto a frequência de "
            "vitórias da Seleção B aumenta. Embora ainda existam empates e resultados "
            "inesperados, a tendência geral mostra que o ranking possui informação "
            "preditiva relevante.",
            styles,
        )
    )

    model_table = [
        ["Modelo", "Conjunto", "Acurácia", "Macro-F1"],
        ["Baseline", "Validação 2014", "0,4531", "0,2079"],
        ["Baseline", "Teste 2018", "0,3906", "0,1873"],
        ["Regressão logística", "Validação 2014", "0,4531", "0,3789"],
        ["Regressão logística", "Teste 2018", "0,5000", "0,4761"],
        ["Árvore de decisão", "Validação 2014", "0,3438", "0,3524"],
        ["Árvore de decisão", "Teste 2018", "0,4062", "0,3988"],
        ["Random forest", "Validação 2014", "0,4844", "0,4127"],
        ["Random forest", "Teste 2018", "0,4219", "0,3668"],
    ]
    story.append(styled_table(model_table, [5.0 * cm, 4.0 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(figure("modelagem/graficos/comparacao_modelos_macro_f1.png", "Figura 4: comparação dos modelos por macro-F1."))
    story.append(figure("modelagem/graficos/matriz_confusao_melhor_modelo_2018.png", "Figura 5: matriz de confusão do melhor modelo no teste de 2018.", width=11 * cm))

    hypothesis_table = [
        ["Grupo de atributos", "Acurácia", "Macro-F1", "Variáveis"],
        ["Ranking FIFA", "0,5938", "0,4799", "6"],
        ["Ranking + forma recente", "0,5312", "0,4617", "15"],
        ["Ranking + forma + ciclo", "0,4688", "0,3931", "28"],
        ["Completo sem força adversários", "0,4375", "0,3887", "29"],
        ["Modelo completo", "0,4531", "0,3789", "32"],
    ]
    story.append(styled_table(hypothesis_table, [6.5 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(figure("modelagem/graficos/importancia_variaveis_random_forest.png", "Figura 6: principais variáveis segundo o random forest."))
    story.append(
        paragraph(
            "A avaliação das hipóteses indica que H1 foi a mais apoiada, pois o grupo "
            "de ranking FIFA obteve o melhor desempenho na validação. H2 recebeu apoio "
            "parcial: a forma recente apresentou desempenho próximo ao ranking, mas "
            "não o superou. H3 também recebeu apoio parcial, já que variáveis do ciclo "
            "tiveram resultado competitivo em alguns cenários, mas sem ganho consistente "
            "na validação. H4 não foi confirmada de forma clara, pois a força dos "
            "adversários recentes não melhorou o modelo completo na validação.",
            styles,
        )
    )

    story.extend(section("6. Discussão e Conclusão", styles))
    story.append(
        paragraph(
            "Os resultados indicam que é possível prever parcialmente os resultados "
            "da Copa usando informações prévias ao torneio. Os modelos treinados "
            "superaram o baseline em macro-F1, sugerindo que os atributos carregam "
            "algum sinal preditivo. No teste final de 2018, a regressão logística "
            "obteve 50% de acurácia e macro-F1 de 0,4761.",
            styles,
        )
    )
    story.append(
        paragraph(
            "A hipótese mais apoiada foi H1, relacionada ao ranking FIFA. O grupo de "
            "atributos de ranking obteve o melhor resultado na validação de 2014. "
            "A inclusão de muitas variáveis de ciclo e forma recente não melhorou "
            "consistentemente o desempenho, possivelmente por causa do tamanho reduzido "
            "da amostra e da alta variabilidade do futebol.",
            styles,
        )
    )
    story.append(
        paragraph(
            "Como limitações, destacam-se o número pequeno de partidas, a ausência de "
            "variáveis contextuais como escalações e lesões, e a dificuldade natural "
            "de prever empates. Ainda assim, o projeto fornece uma base adequada para "
            "investigar fatores associados ao desempenho em Copas do Mundo.",
            styles,
        )
    )
    story.extend(section("Referências e Bases de Dados", styles))
    story.extend(
        reference(
            "Mart Jürisoo. International football results from 1872 to 2026. Kaggle.",
            "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017",
            styles,
        )
    )
    story.extend(
        reference(
            "Lucas Yukio Imafuko. FIFA Men's World Ranking. Kaggle.",
            "https://www.kaggle.com/datasets/lucasyukioimafuko/fifa-mens-world-ranking",
            styles,
        )
    )
    story.extend(
        reference(
            "Scikit-learn. User Guide: supervised learning and model evaluation.",
            "https://scikit-learn.org/stable/user_guide.html",
            styles,
        )
    )
    story.append(
        Paragraph(
            "Hastie, T.; Tibshirani, R.; Friedman, J. The Elements of Statistical Learning. Springer.",
            styles["Reference"],
        )
    )
    story.append(
        Paragraph(
            "James, G.; Witten, D.; Hastie, T.; Tibshirani, R. An Introduction to Statistical Learning. Springer.",
            styles["Reference"],
        )
    )

    return story


def main() -> None:
    """Gera o arquivo PDF."""
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    document.build(build_document(), onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"PDF gerado em: {OUTPUT}")


if __name__ == "__main__":
    main()
