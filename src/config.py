"""Configuração do projeto: caminhos, anos, semente e listas de colunas.

Fica tudo aqui para que os notebooks e os scripts usem as mesmas definições.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DIR_DADOS = RAIZ / "data"
# Pasta com a exportação da Gold. A variável de ambiente permite usar os dados
# fictícios dos testes.
DIR_GOLD = Path(os.environ.get("TC3_DIR_GOLD", DIR_DADOS / "gold"))
DIR_EXTERNO = DIR_DADOS / "external"  # IBGE, Censo e Atlas
DIR_PROCESSADO = DIR_DADOS / "processed"  # bases geradas pelo projeto
DIR_REPORTS = RAIZ / "reports"
DIR_IMAGENS = RAIZ / "images"
DIR_MODELOS = RAIZ / "models"

SEMENTE = 42

# Existem duas provas. A de 2023 (ANO_BASE) dá o histórico e a de 2024 (ANO_ALVO) dá a
# resposta. Nada calculado a partir da prova de 2024 entra no modelo.
#
# A meta do município também fica fora do modelo: a de 2024 foi calculada a partir do
# resultado de 2023, e a de 2025 não. Ela só é usada para comparar com a taxa prevista.
ANO_BASE = 2023
ANO_ALVO = 2024
ANO_PROJECAO = 2025  # ano da meta usada na projeção

PONTO_CORTE_SAEB = 743

# Códigos de rede usados na Gold da Fase 2
REDE_ESTADUAL = 2
REDE_MUNICIPAL = 3
REDE_PUBLICA = 5
NOME_REDE = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada", 5: "Pública", 0: "Total"}

# Arquivos exportados do Databricks (databricks/exportar_gold_fase3.py)
ARQ_ALUNO = "gold_aluno_ml.parquet"
ARQ_INDICADOR_MUNICIPIO = "gold_indicador_municipio.parquet"
ARQ_META_MUNICIPIO = "gold_meta_municipio.parquet"
ARQ_RANKING_UF = "gold_ranking_uf.parquet"
ARQ_EVOLUCAO = "gold_evolucao_temporal.parquet"
ARQ_DISTRIBUICAO = "gold_distribuicao_niveis.parquet"
ARQ_PRIORIZACAO = "gold_priorizacao_municipio.parquet"
ARQ_FEATURES_ML = "gold_features_ml.parquet"
ARQ_META_VS_REALIZADO = "gold_meta_vs_realizado.parquet"
ARQ_PERFIL_ALUNO = "gold_perfil_aluno_municipio.parquet"

ARQ_IBGE = "ibge_socioeconomico_municipio.csv"
# Contexto da escola (Censo Escolar, Saeb 5º ano, Inse), gerado por src/data/externos_inep.py
ARQ_ESCOLA_INEP = "escola_inep.parquet"
ARQ_CENSO_ESCOLAR = ARQ_ESCOLA_INEP
ARQ_ATLAS = "atlas_idhm_municipio.csv"                # opcional, via BigQuery
# Metas do Inep (mesmo arquivo da Fase 2), usado quando não há gold_meta_municipio
ARQ_META_INEP = "inep_meta_alfabetizacao_municipio.csv"

# Quantos alunos usar. A curva de aprendizado mostra que o resultado para de melhorar
# bem antes de 400 mil, e assim o projeto roda num computador comum.
AMOSTRA_TREINO = 400_000
AMOSTRA_BUSCA = 120_000
N_FOLDS = 5
N_TRIALS_OPTUNA = 30
PROPORCAO_TESTE = 0.20

ALVO = "nao_alfabetizado"
GRUPO = "id_municipio"
PESO = "peso_aluno"

# Colunas que não podem entrar no modelo: todas vêm da prova de 2024, a que estamos
# prevendo. O teste tests/test_vazamento.py confere esta lista.
COLUNAS_PROIBIDAS = [
    "proficiencia_portugues", "alfabetizado", "nao_alfabetizado", "faixa_proficiencia",
    "taxa_alfabetizacao", "media_portugues", "gap_meta_ano", "gap_meta_2030",
    "atingiu_meta_ano", "situacao_meta", "risco_alfabetizacao", "posicao_na_uf",
    "quartil_nacional", "dif_vs_uf", "dif_vs_brasil", "taxa_uf", "presenca",
    "preenchimento_caderno", "avaliado",
]

# Variáveis do modelo de aluno, por tipo
FEATURES_NUMERICAS = [
    # escola em 2024: só o cadastro, conhecido antes da prova
    "escola_matriculados",
    # histórico do município e da UF (2023)
    "mun_taxa_rede_ant", "mun_taxa_municipal_ant", "mun_taxa_publica_ant", "mun_media_ant",
    "mun_participacao_ant",
    "uf_taxa_ant",
    # renda e contexto (IBGE: Censo 2022 e PIB 2021)
    "populacao", "densidade", "pib_per_capita", "pct_va_agro", "pct_urbana",
    "taxa_alfab_adultos", "pct_esgoto_adequado",
    # dados das escolas do Inep até 2023, por município e rede
    # (Saeb 5º ano, Ideb, Inse, Censo Escolar)
    "inep_saeb5_lp_2023", "inep_saeb5_lp_2021", "inep_saeb5_lp_tendencia", "inep_ideb_2023",
    "inep_aprovacao_2ano_2023", "inep_inse", "inep_pct_inse_baixo", "inep_pct_integral",
    "inep_pct_biblioteca", "inep_pct_internet_alunos", "inep_pct_coordenador",
    "inep_pct_material_infantil", "inep_alunos_por_turma", "inep_alunos_por_docente",
    "inep_pct_rural",
    # Atlas, se o arquivo existir (via BigQuery)
    "idhm", "idhm_e", "idhm_r", "gini", "renda_pc",
]
FEATURES_CATEGORICAS = ["nome_regiao", "sigla_uf", "rede_nome"]
FEATURES_ALTA_CARDINALIDADE = ["nome_mesorregiao"]
