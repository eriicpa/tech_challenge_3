# Databricks notebook source
# MAGIC %md
# MAGIC # Tech Challenge Fase 3 — Exportação da camada Gold
# MAGIC
# MAGIC Roda no mesmo workspace da Fase 2, depois dos notebooks 01 a 04.
# MAGIC
# MAGIC Faz duas coisas:
# MAGIC
# MAGIC 1. Cria a tabela **gold_aluno_ml**, com uma linha por aluno. A Fase 3 prevê se cada
# MAGIC    aluno vai se alfabetizar, e a Gold da Fase 2 só tinha dados por município. A nova
# MAGIC    tabela sai da Silver (fato_aluno), já com UF, região e mesorregião.
# MAGIC    Cria também a **gold_meta_municipio**, com as metas até 2030.
# MAGIC 2. Exporta as tabelas Gold usadas na Fase 3 em Parquet, um arquivo por tabela, para
# MAGIC    baixar e colocar em `data/gold/` do repositório da Fase 3.
# MAGIC
# MAGIC A nota do aluno continua na tabela porque a análise exploratória usa. Como ela é a
# MAGIC própria resposta em outra escala, o código da Fase 3 impede que ela entre no modelo.

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace", "Catálogo")
dbutils.widgets.text("schema_silver", "tc2_silver", "Schema Silver")
dbutils.widgets.text("schema_gold", "tc2_gold", "Schema Gold")
dbutils.widgets.text("schema_landing", "tc2_landing", "Schema Landing")
dbutils.widgets.text("volume", "arquivos", "Volume")

CATALOGO = dbutils.widgets.get("catalogo")
SCHEMA_SILVER = dbutils.widgets.get("schema_silver")
SCHEMA_GOLD = dbutils.widgets.get("schema_gold")
VOLUME = f"/Volumes/{CATALOGO}/{dbutils.widgets.get('schema_landing')}/{dbutils.widgets.get('volume')}"
DESTINO = f"{VOLUME}/export_fase3"

dbutils.fs.mkdirs(DESTINO)
print("Destino:", DESTINO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. gold_aluno_ml

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOGO}.{SCHEMA_GOLD}.gold_aluno_ml AS
SELECT
    a.ano,
    a.id_aluno,
    a.id_escola,
    a.id_municipio,
    m.nome_municipio,
    m.sigla_uf,
    m.nome_regiao,
    m.nome_mesorregiao,
    a.rede,
    a.rede_nome,
    a.serie,
    a.presenca,
    a.preenchimento_caderno,
    a.peso_aluno,
    a.avaliado,
    a.proficiencia_portugues,
    a.alfabetizado
FROM {CATALOGO}.{SCHEMA_SILVER}.fato_aluno a
LEFT JOIN {CATALOGO}.{SCHEMA_SILVER}.dim_municipio m
  ON a.id_municipio = m.id_municipio
""")

# Metas municipais de 2024 a 2030. A gold_indicador_municipio só traz a meta dos anos
# que já têm resultado, e a Fase 3 também projeta 2025.
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOGO}.{SCHEMA_GOLD}.gold_meta_municipio AS
SELECT id_municipio, rede, rede_nome, ano_meta, meta_taxa
FROM {CATALOGO}.{SCHEMA_SILVER}.dim_meta_municipio
""")

display(spark.sql(f"""
SELECT ano, rede_nome,
       COUNT(*)                                        AS alunos,
       SUM(CASE WHEN avaliado THEN 1 ELSE 0 END)        AS avaliados,
       ROUND(100 * AVG(CASE WHEN avaliado THEN CAST(alfabetizado AS INT) END), 2) AS pct_alfabetizado
FROM {CATALOGO}.{SCHEMA_GOLD}.gold_aluno_ml
GROUP BY ano, rede_nome ORDER BY ano, rede_nome
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exportação
# MAGIC
# MAGIC O Spark grava uma pasta com arquivos `part-*`. Para sair um arquivo por tabela, a
# MAGIC escrita usa uma partição só, e depois o arquivo é copiado com o nome da tabela.
# MAGIC Tabelas que não existirem no workspace são puladas e listadas no final.

# COMMAND ----------

TABELAS = [
    "gold_aluno_ml",
    "gold_meta_municipio",
    "gold_indicador_municipio",
    "gold_meta_vs_realizado",
    "gold_evolucao_temporal",
    "gold_ranking_uf",
    "gold_distribuicao_niveis",
    "gold_perfil_aluno_municipio",
    "gold_features_ml",
    "gold_priorizacao_municipio",
]


def exportar(tabela):
    origem = f"{CATALOGO}.{SCHEMA_GOLD}.{tabela}"
    df = spark.table(origem)
    temporario = f"{DESTINO}/_tmp_{tabela}"
    df.coalesce(1).write.mode("overwrite").parquet(temporario)
    parte = [f.path for f in dbutils.fs.ls(temporario) if f.name.endswith(".parquet")][0]
    final = f"{DESTINO}/{tabela}.parquet"
    dbutils.fs.cp(parte, final)
    dbutils.fs.rm(temporario, recurse=True)
    tamanho = [f.size for f in dbutils.fs.ls(DESTINO) if f.name == f"{tabela}.parquet"][0]
    return df.count(), len(df.columns), tamanho


resumo, ausentes = [], []
for tabela in TABELAS:
    if not spark.catalog.tableExists(f"{CATALOGO}.{SCHEMA_GOLD}.{tabela}"):
        ausentes.append(tabela)
        continue
    linhas, colunas, tamanho = exportar(tabela)
    resumo.append((tabela, linhas, colunas, round(tamanho / 1024 ** 2, 2)))
    print(f"{tabela:32s} {linhas:>10,} linhas  {colunas:>3} colunas  {tamanho / 1024**2:8.2f} MB")

if ausentes:
    print("\nNão encontradas (rode os notebooks da Fase 2 que as criam):", ausentes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Colunas exportadas
# MAGIC
# MAGIC Lista as colunas de cada tabela, para conferir com o que o código da Fase 3 lê.

# COMMAND ----------

for tabela, *_ in resumo:
    campos = spark.table(f"{CATALOGO}.{SCHEMA_GOLD}.{tabela}").dtypes
    print(f"{tabela}: " + ", ".join(f"{n}:{t}" for n, t in campos))
