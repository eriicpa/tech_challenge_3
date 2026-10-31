"""Dados opcionais do Atlas do Desenvolvimento Humano, via BigQuery da Base dos Dados.

Traz IDHM, renda per capita e Gini por município (Censo 2010). Usa o mesmo projeto do
BigQuery da Fase 2. Antes de rodar, o script estima o tamanho da consulta (dry run) e só
pede as colunas usadas. Como os nomes de coluna da Base dos Dados mudam entre versões,
ele lê o esquema da tabela e usa só as colunas que existem.

Uso:
    gcloud auth application-default login
    python -m src.data.externos_bigquery --projeto SEU_PROJETO_GCP
"""
from __future__ import annotations

import argparse
import sys

from src.config import ARQ_ATLAS, DIR_EXTERNO

TABELA_ATLAS = "basedosdados.br_pnud_atlas.municipio"

# nome na Base dos Dados -> nome no projeto
COLUNAS_ATLAS = {
    "id_municipio": "id_municipio",
    "idhm": "idhm",
    "idhm_e": "idhm_e",
    "idhm_r": "idhm_r",
    "indice_gini": "gini",
    "renda_pc": "renda_pc",
}


def cliente_bigquery(projeto):
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("Instale: pip install google-cloud-bigquery db-dtypes pyarrow")
    return bigquery.Client(project=projeto)


def colunas_existentes(cliente, tabela, desejadas):
    esquema = {campo.name for campo in cliente.get_table(tabela).schema}
    faltando = [c for c in desejadas if c not in esquema]
    if faltando:
        print(f"  [{tabela}] colunas ausentes, ignoradas: {faltando}")
    return {c: n for c, n in desejadas.items() if c in esquema}


def executar(cliente, sql, rotulo):
    from google.cloud import bigquery

    estimativa = cliente.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"  [{rotulo}] vai processar {estimativa.total_bytes_processed / 1024**2:.1f} MB")
    return cliente.query(sql).result().to_dataframe()


def extrair_atlas(cliente):
    mapa = colunas_existentes(cliente, TABELA_ATLAS, COLUNAS_ATLAS)
    sql = f"""
        SELECT {", ".join(f"{c} AS {n}" for c, n in mapa.items())}
        FROM `{TABELA_ATLAS}`
        WHERE ano = 2010
    """
    df = executar(cliente, sql, "atlas")
    df["id_municipio"] = df["id_municipio"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projeto", required=True, help="ID do projeto GCP")
    args = parser.parse_args()

    DIR_EXTERNO.mkdir(parents=True, exist_ok=True)
    atlas = extrair_atlas(cliente_bigquery(args.projeto))
    atlas.to_csv(DIR_EXTERNO / ARQ_ATLAS, index=False)
    print(f"  {len(atlas):,} municípios -> {DIR_EXTERNO / ARQ_ATLAS}")


if __name__ == "__main__":
    main()
