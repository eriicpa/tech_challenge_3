"""Leitura da camada Gold exportada e das fontes externas."""
from __future__ import annotations

import pandas as pd

from src import config as C


def _localizar(arquivo: str):
    """Aceita a exportação em Parquet ou em CSV (o Databricks exporta nos dois)."""
    parquet = C.DIR_GOLD / arquivo
    csv = parquet.with_suffix(".csv")
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    return None


def _ler(caminho, colunas=None) -> pd.DataFrame:
    if caminho.suffix == ".parquet":
        return pd.read_parquet(caminho, columns=colunas)
    # O CSV do Databricks escreve nulo como "null"
    return pd.read_csv(caminho, usecols=colunas, na_values=["null", "NULL", ""])


def _ler_gold(arquivo: str, colunas: list[str] | None = None) -> pd.DataFrame:
    caminho = _localizar(arquivo)
    if caminho is None:
        raise FileNotFoundError(
            f"{C.DIR_GOLD / arquivo} não encontrado. Exporte a Gold com "
            f"databricks/exportar_gold_fase3.py e copie os arquivos para data/gold/ "
            f"(instruções em data/README.md).")
    return _ler(caminho, colunas)


def carregar_alunos(anos: list[int] | None = None) -> pd.DataFrame:
    """Microdados de aluno (gold_aluno_ml), com tipos compactos."""
    df = _ler_gold(C.ARQ_ALUNO)
    if anos is not None:
        df = df[df["ano"].isin(anos)]
    for coluna in ("ano", "rede", "serie"):
        df[coluna] = df[coluna].astype("int16")
    df["id_municipio"] = df["id_municipio"].astype("int32")
    df["id_escola"] = pd.to_numeric(df["id_escola"], errors="coerce").astype("Int64")
    df["peso_aluno"] = df["peso_aluno"].astype("float32").fillna(1.0)
    df["avaliado"] = df["avaliado"].astype(bool)
    df = df.drop(columns=[c for c in ("faixa_proficiencia",) if c in df.columns])

    # A fato_aluno da Silver não tem território. Ele vem da gold_indicador_municipio.
    territorio = ["nome_municipio", "sigla_uf", "nome_regiao", "nome_mesorregiao"]
    faltando = [c for c in territorio if c not in df.columns]
    if faltando:
        dim = (carregar_indicador_municipio().drop_duplicates("id_municipio")
               .set_index("id_municipio")[faltando])
        df = df.join(dim, on="id_municipio")
    if "rede_nome" not in df.columns:
        df["rede_nome"] = df["rede"].map(C.NOME_REDE)
    for coluna in ("sigla_uf", "nome_regiao", "rede_nome", "nome_mesorregiao"):
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("category")
    return df.reset_index(drop=True)


def carregar_indicador_municipio() -> pd.DataFrame:
    df = _ler_gold(C.ARQ_INDICADOR_MUNICIPIO)
    df["ano"] = df["ano"].astype(int)
    df["rede"] = df["rede"].astype(int)
    df["id_municipio"] = df["id_municipio"].astype(int)
    return df


def carregar_metas_municipio() -> pd.DataFrame:
    """Metas municipais de 2024 a 2030, uma linha por município e ano.

    Usa a gold_meta_municipio se ela foi exportada. Se não, monta a mesma tabela a partir
    do arquivo de metas do Inep, usando a meta declarada mais recente, como na Fase 2.
    """
    if _localizar(C.ARQ_META_MUNICIPIO) is not None:
        df = _ler_gold(C.ARQ_META_MUNICIPIO)
    else:
        bruto = pd.read_csv(C.DIR_EXTERNO / C.ARQ_META_INEP)
        colunas = [c for c in bruto.columns if c.startswith("meta_alfabetizacao_")]
        df = bruto.melt(id_vars=["ano", "id_municipio", "rede"], value_vars=colunas,
                        var_name="ano_meta", value_name="meta_taxa").dropna(subset=["meta_taxa"])
        df["ano_meta"] = df["ano_meta"].str[-4:].astype(int)
        df = (df.sort_values("ano", ascending=False)
              .drop_duplicates(["id_municipio", "rede", "ano_meta"]).drop(columns="ano"))
        df["rede_nome"] = df["rede"]
        df["rede"] = df["rede"].str.lower().map({"municipal": C.REDE_MUNICIPAL,
                                                 "estadual": C.REDE_ESTADUAL})
    df["id_municipio"] = df["id_municipio"].astype(int)
    df["ano_meta"] = df["ano_meta"].astype(int)
    return df


def carregar_gold_opcional(arquivo: str) -> pd.DataFrame | None:
    """Tabelas Gold usadas só na análise exploratória."""
    caminho = _localizar(arquivo)
    return _ler(caminho) if caminho is not None else None


def carregar_ibge() -> pd.DataFrame:
    caminho = C.DIR_EXTERNO / C.ARQ_IBGE
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não encontrado. Rode: python -m src.data.externos_ibge")
    return pd.read_csv(caminho)


def carregar_censo_escolar() -> pd.DataFrame | None:
    caminho = C.DIR_EXTERNO / C.ARQ_CENSO_ESCOLAR
    return pd.read_parquet(caminho) if caminho.exists() else None


def carregar_atlas() -> pd.DataFrame | None:
    caminho = C.DIR_EXTERNO / C.ARQ_ATLAS
    return pd.read_csv(caminho) if caminho.exists() else None
