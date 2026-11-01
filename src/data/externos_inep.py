"""Dados das escolas publicados pelo Inep, todos de 2023 ou antes.

| Fonte | Arquivo | O que traz |
|---|---|---|
| Saeb/Ideb anos iniciais por escola | divulgacao_anos_iniciais_escolas_2025.xlsx | nota do 5º ano em 2019, 2021 e 2023, aprovação e Ideb 2023 |
| Indicador de Nível Socioeconômico (Inse) | INSE_2023_escolas.xlsx | nível socioeconômico dos alunos da escola, Saeb 2023 |
| Censo Escolar | microdados_ed_basica_2023.csv | infraestrutura, matrículas, docentes, turmas, tempo integral |

Na base de alunos o código da escola é fictício (e muda a cada ano), então esses dados
entram agregados por município e rede, em src/preprocessing/base_analitica.py.

O arquivo do Ideb é a edição 2025 e tem colunas de 2025. Só as colunas até 2023 são
lidas, porque a prova que o modelo prevê é a de 2024.

Downloads (públicos), salvos em data/external/inep_raw/:
    https://download.inep.gov.br/ideb/resultados/divulgacao_anos_iniciais_escolas_2025.zip
    https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2023/nivel_socioeconomico/INSE_2023_escolas.xlsx
    https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip

Uso:
    python -m src.data.externos_inep --baixar    # baixa e monta
    python -m src.data.externos_inep             # só monta, se os arquivos já estão lá
"""
from __future__ import annotations

import argparse
import zipfile

import numpy as np
import requests
import pandas as pd

from src.config import ARQ_ESCOLA_INEP, DIR_EXTERNO

DIR_BRUTO = DIR_EXTERNO / "inep_raw"
ARQ_IDEB = DIR_BRUTO / "ideb/divulgacao_anos_iniciais_escolas_2025/divulgacao_anos_iniciais_escolas_2025.xlsx"
ARQ_INSE = DIR_BRUTO / "inse/INSE_2023_escolas.xlsx"
ARQ_CENSO = DIR_BRUTO / "censo/microdados_censo_escolar_2023/dados/microdados_ed_basica_2023.csv"

ANO_LIMITE = 2023  # nada depois deste ano entra

DOWNLOADS = {
    "ideb": "https://download.inep.gov.br/ideb/resultados/divulgacao_anos_iniciais_escolas_2025.zip",
    "inse": "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2023/nivel_socioeconomico/INSE_2023_escolas.xlsx",
    "censo": "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip",
}


def baixar():
    """Baixa e descompacta os três arquivos (cerca de 150 MB compactados)."""
    for pasta, url in DOWNLOADS.items():
        destino = DIR_BRUTO / pasta / url.rsplit("/", 1)[-1]
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not destino.exists():
            print(f"Baixando {url}")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(destino, "wb") as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
        if destino.suffix == ".zip":
            with zipfile.ZipFile(destino) as z:
                z.extractall(destino.parent)


def _numero(serie: pd.Series) -> pd.Series:
    """O Inep usa '-' e '--' para ausente e vírgula decimal em algumas planilhas."""
    return pd.to_numeric(serie.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


def ler_ideb() -> pd.DataFrame:
    bruto = pd.read_excel(ARQ_IDEB, header=9, dtype=str)
    colunas = {
        "ID_ESCOLA": "id_escola",
        "VL_NOTA_PORTUGUES_2019": "saeb5_lp_2019",
        "VL_NOTA_PORTUGUES_2021": "saeb5_lp_2021",
        "VL_NOTA_PORTUGUES_2023": "saeb5_lp_2023",
        "VL_NOTA_MATEMATICA_2023": "saeb5_mt_2023",
        "VL_OBSERVADO_2023": "ideb_2023",
        "VL_APROVACAO_2023_SI_4": "aprovacao_ai_2023",
        "VL_APROVACAO_2023_2": "aprovacao_2ano_2023",
    }
    assert all(int(c[-4:]) <= ANO_LIMITE for c in colunas if c[-4:].isdigit())
    df = bruto[list(colunas)].rename(columns=colunas)
    df["id_escola"] = pd.to_numeric(df["id_escola"], errors="coerce")
    df = df.dropna(subset=["id_escola"])
    for c in df.columns.drop("id_escola"):
        df[c] = _numero(df[c])
    df["saeb5_lp_tendencia"] = df["saeb5_lp_2023"] - df["saeb5_lp_2019"]
    return df.drop_duplicates("id_escola").astype({"id_escola": "int64"})


def ler_inse() -> pd.DataFrame:
    bruto = pd.read_excel(ARQ_INSE)
    niveis = [c for c in bruto.columns if c.startswith("PC_NIVEL_")]
    df = pd.DataFrame({
        "id_escola": pd.to_numeric(bruto["ID_ESCOLA"], errors="coerce"),
        "inse_valor": _numero(bruto["MEDIA_INSE"]),
        # Níveis I e II do Inse: alunos com menor acesso a bens e escolaridade dos pais
        "inse_pct_niveis_baixos": bruto[niveis[:2]].apply(_numero).sum(axis=1, min_count=1),
    })
    return df.dropna(subset=["id_escola"]).drop_duplicates("id_escola").astype({"id_escola": "int64"})


def ler_censo() -> pd.DataFrame:
    usar = ["CO_ENTIDADE", "CO_MUNICIPIO", "TP_DEPENDENCIA", "TP_LOCALIZACAO", "TP_LOCALIZACAO_DIFERENCIADA", "IN_INTERNET_ALUNOS",
            "IN_BANDA_LARGA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_LABORATORIO_INFORMATICA",
            "IN_QUADRA_ESPORTES", "IN_AGUA_POTAVEL", "IN_ESGOTO_REDE_PUBLICA",
            "IN_MATERIAL_PED_INFANTIL", "IN_PROF_COORDENADOR", "IN_PROF_PSICOLOGO",
            "QT_MAT_FUND_AI", "QT_MAT_FUND_AI_INT", "QT_DOC_FUND_AI", "QT_TUR_FUND_AI",
            "QT_MAT_BAS"]
    bruto = pd.read_csv(ARQ_CENSO, sep=";", encoding="latin1", usecols=usar, low_memory=False)
    bruto = bruto[bruto["QT_MAT_FUND_AI"].fillna(0) > 0]
    df = pd.DataFrame({
        "id_escola": bruto["CO_ENTIDADE"].astype("int64"),
        "id_municipio": bruto["CO_MUNICIPIO"].astype("int64"),
        "rede": bruto["TP_DEPENDENCIA"].astype("int64"),
        "esc_localizacao": bruto["TP_LOCALIZACAO"].map({1: "Urbana", 2: "Rural"}),
        "esc_localizacao_diferenciada": (bruto["TP_LOCALIZACAO_DIFERENCIADA"].fillna(0) > 0).astype(float),
        "esc_internet_alunos": bruto["IN_INTERNET_ALUNOS"],
        "esc_banda_larga": bruto["IN_BANDA_LARGA"],
        "esc_biblioteca": bruto["IN_BIBLIOTECA_SALA_LEITURA"],
        "esc_lab_informatica": bruto["IN_LABORATORIO_INFORMATICA"],
        "esc_quadra": bruto["IN_QUADRA_ESPORTES"],
        "esc_agua_potavel": bruto["IN_AGUA_POTAVEL"],
        "esc_esgoto_rede": bruto["IN_ESGOTO_REDE_PUBLICA"],
        "esc_material_infantil": bruto["IN_MATERIAL_PED_INFANTIL"],
        "esc_coordenador": bruto["IN_PROF_COORDENADOR"],
        "esc_psicologo": bruto["IN_PROF_PSICOLOGO"],
        "esc_matriculas_ai": bruto["QT_MAT_FUND_AI"],
        "esc_alunos_por_docente": bruto["QT_MAT_FUND_AI"] / bruto["QT_DOC_FUND_AI"].replace(0, np.nan),
        "esc_alunos_por_turma": bruto["QT_MAT_FUND_AI"] / bruto["QT_TUR_FUND_AI"].replace(0, np.nan),
        "esc_pct_integral": 100 * bruto["QT_MAT_FUND_AI_INT"].fillna(0) / bruto["QT_MAT_FUND_AI"],
    })
    return df.drop_duplicates("id_escola")


def montar() -> pd.DataFrame:
    censo, ideb, inse = ler_censo(), ler_ideb(), ler_inse()
    print(f"Censo Escolar 2023: {len(censo):>7,} escolas com anos iniciais")
    print(f"Saeb/Ideb 5º ano  : {len(ideb):>7,} escolas | com nota LP 2023: {ideb['saeb5_lp_2023'].notna().sum():,}")
    print(f"Inse 2023         : {len(inse):>7,} escolas")
    # Só escolas presentes no Censo: é ele que diz município e rede de cada uma
    return censo.merge(ideb, on="id_escola", how="left").merge(inse, on="id_escola", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baixar", action="store_true", help="baixa os arquivos do Inep antes")
    if parser.parse_args().baixar:
        baixar()
    base = montar()
    destino = DIR_EXTERNO / ARQ_ESCOLA_INEP
    base.to_parquet(destino, index=False)
    print(f"\n{len(base):,} escolas salvas em {destino}")
    print(base.describe().T[["count", "mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
