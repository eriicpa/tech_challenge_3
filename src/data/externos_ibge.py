"""Dados socioeconômicos dos municípios, baixados da API SIDRA do IBGE.

| Variável              | Tabela SIDRA | Fonte                         |
|-----------------------|--------------|-------------------------------|
| populacao, densidade  | 4714         | Censo 2022                    |
| pct_urbana            | 9923         | Censo 2022                    |
| taxa_alfab_adultos    | 9543         | Censo 2022 (15 anos ou mais)  |
| pct_esgoto_adequado   | 6805         | Censo 2022                    |
| pib, va_agro, va_total| 5938         | PIB dos Municípios, 2021      |

Todos são anteriores às provas de 2023 e 2024. Não precisa de login.

Uso:
    python -m src.data.externos_ibge
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from src.config import ARQ_IBGE, DIR_EXTERNO

URL = "https://apisidra.ibge.gov.br/values"


def consultar_sidra(caminho: str, tentativas: int = 3) -> pd.DataFrame:
    """Chama a API SIDRA e devolve id_municipio e valor."""
    for tentativa in range(1, tentativas + 1):
        try:
            resposta = requests.get(f"{URL}/{caminho}", timeout=120)
            resposta.raise_for_status()
            linhas = resposta.json()[1:]  # a primeira linha é o cabeçalho
            break
        except (requests.RequestException, ValueError):
            if tentativa == tentativas:
                raise
            time.sleep(2 ** tentativa)
    df = pd.DataFrame(linhas)
    df = df.rename(columns={"D1C": "id_municipio", "V": "valor"})
    df["id_municipio"] = df["id_municipio"].astype(int)
    # "-", "..." e "X" são valores ausentes ou sigilosos no SIDRA
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def serie(caminho: str, nome: str, filtro: dict | None = None) -> pd.Series:
    df = consultar_sidra(caminho)
    for coluna, valor in (filtro or {}).items():
        df = df[df[coluna] == valor]
    print(f"  {nome:22s} {df['valor'].notna().sum():>5} municípios")
    return df.set_index("id_municipio")["valor"].rename(nome)


def extrair() -> pd.DataFrame:
    print("Consultando a API SIDRA do IBGE...")
    colunas = [
        serie("t/4714/n6/all/v/93/p/2022", "populacao"),
        serie("t/4714/n6/all/v/614/p/2022", "densidade"),
        serie("t/9923/n6/all/v/1000093/p/2022/c1/1", "pct_urbana"),
        serie("t/9543/n6/all/v/2513/p/2022/c2/6794/c86/95251/c287/100362", "taxa_alfab_adultos"),
        serie("t/6805/n6/all/v/1000381/p/2022/c11558/46290", "pct_esgoto_adequado"),
        # PIB e valores adicionados vêm em mil reais
        serie("t/5938/n6/all/v/37/p/2021", "pib"),
        serie("t/5938/n6/all/v/498/p/2021", "va_total"),
        serie("t/5938/n6/all/v/513/p/2021", "va_agro"),
    ]
    base = pd.concat(colunas, axis=1).reset_index()
    base["pib_per_capita"] = base["pib"] * 1000 / base["populacao"]
    base["pct_va_agro"] = 100 * base["va_agro"] / base["va_total"]
    return base[["id_municipio", "populacao", "densidade", "pct_urbana", "taxa_alfab_adultos",
                 "pct_esgoto_adequado", "pib_per_capita", "pct_va_agro"]]


def main():
    DIR_EXTERNO.mkdir(parents=True, exist_ok=True)
    base = extrair()
    destino = DIR_EXTERNO / ARQ_IBGE
    base.to_csv(destino, index=False)
    print(f"\n{len(base)} municípios salvos em {destino}")
    print(base.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
