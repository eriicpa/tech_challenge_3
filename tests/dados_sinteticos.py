"""Gera uma Gold fictícia, com o mesmo formato da exportação real.

Serve para testar o código sem depender do Databricks. Os números não têm valor de
análise: são gerados por uma regra simples (qualidade do município, efeito da escola e
variação do aluno), só para o modelo ter o que aprender.

Uso:
    python -m tests.dados_sinteticos --destino data/sintetico --municipios 600
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

UF = {11: ("RO", "Norte"), 12: ("AC", "Norte"), 13: ("AM", "Norte"), 14: ("RR", "Norte"),
      15: ("PA", "Norte"), 16: ("AP", "Norte"), 17: ("TO", "Norte"), 21: ("MA", "Nordeste"),
      22: ("PI", "Nordeste"), 23: ("CE", "Nordeste"), 24: ("RN", "Nordeste"),
      25: ("PB", "Nordeste"), 26: ("PE", "Nordeste"), 27: ("AL", "Nordeste"),
      28: ("SE", "Nordeste"), 29: ("BA", "Nordeste"), 31: ("MG", "Sudeste"),
      32: ("ES", "Sudeste"), 33: ("RJ", "Sudeste"), 35: ("SP", "Sudeste"), 41: ("PR", "Sul"),
      42: ("SC", "Sul"), 43: ("RS", "Sul"), 50: ("MS", "Centro-Oeste"),
      51: ("MT", "Centro-Oeste"), 52: ("GO", "Centro-Oeste"), 53: ("DF", "Centro-Oeste")}


def gerar(destino: Path, n_municipios: int = 600, semente: int = 7):
    rng = np.random.default_rng(semente)
    ibge = pd.read_csv(C.DIR_EXTERNO / C.ARQ_IBGE)
    mun = ibge.sample(n=n_municipios, random_state=semente).reset_index(drop=True)
    mun["id_uf"] = mun["id_municipio"] // 100000
    mun["sigla_uf"] = mun["id_uf"].map(lambda u: UF[u][0])
    mun["nome_regiao"] = mun["id_uf"].map(lambda u: UF[u][1])
    mun["nome_mesorregiao"] = mun["sigla_uf"] + " meso " + rng.integers(1, 5, len(mun)).astype(str)
    mun["nome_municipio"] = "Município " + mun["id_municipio"].astype(str)
    z = lambda s: (s - s.mean()) / s.std()
    mun["qualidade"] = (0.6 * z(mun["taxa_alfab_adultos"]) + 0.3 * z(np.log(mun["pib_per_capita"]))
                        + rng.normal(0, 0.6, len(mun)))

    escolas, id_escola = [], 10_000_000
    for m in mun.itertuples():
        for _ in range(rng.integers(1, 7)):
            id_escola += 1
            escolas.append((id_escola, m.id_municipio, int(rng.choice([2, 3], p=[0.25, 0.75])),
                            rng.normal(0, 0.7), int(rng.integers(10, 80))))
    escolas = pd.DataFrame(escolas, columns=["id_escola", "id_municipio", "rede", "efeito", "tam"])
    escolas = escolas.merge(mun, on="id_municipio")

    alunos, id_aluno = [], 0
    for ano, efeito_ano in ((2023, 0.0), (2024, 0.15)):
        for e in escolas.itertuples():
            n = max(3, int(e.tam * rng.uniform(0.8, 1.2)))
            prof = rng.normal(735 + 35 * e.qualidade + 25 * e.efeito + 30 * efeito_ano, 65, n)
            avaliado = rng.random(n) < 0.92
            alunos.append(pd.DataFrame({
                "ano": ano, "id_aluno": np.arange(id_aluno, id_aluno + n), "id_escola": e.id_escola,
                "id_municipio": e.id_municipio, "nome_municipio": e.nome_municipio,
                "sigla_uf": e.sigla_uf, "nome_regiao": e.nome_regiao,
                "nome_mesorregiao": e.nome_mesorregiao, "rede": e.rede,
                "rede_nome": C.NOME_REDE[e.rede], "serie": 2, "presenca": avaliado.astype(int),
                "preenchimento_caderno": avaliado.astype(int),
                "peso_aluno": rng.uniform(0.9, 1.1, n), "avaliado": avaliado,
                "proficiencia_portugues": np.where(avaliado, prof, np.nan),
                "alfabetizado": np.where(avaliado, prof >= C.PONTO_CORTE_SAEB, np.nan)}))
            id_aluno += n
    alunos = pd.concat(alunos, ignore_index=True)

    # Indicador municipal por rede, calculado dos alunos como a Fase 2 faz
    av = alunos[alunos["avaliado"]].copy()
    av5 = av.assign(rede=5)
    linhas = []
    for (ano, idm, rede), g in pd.concat([av, av5]).groupby(["ano", "id_municipio", "rede"]):
        linhas.append({"ano": ano, "id_municipio": idm, "rede": rede,
                       "taxa_alfabetizacao": round(100 * g["alfabetizado"].mean(), 2),
                       "media_portugues": round(g["proficiencia_portugues"].mean(), 2)})
    ind = pd.DataFrame(linhas).merge(
        mun[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "nome_mesorregiao"]],
        on="id_municipio")
    ind["rede_nome"] = ind["rede"].map(C.NOME_REDE)
    ind["serie"] = 2
    part = alunos.groupby(["ano", "id_municipio"])["avaliado"].mean().mul(100).rename(
        "percentual_participacao").reset_index().assign(rede=3)
    ind = ind.merge(part, on=["ano", "id_municipio", "rede"], how="left")
    uf = ind.groupby(["ano", "sigla_uf", "rede"])["taxa_alfabetizacao"].mean().rename("taxa_uf")
    ind = ind.join(uf, on=["ano", "sigla_uf", "rede"])

    base23 = ind[(ind["ano"] == 2023) & (ind["rede"] == 3)].set_index("id_municipio")["taxa_alfabetizacao"]
    metas = []
    for idm, t in base23.items():
        for i, ano_meta in enumerate(range(2024, 2031)):
            metas.append({"id_municipio": idm, "rede": 3, "rede_nome": "Municipal",
                          "ano_meta": ano_meta,
                          "meta_taxa": round(min(80, t + (80 - t) * (i + 1) / 7 + 2), 2)})
    metas = pd.DataFrame(metas)
    meta_ano = metas.rename(columns={"ano_meta": "ano", "meta_taxa": "meta_taxa_ano"})
    ind = ind.merge(meta_ano[["id_municipio", "rede", "ano", "meta_taxa_ano"]],
                    on=["id_municipio", "rede", "ano"], how="left")
    ind["meta_taxa_2030"] = 80.0
    ind["gap_meta_ano"] = ind["taxa_alfabetizacao"] - ind["meta_taxa_ano"]
    ind["atingiu_meta_ano"] = ind["gap_meta_ano"] >= 0

    destino.mkdir(parents=True, exist_ok=True)
    alunos.to_parquet(destino / C.ARQ_ALUNO, index=False)
    ind.to_parquet(destino / C.ARQ_INDICADOR_MUNICIPIO, index=False)
    metas.to_parquet(destino / C.ARQ_META_MUNICIPIO, index=False)
    print(f"{len(alunos):,} alunos, {len(ind):,} linhas de indicador, {len(metas):,} metas -> {destino}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destino", type=Path, default=C.DIR_DADOS / "sintetico")
    parser.add_argument("--municipios", type=int, default=600)
    args = parser.parse_args()
    gerar(args.destino, args.municipios)
