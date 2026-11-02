"""Agrupamento de municípios por perfil (K-Means).

Responde "quais regiões têm padrões semelhantes?" sem partir do mapa: municípios de
estados diferentes caem no mesmo grupo se o perfil for parecido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C

FEATURES_PERFIL = [
    "taxa_atual", "variacao_pp", "mun_participacao_ant", "desvio_taxa_escolas_ant",
    "taxa_alfab_adultos", "pib_per_capita", "pct_urbana", "pct_esgoto_adequado", "densidade",
    "idhm", "gini",
]
LOG = ["pib_per_capita", "densidade"]


def preparar(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    colunas = [c for c in FEATURES_PERFIL if c in base.columns and base[c].notna().any()]
    X = base[colunas].copy()
    for c in LOG:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X, colunas


def pipeline_perfil(k: int) -> Pipeline:
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=C.SEMENTE)),
    ])


def escolher_k(X: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    """Inércia (cotovelo), silhueta e Davies-Bouldin para cada número de grupos."""
    linhas = []
    for k in ks:
        pipe = pipeline_perfil(k).fit(X)
        Z = pipe[:-1].transform(X)
        rotulos = pipe["kmeans"].labels_
        linhas.append({"k": k, "inercia": pipe["kmeans"].inertia_,
                       "silhueta": silhouette_score(Z, rotulos, sample_size=min(4000, len(Z)),
                                                    random_state=C.SEMENTE),
                       "davies_bouldin": davies_bouldin_score(Z, rotulos)})
    return pd.DataFrame(linhas)


def agrupar(base: pd.DataFrame, k: int):
    """Roda o K-Means e numera os grupos pela taxa média (0 = menor taxa)."""
    X, colunas = preparar(base)
    pipe = pipeline_perfil(k).fit(X)
    rotulos = pipe["kmeans"].labels_
    ordem = (pd.Series(base["taxa_atual"].to_numpy()).groupby(rotulos).mean()
             .sort_values().index.tolist())
    remapear = {antigo: novo for novo, antigo in enumerate(ordem)}
    rotulos = np.vectorize(remapear.get)(rotulos)

    Z = pipe[:-1].transform(X)
    pca = PCA(n_components=2, random_state=C.SEMENTE).fit(Z)
    coords = pca.transform(Z)
    resultado = base.assign(cluster=rotulos, pc1=coords[:, 0], pc2=coords[:, 1])
    return resultado, pipe, pca, colunas


def perfil_clusters(resultado: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    perfil = resultado.groupby("cluster")[colunas].median().round(2)
    perfil.insert(0, "municipios", resultado.groupby("cluster").size())
    return perfil
