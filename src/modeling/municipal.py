"""Resultados por município.

Quem decide olha para municípios, não para alunos. Aqui ficam:

1. a taxa prevista de cada município, que é a média das previsões dos seus alunos;
2. a chance de o município ficar abaixo da meta, comparando a taxa prevista com a meta;
3. um classificador direto de meta, usado só como comparação. Ele funciona para 2024,
   mas não para 2025, porque a meta de 2025 está muito mais longe da taxa anterior do
   que qualquer caso visto no treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src.evaluation.metricas import metricas_classificacao
from src.modeling.treino import construir_pipeline, criar_xgboost
from src.preprocessing.features import colunas_entrada

ALVO_MUNICIPAL = "nao_atingiu_meta"

FEATURES_MUNICIPAIS_NUM = [
    "mun_taxa_municipal_ant", "mun_taxa_publica_ant", "mun_taxa_estadual_ant", "mun_media_ant",
    "mun_participacao_ant", "uf_taxa_ant", "meta_taxa_ano", "n_escolas",
    "desvio_taxa_escolas_ant", "pct_escolas_abaixo_50_ant", "populacao", "densidade",
    "pib_per_capita", "pct_va_agro", "pct_urbana", "taxa_alfab_adultos", "pct_esgoto_adequado",
    "idhm", "idhm_e", "idhm_r", "gini", "renda_pc",
    "inep_saeb5_lp_2023", "inep_saeb5_lp_2021", "inep_saeb5_lp_tendencia", "inep_ideb_2023",
    "inep_aprovacao_2ano_2023", "inep_inse", "inep_pct_inse_baixo", "inep_pct_integral",
    "inep_pct_biblioteca", "inep_pct_internet_alunos", "inep_pct_coordenador",
    "inep_pct_material_infantil", "inep_alunos_por_turma", "inep_alunos_por_docente",
    "inep_pct_rural",
]
FEATURES_MUNICIPAIS_CAT = ["nome_regiao", "sigla_uf"]


def grupos_municipais(base: pd.DataFrame) -> dict:
    def disponiveis(lista):
        return [c for c in lista if c in base.columns and base[c].notna().any()]
    return {"numericas": disponiveis(FEATURES_MUNICIPAIS_NUM),
            "categoricas": disponiveis(FEATURES_MUNICIPAIS_CAT),
            "alta_cardinalidade": []}


def modelos_municipais():
    return {
        "Regressão Logística": (LogisticRegression(C=0.5, max_iter=2000), True),
        "XGBoost": (criar_xgboost({"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4,
                                   "min_child_weight": 5, "subsample": 0.8,
                                   "colsample_bytree": 0.8, "reg_lambda": 5.0}), False),
    }


def validar_modelos_municipais(base_treino: pd.DataFrame, n_folds=C.N_FOLDS):
    """Validação cruzada com uma linha por município."""
    grupos = grupos_municipais(base_treino)
    X = base_treino[colunas_entrada(grupos)]
    y = base_treino[ALVO_MUNICIPAL].to_numpy()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=C.SEMENTE)
    resumo, oofs = [], {}
    for nome, (estimador, escalar) in modelos_municipais().items():
        oof = np.zeros(len(X))
        for i_treino, i_valid in cv.split(X, y):
            modelo = construir_pipeline(grupos, clone(estimador), escalar)
            modelo.fit(X.iloc[i_treino], y[i_treino])
            oof[i_valid] = modelo.predict_proba(X.iloc[i_valid])[:, 1]
        oofs[nome] = oof
        resumo.append({"modelo": nome, **metricas_classificacao(y, oof)})
    return pd.DataFrame(resumo), oofs, grupos


def treinar_modelo_municipal(base: pd.DataFrame, nome: str, grupos: dict):
    estimador, escalar = modelos_municipais()[nome]
    modelo = construir_pipeline(grupos, clone(estimador), escalar)
    return modelo.fit(base[colunas_entrada(grupos)], base[ALVO_MUNICIPAL])


def agregar_previsao_aluno(base_aluno: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Taxa prevista de cada município a partir das previsões dos alunos.

    Usa o peso amostral, como a taxa oficial. Se houver o resultado real, traz também a
    taxa observada, para comparar.
    """
    df = base_aluno[["id_municipio", C.PESO]].copy()
    df["proba"] = proba
    if C.ALVO in base_aluno.columns:
        df["y"] = base_aluno[C.ALVO].to_numpy()
    df["_wp"] = df["proba"] * df[C.PESO]
    agregado = df.groupby("id_municipio").agg(alunos=(C.PESO, "size"), _w=(C.PESO, "sum"),
                                              _wp=("_wp", "sum"))
    agregado["taxa_prevista"] = 100 * (1 - agregado["_wp"] / agregado["_w"])
    if "y" in df.columns:
        df["_wy"] = df["y"] * df[C.PESO]
        agregado["taxa_observada"] = 100 * (1 - df.groupby("id_municipio")["_wy"].sum()
                                            / agregado["_w"])
    return agregado.drop(columns=["_w", "_wp"]).reset_index()


def prob_abaixo_meta(taxa_prevista, meta, sigma: float):
    """Chance de a taxa real ficar abaixo da meta.

    Supõe que o erro da taxa prevista segue uma curva normal com desvio sigma (o erro
    medido no teste). Como a meta não entra no modelo, funciona para qualquer meta futura.
    """
    return norm.cdf((np.asarray(meta, dtype=float) - np.asarray(taxa_prevista, dtype=float)) / sigma)
