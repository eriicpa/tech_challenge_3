"""Tratamento dos dados que fica dentro do pipeline do modelo.

Como tudo está dentro do Pipeline do scikit-learn, cada etapa aprende só com os dados
de treino:

1. CriadorFeatures: passa população, densidade, PIB e tamanho da escola para
   logaritmo. Não aprende nada dos dados, então não vaza informação.
2. Numéricas: preenche nulos com a mediana, marca onde havia nulo e padroniza.
3. Categóricas com poucas categorias: preenche nulos e faz one-hot.
4. Mesorregião (cerca de 140 categorias): target encoding com validação cruzada
   interna, para cada linha não ver o próprio alvo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src import config as C


# Colunas que o CriadorFeatures troca pela versão em logaritmo
COLUNAS_SUBSTITUIDAS = ("populacao", "densidade", "pib_per_capita", "escola_matriculados")


class CriadorFeatures(BaseEstimator, TransformerMixin):
    """Cria variáveis novas. Não aprende nada no fit."""

    def fit(self, X, y=None):
        self.colunas_entrada_ = list(X.columns)
        return self

    def transform(self, X):
        X = X[self.colunas_entrada_].copy()
        if {"meta_taxa_ano", "mun_taxa_municipal_ant"} <= set(X.columns):
            # Quanto a rede municipal precisa subir para chegar à meta
            X["gap_meta_vs_ant"] = X["meta_taxa_ano"] - X["mun_taxa_municipal_ant"]
        for coluna in ("populacao", "densidade", "escola_matriculados"):
            if coluna in X.columns:
                X[f"log_{coluna}"] = np.log1p(X[coluna].clip(lower=0))
        if "pib_per_capita" in X.columns:
            X["log_pib_per_capita"] = np.log(X["pib_per_capita"].clip(lower=1))
        # Categorias do pandas viram texto para o preenchimento de nulos aceitar valores novos
        for coluna in X.select_dtypes(include=["category", "string"]).columns:
            X[coluna] = X[coluna].astype(object).where(X[coluna].notna(), np.nan)
        return X.drop(columns=[c for c in COLUNAS_SUBSTITUIDAS if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        # Calculado sem chamar transform: com set_output("pandas"), o transform chama este
        # método, e os dois ficariam se chamando sem parar.
        entrada = list(self.colunas_entrada_)
        criadas = [c for c in _numericas_apos_criacao(entrada) if c not in entrada]
        return np.asarray([c for c in entrada if c not in COLUNAS_SUBSTITUIDAS] + criadas)


def selecionar_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """Variáveis disponíveis nesta base, separadas por tipo.

    As fontes opcionais só entram se o arquivo existir e a coluna tiver algum valor.
    """
    def disponiveis(lista):
        return [c for c in lista if c in df.columns and df[c].notna().any()]

    numericas = disponiveis(C.FEATURES_NUMERICAS)
    categoricas = disponiveis(C.FEATURES_CATEGORICAS)
    alta = disponiveis(C.FEATURES_ALTA_CARDINALIDADE)
    proibidas = set(numericas + categoricas + alta) & set(C.COLUNAS_PROIBIDAS)
    if proibidas:
        raise ValueError(f"Coluna com vazamento selecionada como feature: {proibidas}")
    return {"numericas": numericas, "categoricas": categoricas, "alta_cardinalidade": alta}


def _numericas_apos_criacao(numericas: list[str]) -> list[str]:
    """Nomes das variáveis numéricas depois do CriadorFeatures."""
    trocadas = {"populacao": "log_populacao", "densidade": "log_densidade",
                "pib_per_capita": "log_pib_per_capita",
                "escola_matriculados": "log_escola_matriculados"}
    saida = [trocadas.get(c, c) for c in numericas]
    if {"meta_taxa_ano", "mun_taxa_municipal_ant"} <= set(numericas):
        saida.append("gap_meta_vs_ant")
    return saida


def construir_preprocessador(grupos: dict[str, list[str]], escalar: bool = True) -> Pipeline:
    """Junta o CriadorFeatures e o ColumnTransformer e devolve um DataFrame com nomes."""
    numericas = _numericas_apos_criacao(grupos["numericas"])
    transformadores = [
        ("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler() if escalar else "passthrough"),
        ]), numericas),
    ]
    if grupos["categoricas"]:
        transformadores.append(("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                        sparse_output=False)),
        ]), grupos["categoricas"]))
    if grupos["alta_cardinalidade"]:
        transformadores.append(("alta", Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("codificar", TargetEncoder(
                target_type="binary", cv=KFold(5, shuffle=True, random_state=C.SEMENTE))),
        ]), grupos["alta_cardinalidade"]))

    colunas = ColumnTransformer(transformadores, remainder="drop",
                                verbose_feature_names_out=False)
    return Pipeline([("criar", CriadorFeatures()), ("colunas", colunas)]).set_output(
        transform="pandas")


def colunas_entrada(grupos: dict[str, list[str]]) -> list[str]:
    return grupos["numericas"] + grupos["categoricas"] + grupos["alta_cardinalidade"]
