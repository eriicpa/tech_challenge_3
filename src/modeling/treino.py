"""Divisão dos dados, modelos, validação cruzada e Optuna.

A validação separa municípios inteiros. Alunos da mesma cidade têm as mesmas
informações de contexto; se a cidade estivesse no treino e na validação, o resultado
sairia bom demais. Com StratifiedGroupKFold, cada rodada valida em cidades que o
modelo não viu.
"""
from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config as C
from src.evaluation.metricas import metricas_classificacao
from src.preprocessing.features import colunas_entrada, construir_preprocessador


# Divisão treino / teste
def dividir_municipios(territorio: pd.DataFrame, proporcao_teste=C.PROPORCAO_TESTE,
                       semente=C.SEMENTE) -> pd.DataFrame:
    """Sorteia 20% dos municípios de cada UF para o teste.

    O teste fica guardado até o fim. Sortear dentro de cada UF garante que todo estado
    tenha municípios no treino e no teste.
    """
    rng = np.random.default_rng(semente)
    territorio = territorio.drop_duplicates("id_municipio")[["id_municipio", "sigla_uf"]]
    teste = []
    for _, parte in territorio.groupby("sigla_uf", observed=True):
        ids = parte["id_municipio"].to_numpy()
        n = max(1, int(round(len(ids) * proporcao_teste)))
        teste.extend(rng.choice(ids, size=n, replace=False))
    territorio = territorio.assign(conjunto="treino")
    territorio.loc[territorio["id_municipio"].isin(teste), "conjunto"] = "teste"
    return territorio.reset_index(drop=True)


def amostrar(df: pd.DataFrame, n: int, semente=C.SEMENTE) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=semente)


# Modelos
def catalogo_modelos(n_jobs: int = -1) -> dict[str, tuple[object, bool]]:
    """Modelos comparados. O True/False diz se as variáveis numéricas são padronizadas.

    A padronização só faz diferença na regressão logística. Nas árvores fica desligada,
    para as variáveis continuarem na escala original.
    """
    return {
        "Baseline (proporção)": (DummyClassifier(strategy="prior"), False),
        "Regressão Logística": (LogisticRegression(C=1.0, max_iter=2000), True),
        "Random Forest": (RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, max_features="sqrt", n_jobs=n_jobs,
            random_state=C.SEMENTE), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
            random_state=C.SEMENTE), False),
        "XGBoost": (XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=n_jobs,
            eval_metric="logloss", random_state=C.SEMENTE), False),
    }


def construir_pipeline(grupos: dict, estimador, escalar: bool) -> Pipeline:
    return Pipeline([("preprocessamento", construir_preprocessador(grupos, escalar=escalar)),
                     ("modelo", estimador)])


def treinar(pipeline: Pipeline, X, y, peso=None) -> Pipeline:
    if peso is None:
        return pipeline.fit(X, y)
    return pipeline.fit(X, y, modelo__sample_weight=np.asarray(peso))


# Validação cruzada por grupo
def validacao_cruzada(fabrica, base: pd.DataFrame, grupos: dict, n_folds=C.N_FOLDS,
                      semente=C.SEMENTE, usar_peso=True, callback=None):
    """Validação cruzada separando municípios.

    fabrica() cria um pipeline novo em cada rodada, para nada aprendido numa rodada
    passar para a seguinte. Devolve as métricas de cada rodada e a previsão de validação
    de cada linha, usada depois para escolher o limiar sem tocar no teste.
    """
    X = base[colunas_entrada(grupos)]
    y = base[C.ALVO].to_numpy()
    peso = base[C.PESO].to_numpy() if usar_peso else None
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=semente)

    oof = np.full(len(base), np.nan)
    linhas = []
    for dobra, (i_treino, i_valid) in enumerate(cv.split(X, y, base[C.GRUPO])):
        inicio = time.perf_counter()
        modelo = treinar(fabrica(), X.iloc[i_treino], y[i_treino],
                         None if peso is None else peso[i_treino])
        proba_treino = modelo.predict_proba(X.iloc[i_treino])[:, 1]
        oof[i_valid] = modelo.predict_proba(X.iloc[i_valid])[:, 1]
        p_valid = None if peso is None else peso[i_valid]
        p_treino = None if peso is None else peso[i_treino]
        m_valid = metricas_classificacao(y[i_valid], oof[i_valid], p_valid)
        m_treino = metricas_classificacao(y[i_treino], proba_treino, p_treino)
        linhas.append({"dobra": dobra + 1, "segundos": time.perf_counter() - inicio,
                       "auc_treino": m_treino["roc_auc"], **m_valid})
        if callback is not None:
            callback(dobra, m_valid["roc_auc"])
    return pd.DataFrame(linhas), oof


def comparar_modelos(base, grupos, n_folds=C.N_FOLDS, modelos=None, verbose=True):
    """Roda a mesma validação cruzada em todos os modelos."""
    modelos = modelos or catalogo_modelos()
    resumo, oofs = [], {}
    for nome, (estimador, escalar) in modelos.items():
        from sklearn.base import clone

        def fabrica(est=estimador, esc=escalar):
            return construir_pipeline(grupos, clone(est), esc)

        dobras, oof = validacao_cruzada(fabrica, base, grupos, n_folds)
        oofs[nome] = oof
        linha = {"modelo": nome}
        for metrica in ("roc_auc", "pr_auc", "log_loss", "brier", "f1", "auc_treino", "segundos"):
            linha[f"{metrica}_media"] = dobras[metrica].mean()
            linha[f"{metrica}_desvio"] = dobras[metrica].std()
        linha["gap_treino_valid"] = linha["auc_treino_media"] - linha["roc_auc_media"]
        resumo.append(linha)
        if verbose:
            print(f"{nome:24s} AUC {linha['roc_auc_media']:.4f} ± {linha['roc_auc_desvio']:.4f}"
                  f" | treino {linha['auc_treino_media']:.4f}"
                  f" | {linha['segundos_media']:.1f}s/dobra")
    return pd.DataFrame(resumo).sort_values("roc_auc_media", ascending=False), oofs


# Otimização de hiperparâmetros (Optuna)
def espaco_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }


def criar_xgboost(params: dict, n_jobs=-1) -> XGBClassifier:
    return XGBClassifier(**params, tree_method="hist", eval_metric="logloss", n_jobs=n_jobs,
                         random_state=C.SEMENTE)


def otimizar_xgboost(base, grupos, n_trials=C.N_TRIALS_OPTUNA, n_folds=3, timeout=None):
    """Busca de hiperparâmetros do XGBoost com Optuna.

    O objetivo é a AUC média na validação por município. Tentativas que começam mal são
    interrompidas na primeira rodada. Todas usam as mesmas divisões, então a comparação
    é justa.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial):
        params = espaco_xgboost(trial)

        def relatar(dobra, auc):
            trial.report(auc, dobra)
            if trial.should_prune():
                raise optuna.TrialPruned()

        dobras, _ = validacao_cruzada(
            lambda: construir_pipeline(grupos, criar_xgboost(params), False),
            base, grupos, n_folds=n_folds, callback=relatar)
        trial.set_user_attr("auc_treino", dobras["auc_treino"].mean())
        return dobras["roc_auc"].mean()

    estudo = optuna.create_study(
        direction="maximize", study_name="xgboost_alfabetizacao",
        sampler=optuna.samplers.TPESampler(seed=C.SEMENTE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    estudo.optimize(objetivo, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return estudo
