"""Métricas, escolha do limiar, intervalo de confiança e PSI.

A classe positiva é "não alfabetizado", a que a política precisa encontrar. Recall e
precisão são calculados sobre ela.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def metricas_classificacao(y, proba, peso=None, limiar=0.5) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = (proba >= limiar).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba, sample_weight=peso),
        "pr_auc": average_precision_score(y, proba, sample_weight=peso),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), sample_weight=peso),
        "brier": brier_score_loss(y, proba, sample_weight=peso),
        "acuracia": accuracy_score(y, pred, sample_weight=peso),
        "acuracia_balanceada": balanced_accuracy_score(y, pred, sample_weight=peso),
        "precisao": precision_score(y, pred, sample_weight=peso, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=peso),
        "f1": f1_score(y, pred, sample_weight=peso),
        "limiar": limiar,
    }


def escolher_limiar(y, proba, peso=None, recall_minimo: float | None = None) -> float:
    """Escolhe o limiar a partir das previsões da validação, nunca do teste.

    Sem recall mínimo, usa o limiar com o melhor F1. Com recall mínimo, usa o limiar mais
    alto que ainda encontra essa fração dos alunos em risco.
    """
    precisao, recall, limiares = precision_recall_curve(y, proba, sample_weight=peso)
    precisao, recall = precisao[:-1], recall[:-1]
    if recall_minimo is not None:
        ok = np.where(recall >= recall_minimo)[0]
        return float(limiares[ok[-1]]) if len(ok) else float(limiares[0])
    f1 = 2 * precisao * recall / np.clip(precisao + recall, 1e-12, None)
    return float(limiares[np.nanargmax(f1)])


def ic_bootstrap_por_grupo(y, proba, grupos, peso=None, metrica=roc_auc_score,
                           n=300, semente=42) -> tuple[float, float]:
    """Intervalo de 95% sorteando municípios inteiros.

    Alunos da mesma cidade se parecem entre si. Sortear aluno por aluno daria um intervalo
    estreito demais; sortear municípios leva isso em conta.
    """
    rng = np.random.default_rng(semente)
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba), "g": np.asarray(grupos),
                       "w": np.ones(len(y)) if peso is None else np.asarray(peso)})
    indices = df.groupby("g").indices
    chaves = np.array(list(indices))
    valores = []
    for _ in range(n):
        sorteio = rng.choice(chaves, size=len(chaves), replace=True)
        linhas = np.concatenate([indices[k] for k in sorteio])
        amostra = df.iloc[linhas]
        if amostra["y"].nunique() < 2:
            continue
        valores.append(metrica(amostra["y"], amostra["p"], sample_weight=amostra["w"]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def metricas_por_segmento(df: pd.DataFrame, segmento: str, y="y", proba="proba", peso="peso",
                          limiar=0.5) -> pd.DataFrame:
    """Métricas por região ou rede, para ver onde o modelo erra mais."""
    linhas = []
    for valor, parte in df.groupby(segmento, observed=True):
        if parte[y].nunique() < 2:
            continue
        m = metricas_classificacao(parte[y], parte[proba], parte[peso], limiar)
        linhas.append({segmento: valor, "alunos": len(parte),
                       "taxa_nao_alfab_real": np.average(parte[y], weights=parte[peso]),
                       "taxa_nao_alfab_prevista": np.average(parte[proba], weights=parte[peso]),
                       **{k: m[k] for k in ("roc_auc", "recall", "precisao", "f1", "brier")}})
    return pd.DataFrame(linhas)


def psi(esperado, observado, bins: int = 10) -> float:
    """PSI: mede quanto a distribuição de uma variável mudou entre duas bases.

    Usado para comparar a base de treino com a de projeção. Abaixo de 0,1 está estável;
    entre 0,1 e 0,25, atenção; acima de 0,25, mudou.
    """
    esperado = pd.Series(esperado).dropna()
    observado = pd.Series(observado).dropna()
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(esperado, cortes)[0] / len(esperado)
    o = np.histogram(observado, cortes)[0] / len(observado)
    e, o = np.clip(e, 1e-6, None), np.clip(o, 1e-6, None)
    return float(np.sum((o - e) * np.log(o / e)))
