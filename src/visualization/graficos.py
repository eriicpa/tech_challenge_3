"""Gráficos do projeto. Todos salvam PNG em images/ para uso no README e no vídeo."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src import config as C

# Cores em ordem fixa (as três primeiras são seguras para daltonismo)
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"
SEQUENCIAL = "Blues"


def estilo():
    plt.rcParams.update({
        "figure.figsize": (10, 5), "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
        "axes.edgecolor": TEXTO_SECUNDARIO, "axes.labelcolor": TEXTO,
        "xtick.color": TEXTO_SECUNDARIO, "ytick.color": TEXTO_SECUNDARIO,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
        "axes.prop_cycle": matplotlib.cycler(color=CORES), "lines.linewidth": 2,
        "legend.frameon": False,
    })


estilo()


def salvar(fig, nome: str):
    C.DIR_IMAGENS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.DIR_IMAGENS / f"{nome}.png")
    return fig


# Análise exploratória
def distribuicao_proficiencia(alunos: pd.DataFrame, ano: int):
    dados = alunos[(alunos["ano"] == ano) & alunos["avaliado"]]
    fig, ax = plt.subplots()
    for i, (rede, parte) in enumerate(dados.groupby("rede_nome", observed=True)):
        ax.hist(parte["proficiencia_portugues"], bins=80, alpha=0.55, density=True,
                color=CORES[i], label=str(rede), weights=parte["peso_aluno"])
    ax.axvline(C.PONTO_CORTE_SAEB, color=TEXTO, linestyle="--", linewidth=1.2)
    ax.annotate(f"corte de alfabetização ({C.PONTO_CORTE_SAEB})",
                (C.PONTO_CORTE_SAEB, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=TEXTO_SECUNDARIO)
    ax.set(title=f"Proficiência em leitura dos alunos do 2º ano, {ano}",
           xlabel="Proficiência (escala Saeb)", ylabel="Densidade")
    ax.legend(title="Rede")
    return salvar(fig, "eda_distribuicao_proficiencia")


def taxa_por_categoria(df: pd.DataFrame, categoria: str, valor: str, titulo: str, nome: str,
                       referencia: float | None = None):
    ordem = df.sort_values(valor)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(ordem))))
    ax.barh(ordem[categoria].astype(str), ordem[valor], color=CORES[0], height=0.7)
    if referencia is not None:
        ax.axvline(referencia, color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
    for y, v in enumerate(ordem[valor]):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", color=TEXTO_SECUNDARIO, fontsize=8)
    ax.set(title=titulo, xlabel="%")
    ax.grid(axis="y", visible=False)
    return salvar(fig, nome)


def dispersao_relacao(df, x, y, titulo, nome, xlabel=None, ylabel=None, cor_por=None,
                      logx=False):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if cor_por:
        for i, (grupo, parte) in enumerate(df.groupby(cor_por, observed=True)):
            ax.scatter(parte[x], parte[y], s=9, alpha=0.45, color=CORES[i % len(CORES)],
                       label=str(grupo), linewidths=0)
        ax.legend(title=cor_por, markerscale=2.5, ncol=1)
    else:
        ax.scatter(df[x], df[y], s=9, alpha=0.35, color=CORES[0], linewidths=0)
    validos = df[[x, y]].dropna()
    if logx:
        ax.set_xscale("log")
    xs = np.log(validos[x].clip(lower=1e-9)) if logx else validos[x]
    coef = np.polyfit(xs, validos[y], 1)
    grade = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(np.exp(grade) if logx else grade, np.polyval(coef, grade), color=TEXTO, linewidth=1.5)
    r = validos.corr(method="spearman").iloc[0, 1]
    ax.set(title=f"{titulo}  (Spearman ρ = {r:.2f})", xlabel=xlabel or x, ylabel=ylabel or y)
    return salvar(fig, nome)


def matriz_correlacao(df: pd.DataFrame, colunas: list[str], nome: str, titulo: str):
    corr = df[colunas].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.62 * len(colunas) + 3, 0.55 * len(colunas) + 2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(colunas)), colunas, rotation=60, ha="right")
    ax.set_yticks(range(len(colunas)), colunas)
    for i in range(len(colunas)):
        for j in range(len(colunas)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else TEXTO)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title(titulo)
    return salvar(fig, nome)


def boxplot_por_grupo(df, valor, grupo, titulo, nome, ylabel=None):
    ordem = df.groupby(grupo, observed=True)[valor].median().sort_values().index
    fig, ax = plt.subplots()
    dados = [df.loc[df[grupo] == g, valor].dropna() for g in ordem]
    caixa = ax.boxplot(dados, tick_labels=[str(g) for g in ordem], patch_artist=True,
                       showfliers=False, widths=0.55)
    for patch in caixa["boxes"]:
        patch.set(facecolor=CORES[0], alpha=0.35, edgecolor=CORES[0])
    for mediana in caixa["medians"]:
        mediana.set(color=TEXTO, linewidth=1.5)
    ax.set(title=titulo, ylabel=ylabel or valor)
    return salvar(fig, nome)


# Modelagem
def comparacao_modelos(resumo: pd.DataFrame):
    ordem = resumo.sort_values("roc_auc_media")
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(ordem))
    ax.barh(y - 0.18, ordem["auc_treino_media"], height=0.34, color=CORES[1], label="Treino")
    ax.barh(y + 0.18, ordem["roc_auc_media"], height=0.34, color=CORES[0],
            xerr=ordem["roc_auc_desvio"], label="Validação (municípios não vistos)")
    ax.set_yticks(y, ordem["modelo"])
    for yi, v, d in zip(y, ordem["roc_auc_media"], ordem["roc_auc_desvio"].fillna(0)):
        ax.text(v + d + 0.008, yi + 0.18, f"{v:.3f}", va="center", fontsize=8, color=TEXTO_SECUNDARIO)
    ax.set(xlim=(0.45, 1), xlabel="ROC-AUC", title="Comparação de modelos (validação cruzada por município)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    return salvar(fig, "modelo_comparacao")


def curvas_roc_pr(y, probas: dict, peso=None, nome="modelo_curvas_roc_pr"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for i, (rotulo, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y, p, sample_weight=peso)
        a1.plot(fpr, tpr, color=CORES[i], label=rotulo)
        prec, rec, _ = precision_recall_curve(y, p, sample_weight=peso)
        a2.plot(rec, prec, color=CORES[i], label=rotulo)
    a1.plot([0, 1], [0, 1], color=GRADE, linestyle="--")
    prevalencia = np.average(y, weights=peso)
    a2.axhline(prevalencia, color=GRADE, linestyle="--")
    a1.set(title="Curva ROC", xlabel="Falsos positivos", ylabel="Verdadeiros positivos")
    a2.set(title="Precisão × Recall (classe: não alfabetizado)", xlabel="Recall", ylabel="Precisão")
    a1.legend(loc="lower right")
    return salvar(fig, nome)


def matriz_confusao(y, proba, limiar, peso=None):
    pred = (np.asarray(proba) >= limiar).astype(int)
    m = confusion_matrix(y, pred, sample_weight=peso, normalize="true")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(m, cmap=SEQUENCIAL, vmin=0, vmax=1)
    rotulos = ["Alfabetizado", "Não alfabetizado"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:.1%}", ha="center", va="center",
                    color="white" if m[i, j] > 0.5 else TEXTO, fontsize=12)
    ax.set(xlabel="Previsto", ylabel="Real", title=f"Matriz de confusão (limiar {limiar:.2f})")
    ax.grid(False)
    return salvar(fig, "modelo_matriz_confusao")


def curva_calibracao(y, proba, peso=None):
    reais, previstos = calibration_curve(y, proba, n_bins=15, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GRADE, linestyle="--", label="Calibração perfeita")
    ax.plot(previstos, reais, marker="o", markersize=6, color=CORES[0], label="Modelo")
    ax.set(xlabel="Probabilidade prevista", ylabel="Frequência observada",
           title="Calibração no conjunto de teste")
    ax.legend()
    return salvar(fig, "modelo_calibracao")


def curva_aprendizado(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["n"], df["auc_treino"], marker="o", markersize=6, color=CORES[1], label="Treino")
    ax.plot(df["n"], df["auc_valid"], marker="o", markersize=6, color=CORES[0],
            label="Validação (municípios não vistos)")
    ax.set_xscale("log")
    ax.set(xlabel="Alunos no treino (escala log)", ylabel="ROC-AUC",
           title="Curva de aprendizado")
    ax.legend()
    return salvar(fig, "modelo_curva_aprendizado")


def historico_optuna(estudo):
    valores = [t.value for t in estudo.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(len(valores)), valores, s=18, color=CORES[0], label="Tentativa")
    ax.plot(np.maximum.accumulate(valores), color=CORES[1], label="Melhor até aqui")
    ax.set(xlabel="Tentativa", ylabel="ROC-AUC (validação)", title="Otimização com Optuna")
    ax.legend()
    return salvar(fig, "modelo_optuna_historico")


def importancia(serie: pd.Series, titulo: str, nome: str, top=20, erro=None):
    s = serie.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(s) + 1.5))
    ax.barh(s.index, s.values, color=CORES[0], height=0.7,
            xerr=None if erro is None else erro.reindex(s.index).values)
    ax.set(title=titulo)
    ax.grid(axis="y", visible=False)
    return salvar(fig, nome)


def desempenho_por_segmento(tabela: pd.DataFrame, segmento: str, nome: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tabela))
    ax.bar(x - 0.2, 100 * tabela["taxa_nao_alfab_real"], width=0.4, color=CORES[1], label="Real")
    ax.bar(x + 0.2, 100 * tabela["taxa_nao_alfab_prevista"], width=0.4, color=CORES[0],
           label="Prevista")
    ax.set_xticks(x, tabela[segmento].astype(str))
    for xi, auc in zip(x, tabela["roc_auc"]):
        ax.text(xi, 2, f"AUC {auc:.2f}", ha="center", color="white", fontsize=8)
    ax.set(ylabel="% não alfabetizados", title=f"Real × previsto por {segmento} (teste)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    return salvar(fig, nome)


# Aplicação estratégica
def previsto_vs_observado_municipio(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(df["taxa_prevista"], df["taxa_observada"], s=np.clip(df["alunos"] / 8, 6, 120),
               alpha=0.45, color=CORES[0], linewidths=0)
    ax.plot([0, 100], [0, 100], color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
    r = np.corrcoef(df["taxa_prevista"], df["taxa_observada"])[0, 1]
    mae = np.mean(np.abs(df["taxa_prevista"] - df["taxa_observada"]))
    ax.set(xlabel="Taxa prevista pelo modelo (%)", ylabel="Taxa observada (%)",
           title=f"Municípios de teste: r = {r:.2f}, erro médio = {mae:.1f} p.p.",
           xlim=(0, 100), ylim=(0, 100))
    return salvar(fig, "aplicacao_previsto_vs_observado")


def escolha_k(tabela: pd.DataFrame, k_escolhido: int):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(tabela["k"], tabela["inercia"], marker="o", markersize=6, color=CORES[0])
    a1.set(title="Cotovelo (inércia)", xlabel="k")
    a2.plot(tabela["k"], tabela["silhueta"], marker="o", markersize=6, color=CORES[0])
    a2.set(title="Silhueta", xlabel="k")
    for a in (a1, a2):
        a.axvline(k_escolhido, color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
    return salvar(fig, "cluster_escolha_k")


def clusters_pca(resultado: pd.DataFrame, nomes: dict, pca):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(resultado["cluster"].unique()):
        parte = resultado[resultado["cluster"] == c]
        ax.scatter(parte["pc1"], parte["pc2"], s=8, alpha=0.5, color=CORES[c % len(CORES)],
                   label=f"{c}: {nomes.get(c, '')} ({len(parte)})", linewidths=0)
    v = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({v[0]:.0%} da variância)", ylabel=f"PC2 ({v[1]:.0%})",
           title="Perfis de municípios (K-Means, projeção PCA)")
    ax.legend(markerscale=2.5, loc="best")
    return salvar(fig, "cluster_pca")


def clusters_por_regiao(resultado: pd.DataFrame, nomes: dict):
    tabela = pd.crosstab(resultado["nome_regiao"], resultado["cluster"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    esquerda = np.zeros(len(tabela))
    for c in tabela.columns:
        ax.barh(tabela.index, tabela[c], left=esquerda, color=CORES[c % len(CORES)],
                label=f"{c}: {nomes.get(c, '')}", edgecolor="white", linewidth=2)
        esquerda += tabela[c].to_numpy()
    ax.set(xlabel="% dos municípios da região", title="Composição de perfis por região")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", visible=False)
    return salvar(fig, "cluster_por_regiao")


def ranking_risco(df: pd.DataFrame, top=20):
    top_df = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.33 * top + 1.5))
    rotulos = top_df["nome_municipio"].astype(str) + " (" + top_df["sigla_uf"].astype(str) + ")"
    ax.barh(rotulos, top_df["taxa_prevista"], color=CORES[7], height=0.7, label="Taxa prevista")
    ax.scatter(top_df["meta_taxa_ano"], rotulos, color=TEXTO, marker="|", s=160, zorder=3,
               label="Meta")
    ax.set(xlabel="% alfabetizados", xlim=(0, 100),
           title=f"{top} municípios com maior distância prevista da meta de {C.ANO_PROJECAO}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.grid(axis="y", visible=False)
    return salvar(fig, "aplicacao_ranking_risco")
