"""Treino completo, de ponta a ponta.

Roda todas as etapas e salva o modelo, as tabelas e os gráficos que os notebooks usam.

    python -m src.pipeline            # completo, cerca de 10 minutos
    python -m src.pipeline --rapido   # amostras menores, para testar

Etapas:
 1. monta as bases (aluno, município e projeção) a partir da Gold
 2. separa treino e teste por município
 3. compara os modelos
 4. ajusta o XGBoost com Optuna
 5. curva de aprendizado
 6. escolhe o modelo final e o limiar e avalia no teste
 7. permutation importance e SHAP
 8. resultado por município, metas e projeção de 2025
 9. agrupamento de municípios (K-Means)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

from src import config as C
from src.data import carregar
from src.evaluation import metricas as M
from src.modeling import agrupamento, municipal, treino
from src.preprocessing import base_analitica as B
from src.preprocessing.features import colunas_entrada, selecionar_features
from src.visualization import graficos as G


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def salvar_json(obj, nome):
    C.DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    with open(C.DIR_REPORTS / nome, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def etapa_bases():
    log("Carregando Gold e fontes externas")
    alunos = carregar.carregar_alunos()
    indicador = carregar.carregar_indicador_municipio()
    metas = carregar.carregar_metas_municipio()
    ibge = carregar.carregar_ibge()
    censo, atlas = carregar.carregar_censo_escolar(), carregar.carregar_atlas()

    base = B.construir_base_aluno(alunos, indicador, metas, ibge, censo, atlas)
    projecao = B.construir_base_projecao(alunos, indicador, metas, ibge, censo, atlas)
    mun = B.construir_base_municipio(indicador, metas, ibge, alunos, atlas, censo_escolar=censo)
    mun_proj = B.construir_base_municipio(indicador, metas, ibge, alunos, atlas,
                                          ano_base=C.ANO_ALVO, ano_alvo=C.ANO_PROJECAO,
                                          rotulado=False, censo_escolar=censo)
    log(f"Base de alunos: {len(base):,} | projeção: {len(projecao):,} | municípios: {len(mun):,}")
    return alunos, indicador, base, projecao, mun, mun_proj


def etapa_divisao(base):
    divisao = treino.dividir_municipios(base[["id_municipio", "sigla_uf"]])
    C.DIR_PROCESSADO.mkdir(parents=True, exist_ok=True)
    divisao.to_csv(C.DIR_PROCESSADO / "divisao_municipios.csv", index=False)
    teste_ids = set(divisao.loc[divisao["conjunto"] == "teste", "id_municipio"])
    em_teste = base["id_municipio"].isin(teste_ids)
    log(f"Treino: {(~em_teste).sum():,} alunos | Teste: {em_teste.sum():,} alunos "
        f"({len(teste_ids)} municípios guardados)")
    return base[~em_teste].reset_index(drop=True), base[em_teste].reset_index(drop=True), teste_ids


def etapa_curva_aprendizado(treino_df, grupos, params, tamanhos):
    """Treina com quantidades crescentes de alunos e valida sempre nos mesmos municípios."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEMENTE)
    i_tr, i_va = next(gss.split(treino_df, groups=treino_df[C.GRUPO]))
    tr, va = treino_df.iloc[i_tr], treino_df.iloc[i_va]
    va = treino.amostrar(va, 100_000)
    cols = colunas_entrada(grupos)
    linhas = []
    for n in tamanhos:
        if n > len(tr):
            break
        parte = tr.sample(n=n, random_state=C.SEMENTE)
        modelo = treino.treinar(treino.construir_pipeline(grupos, treino.criar_xgboost(params), False),
                                parte[cols], parte[C.ALVO], parte[C.PESO])
        auc_tr = M.metricas_classificacao(parte[C.ALVO], modelo.predict_proba(parte[cols])[:, 1],
                                          parte[C.PESO])["roc_auc"]
        auc_va = M.metricas_classificacao(va[C.ALVO], modelo.predict_proba(va[cols])[:, 1],
                                          va[C.PESO])["roc_auc"]
        linhas.append({"n": n, "auc_treino": auc_tr, "auc_valid": auc_va})
        log(f"  n={n:>7,}  treino {auc_tr:.4f}  validação {auc_va:.4f}")
    return pd.DataFrame(linhas)


def nomear_clusters(perfil: pd.DataFrame) -> dict:
    """Dá nome a cada grupo a partir das medianas (o K-Means só devolve números).

    Os grupos se separam mais pelo contexto do que pelo resultado, então o nome descreve
    o contexto.
    """
    nomes = {}
    restantes = list(perfil.index)
    if "taxa_alfab_adultos" in perfil and len(restantes) >= 3:
        baixo = perfil.loc[restantes, "taxa_alfab_adultos"].idxmin()
        nomes[baixo] = "Alto analfabetismo adulto e pouca infraestrutura"
        restantes.remove(baixo)
        urbano = perfil.loc[restantes, "pct_urbana"].idxmax()
        nomes[urbano] = "Urbano com infraestrutura"
        restantes.remove(urbano)
        for c in restantes:
            nomes[c] = "Interior pouco povoado e de renda mais alta" if len(restantes) == 1 else f"Interior {c}"
    return {c: nomes.get(c, f"Perfil {c}") for c in perfil.index}


def main(rapido=False):
    inicio = time.perf_counter()
    amostra_treino = 60_000 if rapido else C.AMOSTRA_TREINO
    amostra_busca = 30_000 if rapido else C.AMOSTRA_BUSCA
    n_trials = 6 if rapido else C.N_TRIALS_OPTUNA
    resultados = {"configuracao": {"ano_base": C.ANO_BASE, "ano_alvo": C.ANO_ALVO,
                                   "amostra_treino": amostra_treino, "amostra_busca": amostra_busca,
                                   "n_trials": n_trials, "rapido": rapido}}

    alunos, indicador, base, projecao, mun, mun_proj = etapa_bases()
    grupos = selecionar_features(base)
    resultados["features"] = grupos
    log(f"Features: {sum(len(v) for v in grupos.values())} de entrada")

    treino_df, teste_df, teste_ids = etapa_divisao(base)
    resultados["divisao"] = {"alunos_treino": len(treino_df), "alunos_teste": len(teste_df),
                             "municipios_teste": len(teste_ids),
                             "prevalencia_treino": float(np.average(treino_df[C.ALVO], weights=treino_df[C.PESO])),
                             "prevalencia_teste": float(np.average(teste_df[C.ALVO], weights=teste_df[C.PESO]))}

    # 3. Comparação de modelos
    log("Comparando modelos (validação cruzada por município)")
    busca = treino.amostrar(treino_df, amostra_busca)
    comparacao, _ = treino.comparar_modelos(busca, grupos)
    comparacao.to_csv(C.DIR_REPORTS / "comparacao_modelos.csv", index=False)
    G.comparacao_modelos(comparacao)
    resultados["comparacao"] = comparacao.to_dict(orient="records")

    # 4. Optuna
    log(f"Otimizando XGBoost com Optuna ({n_trials} tentativas)")
    estudo = treino.otimizar_xgboost(busca, grupos, n_trials=n_trials)
    melhores = estudo.best_params
    G.historico_optuna(estudo)
    estudo.trials_dataframe().to_csv(C.DIR_REPORTS / "optuna_tentativas.csv", index=False)
    auc_padrao = comparacao.set_index("modelo").loc["XGBoost", "roc_auc_media"]
    resultados["optuna"] = {"melhor_auc_cv": estudo.best_value, "melhores_parametros": melhores,
                            "auc_xgboost_padrao_cv": auc_padrao,
                            "tentativas": len(estudo.trials),
                            "podadas": sum(t.state.name == "PRUNED" for t in estudo.trials)}
    log(f"Melhor AUC na busca: {estudo.best_value:.4f} (padrão: {auc_padrao:.4f})")

    # 5. Curva de aprendizado
    log("Curva de aprendizado")
    tamanhos = [2_000, 5_000, 15_000, 40_000, 100_000, 250_000, amostra_treino]
    curva = etapa_curva_aprendizado(treino.amostrar(treino_df, int(amostra_treino * 1.3)), grupos,
                                    melhores, sorted(set(t for t in tamanhos if t <= amostra_treino)))
    G.curva_aprendizado(curva)
    resultados["curva_aprendizado"] = curva.to_dict(orient="records")

    # 6. Escolha do modelo final e do limiar, os dois na validação
    log("Validação cruzada dos candidatos finais (XGBoost otimizado × Regressão Logística)")
    final_df = treino.amostrar(treino_df, amostra_treino)
    candidatos = {
        "XGBoost otimizado": lambda: treino.construir_pipeline(
            grupos, treino.criar_xgboost(melhores), False),
        "Regressão Logística": lambda: treino.construir_pipeline(
            grupos, treino.catalogo_modelos()["Regressão Logística"][0], True),
    }
    selecao, oofs = {}, {}
    for nome, fab in candidatos.items():
        dobras, oofs[nome] = treino.validacao_cruzada(fab, final_df, grupos)
        selecao[nome] = dobras[["roc_auc", "pr_auc", "log_loss", "brier", "auc_treino"]].mean().to_dict()
        selecao[nome]["roc_auc_desvio"] = dobras["roc_auc"].std()
        log(f"  {nome:22s} AUC {selecao[nome]['roc_auc']:.4f} ± {selecao[nome]['roc_auc_desvio']:.4f}")
    escolhido = max(selecao, key=lambda n: selecao[n]["roc_auc"])
    outro = next(n for n in candidatos if n != escolhido)
    fabrica = candidatos[escolhido]
    oof = oofs[escolhido]
    limiar = M.escolher_limiar(final_df[C.ALVO], oof, final_df[C.PESO])
    limiar_recall80 = M.escolher_limiar(final_df[C.ALVO], oof, final_df[C.PESO], recall_minimo=0.80)
    resultados["selecao_final"] = selecao
    resultados["modelo_final"] = escolhido
    resultados["limiar"] = {"f1_max": limiar, "recall_80": limiar_recall80}
    log(f"Modelo final: {escolhido} | limiar F1 {limiar:.3f} | limiar recall 80% {limiar_recall80:.3f}")

    log("Treinando modelo final e avaliando no teste")
    cols = colunas_entrada(grupos)
    modelo = treino.treinar(fabrica(), final_df[cols], final_df[C.ALVO], final_df[C.PESO])
    comparado = treino.treinar(candidatos[outro](), final_df[cols], final_df[C.ALVO], final_df[C.PESO])

    y_te, w_te = teste_df[C.ALVO].to_numpy(), teste_df[C.PESO].to_numpy()
    p_te = modelo.predict_proba(teste_df[cols])[:, 1]
    p_outro = comparado.predict_proba(teste_df[cols])[:, 1]
    p_base = np.full(len(y_te), np.average(final_df[C.ALVO], weights=final_df[C.PESO]))
    teste_metricas = M.metricas_classificacao(y_te, p_te, w_te, limiar)
    teste_metricas["roc_auc_ic95"] = M.ic_bootstrap_por_grupo(y_te, p_te, teste_df[C.GRUPO], w_te,
                                                              n=100 if rapido else 300)
    resultados["teste"] = {
        escolhido: teste_metricas,
        f"{escolhido} (limiar recall 80%)": M.metricas_classificacao(y_te, p_te, w_te, limiar_recall80),
        outro: M.metricas_classificacao(y_te, p_outro, w_te, limiar),
        "Baseline (proporção)": M.metricas_classificacao(y_te, p_base, w_te, limiar),
    }
    log(f"Teste: AUC {teste_metricas['roc_auc']:.4f} "
        f"(IC95 {teste_metricas['roc_auc_ic95'][0]:.4f}-{teste_metricas['roc_auc_ic95'][1]:.4f}) "
        f"| F1 {teste_metricas['f1']:.4f} | recall {teste_metricas['recall']:.4f}")
    G.curvas_roc_pr(y_te, {escolhido: p_te, outro: p_outro}, w_te)
    G.matriz_confusao(y_te, p_te, limiar, w_te)
    G.curva_calibracao(y_te, p_te, w_te)

    avaliacao = teste_df[["id_municipio", "nome_regiao", "rede_nome", C.ALVO, C.PESO]].rename(
        columns={C.ALVO: "y", C.PESO: "peso"}).assign(proba=p_te)
    for seg in ("nome_regiao", "rede_nome"):
        tabela = M.metricas_por_segmento(avaliacao, seg, limiar=limiar)
        tabela.to_csv(C.DIR_REPORTS / f"desempenho_por_{seg}.csv", index=False)
        G.desempenho_por_segmento(tabela, seg, f"modelo_desempenho_{seg}")

    C.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "nome": escolhido, "grupos": grupos, "limiar": limiar,
                 "parametros": melhores if escolhido.startswith("XGBoost") else None},
                C.DIR_MODELOS / "modelo_aluno.joblib")

    # 7. Interpretabilidade
    log("Permutation importance e SHAP")
    amostra_imp = treino.amostrar(teste_df, 20_000 if not rapido else 5_000)
    perm = permutation_importance(modelo, amostra_imp[cols], amostra_imp[C.ALVO], scoring="roc_auc",
                                  n_repeats=5, random_state=C.SEMENTE, n_jobs=1,
                                  sample_weight=amostra_imp[C.PESO])
    imp_perm = pd.DataFrame({"feature": cols, "queda_auc": perm.importances_mean,
                             "desvio": perm.importances_std}).sort_values("queda_auc", ascending=False)
    imp_perm.to_csv(C.DIR_REPORTS / "importancia_permutacao.csv", index=False)
    G.importancia(imp_perm.set_index("feature")["queda_auc"],
                  "Permutation importance (queda de AUC no teste)", "interp_permutation_importance",
                  erro=imp_perm.set_index("feature")["desvio"])

    import shap
    import matplotlib.pyplot as plt
    pre, estimador = modelo["preprocessamento"], modelo["modelo"]
    amostra_shap = treino.amostrar(teste_df, 3_000)
    X_shap = pre.transform(amostra_shap[cols])
    if hasattr(estimador, "get_booster"):
        explicador = shap.TreeExplainer(estimador)
    else:
        fundo = pre.transform(treino.amostrar(final_df, 1_000, semente=1)[cols])
        explicador = shap.LinearExplainer(estimador, shap.maskers.Independent(fundo, max_samples=1000))
    valores = explicador(X_shap)
    shap_medio = pd.Series(np.abs(valores.values).mean(axis=0), index=X_shap.columns).sort_values(
        ascending=False)
    shap_medio.rename("shap_medio_abs").to_csv(C.DIR_REPORTS / "importancia_shap.csv")
    for tipo, nome in ((None, "interp_shap_beeswarm"), ("bar", "interp_shap_barras")):
        plt.figure()
        shap.summary_plot(valores, X_shap, plot_type=tipo, max_display=15, show=False)
        G.salvar(plt.gcf(), nome)
        plt.close("all")
    for i, feat in enumerate(shap_medio.index[:2]):
        plt.figure()
        shap.plots.scatter(valores[:, feat], color=valores, show=False)
        G.salvar(plt.gcf(), f"interp_shap_dependencia_{i + 1}")
        plt.close("all")
    risco_idx = int(np.argmax(modelo.predict_proba(amostra_shap[cols])[:, 1]))
    plt.figure()
    shap.plots.waterfall(valores[risco_idx], max_display=12, show=False)
    G.salvar(plt.gcf(), "interp_shap_waterfall_aluno_risco")
    plt.close("all")
    resultados["importancia"] = {"permutacao_top10": imp_perm.head(10).to_dict(orient="records"),
                                 "shap_top10": shap_medio.head(10).to_dict()}

    # 8. Aplicação municipal
    log("Agregando previsões por município (teste)")
    agregado = municipal.agregar_previsao_aluno(teste_df, p_te)
    erro = (agregado["taxa_prevista"] - agregado["taxa_observada"]).abs()
    resultados["municipal_agregado_teste"] = {
        "municipios": len(agregado), "mae_pp": float(erro.mean()),
        "correlacao": float(agregado[["taxa_prevista", "taxa_observada"]].corr().iloc[0, 1])}
    G.previsto_vs_observado_municipio(agregado)

    # A meta é da rede municipal, então a taxa prevista usa só os alunos dessa rede.
    # sigma é o erro típico da taxa prevista no teste.
    mun_teste = mun[mun["id_municipio"].isin(teste_ids)].reset_index(drop=True)
    rede_mun = (teste_df["rede"] == C.REDE_MUNICIPAL).to_numpy()
    via_taxa = (municipal.agregar_previsao_aluno(teste_df[rede_mun], p_te[rede_mun])
                .merge(mun_teste[["id_municipio", "taxa_alvo", "meta_taxa_ano", "nao_atingiu_meta"]],
                       on="id_municipio"))
    sigma = float(np.sqrt(np.mean((via_taxa["taxa_prevista"] - via_taxa["taxa_alvo"]) ** 2)))
    via_taxa["prob"] = municipal.prob_abaixo_meta(via_taxa["taxa_prevista"],
                                                  via_taxa["meta_taxa_ano"], sigma)

    log("Classificador direto de meta municipal (comparação)")
    mun_treino = mun[~mun["id_municipio"].isin(teste_ids)].reset_index(drop=True)
    resumo_mun, _, grupos_mun = municipal.validar_modelos_municipais(mun_treino)
    melhor_mun = resumo_mun.sort_values("roc_auc", ascending=False).iloc[0]["modelo"]
    modelo_mun = municipal.treinar_modelo_municipal(mun_treino, melhor_mun, grupos_mun)
    p_mun = modelo_mun.predict_proba(mun_teste[colunas_entrada(grupos_mun)])[:, 1]
    gap_treino = mun["gap_meta_vs_ant"]
    gap_proj = mun_proj["gap_meta_vs_ant"]
    resultados["municipal_meta"] = {
        "pct_nao_atingiu_meta": float(mun["nao_atingiu_meta"].mean()),
        "sigma_pp": sigma,
        "via_taxa_prevista_teste": M.metricas_classificacao(via_taxa["nao_atingiu_meta"],
                                                            via_taxa["prob"]),
        "classificador_direto_cv": resumo_mun.to_dict(orient="records"),
        "classificador_direto_escolhido": melhor_mun,
        "classificador_direto_teste": M.metricas_classificacao(mun_teste["nao_atingiu_meta"], p_mun),
        "gap_meta_vs_ant_2024": [float(gap_treino.min()), float(gap_treino.max())],
        "gap_meta_vs_ant_2025": [float(gap_proj.min()), float(gap_proj.max())]}
    mm = resultados["municipal_meta"]
    log(f"  via taxa prevista: AUC {mm['via_taxa_prevista_teste']['roc_auc']:.3f} "
        f"(sigma {sigma:.1f} p.p.) | classificador direto: AUC "
        f"{mm['classificador_direto_teste']['roc_auc']:.3f}")

    log(f"Projeção {C.ANO_PROJECAO}: refit com todos os alunos")
    # O RS caiu quase 19 pontos em 2024. Se a projeção repetisse o efeito "ser do RS"
    # aprendido em 2024, a queda contaria duas vezes. Para UFs assim, a UF vira
    # "desconhecida" na projeção e os municípios recebem um alerta.
    uf_var = (indicador[indicador["rede"] == C.REDE_PUBLICA].groupby(["sigla_uf", "ano"])["taxa_uf"]
              .first().unstack())
    ufs_quebra = sorted(uf_var.index[(uf_var[C.ANO_ALVO] - uf_var[C.ANO_BASE]) < -10])
    resultados["ufs_quebra_serie"] = ufs_quebra
    log(f"  UFs com quebra de série (queda > 10 p.p.): {ufs_quebra}")
    projecao = projecao.copy()
    projecao["sigla_uf"] = projecao["sigla_uf"].astype(object).where(
        ~projecao["sigla_uf"].isin(ufs_quebra), np.nan)

    todos = base[cols]
    amostra_total = treino.amostrar(base, amostra_treino)
    modelo_total = treino.treinar(fabrica(), amostra_total[cols], amostra_total[C.ALVO],
                                  amostra_total[C.PESO])
    p_proj = modelo_total.predict_proba(projecao[cols])[:, 1]
    proj_mun = (projecao["rede"] == C.REDE_MUNICIPAL).to_numpy()
    agregado_proj = municipal.agregar_previsao_aluno(projecao[proj_mun], p_proj[proj_mun])
    risco = (mun_proj.merge(agregado_proj, on="id_municipio", how="left")
             .assign(gap_previsto=lambda d: d["meta_taxa_ano"] - d["taxa_prevista"],
                     alerta_quebra_serie=lambda d: d["sigla_uf"].isin(ufs_quebra)))
    risco["prob_nao_atingir_meta"] = municipal.prob_abaixo_meta(
        risco["taxa_prevista"], risco["meta_taxa_ano"], sigma)
    risco["faixa_risco"] = pd.cut(risco["prob_nao_atingir_meta"], [-0.01, 0.4, 0.6, 0.8, 1.0],
                                  labels=["Baixo", "Moderado", "Alto", "Muito alto"])
    risco = risco.dropna(subset=["taxa_prevista"]).sort_values("gap_previsto", ascending=False)
    # O PSI só faz sentido em variáveis contínuas; a taxa da UF tem 27 valores e fica de fora
    psi = {c: M.psi(todos[c], projecao[c]) for c in grupos["numericas"]
           if c in projecao.columns and todos[c].nunique() > 50}
    resultados["projecao"] = {
        "ano": C.ANO_PROJECAO, "municipios": len(risco),
        "pct_previstos_abaixo_meta": float((risco["gap_previsto"] > 0).mean()),
        "faixas": {str(k): int(v) for k, v in risco["faixa_risco"].value_counts().items()},
        "municipios_alerta_quebra": int(risco["alerta_quebra_serie"].sum()),
        "psi_max": max(psi.values()) if psi else None,
        "psi_features_instaveis": {k: v for k, v in psi.items() if v > 0.1}}
    colunas_risco = ["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao",
                     "mun_taxa_municipal_ant", "taxa_prevista", "meta_taxa_ano", "gap_previsto",
                     "prob_nao_atingir_meta", "faixa_risco", "alerta_quebra_serie"]
    risco[colunas_risco].rename(columns={"mun_taxa_municipal_ant": f"taxa_{C.ANO_ALVO}"}).to_csv(
        C.DIR_REPORTS / f"risco_municipios_{C.ANO_PROJECAO}.csv", index=False)
    G.ranking_risco(risco[~risco["alerta_quebra_serie"]])

    # 9. Agrupamento
    log("Agrupamento de municípios")
    perfil_base = mun.assign(taxa_atual=mun["taxa_alvo"],
                             variacao_pp=mun["taxa_alvo"] - mun["mun_taxa_municipal_ant"])
    X_perfil, _ = agrupamento.preparar(perfil_base)
    escolha = agrupamento.escolher_k(X_perfil)
    k = int(escolha.loc[escolha["silhueta"].idxmax(), "k"]) if rapido else max(
        3, int(escolha.loc[escolha["silhueta"].idxmax(), "k"]))
    resultado, _, pca, colunas = agrupamento.agrupar(perfil_base, k)
    perfil = agrupamento.perfil_clusters(resultado, colunas)
    nomes = nomear_clusters(perfil)
    perfil.insert(0, "perfil", pd.Series(nomes))
    perfil.to_csv(C.DIR_REPORTS / "perfil_clusters.csv")
    resultado[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "cluster"]].to_csv(
        C.DIR_REPORTS / "clusters_municipios.csv", index=False)
    G.escolha_k(escolha, k)
    G.clusters_pca(resultado, nomes, pca)
    G.clusters_por_regiao(resultado, nomes)
    resultados["clusters"] = {"k": k, "escolha_k": escolha.to_dict(orient="records"),
                              "perfil": perfil.reset_index().to_dict(orient="records")}

    resultados["duracao_s"] = time.perf_counter() - inicio
    salvar_json(resultados, "metricas.json")
    log(f"Concluído em {resultados['duracao_s'] / 60:.1f} min. Resultados em reports/ e images/.")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapido", action="store_true", help="amostras e buscas menores")
    main(parser.parse_args().rapido)
