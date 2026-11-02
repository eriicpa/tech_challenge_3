"""Testes das regras contra vazamento de dados.

Usam a Gold fictícia, então não precisam da exportação do Databricks:

    python -m pytest tests
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from tests.dados_sinteticos import gerar


@pytest.fixture(scope="session")
def bases(tmp_path_factory):
    destino = tmp_path_factory.mktemp("gold")
    gerar(destino, n_municipios=150, semente=3)
    C.DIR_GOLD = destino
    from src.data import carregar as L
    from src.preprocessing import base_analitica as B
    alunos = L.carregar_alunos()
    ind, metas, ibge = L.carregar_indicador_municipio(), L.carregar_metas_municipio(), L.carregar_ibge()
    base = B.construir_base_aluno(alunos, ind, metas, ibge)
    return alunos, ind, base


def test_nenhuma_coluna_proibida_vira_feature(bases):
    from src.preprocessing.features import colunas_entrada, selecionar_features
    _, _, base = bases
    features = colunas_entrada(selecionar_features(base))
    assert not set(features) & set(C.COLUNAS_PROIBIDAS)


def test_base_nao_carrega_resultado_do_ciclo_alvo(bases):
    _, _, base = bases
    assert "proficiencia_portugues" not in base.columns
    assert "alfabetizado" not in base.columns
    assert (base["ano"] == C.ANO_ALVO).all()


def test_escola_liga_ao_municipio_do_mesmo_ano():
    """O código da escola muda de ano para ano: o município tem que vir do próprio ano."""
    from src.preprocessing.base_analitica import resultado_escolas
    alunos = pd.DataFrame({
        "ano": [C.ANO_BASE, C.ANO_BASE, C.ANO_ALVO],
        "id_escola": [1, 1, 1],
        "id_municipio": [10, 10, 20],
        "avaliado": True, "alfabetizado": [True, False, True], "peso_aluno": 1.0,
    })
    escolas = resultado_escolas(alunos, C.ANO_BASE)
    assert escolas["id_municipio"].tolist() == [10]
    assert escolas["taxa"].iloc[0] == pytest.approx(50)


def test_divisao_nao_mistura_municipios(bases):
    from src.modeling.treino import dividir_municipios
    _, _, base = bases
    divisao = dividir_municipios(base[["id_municipio", "sigla_uf"]])
    teste = set(divisao.loc[divisao["conjunto"] == "teste", "id_municipio"])
    treino = set(divisao.loc[divisao["conjunto"] == "treino", "id_municipio"])
    assert not teste & treino
    assert divisao.groupby("sigla_uf")["conjunto"].apply(lambda s: (s == "teste").any()).all()


def test_dobras_da_validacao_separam_municipios(bases):
    from sklearn.model_selection import StratifiedGroupKFold
    _, _, base = bases
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEMENTE)
    for i_tr, i_va in cv.split(base, base[C.ALVO], base[C.GRUPO]):
        assert not set(base[C.GRUPO].iloc[i_tr]) & set(base[C.GRUPO].iloc[i_va])


def test_preprocessamento_aprende_so_no_treino(bases):
    """O preenchimento de nulos tem que usar a mediana do treino, mesmo no teste."""
    from src.preprocessing.features import colunas_entrada, construir_preprocessador, selecionar_features
    _, _, base = bases
    grupos = selecionar_features(base)
    cols = colunas_entrada(grupos)
    metade = len(base) // 2
    treino, teste = base.iloc[:metade].copy(), base.iloc[metade:].copy()
    teste["mun_taxa_rede_ant"] = np.nan
    pre = construir_preprocessador(grupos, escalar=False).fit(treino[cols], treino[C.ALVO])
    saida = pre.transform(teste[cols])
    assert saida["mun_taxa_rede_ant"].nunique() == 1
    assert saida["mun_taxa_rede_ant"].iloc[0] == pytest.approx(treino["mun_taxa_rede_ant"].median())
