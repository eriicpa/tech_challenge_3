"""Monta as bases do modelo a partir da Gold.

Regra principal: cada aluno de 2024 só recebe informações que já existiam antes da
prova.

* o desempenho do município e da UF vem sempre do ano anterior;
* de 2024 entra só o número de matrículas da escola, conhecido antes da prova;
* não existe histórico da escola: o código fictício da escola muda de um ano para o
  outro (só 2% dos códigos ficam no mesmo município), então não dá para ligar a escola
  de 2024 à de 2023;
* os dados socioeconômicos e das escolas são de antes das provas (Censo 2022, PIB 2021,
  Inep até 2023).

Isso evita o erro mais fácil de cometer com a Gold: usar a taxa do município no próprio
ano, que é calculada com os mesmos alunos que o modelo tenta prever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

def _media_ponderada(valor: pd.Series, peso: pd.Series, chave: pd.Series) -> pd.Series:
    numerador = (valor * peso).groupby(chave, observed=True).sum()
    denominador = peso.where(valor.notna()).groupby(chave, observed=True).sum()
    return numerador / denominador.replace(0, np.nan)


def resultado_escolas(alunos: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Resultado de cada escola no ano informado, com o município dela.

    O código da escola só vale dentro do mesmo ano, por isso o município vem dos alunos
    daquele ano.
    """
    coorte = alunos[alunos["ano"] == ano]
    avaliados = coorte[coorte["avaliado"]]
    alfabetizado = avaliados["alfabetizado"].astype(float)
    escola = avaliados["id_escola"]

    hist = pd.DataFrame({
        "taxa": 100 * _media_ponderada(alfabetizado, avaliados["peso_aluno"], escola),
        "avaliados": avaliados.groupby("id_escola").size(),
    })
    municipio = coorte.drop_duplicates("id_escola").set_index("id_escola")["id_municipio"]
    return hist.join(municipio).reset_index()


def historico_municipio(indicador: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Resultado do município no ano informado, com uma coluna por rede."""
    base = indicador[indicador["ano"] == ano]

    def por_rede(coluna, rede):
        return (base[base["rede"] == rede].drop_duplicates("id_municipio")
                .set_index("id_municipio")[coluna])

    hist = pd.DataFrame({
        "mun_taxa_estadual_ant": por_rede("taxa_alfabetizacao", C.REDE_ESTADUAL),
        "mun_taxa_municipal_ant": por_rede("taxa_alfabetizacao", C.REDE_MUNICIPAL),
        "mun_taxa_publica_ant": por_rede("taxa_alfabetizacao", C.REDE_PUBLICA),
        "mun_media_ant": por_rede("media_portugues", C.REDE_PUBLICA),
        "mun_participacao_ant": por_rede("percentual_participacao", C.REDE_MUNICIPAL),
    })
    # Município sem a linha "pública" usa a rede municipal, que tem a maior parte
    # dos alunos do 2º ano.
    hist["mun_taxa_publica_ant"] = hist["mun_taxa_publica_ant"].fillna(hist["mun_taxa_municipal_ant"])
    hist["mun_media_ant"] = hist["mun_media_ant"].fillna(por_rede("media_portugues", C.REDE_MUNICIPAL))

    uf = (base[base["rede"] == C.REDE_PUBLICA].groupby("sigla_uf", observed=True)["taxa_uf"]
          .first().rename("uf_taxa_ant"))
    territorio = base.drop_duplicates("id_municipio").set_index("id_municipio")[["sigla_uf"]]
    hist = hist.join(territorio).join(uf, on="sigla_uf").drop(columns="sigla_uf")
    return hist.reset_index()


def metas(metas_municipio: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Meta da rede municipal para o ano informado."""
    m = metas_municipio[(metas_municipio["rede"] == C.REDE_MUNICIPAL)
                        & (metas_municipio["ano_meta"] == ano)]
    return (m.drop_duplicates("id_municipio")[["id_municipio", "meta_taxa"]]
            .rename(columns={"meta_taxa": "meta_taxa_ano"}))


# Dados das escolas do Inep, agregados por município e rede. Na base de alunos, o
# código da escola é fictício (60000001 em diante), então não dá para ligar o aluno à
# escola real. A ligação é feita por município e rede, e cada escola pesa pelas suas
# matrículas nos anos iniciais.
CONTEXTO_ESCOLAR = {
    "inep_saeb5_lp_2023": "saeb5_lp_2023",
    "inep_saeb5_lp_2021": "saeb5_lp_2021",
    "inep_saeb5_lp_tendencia": "saeb5_lp_tendencia",
    "inep_ideb_2023": "ideb_2023",
    "inep_aprovacao_2ano_2023": "aprovacao_2ano_2023",
    "inep_inse": "inse_valor",
    "inep_pct_inse_baixo": "inse_pct_niveis_baixos",
    "inep_pct_integral": "esc_pct_integral",
    "inep_pct_biblioteca": "esc_biblioteca",
    "inep_pct_internet_alunos": "esc_internet_alunos",
    "inep_pct_coordenador": "esc_coordenador",
    "inep_pct_material_infantil": "esc_material_infantil",
    "inep_alunos_por_turma": "esc_alunos_por_turma",
    "inep_alunos_por_docente": "esc_alunos_por_docente",
    "inep_pct_rural": "esc_rural",
}


def contexto_escolar_rede(escolas: pd.DataFrame) -> pd.DataFrame:
    """Média das escolas de cada município e rede, ponderada pelas matrículas."""
    escolas = escolas.assign(esc_rural=(escolas["esc_localizacao"] == "Rural").astype(float))
    peso = escolas["esc_matriculas_ai"]
    chave = [escolas["id_municipio"], escolas["rede"]]
    colunas = {}
    for novo, origem in CONTEXTO_ESCOLAR.items():
        valor = escolas[origem]
        numerador = (valor * peso).groupby(chave).sum(min_count=1)
        denominador = peso.where(valor.notna()).groupby(chave).sum()
        colunas[novo] = numerador / denominador.replace(0, np.nan)
    contexto = pd.DataFrame(colunas)
    contexto.index.names = ["id_municipio", "rede"]
    return contexto.reset_index()


def contexto_externo(ibge: pd.DataFrame, atlas: pd.DataFrame | None) -> pd.DataFrame:
    externo = ibge.copy()
    if atlas is not None:
        externo = externo.merge(atlas, on="id_municipio", how="left")
    return externo


def _anexar_contexto(base, alunos_ano_base, indicador, metas_municipio, ano_base, ano_meta,
                     ibge, censo_escolar, atlas):
    municipio = historico_municipio(indicador, ano_base)

    base = (base
            .merge(municipio, on="id_municipio", how="left")
            .merge(metas(metas_municipio, ano_meta), on="id_municipio", how="left")
            .merge(contexto_externo(ibge, atlas), on="id_municipio", how="left"))
    if censo_escolar is not None:
        base = base.merge(contexto_escolar_rede(censo_escolar), on=["id_municipio", "rede"],
                          how="left")

    # Taxa do município na mesma rede do aluno
    base["mun_taxa_rede_ant"] = np.select(
        [base["rede"] == C.REDE_ESTADUAL, base["rede"] == C.REDE_MUNICIPAL],
        [base["mun_taxa_estadual_ant"], base["mun_taxa_municipal_ant"]],
        default=base["mun_taxa_publica_ant"])
    base["mun_taxa_rede_ant"] = base["mun_taxa_rede_ant"].fillna(base["mun_taxa_publica_ant"])
    return base


def construir_base_aluno(alunos, indicador, metas_municipio, ibge, censo_escolar=None,
                         atlas=None, ano_base=C.ANO_BASE, ano_alvo=C.ANO_ALVO):
    """Base de treino: alunos avaliados no ano previsto, com o histórico do ano anterior.

    Só entra a rede pública (estadual e municipal), que é o foco do Compromisso.
    """
    coorte = alunos[(alunos["ano"] == ano_alvo)
                    & alunos["rede"].isin([C.REDE_ESTADUAL, C.REDE_MUNICIPAL])]
    matriculados = coorte.groupby("id_escola").size().rename("escola_matriculados")

    base = coorte[coorte["avaliado"] & coorte["alfabetizado"].notna()].copy()
    base[C.ALVO] = (~base["alfabetizado"].astype(bool)).astype("int8")
    base = base.drop(columns=[c for c in C.COLUNAS_PROIBIDAS if c in base.columns and c != C.ALVO])
    base = base.merge(matriculados, on="id_escola", how="left")

    base = _anexar_contexto(base, alunos, indicador, metas_municipio, ano_base, ano_alvo,
                            ibge, censo_escolar, atlas)
    return base.reset_index(drop=True)


def construir_base_projecao(alunos, indicador, metas_municipio, ibge, censo_escolar=None,
                            atlas=None, ano_base=C.ANO_ALVO, ano_projecao=C.ANO_PROJECAO):
    """Base para prever o ano seguinte, que ainda não tem resultado.

    Os alunos do ano seguinte ainda não existem nos dados, então os alunos do último ano
    servem de retrato das escolas. O histórico passa a ser o do último ano e a meta, a do
    ano projetado.
    """
    coorte = alunos[(alunos["ano"] == ano_base)
                    & alunos["rede"].isin([C.REDE_ESTADUAL, C.REDE_MUNICIPAL])]
    matriculados = coorte.groupby("id_escola").size().rename("escola_matriculados")
    base = coorte[coorte["avaliado"]].copy()
    base = base.drop(columns=[c for c in C.COLUNAS_PROIBIDAS if c in base.columns])
    base = base.merge(matriculados, on="id_escola", how="left")
    base = _anexar_contexto(base, alunos, indicador, metas_municipio, ano_base, ano_projecao,
                            ibge, censo_escolar, atlas)
    return base.reset_index(drop=True)


def construir_base_municipio(indicador, metas_municipio, ibge, alunos=None, atlas=None,
                             ano_base=C.ANO_BASE, ano_alvo=C.ANO_ALVO, rotulado=True,
                             censo_escolar=None):
    """Uma linha por município: histórico do ano anterior e, se houver, o resultado do ano.

    O resultado usado é o da rede municipal, que é a rede com meta.
    """
    base = historico_municipio(indicador, ano_base)
    territorio = (indicador.drop_duplicates("id_municipio")
                  .set_index("id_municipio")[["nome_municipio", "sigla_uf", "nome_regiao",
                                              "nome_mesorregiao"]])
    base = (base.join(territorio, on="id_municipio")
            .merge(metas(metas_municipio, ano_alvo), on="id_municipio", how="left")
            .merge(contexto_externo(ibge, atlas), on="id_municipio", how="left"))
    base["gap_meta_vs_ant"] = base["meta_taxa_ano"] - base["mun_taxa_municipal_ant"]
    if censo_escolar is not None:
        # A meta é da rede municipal, então entra o contexto das escolas municipais
        contexto = contexto_escolar_rede(censo_escolar)
        contexto = contexto[contexto["rede"] == C.REDE_MUNICIPAL].drop(columns="rede")
        base = base.merge(contexto, on="id_municipio", how="left")

    if alunos is not None:
        # Diferença entre escolas do mesmo município em 2023: a mesma média pode esconder
        # escolas muito diferentes.
        escolas = resultado_escolas(alunos, ano_base)
        disp = escolas.groupby("id_municipio").agg(
            n_escolas=("id_escola", "size"),
            desvio_taxa_escolas_ant=("taxa", "std"),
            pct_escolas_abaixo_50_ant=("taxa", lambda s: 100 * (s < 50).mean()))
        base = base.merge(disp, on="id_municipio", how="left")

    if rotulado:
        resultado = indicador[(indicador["ano"] == ano_alvo)
                              & (indicador["rede"] == C.REDE_MUNICIPAL)]
        resultado = (resultado.drop_duplicates("id_municipio")
                     .set_index("id_municipio")["taxa_alfabetizacao"].rename("taxa_alvo"))
        base = base.join(resultado, on="id_municipio")
        base = base[base["taxa_alvo"].notna() & base["meta_taxa_ano"].notna()].copy()
        base["nao_atingiu_meta"] = (base["taxa_alvo"] < base["meta_taxa_ano"]).astype("int8")
    else:
        base = base[base["meta_taxa_ano"].notna()].copy()
    return base.reset_index(drop=True)
