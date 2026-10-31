# Dados

| Pasta | O que tem | No Git |
|---|---|---|
| gold/ | exportação da camada Gold da Fase 2 | não, são uns 70 MB |
| external/ | IBGE, metas e dados das escolas do Inep | sim |
| processed/ | divisão treino/teste por município | só divisao_municipios.csv |

## gold/

Vem do workspace Databricks da Fase 2
([eriicpa/tech_challenge_2_pipeline_dados](https://github.com/eriicpa/tech_challenge_2_pipeline_dados)).
O notebook databricks/exportar_gold_fase3.py cria a gold_aluno_ml, com uma linha por aluno, e
exporta as tabelas em Parquet para o volume export_fase3. De lá baixei os arquivos para esta pasta.

| Arquivo | Uma linha por | Uso aqui |
|---|---|---|
| gold_aluno_ml | aluno e ano (3.867.999 linhas) | alvo e escola do aluno |
| gold_indicador_municipio | município, rede e ano | histórico do município, meta, território |
| gold_meta_municipio | município e ano da meta | metas até 2030 |
| outras gold_* | UF ou município | análise exploratória |

A fato_aluno da Fase 2 não tem território, então UF, região e mesorregião vêm da
gold_indicador_municipio na hora de carregar. O código lê Parquet ou CSV com o mesmo nome.

## external/

| Arquivo | Fonte | Gerado por |
|---|---|---|
| ibge_socioeconomico_municipio.csv | IBGE, Censo 2022 (tabelas 4714, 9923, 9543 e 6805) e PIB dos Municípios 2021 (5938) | src/data/externos_ibge.py |
| inep_meta_alfabetizacao_municipio.csv | Inep, via Base dos Dados | o mesmo arquivo da landing da Fase 2 |
| escola_inep.parquet | Inep, Censo Escolar 2023, Ideb e Saeb do 5º ano até 2023, Inse 2023 | src/data/externos_inep.py |

Os arquivos brutos do Inep (uns 350 MB) ficam em inep_raw/, fora do Git.

O código da escola na base de alunos é fictício e não liga com o código real do Censo, por isso os
dados das escolas entram agregados por município e rede.

## processed/

divisao_municipios.csv guarda quais municípios ficaram no teste, para qualquer nova execução avaliar
nos mesmos.
