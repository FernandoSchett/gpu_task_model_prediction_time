# Guia de Estrutura das Analises

Este documento descreve a estrutura desejada para `resultados/analises_regressao/`.
Ainda e uma proposta de reorganizacao. O codigo sera ajustado depois do aval.

## Objetivo

A pasta `analises_regressao/` deve separar claramente:

- resultados da Pipeline A;
- resultados da Pipeline B;
- resultados da Pipeline C;
- comparacoes globais entre pipelines.

Estrutura desejada:

```text
resultados/
  analises_regressao/
    pipeline_A/
    pipeline_B/
    pipeline_C/
    comparacoes_pipelines/
    analise_dependencia/
```

## Raiz: `analises_regressao/`

```text
resultados/analises_regressao/
  pipeline_A/
  pipeline_B/
  pipeline_C/
  comparacoes_pipelines/
  analise_dependencia/
```

Conteudo:

- `pipeline_A/`: regressores classicos + modelos sequenciais.
- `pipeline_B/`: modelos de valores extremos, GEV/Gumbel/GPD.
- `pipeline_C/`: modelos CNN 2D.
- `comparacoes_pipelines/`: comparacoes entre Pipeline A e Pipeline C.
- `analise_dependencia/`: metricas e rankings de dependencia dos dados.

## Pipeline A

Pipeline A concentra modelos de Machine Learning:

- regressores classicos;
- modelos sequenciais;
- comparacao local entre classicos e sequenciais.

Estrutura:

```text
resultados/analises_regressao/pipeline_A/
  sem_telemetria/
  com_telemetria/
  rankings/
```

### Condicoes

```text
pipeline_A/
  sem_telemetria/
  com_telemetria/
```

Dentro de cada condicao:

```text
<condicao>/
  dataset_summary.csv
  analysis_jobs.csv
  training_summary.csv
  sequential_summary.csv
  <recorte>/
    <target>/
      nao_sequenciais/
      sequenciais/
```

Exemplo:

```text
pipeline_A/
  sem_telemetria/
    geral/
      response_time_us/
        nao_sequenciais/
          regression_metrics.csv
          trained_models/
          model_diagnostics/
        sequenciais/
          sequential_metrics.csv
          sequence_metadata.json
          *.keras
          model_diagnostics/
```

### Recortes

Recortes esperados:

```text
geral/
perfil_gpu_10/
perfil_gpu_50/
perfil_gpu_100/
perfil_gpu_120/
kernel_busy_wait/
kernel_compute/
kernel_memory/
kernel_mixed/
perfil_gpu_<N>_kernel_<tipo>/
```

### Targets

Targets esperados:

```text
response_time_us/
queueing_delay_us/
slowdown/
```

Por padrao, o projeto roda apenas:

```text
response_time_us
```

### Modelos nao sequenciais

Pasta:

```text
pipeline_A/<condicao>/<recorte>/<target>/nao_sequenciais/
```

Conteudo:

```text
regression_metrics.csv
trained_models/
model_diagnostics/
mae_comparison.png
rmse_comparison.png
r2_comparison.png
```

Modelos:

- Linear Regression
- Ridge Regression
- Polynomial Ridge
- Decision Tree
- Random Forest
- Gradient Boosting
- kNN
- LightGBM
- XGBoost
- CatBoost
- LightGBM/XGBoost quantile models

### Modelos sequenciais

Pasta:

```text
pipeline_A/<condicao>/<recorte>/<target>/sequenciais/
```

Conteudo:

```text
sequential_metrics.csv
sequence_metadata.json
lstm.keras
gru.keras
temporal_cnn.keras
model_diagnostics/
mae_comparison.png
rmse_comparison.png
r2_comparison.png
```

Modelagem sequencial desejada:

- nao cruzar arquivos/experimentos independentes;
- ordenar por `submit_time_ns`, `execution_order`, `completion_time_ns`;
- usar historico anterior + features do kernel atual;
- prever o target do kernel atual.

### Rankings locais da Pipeline A

Pasta:

```text
pipeline_A/rankings/
```

Arquivos:

```text
melhores_modelos_nao_sequenciais.csv
melhores_modelos_sequenciais.csv
melhores_modelos_pipeline_A.csv
top_nao_sequenciais_response_time_us.png
top_sequenciais_response_time_us.png
top_pipeline_A_response_time_us.png
```

Significado:

- `melhores_modelos_nao_sequenciais.csv`: ranking so entre modelos classicos.
- `melhores_modelos_sequenciais.csv`: ranking so entre LSTM/GRU/Temporal CNN.
- `melhores_modelos_pipeline_A.csv`: melhor modelo entre classicos e sequenciais para cada recorte/target.

## Pipeline B

Pipeline B concentra modelos de valores extremos.

Estrutura:

```text
resultados/analises_regressao/pipeline_B/
  sem_telemetria/
  com_telemetria/
  rankings/
```

Dentro de cada condicao:

```text
pipeline_B/
  sem_telemetria/
    <recorte>/
      <target>/
        extreme_values/
```

Conteudo de `extreme_values/`:

```text
extreme_value_summary.csv
extreme_value_quantiles.csv
gev_hist_fit.png
gev_qq.png
gev_pp.png
gumbel_hist_fit.png
gumbel_qq.png
gumbel_pp.png
gpd_excess_hist_fit.png
gpd_qq.png
gpd_pp.png
```

Modelos:

- GEV sobre maximos por bloco;
- Gumbel sobre maximos por bloco;
- GPD/POT sobre excessos declusterizados.

Rankings locais:

```text
pipeline_B/rankings/
  extreme_value_rankings.csv
  top_extremos_response_time_us.png
```

## Pipeline C

Pipeline C concentra modelos CNN 2D.

Estrutura:

```text
resultados/analises_regressao/pipeline_C/
  sem_telemetria/
  com_telemetria/
  rankings/
```

Dentro de cada condicao:

```text
pipeline_C/
  sem_telemetria/
    <recorte>/
      <target>/
        2d_models/
```

Conteudo de `2d_models/`:

```text
cnn2d_x.npy
cnn2d_y.npy
cnn2d_dataset_metadata.json
cnn2d_architecture_metrics.csv
trained_models/
  *.keras
  *.json
  *_metrics.json
model_diagnostics/
  *_predicted_vs_actual.png
  *_error_distribution.png
  *_loss.png
predictions/
  *_predictions.csv
```

Observacao:

- `cnn2d_x.npy` e `cnn2d_y.npy` devem ser memmap em disco.
- Isso evita estouro de RAM quando a quantidade de janelas e grande.

Rankings locais:

```text
pipeline_C/rankings/
  melhores_modelos_2d.csv
  cnn2d_architecture_rankings.csv
  top_cnn2d_response_time_us.png
```

## Comparacoes entre Pipelines

Comparacoes globais ficam fora das pastas A/B/C.

Pasta:

```text
resultados/analises_regressao/comparacoes_pipelines/
```

Conteudo:

```text
best_model_rankings.csv
best_model_top_response_time_us.png
best_model_top_queueing_delay_us.png
best_model_top_slowdown.png
best_model_condition_overview.png
```

Escopo:

- comparar Pipeline A e Pipeline C;
- indicar quais modelos tiveram melhor R2 por recorte/target;
- comparar classicos, sequenciais e CNN 2D.

Pipeline B nao entra nessa comparacao de R2, porque seu objetivo e outro: modelagem de cauda/pior caso.

## Analise de Dependencia

A analise de dependencia deve ficar na raiz da estrutura de analises, separada das pipelines.

Pasta:

```text
resultados/analises_regressao/analise_dependencia/
```

Conteudo:

```text
dependency_rankings.csv
dependency_top_response_time_us.png
dependency_top_queueing_delay_us.png
dependency_top_slowdown.png
dependency_condition_overview.png
```

Metricas esperadas:

- autocorrelacao por lag;
- Durbin-Watson;
- razao aproximada de amostra efetiva;
- Pearson feature x target.

Arquivos por recorte/target podem ser copiados ou referenciados a partir das pastas da Pipeline A, mas o ranking global deve ficar em:

```text
analise_dependencia/
```

## Nova arvore esperada

Resumo:

```text
resultados/analises_regressao/
  pipeline_A/
    sem_telemetria/
      <recorte>/<target>/nao_sequenciais/
      <recorte>/<target>/sequenciais/
    com_telemetria/
      <recorte>/<target>/nao_sequenciais/
      <recorte>/<target>/sequenciais/
    rankings/
      melhores_modelos_nao_sequenciais.csv
      melhores_modelos_sequenciais.csv
      melhores_modelos_pipeline_A.csv

  pipeline_B/
    sem_telemetria/
      <recorte>/<target>/extreme_values/
    com_telemetria/
      <recorte>/<target>/extreme_values/
    rankings/

  pipeline_C/
    sem_telemetria/
      <recorte>/<target>/2d_models/
    com_telemetria/
      <recorte>/<target>/2d_models/
    rankings/

  comparacoes_pipelines/
    best_model_rankings.csv
    *.png

  analise_dependencia/
    dependency_rankings.csv
    *.png
```

## Regras para implementar depois

Quando o codigo for ajustado:

1. `A1_gerar_manifesto_analise.py` deve gerar jobs com `output_dir` apontando para `pipeline_A/<condicao>/...` por padrao.
2. Pipeline B deve usar manifesto proprio ou converter paths para `pipeline_B/<condicao>/...`.
3. Pipeline C deve usar manifesto proprio ou converter paths para `pipeline_C/<condicao>/...`.
4. Comparador global deve escrever somente em `comparacoes_pipelines/`.
5. Rankings locais devem ficar dentro de `pipeline_A/rankings/`, `pipeline_B/rankings/` e `pipeline_C/rankings/`.
6. Dependencia global deve sair em `analise_dependencia/`.
7. Scripts devem preservar compatibilidade por uma fase, ou fornecer migracao dos resultados antigos.

## Estado migrado

Os scripts e resultados atuais devem seguir:

```text
resultados/analises_regressao/
  pipeline_A/
  pipeline_B/
  pipeline_C/
  comparacoes_pipelines/
  analise_dependencia/
```

Se aparecer caminho legado, use `scripts/py_outros/migrar_analises_para_nova_estrutura.py`.
