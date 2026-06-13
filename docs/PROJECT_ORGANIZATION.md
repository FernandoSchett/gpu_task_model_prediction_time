# Organizacao do projeto

Este arquivo descreve como o projeto esta organizado. Use este guia antes de implementar novas pipelines, alterar estrutura de resultados ou mexer em scripts de execucao.

## Visao geral

O projeto tem tres partes principais:

- `experimentos/`: configuracoes JSON dos experimentos brutos.
- `scripts/`: scripts de execucao, pipelines e utilitarios Python.
- `resultados/`: dados brutos e resultados das analises.

Arquivos de configuracao:

- `.env`: parametros das analises e pipelines. Nao deve controlar parametros fisicos da coleta bruta.
- `experimentos/*.json`: fonte oficial dos parametros dos experimentos brutos.
- `requirements.txt`: dependencias Python.

## Experimentos brutos

Os experimentos brutos sao definidos por JSON em `experimentos/`.

Exemplos:

- `experimentos/sweep_padrao.json`: sweep principal.
- `experimentos/slowdown_test.json`: teste pequeno baseline/stress.

Parametros que devem ficar no JSON, nao no `.env`:

- `output_dir`
- `default_device`
- `sync_mode`
- `warmup_kernels`
- `flush_every`
- `gpu_telemetry`
- `gpu_telemetry_during`
- `telemetry_interval_ms`
- `seeds`
- `kernels_per_thread`
- `gpu_load_profiles`
- `blocks_x`
- `threads_per_block`
- `grid_z`
- `kernel_types`

Scripts brutos:

- `scripts/rodar_experimentos.sh`: roda um sweep a partir de um JSON.
- `scripts/rodar_experimentos_com_telemetria.sh`: cria JSON temporario com telemetria ligada e roda o sweep.
- `scripts/rodar_sweep_normal_e_telemetria.sh`: roda sweep sem telemetria e com telemetria.
- `scripts/rodar_experimentos_2seeds_e_pipelines.sh`: roda seeds 67 e 42, com/sem telemetria, depois pipelines.
- `scripts/run_slowdown_test.sh`: roda `slowdown_test.json`.

Estrutura esperada dos dados brutos:

```text
resultados/
  sweep_moderado_sem_estimativas_<timestamp>/
    resultados_experimentos_*.csv
  sweep_moderado_sem_estimativas_telemetry_<timestamp>/
    resultados_experimentos_*.csv
```

## Analises

Raiz padrao das analises:

```text
resultados/analises_regressao/
```

As analises se dividem em duas condicoes principais:

```text
resultados/analises_regressao/
  sem_telemetria_sweep_moderado_sem_estimativas_agrupado/
  com_telemetria_sweep_moderado_sem_estimativas_agrupado/
```

Cada condicao contem recortes:

- `geral`: todos os dados da condicao.
- `perfil_gpu_10`, `perfil_gpu_50`, `perfil_gpu_100`, `perfil_gpu_120`: por demanda alvo de GPU.
- `kernel_busy_wait`, `kernel_compute`, `kernel_memory`, `kernel_mixed`: por tipo de kernel.
- `perfil_gpu_<N>_kernel_<tipo>`: demanda GPU + tipo de kernel.

Cada recorte contem alvos:

```text
<condicao>/<recorte>/response_time_us/
<condicao>/<recorte>/queueing_delay_us/
<condicao>/<recorte>/slowdown/
```

Por padrao, `TARGETS="response_time_us"` no `.env`. Para rodar todos:

```bash
TARGETS="response_time_us queueing_delay_us slowdown" bash scripts/rodar_todas_pipelines.sh
```

## Targets

- `response_time_us`: tempo total observado.
- `queueing_delay_us`: atraso extra estimado. Hoje e calculado como `response_time_us - requested_busy_wait_us`.
- `slowdown`: desaceleracao relativa. Hoje e calculado como `response_time_us / requested_busy_wait_us`.

`queueing_delay_us` e `slowdown` sao derivados de `response_time_us`. Eles ajudam na interpretacao, mas nao sao alvos independentes.

## Pipeline A - Machine Learning

Entrada:

- CSVs brutos em `resultados/sweep_*`.
- Manifesto `analysis_jobs.csv`, gerado por `A1_gerar_manifesto_analise.py`.

Scripts:

- `scripts/rodar_pipeline_a_machine_learning.sh`: entrada principal.
- `scripts/py_pipeline_A/A1_gerar_manifesto_analise.py`: gera `dataset_summary.csv` e `analysis_jobs.csv`.
- `scripts/py_pipeline_A/A2_regressores_classicos.py`: regressores classicos, dependencia, metricas e graficos.
- `scripts/py_pipeline_A/A3_rankings_regressores.py`: wrapper do comparador central.
- `scripts/py_pipeline_A/A4_rankings_dependencia.py`: rankings de dependencia.
- `scripts/py_pipeline_A/A5_modelos_sequenciais.py`: LSTM, GRU e Temporal CNN.
- `scripts/py_pipeline_A/A7_usar_modelos_treinados.py`: inferencia usando modelos treinados.

Saidas por alvo:

```text
<condicao>/<recorte>/<target>/
  regression_metrics.csv
  dependency_metrics.csv
  trained_models/
  model_diagnostics/
  sequential_models/
```

Rankings locais da Pipeline A:

```text
resultados/analises_regressao/melhores_modelos_nao_sequenciais/
resultados/analises_regressao/melhores_modelos_sequenciais/
resultados/analises_regressao/pipeline_a_model_rankings/
```

## Pipeline B - Valores extremos

Objetivo:

- Estimar caudas e pior caso usando EVT.
- Ajustar GEV, Gumbel e GPD/POT com declustering.

Scripts:

- `scripts/rodar_pipeline_b_extremos.sh`: entrada principal.
- `scripts/py_pipeline_B/B1_valores_extremos.py`: ajuste EVT, metricas e graficos.

Saidas por alvo:

```text
<condicao>/<recorte>/<target>/extreme_values/
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

Modelos da Pipeline B:

- `gev`: GEV sobre maximos por bloco.
- `gumbel`: Gumbel sobre maximos por bloco.
- `gpd`: GPD/POT sobre excessos declusterizados.

## Pipeline C - CNN 2D

Objetivo:

- Transformar execucoes em tensores 2D.
- Eixo 1: workers concorrentes.
- Eixo 2: janela temporal.
- Canais: features numericas ja usadas nas outras pipelines.

Scripts:

- `scripts/rodar_pipeline_c_cnn2d.sh`: entrada principal.
- `scripts/py_pipeline_C/C1_preprocessamento_2d.py`: gera tensores.
- `scripts/py_pipeline_C/C2_treinar_cnn_2d.py`: treina arquiteturas CNN 2D.
- `scripts/py_pipeline_C/C3_rankings_e_graficos.py`: rankings e graficos locais da Pipeline C.

Saidas por alvo:

```text
<condicao>/<recorte>/<target>/2d_models/
  cnn2d_dataset.npz
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
```

Ranking local da Pipeline C:

```text
resultados/analises_regressao/2d_models/
  best_model_rankings.csv
  best_cnn2d_rankings.csv
  cnn2d_all_architecture_metrics.csv
  *.png
```

## Comparador global

Script:

```text
scripts/py_outros/comparar_modelos_pipelines.py
```

Funcao:

- Junta resultados da Pipeline A e Pipeline C.
- Gera rankings locais por familia.
- Gera ranking global comparando todos os modelos disponiveis.

Saida global:

```text
resultados/analises_regressao/
  best_model_rankings.csv
  best_model_top_<target>.png
  best_model_condition_overview.png
```

## Padrao para nova pipeline

Ao criar uma Pipeline D, E etc:

1. Criar pasta Python:

```text
scripts/py_pipeline_D/
```

2. Criar shell de entrada:

```text
scripts/rodar_pipeline_d_<nome>.sh
```

3. Usar `analysis_jobs.csv` como contrato de entrada sempre que possivel.

4. Respeitar estrutura:

```text
<condicao>/<recorte>/<target>/<nome_da_pipeline>/
```

5. Gerar metricas comparaveis quando for modelo preditivo:

- `MAE`
- `RMSE`
- `R2`

6. Salvar modelos e cache dentro da pasta da propria pipeline:

```text
<nome_da_pipeline>/trained_models/
```

7. Salvar graficos dentro de:

```text
<nome_da_pipeline>/model_diagnostics/
```

8. Criar ranking local da pipeline em:

```text
resultados/analises_regressao/<nome_da_pipeline>/
```

9. Se a pipeline deve competir com A/C, adaptar `scripts/py_outros/comparar_modelos_pipelines.py`.

10. Adicionar variaveis no `.env` em bloco proprio.

11. Atualizar `scripts/README.md` e este arquivo.

## Regras de cache

Padrao esperado:

- Se resultado/modelo ja existe, reaproveitar.
- Ter opcao `*_CACHE=false` ou equivalente para recalcular.
- Ter opcao `*_MODEL_ONLY=<nome>` para rodar um modelo/arquitetura.
- Ter opcao `*_FORCE_MODEL=true` para retreinar mesmo se ja existir.

## Scripts auxiliares

Ficam em:

```text
scripts/py_outros/
```

Uso atual:

- `A0_analisar_slowdown.py`: analise simples de slowdown.
- `A6_gerar_gantt_csv.py`: gera CSV para visualizacao tipo Gantt.
- `comparar_modelos_pipelines.py`: ranking global entre pipelines.

## Nao confundir

- `scripts/py_pipeline_A/`: Machine Learning.
- `scripts/py_pipeline_B/`: valores extremos.
- `scripts/py_pipeline_C/`: CNN 2D.
- `scripts/py_outros/`: utilitarios compartilhados ou fora das pipelines.
- `.env`: parametros de analise/pipelines.
- `experimentos/*.json`: parametros de coleta bruta.
