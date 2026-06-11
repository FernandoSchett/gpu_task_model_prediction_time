# Scripts principais

Agora a organizacao segue duas pipelines:

| Pipeline | Script | O que faz |
| --- | --- | --- |
| A - Machine Learning | `rodar_pipeline_a_machine_learning.sh` | Roda a analise com e sem telemetria, os regressores classicos e os modelos sequenciais. |
| B - Valores extremos | `rodar_pipeline_b_extremos.sh` | Roda declustering + GEV/GPD para estimar pior caso, como p95/p99/p999. |
| A + B | `rodar_todas_pipelines.sh` | Roda Pipeline A e depois Pipeline B. |

## Pipeline A

Arquivos Python da Pipeline A ficam em `scripts/py_pipeline_A/`:

| Script | Etapa |
| --- | --- |
| `py_pipeline_A/A1_gerar_manifesto_analise.py` | Gera `dataset_summary.csv` e `analysis_jobs.csv` para os recortes com e sem telemetria. |
| `py_pipeline_A/A2_regressores_classicos.py` | Treina/reaproveita regressores classicos e gera metricas/graficos. Tambem gera dependencia quando `DEPENDENCY_ONLY=true`. |
| `py_pipeline_A/A3_rankings_regressores.py` | Gera rankings dos melhores regressores por R2. |
| `py_pipeline_A/A4_rankings_dependencia.py` | Gera rankings de dependencia quando a analise de dependencia foi rodada. |
| `py_pipeline_A/A5_modelos_sequenciais.py` | Treina/reaproveita LSTM, GRU e Temporal CNN. |
| `py_pipeline_A/A7_usar_modelos_treinados.py` | Usa modelos ja treinados para inferencia. |

Auxiliares fora da Pipeline A ficam em `scripts/py_outros/`:

| Script | Etapa |
| --- | --- |
| `py_outros/A0_analisar_slowdown.py` | Analise exploratoria de slowdown. |
| `py_outros/A6_gerar_gantt_csv.py` | Gera CSV auxiliar para visualizacao em Gantt. |

Comando padrao:

```bash
bash scripts/rodar_pipeline_a_machine_learning.sh
```

Rodar so regressores classicos, sem sequenciais:

```bash
PIPELINE_A_SEQUENTIAL=false bash scripts/rodar_pipeline_a_machine_learning.sh
```

Rodar so sequenciais:

```bash
PIPELINE_A_CLASSICAL=false bash scripts/rodar_pipeline_a_machine_learning.sh
```

Rodar a analise de dependencia/Pearson/ACF dentro da Pipeline A:

```bash
DEPENDENCY_ONLY=true PIPELINE_A_SEQUENTIAL=false bash scripts/rodar_pipeline_a_machine_learning.sh
```

As partes pesadas usam cache por padrao. Para forcar recalculo:

```bash
SEQUENCE_CACHE=false bash scripts/rodar_pipeline_a_machine_learning.sh
```

## Pipeline B

Arquivos Python da Pipeline B ficam em `scripts/py_pipeline_B/`:

| Script | Etapa |
| --- | --- |
| `py_pipeline_B/B1_valores_extremos.py` | Ajusta GEV sobre maximos por bloco e GPD sobre excessos declusterizados. |

Comando padrao:

```bash
bash scripts/rodar_pipeline_b_extremos.sh
```

Forcar recalculo:

```bash
EVT_CACHE=false bash scripts/rodar_pipeline_b_extremos.sh
```

## Rodar Tudo

```bash
bash scripts/rodar_todas_pipelines.sh
```

## Experimentos brutos

Estes scripts nao sao pipelines de analise; eles geram dados brutos:

| Script | Uso |
| --- | --- |
| `rodar_experimentos.sh` | Executa um sweep bruto a partir do JSON de configuracao. |
| `rodar_experimentos_com_telemetria.sh` | Executa um sweep bruto com telemetria forcada. |
| `rodar_sweep_normal_e_telemetria.sh` | Executa sweep bruto normal e sweep bruto com telemetria. |
| `run_slowdown_test.sh` | Executa o teste especifico de slowdown. |
