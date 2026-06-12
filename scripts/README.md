# Scripts principais

Agora a organizacao segue tres pipelines:

| Pipeline | Script | O que faz |
| --- | --- | --- |
| A - Machine Learning | `rodar_pipeline_a_machine_learning.sh` | Roda a analise com e sem telemetria, os regressores classicos e os modelos sequenciais. |
| B - Valores extremos | `rodar_pipeline_b_extremos.sh` | Roda declustering + GEV/GPD para estimar pior caso, como p95/p99/p999. |
| C - CNN 2D | `rodar_pipeline_c_cnn2d.sh` | Transforma os dados em tensores worker x tempo x features e treina CNNs 2D. |
| A + B + C | `rodar_todas_pipelines.sh` | Roda Pipeline A, Pipeline B e Pipeline C. |

## Alvos

Por padrao, todas as pipelines rodam apenas `response_time_us`.

Rodar todos os alvos:

```bash
TARGETS="response_time_us queueing_delay_us slowdown" bash scripts/rodar_todas_pipelines.sh
```

Rodar apenas um alvo especifico:

```bash
TARGETS=slowdown bash scripts/rodar_pipeline_a_machine_learning.sh
```

## Pipeline A

Arquivos Python da Pipeline A ficam em `scripts/py_pipeline_A/`:

| Script | Etapa |
| --- | --- |
| `py_pipeline_A/A1_gerar_manifesto_analise.py` | Gera `dataset_summary.csv` e `analysis_jobs.csv` para os recortes com e sem telemetria. |
| `py_pipeline_A/A2_regressores_classicos.py` | Treina/reaproveita regressores classicos e gera metricas/graficos. Tambem gera dependencia quando `DEPENDENCY_ONLY=true`. |
| `py_pipeline_A/A3_rankings_regressores.py` | Wrapper de compatibilidade para o comparador central de modelos. |
| `py_pipeline_A/A4_rankings_dependencia.py` | Gera rankings de dependencia quando a analise de dependencia foi rodada. |
| `py_pipeline_A/A5_modelos_sequenciais.py` | Treina/reaproveita LSTM, GRU e Temporal CNN. |
| `py_pipeline_A/A7_usar_modelos_treinados.py` | Usa modelos ja treinados para inferencia. |

Auxiliares fora da Pipeline A ficam em `scripts/py_outros/`:

| Script | Etapa |
| --- | --- |
| `py_outros/A0_analisar_slowdown.py` | Analise exploratoria de slowdown. |
| `py_outros/A6_gerar_gantt_csv.py` | Gera CSV auxiliar para visualizacao em Gantt. |
| `py_outros/comparar_modelos_pipelines.py` | Compara modelos das Pipelines A e C e gera os graficos/rankings gerais. |

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

Rodar novamente apenas um modelo classico, ignorando cache:

```bash
CLASSICAL_MODEL_ONLY=lightgbm CLASSICAL_FORCE_MODEL=true bash scripts/rodar_pipeline_a_machine_learning.sh
```

Rodar novamente apenas um modelo sequencial, ignorando cache:

```bash
PIPELINE_A_CLASSICAL=false SEQUENCE_MODEL_ONLY=lstm SEQUENCE_FORCE_MODEL=true bash scripts/rodar_pipeline_a_machine_learning.sh
```

Por padrao, os sequenciais usam `SEQUENCE_SPLIT_MODE=random` para ficarem comparaveis aos regressores classicos, que tambem usam split aleatorio. Para testar generalizacao temporal mais dificil:

```bash
SEQUENCE_SPLIT_MODE=chronological bash scripts/rodar_pipeline_a_machine_learning.sh
```

Rodar a analise de dependencia/Pearson/ACF dentro da Pipeline A:

```bash
DEPENDENCY_ONLY=true PIPELINE_A_SEQUENTIAL=false bash scripts/rodar_pipeline_a_machine_learning.sh
```

As partes pesadas usam cache por padrao. Para forcar recalculo:

```bash
SEQUENCE_CACHE=false bash scripts/rodar_pipeline_a_machine_learning.sh
```

O ranking da Pipeline A gera tres saidas:
O ranking geral agora e gerado pelo comparador central e inclui Pipeline A e Pipeline C quando houver resultados 2D.

| Pasta | Conteudo |
| --- | --- |
| `resultados/analises_regressao/melhores_modelos_nao_sequenciais/` | Top apenas dos regressores classicos. |
| `resultados/analises_regressao/melhores_modelos_sequenciais/` | Top apenas de LSTM, GRU e Temporal CNN. |
| `resultados/analises_regressao/2d_models/` | Top apenas da Pipeline C/CNN 2D. |
| `resultados/analises_regressao/pipeline_a_model_rankings/` | Top geral apenas da Pipeline A. |
| `resultados/analises_regressao/` | Top geral misturando Pipeline A e Pipeline C. |

## Pipeline B

Arquivos Python da Pipeline B ficam em `scripts/py_pipeline_B/`:

| Script | Etapa |
| --- | --- |
| `py_pipeline_B/B1_valores_extremos.py` | Ajusta GEV/Gumbel sobre maximos por bloco e GPD sobre excessos declusterizados. |

Comando padrao:

```bash
bash scripts/rodar_pipeline_b_extremos.sh
```

Forcar recalculo:

```bash
EVT_CACHE=false bash scripts/rodar_pipeline_b_extremos.sh
```

Rodar novamente apenas um ajuste da Pipeline B:

```bash
EVT_MODEL_ONLY=gev EVT_FORCE_MODEL=true bash scripts/rodar_pipeline_b_extremos.sh
```

Outras opcoes:

```bash
EVT_MODEL_ONLY=gumbel EVT_FORCE_MODEL=true bash scripts/rodar_pipeline_b_extremos.sh
EVT_MODEL_ONLY=gpd EVT_FORCE_MODEL=true bash scripts/rodar_pipeline_b_extremos.sh
```

## Pipeline C

Arquivos Python da Pipeline C ficam em `scripts/py_pipeline_C/`:

| Script | Etapa |
| --- | --- |
| `py_pipeline_C/C1_preprocessamento_2d.py` | Gera tensores 2D `workers x janela_temporal x features` por recorte/alvo. |
| `py_pipeline_C/C2_treinar_cnn_2d.py` | Treina/reaproveita CNNs 2D com busca simples de arquiteturas. |
| `py_pipeline_C/C3_rankings_e_graficos.py` | Consolida rankings, metricas e graficos da Pipeline C. |

Comando padrao:

```bash
bash scripts/rodar_pipeline_c_cnn2d.sh
```

Variaveis principais:

```bash
CNN2D_WINDOW_SIZE=32
CNN2D_MAX_SAMPLES=120000
CNN2D_MAX_ARCHITECTURES=8
CNN2D_EPOCHS=8
CNN2D_CACHE=true
```

Forcar recalculo:

```bash
CNN2D_CACHE=false bash scripts/rodar_pipeline_c_cnn2d.sh
```

Rodar novamente apenas uma arquitetura CNN 2D, ignorando cache:

```bash
CNN2D_MODEL_ONLY=cnn2d_f16_kt3_d64 CNN2D_FORCE_MODEL=true bash scripts/rodar_pipeline_c_cnn2d.sh
```

Os resultados por recorte/alvo ficam na mesma estrutura da Pipeline A, dentro de `2d_models`:

```text
resultados/analises_regressao/<condicao>_sweep_moderado_sem_estimativas_agrupado/<recorte>/<alvo>/2d_models/
```

Os rankings globais da Pipeline C ficam em:

```text
resultados/analises_regressao/2d_models/
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
