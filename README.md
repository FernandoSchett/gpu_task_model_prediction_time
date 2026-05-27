# CUDA Benchmark

Projeto de execução e análise de benchmarks CUDA para medir latência, slowdown, tempo de resposta e comportamento de execução em GPU.

O objetivo é rodar experimentos normais e com telemetria, preparar os dados e comparar modelos de regressão para prever métricas de desempenho.

## 1. Build

	mkdir -p build && cd build
	cmake .. && make
	cd ..

## 2. Executar Experimentos

Experimentos normais:

	bash scripts/rodar_experimentos.sh

Experimentos com telemetria GPU:

	bash scripts/rodar_experimentos_com_telemetria.sh

Teste de slowdown:

	bash scripts/run_slowdown_test.sh

Sweep normal e com telemetria:

	bash scripts/rodar_sweep_normal_e_telemetria.sh

## 3. Análises

Análise inicial de slowdown:

	python3 scripts/01_analyze_slowdown.py

Preparar dados do sweep:

	python3 scripts/02_gerar_resultados_sweep.py \
	  --results-dir resultados/sweep_moderado_sem_estimativas_20260527_121911 \
	  --analysis-dir resultados/analises_regressao/meu_sweep

Rodar regressão:

	python3 scripts/03_regressor_analysis.py compare \
	  --results-dir resultados/sweep_moderado_sem_estimativas_20260527_121911 \
	  --target response_time_us \
	  --cv-folds 5 \
	  --optimize-hyperparams \
	  --optuna-trials 30

Gerar CSV para Gantt:

	python3 scripts/04_build_gantt_csv.py \
	  --results-dir resultados/sweep_moderado_sem_estimativas_20260527_121911 \
	  --output resultados/gantt_intervals.csv

## 4. Execução Automática

Windows:

	.\scripts\analisar_sweeps_normal_e_telemetria.ps1 -CvFolds 5 -OptunaTrials 30

Windows com pasta específica:

	.\scripts\analisar_sweeps_normal_e_telemetria.ps1 `
	  -NormalResultsDir "resultados/sweep_moderado_sem_estimativas_20260527_121911" `
	  -CvFolds 5 `
	  -OptimizeHyperparams $true `
	  -OptunaTrials 30

Linux/Mac:

	bash scripts/analisar_sweeps_normal_e_telemetria.sh

Linux/Mac com parâmetros:

	RESULTS_DIRS="resultados/sweep_x resultados/sweep_y" \
	CV_FOLDS=5 \
	OPTIMIZE_HYPERPARAMS=true \
	OPTUNA_TRIALS=30 \
	bash scripts/analisar_sweeps_normal_e_telemetria.sh

## 5. Parâmetros Principais

- --cv-folds: número de folds da validação cruzada.
- --optimize-hyperparams: ativa otimização com Optuna.
- --optuna-trials: número de tentativas do Optuna.
- --max-rows: limite de linhas usadas na análise.
- --target: variável alvo da regressão.

Targets suportados:

- response_time_us
- queueing_delay_us
- slowdown

## 6. Saídas

- resultados/sweep_moderado_sem_estimativas_*/: experimentos brutos.
- resultados/sweep_moderado_sem_estimativas_telemetry_*/: experimentos com telemetria.
- resultados/analises_regressao/: análises geradas.
- analysis_jobs.csv: jobs de análise.
- dataset_summary.csv: resumo do dataset.
- regression_metrics.csv: métricas dos modelos.
- mae_comparison.png: comparação por MAE.
- rmse_comparison.png: comparação por RMSE.
- r2_comparison.png: comparação por R².
- resultados/gantt_intervals.csv: intervalos para Gantt.

## 7. Modelos Avaliados

- Linear Regression
- Ridge Regression
- Polynomial Ridge
- Decision Tree
- Random Forest
- Gradient Boosting
- kNN

## 8. Quick Start

	bash scripts/rodar_experimentos.sh

	python3 scripts/02_gerar_resultados_sweep.py \
	  --results-dir resultados/sweep_moderado_sem_estimativas_20260527_121911 \
	  --analysis-dir resultados/analises_regressao/analise_rapida

	python3 scripts/03_regressor_analysis.py compare \
	  --results-dir resultados/sweep_moderado_sem_estimativas_20260527_121911 \
	  --analysis-dir resultados/analises_regressao/analise_rapida \
	  --jobs-file resultados/analises_regressao/analise_rapida/analysis_jobs.csv \
	  --cv-folds 5

## 9. Troubleshooting

- Nenhum CSV gputarget encontrado: verifique o caminho em --results-dir.
- Optuna not available: instale com pip install optuna.
- sklearn not available: instale com pip install scikit-learn numpy matplotlib.
- Nenhuma pasta sweep encontrada: informe a pasta manualmente.