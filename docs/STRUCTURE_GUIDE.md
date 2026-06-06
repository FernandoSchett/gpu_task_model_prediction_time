# 📂 Guia de Estrutura de Dados

## Hierarquia de Diretórios

```
tstr_cuda/
├── docs/                              # 📚 Documentação
│   ├── DATA_DICTIONARY.md             # ← Você está aqui
│   ├── data_dictionary.csv            # Versão tabular
│   └── STRUCTURE_GUIDE.md             # Este arquivo
│
├── resultados/                        # 📊 Dados de Experimentos
│   ├── sweep_moderado_sem_estimativas_20260527_104037/
│   │   └── resultados_experimentos_*.csv   # ← Dados brutos
│   └── analises_regressao/            # 🔬 Análises treinadas
│       └── sweep_moderado_sem_estimativas_20260527_104037/
│           └── geral/
│               ├── response_time_us/
│               │   ├── trained_models/          # 🤖 Modelos salvos
│               │   │   ├── linear_regression.pkl
│               │   │   ├── ridge_regression.pkl
│               │   │   ├── random_forest.pkl
│               │   │   ├── lightgbm.pkl
│               │   │   ├── xgboost.pkl
│               │   │   ├── catboost.pkl
│               │   │   ├── tabpfn.pkl
│               │   │   └── models_info.txt
│               │   ├── regression_metrics.csv  # 📈 Métricas
│               │   ├── mae_comparison.png
│               │   ├── rmse_comparison.png
│               │   └── r2_comparison.png
│               ├── queueing_delay_us/          # Outro target
│               └── slowdown/                    # Outro target
│
├── scripts/                           # 🐍 Python Scripts
│   ├── 01_analyze_slowdown.py         # EDA
│   ├── 02_gerar_resultados_sweep.py   # Data prep
│   ├── 03_regressor_analysis.py       # ML training
│   ├── 04_build_gantt_csv.py          # Timeline
│   └── use_trained_models.py          # Model inference
│
└── libs/                              # C++ Implementation
    ├── include/                       # Headers
    └── src/                           # Implementation
```

---

## 📊 Fluxo de Dados

```
┌─────────────────────────────────────────────┐
│ GPU Benchmark Experiments (CUDA)            │
│ (C++ Application)                           │
└────────────────┬────────────────────────────┘
                 │
                 ├─→ Generates: resultados_experimentos_*.csv
                 │   (Raw experimental data with 54 fields)
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 01_analyze_slowdown.py                      │
│ Exploratory Data Analysis                   │
│ - Calculate slowdown statistics             │
│ - Identify outliers                         │
│ - Summary by configuration                  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 02_gerar_resultados_sweep.py                │
│ Data Preparation                            │
│ - Scan sweep directories                    │
│ - Extract GPU targets from filenames        │
│ - Create analysis job manifest              │
│ Output: analysis_jobs.csv                   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 03_regressor_analysis.py (compare mode)     │
│ Machine Learning Training                   │
│                                             │
│ Train 11+ Models:                           │
│ - Linear/Ridge/Poly Ridge (baselines)       │
│ - Random Forest, Gradient Boosting          │
│ - LightGBM, XGBoost, CatBoost              │
│ - LightGBM/XGBoost Quantiles               │
│ - TabPFN (pretrained)                       │
│ - kNN                                       │
│                                             │
│ Outputs:                                    │
│ - regression_metrics.csv                    │
│ - *.png (comparison plots)                  │
│ - trained_models/ folder                    │
│   - *.pkl files (sklearn models)            │
│   - models_info.txt (metadata)              │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ use_trained_models.py                       │
│ Model Inference                             │
│ - Load saved models                         │
│ - Make predictions on new data              │
│ - Generate predictions CSV                  │
└─────────────────────────────────────────────┘
```

---

## 🔄 Ciclo de Análise

### 1️⃣ Preparação (Setup)
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install large models (TabPFN, XGBoost, etc)
.\install_dependencies.ps1
```

### 2️⃣ Coleta de Dados
```bash
# Run GPU benchmark experiments (C++)
# Gera: resultados_experimentos_*.csv em resultados/sweep_*/
```

### 3️⃣ Exploração (EDA)
```bash
# Analyze slowdown patterns
python3 scripts/01_analyze_slowdown.py --results-dir resultados/sweep_x
```

### 4️⃣ Preparação (Data Prep)
```bash
# Generate analysis job manifest
python3 scripts/02_gerar_resultados_sweep.py \
  --results-dir resultados/sweep_x \
  --analysis-dir resultados/analises_regressao
```

### 5️⃣ Treinamento de Modelos (ML)
```bash
# Train all 11+ models with cross-validation
python3 scripts/03_regressor_analysis.py compare \
  --results-dir resultados/sweep_x \
  --cv-folds 5
```

### 6️⃣ Inferência (Prediction)
```bash
# Load trained models and predict on new data
python3 scripts/use_trained_models.py
```

---

## 📋 Campos Principais por Categoria

### 🎯 Prediction Targets (Y)
- `response_time_us` - **Principal**: Tempo total de resposta
- `queueing_delay_us` - Tempo de espera na fila
- `slowdown` - Fator de desaceleração

### 📊 Feature Categories (X)

#### Baseline (Linear)
- `requested_busy_wait_us` - Tempo esperado de execução
- `arrival_wait_ms` - Inter-arrival time
- `mpi_world_size`, `threads_per_process` - Paralelismo

#### Kernel Configuration
- `blocks_x`, `threads_per_block`, `grid_z` - Dimensões CUDA
- `total_cuda_threads`, `total_warps` - Tamanho do kernel
- `kernel_type_*` - One-hot: busy_wait, compute, memory, mixed

#### Hardware
- `sm_count` - Número de Streaming Multiprocessors
- `device_clock_rate_khz` - Frequency da GPU
- `gpu_name`, `cuda_driver_version` - Especificação

#### Derived (Feature Engineering)
- `effective_workers` = mpi_world_size × threads_per_process
- `blocks_per_sm` = total_blocks / sm_count
- `warps_per_block` = ceil(threads_per_block / 32)
- `workers_x_total_warps` = effective_workers × total_warps
- `workers_x_requested_busy_wait_us` = effective_workers × requested_busy_wait_us
- `requested_busy_wait_us_per_arrival_ms` = requested_busy_wait_us / arrival_wait_ms

---

## 🔍 Como Explorar os Dados

### Via Python
```python
import pandas as pd

# Load raw experiment data
df = pd.read_csv("resultados/sweep_x/resultados_experimentos_*.csv")

# Analyze targets
print(df[['response_time_us', 'queueing_delay_us', 'slowdown']].describe())

# Check features
features = [col for col in df.columns if col not in 
            ['response_time_us', 'queueing_delay_us', 'slowdown', 'cuda_error_string']]
print(df[features].dtypes)
```

### Via SQL (Excel/Database)
```sql
SELECT 
  kernel_type,
  COUNT(*) as count,
  AVG(response_time_us) as avg_response,
  MAX(slowdown) as max_slowdown
FROM resultados_experimentos
WHERE cuda_error_code = 0
GROUP BY kernel_type
```

### Via Pandas Profiling
```python
import pandas_profiling
df.profile_report().to_file("report.html")
```

---

## 📈 Modelos Disponíveis

| Modelo | Arquivo | Tipo | Early Stopping | Quantile |
|--------|---------|------|---------------|----------|
| Linear Regression | linear_regression.pkl | Baseline | ❌ | ❌ |
| Ridge Regression | ridge_regression.pkl | Baseline | ❌ | ❌ |
| Polynomial Ridge | polynomial_ridge.pkl | Baseline | ❌ | ❌ |
| Random Forest | random_forest.pkl | Ensemble | ✅ (Optuna) | ❌ |
| Gradient Boosting | gradient_boosting.pkl | Ensemble | ✅ (Optuna) | ❌ |
| LightGBM | lightgbm.pkl | Boosting | ✅ | ✅ (p90/p95/p99) |
| XGBoost | xgboost.pkl | Boosting | ✅ | ✅ (p90/p95/p99) |
| CatBoost | catboost.pkl | Boosting | ✅ | ❌ |
| TabPFN | tabpfn.pkl | Foundation | ❌ | ❌ |
| kNN Regression | (não salvo) | Non-parametric | ❌ | ❌ |

---

## 🔒 Considerações de Data Quality

### Filtragem Automática
O script `03_regressor_analysis.py` automaticamente:
- Remove linhas com `cuda_error_code ≠ 0`
- Remove valores NaN/Inf
- Aplica deterministic sampling com seed=42

### Validações Esperadas
- `mpi_rank` ∈ [0, mpi_world_size)
- `threads_per_block` ≤ 1024
- `response_time_us` ≥ `requested_busy_wait_us`
- `slowdown` ≥ 1.0

---

## 📝 Versioning

- **DATA_DICTIONARY v1.0** - 27 maio 2026
- **54 campos** documentados
- **11+ modelos** suportados
- **3 targets** de previsão

---

## 🤝 Contribuindo com Novos Campos

Se adicionar novos campos ao C++ benchmark:

1. Atualize [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
2. Atualize [data_dictionary.csv](data_dictionary.csv)
3. Atualize feature lists em `03_regressor_analysis.py`
4. Re-run análises com novos dados

---

**Perguntas?** Consulte DATA_DICTIONARY.md para explicações detalhadas!
