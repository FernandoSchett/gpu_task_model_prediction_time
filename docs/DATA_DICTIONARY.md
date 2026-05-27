# 📊 Dicionário de Dados - Experimentos CUDA GPU

## Visão Geral

Este documento descreve todos os campos presentes nos arquivos CSV de resultados gerados pelos experimentos de benchmark CUDA. Cada linha representa a execução de um kernel CUDA em um processador GPU.

**Total de campos:** 54  
**Formato:** CSV (valores separados por vírgula)  
**Encoding:** UTF-8

---

## 1️⃣ Configuração do Experimento

### `experiment_name`
- **Tipo:** String
- **Descrição:** Identificador único do experimento
- **Exemplo:** `s67_gputarget120_r4_t8_k1200_w20_ktmixed_bx64_tpb256_gz1_ku500-2000_am0-0.5`
- **Componentes:**
  - `s67`: Seed global (67)
  - `gputarget120`: Target GPU load (120%)
  - `r4`: 4 ranks (MPI processes)
  - `t8`: 8 threads per rank
  - `k1200`: 1200 kernels por thread
  - `w20`: 20 warmup kernels
  - `ktmixed`: Kernel type (mixed)
  - `bx64`: Blocks X dimension (64)
  - `tpb256`: Threads per block (256)
  - `gz1`: Grid Z dimension (1)
  - `ku500-2000`: Kernel utilization entre 500-2000
  - `am0-0.5`: Arrival wait entre 0-0.5ms

### `global_seed`
- **Tipo:** Inteiro
- **Descrição:** Seed para reprodutibilidade do experimento
- **Intervalo:** 0-∞
- **Exemplo:** 67

### `warmup_kernels`
- **Tipo:** Inteiro
- **Descrição:** Número de kernels de aquecimento (warmup) executados antes da medição
- **Propósito:** Estabilizar GPU state, cache, e thermal profile antes de medições reais
- **Exemplo:** 20

### `global_kernel_id`
- **Tipo:** Inteiro
- **Descrição:** ID global único para cada kernel na sequência de execução
- **Intervalo:** 0-∞
- **Fórmula:** Incrementado sequencialmente para todos os kernels

---

## 2️⃣ MPI (Message Passing Interface) - Distribuição

### `mpi_world_size`
- **Tipo:** Inteiro
- **Descrição:** Número total de MPI ranks (processos paralelos)
- **Exemplo:** 4
- **Nota:** Define o paralelismo horizontal da aplicação

### `mpi_rank`
- **Tipo:** Inteiro
- **Descrição:** ID único do rank MPI (0-indexed)
- **Intervalo:** 0 a (mpi_world_size - 1)
- **Exemplo:** 0, 1, 2, 3

### `effective_workers`
- **Tipo:** Float
- **Descrição:** Número efetivo de workers = mpi_world_size × threads_per_process
- **Fórmula:** `mpi_world_size * threads_per_process`
- **Exemplo:** 32 (4 ranks × 8 threads)

---

## 3️⃣ Threads e Paralelismo

### `threads_per_process`
- **Tipo:** Inteiro
- **Descrição:** Número de threads dentro de cada processo MPI
- **Exemplo:** 8
- **Total workers:** mpi_world_size × threads_per_process

### `host_thread_id`
- **Tipo:** Inteiro
- **Descrição:** ID da thread no host que submete o kernel
- **Intervalo:** 0 a (threads_per_process - 1)

### `kernels_per_thread`
- **Tipo:** Inteiro
- **Descrição:** Número de kernels que cada thread submete
- **Exemplo:** 1200

### `kernel_index_in_thread`
- **Tipo:** Inteiro
- **Descrição:** Índice sequencial do kernel dentro da thread
- **Intervalo:** 0 a (kernels_per_thread - 1)

### `thread_local_kernel_index`
- **Tipo:** Inteiro
- **Descrição:** Índice local do kernel dentro da thread
- **Intervalo:** 0 a (kernels_per_thread - 1)

---

## 4️⃣ Configuração de Hardware

### `gpu_name`
- **Tipo:** String
- **Descrição:** Nome comercial da GPU
- **Exemplo:** `NVIDIA A100`, `NVIDIA RTX 3090`, `NVIDIA H100`

### `cuda_device_id`
- **Tipo:** Inteiro
- **Descrição:** ID do dispositivo CUDA (quando múltiplas GPUs estão presentes)
- **Exemplo:** 0 (primeira GPU)

### `sm_count`
- **Tipo:** Inteiro
- **Descrição:** Número de Streaming Multiprocessors (SMs) na GPU
- **Exemplo:** 108 (para A100), 40 (para RTX 3090)
- **Importância:** Define a capacidade paralela da GPU

### `device_clock_rate_khz`
- **Tipo:** Inteiro
- **Descrição:** Frequency de clock do dispositivo em KHz
- **Exemplo:** 1410000 (1.41 GHz)
- **Conversão:** divide por 1,000,000 para obter GHz

### `cuda_driver_version`
- **Tipo:** String/Inteiro
- **Descrição:** Versão do CUDA driver instalado
- **Exemplo:** 11060 (CUDA 11.6)

### `cuda_runtime_version`
- **Tipo:** String/Inteiro
- **Descrição:** Versão da CUDA runtime
- **Exemplo:** 11060 (CUDA 11.6)

---

## 5️⃣ Configuração de Kernel

### `kernel_type`
- **Tipo:** Categorias: `busy_wait`, `compute`, `memory`, `mixed`
- **Descrição:** Tipo de operação que o kernel executa
  - **busy_wait:** Spin loop (aguarda em loop)
  - **compute:** Operações computacionais (math-heavy)
  - **memory:** Operações com memória (memory-heavy)
  - **mixed:** Combinação de compute e memory
- **Exemplo:** `mixed`

### `blocks_x`
- **Tipo:** Inteiro
- **Descrição:** Dimensão X do grid (número de blocos em X)
- **Exemplo:** 64

### `threads_per_block`
- **Tipo:** Inteiro
- **Descrição:** Número de threads por bloco
- **Restrição:** Máximo 1024 para a maioria de GPUs
- **Exemplo:** 256
- **Warps por bloco:** threads_per_block / 32

### `grid_z`
- **Tipo:** Inteiro
- **Descrição:** Dimensão Z do grid (terceira dimensão)
- **Exemplo:** 1 (geralmente 1 para grids 2D)

### `total_blocks`
- **Tipo:** Inteiro
- **Descrição:** Número total de blocos no kernel
- **Fórmula:** `blocks_x * grid_z`
- **Exemplo:** 64

### `total_cuda_threads`
- **Tipo:** Inteiro
- **Descrição:** Número total de threads CUDA lançadas
- **Fórmula:** `total_blocks * threads_per_block`
- **Exemplo:** 16384 (64 × 256)

### `total_warps`
- **Tipo:** Inteiro
- **Descrição:** Número total de warps (grupos de 32 threads)
- **Fórmula:** `total_blocks * warps_per_block`
- **Exemplo:** 512

### `warps_per_block`
- **Tipo:** Float
- **Descrição:** Número de warps por bloco
- **Fórmula:** `ceil(threads_per_block / 32)`
- **Exemplo:** 8 (256 threads = 8 warps)

### `blocks_per_sm`
- **Tipo:** Float
- **Descrição:** Número de blocos por SM (Streaming Multiprocessor)
- **Fórmula:** `total_blocks / sm_count`
- **Exemplo:** 0.593 (64 blocos / 108 SMs)

### `total_blocks_per_sm`
- **Tipo:** Float
- **Descrição:** Mesma que `blocks_per_sm`
- **Fórmula:** `total_blocks / sm_count`

### `workers_x_blocks_per_sm`
- **Tipo:** Float
- **Descrição:** effective_workers × blocks_per_sm
- **Fórmula:** `effective_workers * blocks_per_sm`
- **Uso:** Feature derivada para ML

### `workers_x_total_warps`
- **Tipo:** Float
- **Descrição:** effective_workers × total_warps
- **Fórmula:** `effective_workers * total_warps`
- **Uso:** Feature derivada para ML

---

## 6️⃣ Timing - Tempos de Chegada

### `arrival_wait_ms`
- **Tipo:** Float
- **Descrição:** Tempo de espera entre submissões de kernels (em milissegundos)
- **Intervalo:** 0.0 - 1000.0 ms
- **Propósito:** Simula padrões de chegada (inter-arrival time)
- **Exemplo:** 0.25

### `requested_busy_wait_us`
- **Tipo:** Float
- **Descrição:** Tempo que o kernel aguarda em busy-wait (spin loop) em microsegundos
- **Intervalo:** Tipicamente 100-10000 µs
- **Exemplo:** 1500.0
- **Derivadas:**
  - `requested_busy_wait_s`: Conversão para segundos
  - `workers_x_requested_busy_wait_us`: effective_workers × requested_busy_wait_us
  - `requested_busy_wait_us_per_arrival_ms`: requested_busy_wait_us / arrival_wait_ms

### `measurement_start_time_ns`
- **Tipo:** Inteiro (nanosegundos)
- **Descrição:** Timestamp absoluto do início da medição
- **Unidade:** nanosegundos
- **Nota:** Usado para sincronização entre hosts

### `time_since_experiment_start_us`
- **Tipo:** Float
- **Descrição:** Tempo decorrido desde o início do experimento (em microsegundos)
- **Exemplo:** 125478.5

### `rank_local_submitted_count`
- **Tipo:** Inteiro
- **Descrição:** Número sequencial de kernels submetidos por este rank
- **Intervalo:** 0 a kernels_per_thread

### `rank_local_completed_count`
- **Tipo:** Inteiro
- **Descrição:** Número sequencial de kernels completados por este rank
- **Intervalo:** 0 a kernels_per_thread

---

## 7️⃣ Timing - Eventos CUDA

### `submit_time_ns`
- **Tipo:** Inteiro (nanosegundos)
- **Descrição:** Timestamp quando o kernel foi submetido à fila CUDA
- **Unidade:** nanosegundos (ns)

### `launch_return_time_ns`
- **Tipo:** Inteiro (nanosegundos)
- **Descrição:** Timestamp quando a chamada de lançamento do kernel retornou
- **Unidade:** nanosegundos

### `completion_time_ns`
- **Tipo:** Inteiro (nanosegundos)
- **Descrição:** Timestamp quando o kernel completou sua execução
- **Unidade:** nanosegundos

### `host_submit_time_ns`
- **Tipo:** Inteiro (nanosegundos)
- **Descrição:** Timestamp no host quando o kernel foi submetido
- **Unidade:** nanosegundos

### `host_completion_time_ns`
- **Tipo:** Inteiro (nanosegundos)
- **Descrição:** Timestamp no host quando o kernel completou
- **Unidade:** nanosegundos

### `logical_stream_id`
- **Tipo:** Inteiro
- **Descrição:** ID da stream CUDA (para ordenação de kernels)
- **Exemplo:** 0, 1, 2...
- **Nota:** Determina a ordem de execução

---

## 8️⃣ Métricas de Performance

### `launch_overhead_us`
- **Tipo:** Float
- **Descrição:** Overhead de lançamento em microsegundos
- **Fórmula:** `launch_return_time_ns - submit_time_ns` (em µs)
- **Propósito:** Mede tempo gasto em syscalls e scheduler
- **Exemplo:** 45.5

### `response_time_us`
- **Tipo:** Float
- **Descrição:** Tempo total de resposta (do submit ao completion) em microsegundos
- **Fórmula:** `completion_time_ns - submit_time_ns` (em µs)
- **Exemplo:** 1523.2
- **Inclui:** queueing_delay + kernel_execution_time

### `queueing_delay_us`
- **Tipo:** Float
- **Descrição:** Tempo que o kernel esperou na fila antes de executar (microsegundos)
- **Fórmula:** `response_time_us - requested_busy_wait_us`
- **Exemplo:** 23.2
- **Propósito:** Mede contenção de GPU

### `cuda_event_elapsed_time_us`
- **Tipo:** Float
- **Descrição:** Tempo de execução reportado pelos eventos CUDA (microsegundos)
- **Unidade:** microsegundos
- **Nota:** Pode diferir de response_time_us em syscalls overhead

### `slowdown`
- **Tipo:** Float
- **Descrição:** Fator de desaceleração
- **Fórmula:** `response_time_us / requested_busy_wait_us`
- **Exemplo:** 1.5 (resposta 1.5x mais lenta que esperado)
- **Interpretação:**
  - 1.0 = sem contention
  - > 1.0 = há contention
  - >> 1.0 = contention severa

### `workers_x_requested_busy_wait_us`
- **Tipo:** Float
- **Descrição:** effective_workers × requested_busy_wait_us
- **Fórmula:** `effective_workers * requested_busy_wait_us`
- **Uso:** Feature para ML que considera paralelismo

### `requested_busy_wait_us_per_arrival_ms`
- **Tipo:** Float
- **Descrição:** Razão de busy_wait para inter-arrival time
- **Fórmula:** `requested_busy_wait_us / arrival_wait_ms`
- **Interpretação:** Se > 1.0, kernel dura mais que intervalo de chegada

---

## 9️⃣ Erro e Status

### `cuda_error_code`
- **Tipo:** Inteiro
- **Descrição:** Código de erro CUDA
- **Valores comuns:**
  - `0`: Sucesso (cudaSuccess)
  - `1`: InvalidValue
  - `2`: MemoryAllocation
  - `3`: NotInitialized
  - `4`: Deinitialized
  - `7`: CudaRuntimeError
  - (ver [CUDA Error Codes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94cf4891e86))
- **Exemplo:** 0 (sucesso)

### `cuda_error_string`
- **Tipo:** String
- **Descrição:** Descrição textual do erro CUDA
- **Exemplo:** `"cudaSuccess"`, `"cudaErrorInvalidValue"`
- **Nota:** Vazio ou omitido se sucesso

---

## 🔟 Features Derivadas para ML

Durante o pré-processamento para análise de regressão, as seguintes features são derivadas:

| Feature | Fórmula | Uso |
|---------|---------|-----|
| `kernel_type_busy_wait` | 1 se kernel_type=="busy_wait" | One-hot encoding |
| `kernel_type_compute` | 1 se kernel_type=="compute" | One-hot encoding |
| `kernel_type_memory` | 1 se kernel_type=="memory" | One-hot encoding |
| `kernel_type_mixed` | 1 se kernel_type=="mixed" | One-hot encoding |
| `target_gpu_demand_percent` | Extraído de experiment_name | Carga GPU alvo |

---

## 📋 Exemplo de Linha Completa

```csv
experiment_name,global_seed,warmup_kernels,mpi_world_size,mpi_rank,threads_per_process,...
s67_gputarget120_r4_t8_k1200_w20_ktmixed_bx64_tpb256_gz1_ku500-2000_am0-0.5,67,20,4,0,8,...
```

---

## 📊 Estatísticas Típicas

| Campo | Min | Max | Mean | Unidade |
|-------|-----|-----|------|---------|
| arrival_wait_ms | 0.0 | 0.5 | 0.25 | ms |
| requested_busy_wait_us | 500 | 2000 | 1250 | µs |
| response_time_us | 523 | 15000 | 2500 | µs |
| queueing_delay_us | 0 | 13000 | 1250 | µs |
| slowdown | 1.0 | 15.0 | 2.0 | ratio |
| launch_overhead_us | 10 | 200 | 50 | µs |

---

## 🎯 Targets Comuns para Regressão

1. **response_time_us**: Tempo total de resposta (alvo primário)
2. **queueing_delay_us**: Tempo de espera na fila
3. **slowdown**: Fator de desaceleração

---

## 💾 Como Usar Este Dicionário

1. **Para análise:** Consulte a seção correspondente ao campo de interesse
2. **Para engenharia de features:** Veja "Features Derivadas para ML"
3. **Para compreender timeouts:** Consulte "Métricas de Performance"
4. **Para debugging:** Verifique "Erro e Status"

---

**Última atualização:** 27 de maio de 2026  
**Versão:** 1.0
