## Dicionário de Dados

| Campo | Tipo | Unidade | Descrição |
|---|---|---:|---|
| experiment_name | string | - | Nome do experimento executado. |
| global_seed | inteiro | - | Seed global usada para reprodutibilidade. |
| warmup_kernels | inteiro | kernels | Quantidade de kernels de aquecimento antes da medição. |
| mpi_world_size | inteiro | processos | Número total de processos MPI. |
| mpi_rank | inteiro | - | Rank MPI responsável pela linha medida. |
| threads_per_process | inteiro | threads | Número de threads por processo MPI. |
| kernels_per_thread | inteiro | kernels | Número de kernels lançados por thread. |
| host_thread_id | inteiro | - | Identificador da thread no host. |
| kernel_index_in_thread | inteiro | - | Índice do kernel dentro da thread. |
| thread_local_kernel_index | inteiro | - | Índice local do kernel na thread. |
| global_kernel_id | inteiro | - | Identificador global do kernel no experimento. |
| cuda_device_id | inteiro | - | ID da GPU CUDA usada. |
| arrival_wait_ms | float | ms | Intervalo de espera entre chegadas/submissões de kernels. |
| requested_busy_wait_us | float | us | Tempo de execução solicitado para o kernel busy-wait. |
| kernel_type | string | - | Tipo de kernel executado: busy_wait, compute, memory ou mixed. |
| gpu_name | string | - | Nome da GPU usada. |
| cuda_runtime_version | string/int | - | Versão do CUDA Runtime. |
| cuda_driver_version | string/int | - | Versão do driver CUDA. |
| sm_count | inteiro | SMs | Número de Streaming Multiprocessors da GPU. |
| device_clock_rate_khz | inteiro | kHz | Frequência de clock reportada pela GPU. |
| blocks_x | inteiro | blocos | Número de blocos na dimensão X do grid. |
| threads_per_block | inteiro | threads | Número de threads por bloco CUDA. |
| grid_z | inteiro | blocos | Número de blocos na dimensão Z do grid. |
| total_blocks | inteiro | blocos | Número total de blocos lançados. |
| total_cuda_threads | inteiro | threads | Número total de threads CUDA lançadas. |
| total_warps | float/int | warps | Número total estimado de warps. |
| warps_per_block | float/int | warps/bloco | Número de warps por bloco. |
| blocks_per_sm | float | blocos/SM | Blocos por SM considerando a configuração do kernel. |
| total_blocks_per_sm | float | blocos/SM | Total de blocos distribuídos por SM. |
| effective_workers | inteiro | workers | Número efetivo de workers no host, geralmente MPI x threads. |
| requested_busy_wait_s | float | s | Tempo solicitado convertido para segundos. |
| workers_x_requested_busy_wait_us | float | us | Produto entre workers efetivos e tempo solicitado. |
| workers_x_total_warps | float | warps | Produto entre workers efetivos e total de warps. |
| workers_x_blocks_per_sm | float | blocos/SM | Produto entre workers efetivos e blocos por SM. |
| requested_busy_wait_us_per_arrival_ms | float | us/ms | Razão entre tempo solicitado e intervalo de chegada. |
| logical_stream_id | inteiro | - | Identificador lógico da stream CUDA usada. |
| measurement_start_time_ns | inteiro | ns | Timestamp inicial da medição. |
| time_since_experiment_start_us | float | us | Tempo decorrido desde o início do experimento. |
| rank_local_submitted_count | inteiro | kernels | Número de kernels submetidos localmente pelo rank até o momento. |
| rank_local_completed_count | inteiro | kernels | Número de kernels concluídos localmente pelo rank até o momento. |
| submit_time_ns | inteiro | ns | Timestamp de submissão do kernel. |
| launch_return_time_ns | inteiro | ns | Timestamp em que a chamada de lançamento retornou no host. |
| completion_time_ns | inteiro | ns | Timestamp de conclusão/sincronização do kernel. |
| host_submit_time_ns | inteiro | ns | Timestamp de submissão medido no host. |
| host_completion_time_ns | inteiro | ns | Timestamp de conclusão medido no host. |
| response_time_us | float | us | Tempo de resposta observado do kernel. |
| launch_overhead_us | float | us | Overhead entre submissão e retorno do lançamento CUDA. |
| cuda_event_elapsed_time_us | float | us | Tempo medido por eventos CUDA na GPU. |
| queueing_delay_us | float | us | Atraso estimado associado à fila/contenção antes ou durante execução. |
| slowdown | float | - | Razão entre tempo observado e tempo esperado/solicitado. |
| cuda_error_code | inteiro | - | Código de erro CUDA retornado. |
| cuda_error_string | string | - | Mensagem textual associada ao erro CUDA. |

## Variáveis Deriváveis Relevantes

| Variável | Fórmula aproximada | Uso |
|---|---|---|
| in_flight_kernels_at_launch | kernels com submit_time_ns <= t e completion_time_ns > t | Medir pressão concorrente na GPU. |
| pending_kernels_at_launch | rank_local_submitted_count - rank_local_completed_count | Estimar kernels pendentes no rank. |
| launch_overhead_us | launch_return_time_ns - submit_time_ns | Medir custo de lançamento no host. |
| response_time_us | completion_time_ns - submit_time_ns | Medir tempo total percebido pelo host. |
| gpu_execution_time_us | cuda_event_elapsed_time_us | Medir tempo efetivo na GPU. |
| host_queueing_overhead_us | response_time_us - cuda_event_elapsed_time_us | Estimar espera fora da execução efetiva na GPU. |
| slowdown | response_time_us / requested_busy_wait_us | Medir degradação relativa ao tempo solicitado. |