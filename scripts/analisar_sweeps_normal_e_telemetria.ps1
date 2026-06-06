param(
    [string]$NormalResultsDirs = "",
    [string]$NormalResultsDir = "",
    [string]$TelemetryResultsDirs = "",
    [string]$TelemetryResultsDir = "",
    [string]$AnalysisRoot = "resultados/analises_regressao",
    [string]$Targets = "response_time_us queueing_delay_us slowdown",
    [string]$GpuTargets = "10 50 100 120",
    [int]$CvFolds = 5,
    [bool]$DependencyOnly = $false
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

# Setup paths
$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

# Support both singular and plural parameter names
if ($NormalResultsDir -and -not $NormalResultsDirs) { $NormalResultsDirs = $NormalResultsDir }
if ($TelemetryResultsDir -and -not $TelemetryResultsDirs) { $TelemetryResultsDirs = $TelemetryResultsDir }

# Override from environment variables if set
if ($env:NORMAL_RESULTS_DIRS) { $NormalResultsDirs = $env:NORMAL_RESULTS_DIRS }
if ($env:NORMAL_RESULTS_DIR -and -not $NormalResultsDirs) { $NormalResultsDirs = $env:NORMAL_RESULTS_DIR }
if ($env:TELEMETRY_RESULTS_DIRS) { $TelemetryResultsDirs = $env:TELEMETRY_RESULTS_DIRS }
if ($env:TELEMETRY_RESULTS_DIR -and -not $TelemetryResultsDirs) { $TelemetryResultsDirs = $env:TELEMETRY_RESULTS_DIR }
if ($env:ANALYSIS_ROOT) { $AnalysisRoot = $env:ANALYSIS_ROOT }
if ($env:TARGETS) { $Targets = $env:TARGETS }
if ($env:GPU_TARGETS) { $GpuTargets = $env:GPU_TARGETS }
if ($env:CV_FOLDS) { $CvFolds = [int]$env:CV_FOLDS }
if ($env:DEPENDENCY_ONLY -eq "true" -or $env:DEPENDENCY_ONLY -eq "1") { $DependencyOnly = $true }

# Function to find matching directories
function Get-MatchingDirs {
    param([string]$Pattern)
    
    $dirs = @()
    if (Test-Path "resultados") {
        $dirs = Get-ChildItem -Path "resultados" -Directory -ErrorAction SilentlyContinue | 
            Where-Object { $_.Name -like $Pattern } |
            Sort-Object LastWriteTime |
            Select-Object -ExpandProperty FullName
    }
    return $dirs -join " "
}

# Find directories if not specified
if ([string]::IsNullOrWhiteSpace($NormalResultsDirs)) {
    $NormalResultsDirs = Get-MatchingDirs "sweep_moderado_sem_estimativas_[0-9]*"
}

if ([string]::IsNullOrWhiteSpace($TelemetryResultsDirs)) {
    $TelemetryResultsDirs = Get-MatchingDirs "sweep_moderado_sem_estimativas_telemetry_*"
}

# Function to run analysis
function Invoke-Analysis {
    param(
        [string]$Label,
        [string]$ResultsDirs
    )
    
    if ([string]::IsNullOrWhiteSpace($ResultsDirs)) {
        Write-Error "Pastas do sweep $Label nao encontradas."
        exit 1
    }
    
    $analysisDir = "$AnalysisRoot/${Label}_sweep_moderado_sem_estimativas_agrupado"
    Write-Host "Analise $Label : $ResultsDirs -> $analysisDir"
    
    # First run gerar_resultados_sweep.py
    $targetsArray = $Targets -split '\s+'
    $gpuTargetsArray = $GpuTargets -split '\s+'
    $resultsDirsArray = $ResultsDirs -split '\s+' | Where-Object { $_ }
    
    Write-Host "Gerando resultados do sweep..."
    $geradorArgs = @(
        "scripts/02_gerar_resultados_sweep.py",
        "--analysis-dir", $analysisDir
    )
    
    # Add results dirs as separate arguments
    foreach ($dir in $resultsDirsArray) {
        $geradorArgs += "--results-dir", $dir
    }
    
    # Add targets
    foreach ($target in $targetsArray) {
        $geradorArgs += "--targets", $target
    }
    
    # Add GPU targets
    foreach ($gpuTarget in $gpuTargetsArray) {
        $geradorArgs += "--gpu-targets", $gpuTarget
    }
    
    & python3 $geradorArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Erro ao gerar resultados do sweep"
        exit $LASTEXITCODE
    }
    
    # Build python command arguments for regressor_analysis
    $pythonArgs = @(
        "scripts/03_regressor_analysis.py",
        "compare"
    )
    
    # Add results dirs as separate arguments
    foreach ($dir in $resultsDirsArray) {
        $pythonArgs += "--results-dir", $dir
    }
    
    $pythonArgs += @(
        "--analysis-dir", $analysisDir,
        "--jobs-file", "$analysisDir/analysis_jobs.csv",
        "--first-sweep",
        "--cv-folds", $CvFolds.ToString()
    )

    if ($DependencyOnly) {
        $pythonArgs += "--dependency-only"
    }
    
    # Then run regressor analysis
    Write-Host "Executando analise de regressao..."
    & python3 $pythonArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Erro ao executar analise de regressao"
        exit $LASTEXITCODE
    }

    if (Test-Path "scripts/05_plot_best_model_rankings.py") {
        & python3 "scripts/05_plot_best_model_rankings.py" "--analysis-root" $AnalysisRoot
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Erro ao gerar ranking dos melhores modelos"
            exit $LASTEXITCODE
        }
    }

    if ($DependencyOnly -and (Test-Path "scripts/06_plot_dependency_rankings.py")) {
        & python3 "scripts/06_plot_dependency_rankings.py" "--analysis-root" $AnalysisRoot
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Erro ao gerar ranking de dependencia"
            exit $LASTEXITCODE
        }
    }
}

# Run both analyses
Invoke-Analysis "sem_telemetria" $NormalResultsDirs
Invoke-Analysis "com_telemetria" $TelemetryResultsDirs

Write-Host "Analise concluida com sucesso!" -ForegroundColor Green
