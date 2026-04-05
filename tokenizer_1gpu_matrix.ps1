# 1-GPU tokenizer matrix runner
#
# Exports tokenizer variants from a tokenizer config, then runs train_gpt.py once per
# tokenizer x seed combination and writes per-run + aggregate CSVs.
#
# Usage:
#   .\tokenizer_1gpu_matrix.ps1
#
# Optional env overrides:
#   TOKENIZER_1GPU_MODE=quick|full               (default: quick)
#   TOKENIZER_1GPU_SEEDS=1337,2027               (default: 1337,2027)
#   TOKENIZER_1GPU_CONFIG=./data/tokenizer_specs_1gpu_matrix.json
#   TOKENIZER_1GPU_SWEEP_ID=tok1gpu_20260405
#   TOKENIZER_1GPU_OUTPUT_ROOT=./data/tokenizer_sweeps/<sweep_id>
#   TOKENIZER_1GPU_SKIP_EXPORT=1                 (reuse existing output_root)
#   TOKENIZER_1GPU_TOKENIZER_NAMES=name1,name2   (run subset from manifest tokenizer names)
#   TOKENIZER_1GPU_TRAINER_DOCS=200000           (passed to tokenizer export)
#
# Model/training/export knobs can all be overridden via normal env vars (NUM_LAYERS, QUANT_SCHEME, ...).

$ErrorActionPreference = "Stop"

function Set-EnvVar([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Get-EnvOrDefault([string]$Name, [string]$DefaultValue) {
    $v = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($v)) { return $DefaultValue }
    return $v
}

function Get-Median([double[]]$Values) {
    if ($null -eq $Values -or $Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $n = $sorted.Count
    if (($n % 2) -eq 1) { return [double]$sorted[[int]($n / 2)] }
    return ([double]$sorted[($n / 2) - 1] + [double]$sorted[$n / 2]) / 2.0
}

function Parse-RunLog([string]$LogPath) {
    if (-not (Test-Path $LogPath)) {
        return [pscustomobject]@{
            LogPath = $LogPath
            ValBpbPreQuant = ""
            ValBpbRoundtrip = ""
            ValLossRoundtrip = ""
            TotalSubmissionBytes = ""
            BudgetBytes = ""
            BudgetHeadroomBytes = ""
            UnderBudget = ""
            ParseStatus = "missing_log"
        }
    }

    $lines = Get-Content $LogPath
    $prequantLine = ($lines | Select-String -Pattern '^step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+' | Select-Object -Last 1).Line
    $roundtripLine = ($lines | Select-String -Pattern '^final_.*_roundtrip_exact .*val_loss:[0-9.]+ val_bpb:[0-9.]+' | Select-Object -Last 1).Line
    $budgetLine = ($lines | Select-String -Pattern '^submission_budget .+ total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+' | Select-Object -Last 1).Line

    $preBpb = ""
    $rtBpb = ""
    $rtLoss = ""
    $total = ""
    $budget = ""
    $headroom = ""
    $underBudget = ""
    $status = "ok"

    if ($prequantLine) {
        $m = [regex]::Match($prequantLine, 'val_bpb:(?<b>[0-9.]+)')
        if ($m.Success) { $preBpb = $m.Groups['b'].Value }
    }

    if ($roundtripLine) {
        $m = [regex]::Match($roundtripLine, 'val_bpb:(?<b>[0-9.]+)')
        if ($m.Success) { $rtBpb = $m.Groups['b'].Value }
        $m = [regex]::Match($roundtripLine, 'val_loss:(?<l>[0-9.]+)')
        if ($m.Success) { $rtLoss = $m.Groups['l'].Value }
    } else {
        $status = "missing_roundtrip"
    }

    if ($budgetLine) {
        $m = [regex]::Match($budgetLine, 'total:(?<t>[0-9]+)')
        if ($m.Success) { $total = $m.Groups['t'].Value }
        $m = [regex]::Match($budgetLine, 'budget:(?<b>[0-9]+)')
        if ($m.Success) { $budget = $m.Groups['b'].Value }
        $m = [regex]::Match($budgetLine, 'headroom_bytes:(?<h>[0-9]+)')
        if ($m.Success) {
            $headroom = $m.Groups['h'].Value
            $underBudget = "True"
        } else {
            $m = [regex]::Match($budgetLine, 'over_bytes:(?<o>[0-9]+)')
            if ($m.Success) {
                $headroom = "-" + $m.Groups['o'].Value
                $underBudget = "False"
            }
        }
    }

    return [pscustomobject]@{
        LogPath = $LogPath
        ValBpbPreQuant = $preBpb
        ValBpbRoundtrip = $rtBpb
        ValLossRoundtrip = $rtLoss
        TotalSubmissionBytes = $total
        BudgetBytes = $budget
        BudgetHeadroomBytes = $headroom
        UnderBudget = $underBudget
        ParseStatus = $status
    }
}

function Resolve-TokenizerRuns([string]$ManifestPath, [string]$OutputRoot, [string[]]$TokenizerNameFilter) {
    if (-not (Test-Path $ManifestPath)) {
        throw "Manifest not found: $ManifestPath"
    }
    $manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json

    $tokenizerMap = @{}
    foreach ($tok in $manifest.tokenizers) {
        $tokenizerMap[[string]$tok.name] = $tok
    }

    $runs = @()
    foreach ($ds in $manifest.datasets) {
        $tokName = [string]$ds.tokenizer_name
        if ($TokenizerNameFilter.Count -gt 0 -and -not ($TokenizerNameFilter -contains $tokName)) {
            continue
        }
        if (-not $tokenizerMap.ContainsKey($tokName)) { continue }

        $tok = $tokenizerMap[$tokName]
        $modelPathRel = [string]$tok.model_path
        if ([string]::IsNullOrWhiteSpace($modelPathRel)) {
            continue
        }

        $runs += [pscustomobject]@{
            TokenizerName = $tokName
            DatasetName = [string]$ds.name
            VocabSize = [int]$ds.vocab_size
            DataPath = (Join-Path $OutputRoot ([string]$ds.path))
            TokenizerPath = (Join-Path $OutputRoot $modelPathRel)
            RecommendedBigramVocabSize = [string]$ds.recommended_bigram_vocab_size
        }
    }

    if ($runs.Count -eq 0) {
        throw "No tokenizer datasets selected from manifest: $ManifestPath"
    }
    return @($runs | Sort-Object TokenizerName)
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sweepId = Get-EnvOrDefault "TOKENIZER_1GPU_SWEEP_ID" "tok1gpu_$timestamp"
$configPath = (Resolve-Path (Get-EnvOrDefault "TOKENIZER_1GPU_CONFIG" ".\data\tokenizer_specs_1gpu_matrix.json")).Path
$outputRootDefault = Join-Path ".\data\tokenizer_sweeps" $sweepId
$outputRoot = Get-EnvOrDefault "TOKENIZER_1GPU_OUTPUT_ROOT" $outputRootDefault
$outputRoot = [System.IO.Path]::GetFullPath($outputRoot)
$mode = (Get-EnvOrDefault "TOKENIZER_1GPU_MODE" "quick").ToLowerInvariant()
$skipExport = (Get-EnvOrDefault "TOKENIZER_1GPU_SKIP_EXPORT" "0") -eq "1"
$seedSpec = Get-EnvOrDefault "TOKENIZER_1GPU_SEEDS" "1337,2027"
$seeds = @($seedSpec.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
if ($seeds.Count -eq 0) { throw "No seeds provided in TOKENIZER_1GPU_SEEDS" }

$tokNameSpec = Get-EnvOrDefault "TOKENIZER_1GPU_TOKENIZER_NAMES" ""
$tokFilter = @($tokNameSpec.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })

$repoId = Get-EnvOrDefault "MATCHED_FINEWEB_REPO_ID" "willdepueoai/parameter-golf"
$remoteRoot = Get-EnvOrDefault "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX" "datasets"
$manifestPath = Join-Path $outputRoot "manifest.json"

if (-not $skipExport) {
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $args = @(
        "data/download_hf_docs_and_tokenize.py",
        "--repo-id", $repoId,
        "--remote-root", $remoteRoot,
        "--output-root", $outputRoot,
        "--tokenizer-config", $configPath
    )
    $trainerDocs = Get-EnvOrDefault "TOKENIZER_1GPU_TRAINER_DOCS" ""
    if (-not [string]::IsNullOrWhiteSpace($trainerDocs)) {
        $args += @("--tokenizer-train-docs", $trainerDocs)
    }

    Write-Host "`n=== Exporting tokenizer datasets ==="
    Write-Host "repo=$repoId remote_root=$remoteRoot output_root=$outputRoot"
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Tokenizer export failed with exit code $LASTEXITCODE"
    }
}

$tokenizerRuns = Resolve-TokenizerRuns -ManifestPath $manifestPath -OutputRoot $outputRoot -TokenizerNameFilter $tokFilter

$wallclock = if ($mode -eq "full") { "600" } else { "240" }
$valMaxTokens = if ($mode -eq "full") { "0" } else { "1048576" }
$warmupSteps = if ($mode -eq "full") { "200" } else { "50" }

$baseEnv = @{
    DEVICE = Get-EnvOrDefault "DEVICE" "cuda"
    USE_TORCH_COMPILE = Get-EnvOrDefault "USE_TORCH_COMPILE" "1"
    ITERATIONS = Get-EnvOrDefault "ITERATIONS" "20000"
    MAX_WALLCLOCK_SECONDS = Get-EnvOrDefault "MAX_WALLCLOCK_SECONDS" $wallclock
    WARMUP_STEPS = Get-EnvOrDefault "WARMUP_STEPS" $warmupSteps
    TRAIN_LOG_EVERY = Get-EnvOrDefault "TRAIN_LOG_EVERY" "200"
    TRAIN_BATCH_TOKENS = Get-EnvOrDefault "TRAIN_BATCH_TOKENS" "65536"
    VAL_BATCH_SIZE = Get-EnvOrDefault "VAL_BATCH_SIZE" "131072"
    VAL_LOSS_EVERY = Get-EnvOrDefault "VAL_LOSS_EVERY" "0"
    VAL_MAX_TOKENS = Get-EnvOrDefault "VAL_MAX_TOKENS" $valMaxTokens
    FINAL_ROUNDTRIP_EVAL = Get-EnvOrDefault "FINAL_ROUNDTRIP_EVAL" "1"
    SUBMISSION_SIZE_BUDGET_BYTES = Get-EnvOrDefault "SUBMISSION_SIZE_BUDGET_BYTES" "16000000"

    MODEL_DIM = Get-EnvOrDefault "MODEL_DIM" "512"
    NUM_LAYERS = Get-EnvOrDefault "NUM_LAYERS" "9"
    NUM_HEADS = Get-EnvOrDefault "NUM_HEADS" "8"
    NUM_KV_HEADS = Get-EnvOrDefault "NUM_KV_HEADS" "4"
    MLP_MULT = Get-EnvOrDefault "MLP_MULT" "2"
    TIE_EMBEDDINGS = Get-EnvOrDefault "TIE_EMBEDDINGS" "1"
    RECURRENT_CORE_LAYERS = Get-EnvOrDefault "RECURRENT_CORE_LAYERS" "0"
    RECURRENT_STEPS = Get-EnvOrDefault "RECURRENT_STEPS" "0"
    SHARE_FFN_ACROSS_BLOCKS = Get-EnvOrDefault "SHARE_FFN_ACROSS_BLOCKS" "0"
    USE_SWIGLU = Get-EnvOrDefault "USE_SWIGLU" "1"
    GRAD_CLIP_NORM = Get-EnvOrDefault "GRAD_CLIP_NORM" "1.0"

    EVAL_STRIDE_FRAC = Get-EnvOrDefault "EVAL_STRIDE_FRAC" "0.5"
    EVAL_SEQ_LEN = Get-EnvOrDefault "EVAL_SEQ_LEN" "0"
    EVAL_ROPE_SCALE = Get-EnvOrDefault "EVAL_ROPE_SCALE" "1.0"

    BIGRAM_RANK = Get-EnvOrDefault "BIGRAM_RANK" "32"
    BIGRAM_LR = Get-EnvOrDefault "BIGRAM_LR" "0.04"
    SWA_ENABLED = Get-EnvOrDefault "SWA_ENABLED" "1"
    SWA_COLLECT_EVERY = Get-EnvOrDefault "SWA_COLLECT_EVERY" "10"
    CURRICULUM_ENABLED = Get-EnvOrDefault "CURRICULUM_ENABLED" "0"
    CURRICULUM_MIN_SEQ_LEN = Get-EnvOrDefault "CURRICULUM_MIN_SEQ_LEN" "256"
    CURRICULUM_STEPS = Get-EnvOrDefault "CURRICULUM_STEPS" "5000"

    MUON_MOMENTUM = Get-EnvOrDefault "MUON_MOMENTUM" "0.98"
    MUON_BACKEND_STEPS = Get-EnvOrDefault "MUON_BACKEND_STEPS" "5"
    MUON_MOMENTUM_WARMUP_START = Get-EnvOrDefault "MUON_MOMENTUM_WARMUP_START" "0.85"
    MUON_MOMENTUM_WARMUP_STEPS = Get-EnvOrDefault "MUON_MOMENTUM_WARMUP_STEPS" "500"
    MATRIX_LR = Get-EnvOrDefault "MATRIX_LR" "0.04"
    SCALAR_LR = Get-EnvOrDefault "SCALAR_LR" "0.04"
    EMBED_LR = Get-EnvOrDefault "EMBED_LR" "0.6"
    TIED_EMBED_LR = Get-EnvOrDefault "TIED_EMBED_LR" "0.05"
    WARMDOWN_ITERS = Get-EnvOrDefault "WARMDOWN_ITERS" "3000"

    QUANT_SCHEME = Get-EnvOrDefault "QUANT_SCHEME" "mixed"
    MIXED_LOW_PRECISION_SCHEME = Get-EnvOrDefault "MIXED_LOW_PRECISION_SCHEME" "int4"
    QAT_SCHEME = Get-EnvOrDefault "QAT_SCHEME" "int4"
    QAT_START_STEP = Get-EnvOrDefault "QAT_START_STEP" "4200"
    COMPRESSOR = Get-EnvOrDefault "COMPRESSOR" "zstd"
    WEIGHT_ORDER = Get-EnvOrDefault "WEIGHT_ORDER" "none"
}

Write-Host "`n=== Tokenizer 1-GPU Matrix Config ==="
Write-Host "sweep_id=$sweepId mode=$mode seeds=$seedSpec tokenizers=$($tokenizerRuns.Count)"
Write-Host "output_root=$outputRoot"
Write-Host "config=$configPath"

$results = @()
foreach ($tok in $tokenizerRuns) {
    foreach ($seed in $seeds) {
        $runId = "tok1gpu_${timestamp}_$($tok.TokenizerName)_s$seed"
        Write-Host "`n=== Running tokenizer=$($tok.TokenizerName) seed=$seed (RUN_ID=$runId) ==="

        foreach ($k in $baseEnv.Keys) { Set-EnvVar $k $baseEnv[$k] }
        Set-EnvVar "RUN_ID" $runId
        Set-EnvVar "SEED" $seed
        Set-EnvVar "DATA_PATH" $tok.DataPath
        Set-EnvVar "TOKENIZER_PATH" $tok.TokenizerPath
        Set-EnvVar "VOCAB_SIZE" ([string]$tok.VocabSize)

        $start = Get-Date
        & torchrun --standalone --nnodes=1 --nproc_per_node=1 train_gpt.py
        $exitCode = $LASTEXITCODE
        $elapsedSec = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)

        $logPath = Join-Path "logs" ("{0}.txt" -f $runId)
        $parsed = Parse-RunLog -LogPath $logPath

        $gap = ""
        if ($parsed.ValBpbPreQuant -ne "" -and $parsed.ValBpbRoundtrip -ne "") {
            $gap = [math]::Round(([double]$parsed.ValBpbRoundtrip - [double]$parsed.ValBpbPreQuant), 6)
        }

        $results += [pscustomobject]@{
            TokenizerName = $tok.TokenizerName
            DatasetName = $tok.DatasetName
            Seed = $seed
            RunId = $runId
            ExitCode = $exitCode
            DurationSec = $elapsedSec
            VocabSize = $tok.VocabSize
            DataPath = $tok.DataPath
            TokenizerPath = $tok.TokenizerPath
            ValBpbPreQuant = $parsed.ValBpbPreQuant
            ValBpbRoundtrip = $parsed.ValBpbRoundtrip
            QuantizationGap = $gap
            ValLossRoundtrip = $parsed.ValLossRoundtrip
            TotalSubmissionBytes = $parsed.TotalSubmissionBytes
            BudgetBytes = $parsed.BudgetBytes
            BudgetHeadroomBytes = $parsed.BudgetHeadroomBytes
            UnderBudget = $parsed.UnderBudget
            ParseStatus = $parsed.ParseStatus
            LogPath = $parsed.LogPath
        }

        if ($exitCode -ne 0) {
            Write-Warning "Run failed: tokenizer=$($tok.TokenizerName) seed=$seed exit=$exitCode"
        }
    }
}

Write-Host "`n=== Tokenizer 1-GPU Per-Run Summary ==="
$results | Sort-Object ValBpbRoundtrip | Format-Table -AutoSize

$aggregate = @()
foreach ($g in ($results | Group-Object TokenizerName)) {
    $rows = @($g.Group)
    $okRows = @($rows | Where-Object { $_.ParseStatus -eq "ok" -and $_.ValBpbRoundtrip -ne "" })

    $medianBpb = $null
    $meanBpb = $null
    $medianGap = $null
    $meanGap = $null

    if ($okRows.Count -gt 0) {
        $bpbVals = @($okRows | ForEach-Object { [double]$_.ValBpbRoundtrip })
        $gapVals = @($okRows | Where-Object { $_.QuantizationGap -ne "" } | ForEach-Object { [double]$_.QuantizationGap })
        $medianBpb = Get-Median -Values $bpbVals
        $meanBpb = ($bpbVals | Measure-Object -Average).Average
        if ($gapVals.Count -gt 0) {
            $medianGap = Get-Median -Values $gapVals
            $meanGap = ($gapVals | Measure-Object -Average).Average
        }
    }

    $aggregate += [pscustomobject]@{
        TokenizerName = $g.Name
        Runs = $rows.Count
        OkRuns = $okRows.Count
        MedianValBpbRoundtrip = if ($null -ne $medianBpb) { [math]::Round($medianBpb, 8) } else { "" }
        MeanValBpbRoundtrip = if ($null -ne $meanBpb) { [math]::Round([double]$meanBpb, 8) } else { "" }
        MedianQuantizationGap = if ($null -ne $medianGap) { [math]::Round($medianGap, 8) } else { "" }
        MeanQuantizationGap = if ($null -ne $meanGap) { [math]::Round([double]$meanGap, 8) } else { "" }
    }
}

Write-Host "`n=== Tokenizer 1-GPU Aggregate Summary ==="
$aggregate | Sort-Object MedianValBpbRoundtrip | Format-Table -AutoSize

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$csvRuns = Join-Path "logs" ("tokenizer_1gpu_matrix_{0}.csv" -f $timestamp)
$csvAgg = Join-Path "logs" ("tokenizer_1gpu_matrix_{0}_aggregate.csv" -f $timestamp)
$results | Export-Csv -Path $csvRuns -NoTypeInformation -Encoding UTF8
$aggregate | Export-Csv -Path $csvAgg -NoTypeInformation -Encoding UTF8

Write-Host "Saved per-run CSV: $csvRuns"
Write-Host "Saved aggregate CSV: $csvAgg"
