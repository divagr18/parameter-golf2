# Run a capacity-focused Phase 3 sweep that scales recurrent model size.
# Usage: .\phase3_capacity_matrix.ps1
# Optional:
#   $env:PHASE3_CAPACITY_SEEDS="1337,2027"
#   $env:ITERATIONS="600"
#   $env:VAL_MAX_TOKENS="1048576"

$ErrorActionPreference = "Stop"

function Set-EnvVar([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Get-EnvOrDefault([string]$Name, [string]$DefaultValue) {
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        return $DefaultValue
    }
    return $current
}

function Parse-RunLog([string]$LogPath) {
    if (-not (Test-Path $LogPath)) {
        return [pscustomobject]@{
            LogPath = $LogPath
            ModelParams = ""
            QuantScheme = ""
            CompressorResolved = ""
            ValBpbRoundtrip = ""
            ArtifactBytes = ""
            TotalSubmissionBytes = ""
            BudgetBytes = ""
            BudgetHeadroomBytes = ""
            UnderBudget = ""
            ParseStatus = "missing_log"
        }
    }

    $lines = Get-Content $LogPath
    $paramsLine = ($lines | Select-String -Pattern '^model_params:[0-9]+' | Select-Object -Last 1).Line
    $exportLine = ($lines | Select-String -Pattern '^export_config ' | Select-Object -Last 1).Line
    $artifactLine = ($lines | Select-String -Pattern '^Serialized model .+\+[a-z0-9]+: [0-9]+ bytes' | Select-Object -Last 1).Line
    $totalLine = ($lines | Select-String -Pattern '^Total submission size .+\+[a-z0-9]+: [0-9]+ bytes' | Select-Object -Last 1).Line
    $budgetLine = ($lines | Select-String -Pattern '^submission_budget .+ total:[0-9]+ budget:[0-9]+ (headroom_bytes|over_bytes):[0-9]+' | Select-Object -Last 1).Line
    $roundtripLine = ($lines | Select-String -Pattern '^final_.*_roundtrip_exact .*val_bpb:' | Select-Object -Last 1).Line

    $modelParams = ""
    $quant = ""
    $comp = ""
    $bpb = ""
    $artifact = ""
    $total = ""
    $budget = ""
    $headroom = ""
    $underBudget = ""
    $parseStatus = "ok"

    if ($paramsLine) {
        $m = [regex]::Match($paramsLine, '^model_params:(?<p>[0-9]+)')
        if ($m.Success) { $modelParams = $m.Groups['p'].Value }
    }

    if ($exportLine) {
        $m = [regex]::Match($exportLine, 'quant_scheme:(?<q>[^ ]+)')
        if ($m.Success) { $quant = $m.Groups['q'].Value }
        $m = [regex]::Match($exportLine, 'compressor:(?<c>[^ ]+)')
        if ($m.Success) { $comp = $m.Groups['c'].Value }
    }

    if ($roundtripLine) {
        $m = [regex]::Match($roundtripLine, 'val_bpb:(?<bpb>[0-9.]+)')
        if ($m.Success) { $bpb = $m.Groups['bpb'].Value }
    } else {
        $parseStatus = "missing_roundtrip"
    }

    if ($artifactLine) {
        $m = [regex]::Match($artifactLine, ': (?<bytes>[0-9]+) bytes')
        if ($m.Success) { $artifact = $m.Groups['bytes'].Value }
    }

    if ($totalLine) {
        $m = [regex]::Match($totalLine, ': (?<bytes>[0-9]+) bytes')
        if ($m.Success) { $total = $m.Groups['bytes'].Value }
    }

    if ($budgetLine) {
        $m = [regex]::Match($budgetLine, 'budget:(?<budget>[0-9]+)')
        if ($m.Success) { $budget = $m.Groups['budget'].Value }
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
        ModelParams = $modelParams
        QuantScheme = $quant
        CompressorResolved = $comp
        ValBpbRoundtrip = $bpb
        ArtifactBytes = $artifact
        TotalSubmissionBytes = $total
        BudgetBytes = $budget
        BudgetHeadroomBytes = $headroom
        UnderBudget = $underBudget
        ParseStatus = $parseStatus
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$baseEnv = @{
    DEVICE = Get-EnvOrDefault "DEVICE" "cuda"
    USE_TORCH_COMPILE = Get-EnvOrDefault "USE_TORCH_COMPILE" "0"
    ITERATIONS = Get-EnvOrDefault "ITERATIONS" "500"
    WARMUP_STEPS = Get-EnvOrDefault "WARMUP_STEPS" "20"
    TRAIN_BATCH_TOKENS = Get-EnvOrDefault "TRAIN_BATCH_TOKENS" "8192"
    VAL_LOSS_EVERY = Get-EnvOrDefault "VAL_LOSS_EVERY" "0"
    VAL_BATCH_SIZE = Get-EnvOrDefault "VAL_BATCH_SIZE" "131072"
    VAL_MAX_TOKENS = Get-EnvOrDefault "VAL_MAX_TOKENS" "1048576"
    FINAL_ROUNDTRIP_EVAL = Get-EnvOrDefault "FINAL_ROUNDTRIP_EVAL" "1"
    SUBMISSION_SIZE_BUDGET_BYTES = Get-EnvOrDefault "SUBMISSION_SIZE_BUDGET_BYTES" "16777216"
    QUANT_SCHEME = Get-EnvOrDefault "QUANT_SCHEME" "int8"
    COMPRESSOR = Get-EnvOrDefault "COMPRESSOR" "auto"
    WEIGHT_ORDER = Get-EnvOrDefault "WEIGHT_ORDER" "none"
    MIXED_LOW_PRECISION_SCHEME = Get-EnvOrDefault "MIXED_LOW_PRECISION_SCHEME" "int8"
    NUM_LAYERS = Get-EnvOrDefault "NUM_LAYERS" "9"
}

$tests = @(
    @{ Name = "recur_3x6_d640_share"; ModelDim = "640"; Core = "3"; Steps = "6"; Share = "1" },
    @{ Name = "recur_3x6_d704_share"; ModelDim = "704"; Core = "3"; Steps = "6"; Share = "1" },
    @{ Name = "recur_3x6_d768_share"; ModelDim = "768"; Core = "3"; Steps = "6"; Share = "1" },
    @{ Name = "recur_3x6_d832_share"; ModelDim = "832"; Core = "3"; Steps = "6"; Share = "1" },
    @{ Name = "recur_3x6_d896_share"; ModelDim = "896"; Core = "3"; Steps = "6"; Share = "1" },
    @{ Name = "recur_3x8_d768_share"; ModelDim = "768"; Core = "3"; Steps = "8"; Share = "1" },
    @{ Name = "recur_4x6_d768_share"; ModelDim = "768"; Core = "4"; Steps = "6"; Share = "1" }
)

$seedSpec = Get-EnvOrDefault "PHASE3_CAPACITY_SEEDS" "1337"
$seeds = $seedSpec.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if ($seeds.Count -eq 0) {
    throw "No seeds provided. Set PHASE3_CAPACITY_SEEDS, e.g. 1337,2027"
}

$results = @()

foreach ($test in $tests) {
    foreach ($seed in $seeds) {
        $runId = "phase3cap_${timestamp}_$($test.Name)_s$seed"
        Write-Host "`n=== Running $($test.Name) seed=$seed (RUN_ID=$runId) ==="

        foreach ($k in $baseEnv.Keys) { Set-EnvVar $k $baseEnv[$k] }
        Set-EnvVar "RUN_ID" $runId
        Set-EnvVar "SEED" $seed
        Set-EnvVar "MODEL_DIM" $test.ModelDim
        Set-EnvVar "RECURRENT_CORE_LAYERS" $test.Core
        Set-EnvVar "RECURRENT_STEPS" $test.Steps
        Set-EnvVar "SHARE_FFN_ACROSS_BLOCKS" $test.Share

        $start = Get-Date
        & python train_gpt.py
        $exitCode = $LASTEXITCODE
        $elapsedSec = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)

        $logPath = Join-Path "logs" ("{0}.txt" -f $runId)
        $parsed = Parse-RunLog -LogPath $logPath

        $totalBytesNum = 0
        if ($parsed.TotalSubmissionBytes -ne "") {
            $totalBytesNum = [double]$parsed.TotalSubmissionBytes
        }
        $totalMiB = [math]::Round(($totalBytesNum / 1048576.0), 3)
        $headroomMiB = ""
        if ($parsed.BudgetHeadroomBytes -ne "") {
            $headroomMiB = [math]::Round(([double]$parsed.BudgetHeadroomBytes / 1048576.0), 3)
        }

        $results += [pscustomobject]@{
            Test = $test.Name
            Seed = $seed
            ModelDim = $test.ModelDim
            NumLayers = $baseEnv.NUM_LAYERS
            CoreLayers = $test.Core
            RecurrentSteps = $test.Steps
            ShareFFN = $test.Share
            ModelParams = $parsed.ModelParams
            ExitCode = $exitCode
            DurationSec = $elapsedSec
            ValBpbRoundtrip = $parsed.ValBpbRoundtrip
            TotalSubmissionMiB = $totalMiB
            TotalSubmissionBytes = $parsed.TotalSubmissionBytes
            BudgetMiB = if ($parsed.BudgetBytes -ne "") { [math]::Round(([double]$parsed.BudgetBytes / 1048576.0), 3) } else { "" }
            HeadroomMiB = $headroomMiB
            UnderBudget = $parsed.UnderBudget
            QuantScheme = $parsed.QuantScheme
            CompressorResolved = $parsed.CompressorResolved
            ParseStatus = $parsed.ParseStatus
            LogPath = $parsed.LogPath
        }

        if ($exitCode -ne 0) {
            Write-Warning "Run $($test.Name) seed=$seed failed with exit code $exitCode"
        }
    }
}

Write-Host "`n=== Phase 3 Capacity Per-Run Summary ==="
$results | Sort-Object ValBpbRoundtrip | Format-Table -AutoSize

$aggregate = @()
foreach ($g in ($results | Group-Object Test)) {
    $rows = $g.Group
    $okRows = $rows | Where-Object { $_.ParseStatus -eq "ok" -and $_.ValBpbRoundtrip -ne "" }
    $avgBpb = ""
    $avgTotalMiB = ""
    if ($okRows.Count -gt 0) {
        $avgBpb = [math]::Round((($okRows | ForEach-Object { [double]$_.ValBpbRoundtrip } | Measure-Object -Average).Average), 8)
        $avgTotalMiB = [math]::Round((($okRows | ForEach-Object { [double]$_.TotalSubmissionMiB } | Measure-Object -Average).Average), 3)
    }

    $aggregate += [pscustomobject]@{
        Test = $g.Name
        Runs = $rows.Count
        OkRuns = $okRows.Count
        AvgValBpb = $avgBpb
        AvgSubmissionMiB = $avgTotalMiB
    }
}

Write-Host "`n=== Phase 3 Capacity Aggregate Summary ==="
$aggregate | Sort-Object AvgValBpb | Format-Table -AutoSize

$csvRuns = Join-Path "logs" ("phase3_capacity_matrix_${timestamp}.csv")
$csvAgg = Join-Path "logs" ("phase3_capacity_matrix_${timestamp}_aggregate.csv")
$results | Export-Csv -Path $csvRuns -NoTypeInformation -Encoding UTF8
$aggregate | Export-Csv -Path $csvAgg -NoTypeInformation -Encoding UTF8
Write-Host "Saved per-run CSV: $csvRuns"
Write-Host "Saved aggregate CSV: $csvAgg"
