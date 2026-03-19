# Run stronger Phase 3 matrix with multi-seed comparisons.
# Usage: .\phase3_matrix.ps1
# Optional: $env:PHASE3_SEEDS="1337,2027"

$ErrorActionPreference = "Stop"

function Set-EnvVar([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Parse-RunLog([string]$LogPath) {
    if (-not (Test-Path $LogPath)) {
        return [pscustomobject]@{
            LogPath = $LogPath
            ValBpbRoundtrip = ""
            ArtifactBytes = ""
            TotalSubmissionBytes = ""
            CompressorResolved = ""
            ParseStatus = "missing_log"
        }
    }

    $lines = Get-Content $LogPath
    $exportLine = ($lines | Select-String -Pattern '^export_config ' | Select-Object -Last 1).Line
    $artifactLine = ($lines | Select-String -Pattern '^Serialized model .+\+[a-z0-9]+: [0-9]+ bytes' | Select-Object -Last 1).Line
    $totalLine = ($lines | Select-String -Pattern '^Total submission size .+\+[a-z0-9]+: [0-9]+ bytes' | Select-Object -Last 1).Line
    $roundtripLine = ($lines | Select-String -Pattern '^final_.*_roundtrip_exact .*val_bpb:' | Select-Object -Last 1).Line

    $bpb = ""
    $artifact = ""
    $total = ""
    $comp = ""
    $parseStatus = "ok"

    if ($exportLine) {
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

    return [pscustomobject]@{
        LogPath = $LogPath
        ValBpbRoundtrip = $bpb
        ArtifactBytes = $artifact
        TotalSubmissionBytes = $total
        CompressorResolved = $comp
        ParseStatus = $parseStatus
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$baseEnv = @{
    DEVICE = "cuda"
    USE_TORCH_COMPILE = "0"
    ITERATIONS = "400"
    WARMUP_STEPS = "20"
    TRAIN_BATCH_TOKENS = "8192"
    VAL_LOSS_EVERY = "0"
    VAL_BATCH_SIZE = "131072"
    VAL_MAX_TOKENS = "1048576"
    FINAL_ROUNDTRIP_EVAL = "1"
    QUANT_SCHEME = "int8"
    COMPRESSOR = "auto"
    WEIGHT_ORDER = "none"
    MIXED_LOW_PRECISION_SCHEME = "int8"
}

$tests = @(
    @{ Name = "stacked_512_l9"; ModelDim = "512"; NumLayers = "9"; Core = "0"; Steps = "0"; Share = "0" },
    @{ Name = "recur_2x6_d704_share"; ModelDim = "704"; NumLayers = "9"; Core = "2"; Steps = "6"; Share = "1" },
    @{ Name = "recur_2x6_d704_noshare"; ModelDim = "704"; NumLayers = "9"; Core = "2"; Steps = "6"; Share = "0" },
    @{ Name = "recur_3x6_d640_share"; ModelDim = "640"; NumLayers = "9"; Core = "3"; Steps = "6"; Share = "1" }
)

$seedSpec = [Environment]::GetEnvironmentVariable("PHASE3_SEEDS", "Process")
if ([string]::IsNullOrWhiteSpace($seedSpec)) { $seedSpec = "1337,2027" }
$seeds = $seedSpec.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if ($seeds.Count -eq 0) {
    throw "No seeds provided. Set PHASE3_SEEDS, e.g. 1337,2027"
}

$results = @()

foreach ($test in $tests) {
    foreach ($seed in $seeds) {
        $runId = "phase3_${timestamp}_$($test.Name)_s$seed"
        Write-Host "`n=== Running $($test.Name) seed=$seed (RUN_ID=$runId) ==="

        foreach ($k in $baseEnv.Keys) { Set-EnvVar $k $baseEnv[$k] }
        Set-EnvVar "RUN_ID" $runId
        Set-EnvVar "SEED" $seed
        Set-EnvVar "MODEL_DIM" $test.ModelDim
        Set-EnvVar "NUM_LAYERS" $test.NumLayers
        Set-EnvVar "RECURRENT_CORE_LAYERS" $test.Core
        Set-EnvVar "RECURRENT_STEPS" $test.Steps
        Set-EnvVar "SHARE_FFN_ACROSS_BLOCKS" $test.Share

        $start = Get-Date
        & python train_gpt.py
        $exitCode = $LASTEXITCODE
        $elapsedSec = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)

        $logPath = Join-Path "logs" ("{0}.txt" -f $runId)
        $parsed = Parse-RunLog -LogPath $logPath

        $results += [pscustomobject]@{
            Test = $test.Name
            Seed = $seed
            ModelDim = $test.ModelDim
            NumLayers = $test.NumLayers
            CoreLayers = $test.Core
            RecurrentSteps = $test.Steps
            ShareFFN = $test.Share
            ExitCode = $exitCode
            DurationSec = $elapsedSec
            ValBpbRoundtrip = $parsed.ValBpbRoundtrip
            ArtifactBytes = $parsed.ArtifactBytes
            TotalSubmissionBytes = $parsed.TotalSubmissionBytes
            CompressorResolved = $parsed.CompressorResolved
            ParseStatus = $parsed.ParseStatus
            LogPath = $parsed.LogPath
        }

        if ($exitCode -ne 0) {
            Write-Warning "Run $($test.Name) seed=$seed failed with exit code $exitCode"
        }
    }
}

Write-Host "`n=== Phase 3 Per-Run Summary ==="
$results | Format-Table -AutoSize

$aggregate = @()
foreach ($g in ($results | Group-Object Test)) {
    $rows = $g.Group
    $okRows = $rows | Where-Object { $_.ParseStatus -eq "ok" -and $_.ValBpbRoundtrip -ne "" }
    $avgBpb = ""
    $avgTotal = ""
    if ($okRows.Count -gt 0) {
        $avgBpb = [math]::Round((($okRows | ForEach-Object { [double]$_.ValBpbRoundtrip } | Measure-Object -Average).Average), 8)
        $avgTotal = [math]::Round((($okRows | ForEach-Object { [double]$_.TotalSubmissionBytes } | Measure-Object -Average).Average), 0)
    }

    $aggregate += [pscustomobject]@{
        Test = $g.Name
        Runs = $rows.Count
        OkRuns = $okRows.Count
        AvgValBpb = $avgBpb
        AvgTotalSubmissionBytes = $avgTotal
    }
}

Write-Host "`n=== Phase 3 Aggregate Summary ==="
$aggregate | Sort-Object AvgValBpb | Format-Table -AutoSize

$csvPath = Join-Path "logs" ("phase3_matrix_${timestamp}.csv")
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "Saved per-run CSV: $csvPath"
