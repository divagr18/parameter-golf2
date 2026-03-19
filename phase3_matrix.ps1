# Run 4 Phase 3 architecture smoke variants and summarize results.
# Usage: .\phase3_matrix.ps1

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
    $archLine = ($lines | Select-String -Pattern '^architecture:' | Select-Object -Last 1).Line

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
        Architecture = $archLine
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
    ITERATIONS = "150"
    WARMUP_STEPS = "10"
    TRAIN_BATCH_TOKENS = "8192"
    VAL_LOSS_EVERY = "0"
    VAL_BATCH_SIZE = "131072"
    VAL_MAX_TOKENS = "262144"
    FINAL_ROUNDTRIP_EVAL = "1"
    QUANT_SCHEME = "int8"
    COMPRESSOR = "auto"
    WEIGHT_ORDER = "none"
    MIXED_LOW_PRECISION_SCHEME = "int8"
}

$tests = @(
    @{ Name = "stacked_baseline"; Core = "0"; Steps = "0"; Share = "0" },
    @{ Name = "recur_2x4_share"; Core = "2"; Steps = "4"; Share = "1" },
    @{ Name = "recur_2x6_share"; Core = "2"; Steps = "6"; Share = "1" },
    @{ Name = "recur_2x6_noshare"; Core = "2"; Steps = "6"; Share = "0" }
)

$results = @()

foreach ($test in $tests) {
    $runId = "phase3_${timestamp}_$($test.Name)"
    Write-Host "`n=== Running $($test.Name) (RUN_ID=$runId) ==="

    foreach ($k in $baseEnv.Keys) { Set-EnvVar $k $baseEnv[$k] }
    Set-EnvVar "RUN_ID" $runId
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
        Write-Warning "Run $($test.Name) failed with exit code $exitCode"
    }
}

Write-Host "`n=== Phase 3 Smoke Matrix Summary ==="
$results | Format-Table -AutoSize

$csvPath = Join-Path "logs" ("phase3_matrix_${timestamp}.csv")
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "Saved summary CSV: $csvPath"
