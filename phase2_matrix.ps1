# Run 4 Phase 2 smoke variants sequentially and summarize results.
# Usage: .\phase2_matrix.ps1

$ErrorActionPreference = "Stop"

function Set-EnvVar([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Parse-RunLog([string]$LogPath) {
    if (-not (Test-Path $LogPath)) {
        return [pscustomobject]@{
            LogPath = $LogPath
            QuantScheme = ""
            CompressorResolved = ""
            ValBpbRoundtrip = ""
            ArtifactBytes = ""
            TotalSubmissionBytes = ""
            ParseStatus = "missing_log"
        }
    }

    $lines = Get-Content $LogPath

    $exportLine = ($lines | Select-String -Pattern '^export_config ' | Select-Object -Last 1).Line
    $artifactLine = ($lines | Select-String -Pattern '^Serialized model .+\+[a-z0-9]+: [0-9]+ bytes' | Select-Object -Last 1).Line
    $totalLine = ($lines | Select-String -Pattern '^Total submission size .+\+[a-z0-9]+: [0-9]+ bytes' | Select-Object -Last 1).Line
    $roundtripLine = ($lines | Select-String -Pattern '^final_.*_roundtrip_exact .*val_bpb:' | Select-Object -Last 1).Line

    $quant = ""
    $comp = ""
    $bpb = ""
    $artifact = ""
    $total = ""
    $parseStatus = "ok"

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

    return [pscustomobject]@{
        LogPath = $LogPath
        QuantScheme = $quant
        CompressorResolved = $comp
        ValBpbRoundtrip = $bpb
        ArtifactBytes = $artifact
        TotalSubmissionBytes = $total
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
    WEIGHT_ORDER = "none"
    MIXED_LOW_PRECISION_SCHEME = "int8"
}

$tests = @(
    @{ Name = "int8_zlib"; Quant = "int8"; Compressor = "zlib" },
    @{ Name = "int4_zlib"; Quant = "int4"; Compressor = "zlib" },
    @{ Name = "mixed_zlib"; Quant = "mixed"; Compressor = "zlib" },
    @{ Name = "int8_auto"; Quant = "int8"; Compressor = "auto" }
)

$results = @()

foreach ($test in $tests) {
    $runId = "phase2_${timestamp}_$($test.Name)"
    Write-Host "`n=== Running $($test.Name) (RUN_ID=$runId) ==="

    foreach ($k in $baseEnv.Keys) { Set-EnvVar $k $baseEnv[$k] }
    Set-EnvVar "RUN_ID" $runId
    Set-EnvVar "QUANT_SCHEME" $test.Quant
    Set-EnvVar "COMPRESSOR" $test.Compressor

    $start = Get-Date
    & python train_gpt.py
    $exitCode = $LASTEXITCODE
    $elapsedSec = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)

    $logPath = Join-Path "logs" ("{0}.txt" -f $runId)
    $parsed = Parse-RunLog -LogPath $logPath

    $results += [pscustomobject]@{
        Test = $test.Name
        ExitCode = $exitCode
        DurationSec = $elapsedSec
        QuantScheme = $parsed.QuantScheme
        CompressorResolved = $parsed.CompressorResolved
        ValBpbRoundtrip = $parsed.ValBpbRoundtrip
        ArtifactBytes = $parsed.ArtifactBytes
        TotalSubmissionBytes = $parsed.TotalSubmissionBytes
        ParseStatus = $parsed.ParseStatus
        LogPath = $parsed.LogPath
    }

    if ($exitCode -ne 0) {
        Write-Warning "Run $($test.Name) failed with exit code $exitCode"
    }
}

Write-Host "`n=== Phase 2 Smoke Matrix Summary ==="
$results | Format-Table -AutoSize

$csvPath = Join-Path "logs" ("phase2_matrix_${timestamp}.csv")
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "Saved summary CSV: $csvPath"
