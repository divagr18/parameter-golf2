param(
  [Parameter(Mandatory = $true)]
  [string]$PodId,

  [string]$RemoteWorkDir = "/workspace/parameter-golf",

  [bool]$DoGitPull = $false
)

$ErrorActionPreference = "Stop"

function Expand-Template([string]$Template, [hashtable]$Vars) {
  $out = $Template
  foreach ($k in $Vars.Keys) {
    $out = $out.Replace("{$k}", [string]$Vars[$k])
  }
  return $out
}

$startTemplate = if ($env:RUNPOD_START_CMD_TEMPLATE) { $env:RUNPOD_START_CMD_TEMPLATE } else { "runpodctl pod start --id {POD_ID}" }
$execTemplate  = if ($env:RUNPOD_EXEC_CMD_TEMPLATE)  { $env:RUNPOD_EXEC_CMD_TEMPLATE }  else { 'runpodctl pod exec --id {POD_ID} -- /bin/bash -lc "{CMD}"' }
$stopTemplate  = if ($env:RUNPOD_STOP_CMD_TEMPLATE)  { $env:RUNPOD_STOP_CMD_TEMPLATE }  else { "runpodctl pod stop --id {POD_ID}" }

$startCmd = Expand-Template $startTemplate @{ POD_ID = $PodId }
$stopCmd  = Expand-Template $stopTemplate  @{ POD_ID = $PodId }

$remoteSteps = @(
  "set -euo pipefail",
  "cd $RemoteWorkDir",
  "chmod +x ./scripts/runpod_distill_sweeps.sh"
)
if ($DoGitPull) {
  $remoteSteps += "git pull --ff-only"
}
$remoteSteps += "./scripts/runpod_distill_sweeps.sh"
$remoteCombined = ($remoteSteps -join "; ")
$escapedRemote = $remoteCombined.Replace('"', '\"')
$execCmd = Expand-Template $execTemplate @{ POD_ID = $PodId; CMD = $escapedRemote }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$localLog = "logs/runpod_distill_sweeps_$stamp.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null

Write-Host "=== RunPod Distill Orchestrator ==="
Write-Host "PodId:      $PodId"
Write-Host "Start cmd:  $startCmd"
Write-Host "Exec cmd:   $execCmd"
Write-Host "Stop cmd:   $stopCmd"
Write-Host "Local log:  $localLog"

$stopAttempted = $false
try {
  Write-Host "`n[1/3] Starting pod..."
  Invoke-Expression $startCmd | Tee-Object -FilePath $localLog -Append

  Write-Host "`n[2/3] Running distill sweeps..."
  Invoke-Expression $execCmd | Tee-Object -FilePath $localLog -Append

  Write-Host "`n[3/3] Stopping pod..."
  Invoke-Expression $stopCmd | Tee-Object -FilePath $localLog -Append
  $stopAttempted = $true

  Write-Host "`nDone. See log: $localLog"
}
catch {
  Write-Error $_
  if (-not $stopAttempted) {
    Write-Host "Attempting pod stop after failure..."
    try {
      Invoke-Expression $stopCmd | Tee-Object -FilePath $localLog -Append
    }
    catch {
      Write-Warning "Automatic stop failed. Run manually: $stopCmd"
    }
  }
  throw
}
