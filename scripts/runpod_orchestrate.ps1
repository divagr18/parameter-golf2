param(
  [Parameter(Mandatory = $true)]
  [string]$PodId,

  [string]$RemoteWorkDir = "/workspace/parameter-golf",

  # If your repo isn't up to date on the pod, set this true to run git pull first.
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

# -----------------------------------------------------------------------------
# IMPORTANT:
# You can override these templates via environment variables if your runpodctl
# command shape differs.
# Required placeholders:
#   {POD_ID}
#   {CMD}     (for EXEC only)
# -----------------------------------------------------------------------------

$startTemplate = if ($env:RUNPOD_START_CMD_TEMPLATE) { $env:RUNPOD_START_CMD_TEMPLATE } else { "runpodctl pod start --id {POD_ID}" }
$execTemplate  = if ($env:RUNPOD_EXEC_CMD_TEMPLATE)  { $env:RUNPOD_EXEC_CMD_TEMPLATE }  else { "runpodctl pod exec --id {POD_ID} -- /bin/bash -lc \"{CMD}\"" }
$stopTemplate  = if ($env:RUNPOD_STOP_CMD_TEMPLATE)  { $env:RUNPOD_STOP_CMD_TEMPLATE }  else { "runpodctl pod stop --id {POD_ID}" }

$vars = @{ POD_ID = $PodId }
$startCmd = Expand-Template $startTemplate $vars
$stopCmd  = Expand-Template $stopTemplate $vars

# Build remote command payload.
$remoteSteps = @(
  "set -euo pipefail",
  "cd $RemoteWorkDir",
  "chmod +x ./scripts/runpod_experiment_batch.sh"
)
if ($DoGitPull) {
  $remoteSteps += "git pull --ff-only"
}
$remoteSteps += "./scripts/runpod_experiment_batch.sh"
$remoteCombined = ($remoteSteps -join "; ")

# Escape embedded double quotes for the template execution context.
$escapedRemote = $remoteCombined.Replace('"', '\"')
$execCmd = Expand-Template $execTemplate (@{ POD_ID = $PodId; CMD = $escapedRemote })

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$localLog = "logs/runpod_batch_$stamp.log"
New-Item -ItemType Directory -Path "logs" -Force | Out-Null

Write-Host "=== RunPod Orchestrator ==="
Write-Host "PodId:      $PodId"
Write-Host "Start cmd:  $startCmd"
Write-Host "Exec cmd:   $execCmd"
Write-Host "Stop cmd:   $stopCmd"
Write-Host "Local log:  $localLog"

$stopAttempted = $false
try {
  Write-Host "`n[1/3] Starting pod..."
  Invoke-Expression $startCmd | Tee-Object -FilePath $localLog -Append

  Write-Host "`n[2/3] Running remote experiment batch..."
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
