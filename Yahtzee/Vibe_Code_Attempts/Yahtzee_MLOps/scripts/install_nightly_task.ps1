param(
    [string]$TaskName = "Yahtzee MLOps Nightly Training",
    [string]$Time = "02:00",
    [int]$MinNewEvents = 100,
    [int]$Episodes = 2000,
    [int]$EvalGames = 100
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PipelineScript = Join-Path $ProjectRoot "scripts\run_local_pipeline.ps1"

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PipelineScript`" -MinNewEvents $MinNewEvents -Episodes $Episodes -EvalGames $EvalGames" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Runs the Yahtzee MLOps pipeline once per day overnight." `
    -Force

Write-Host "Installed scheduled task '$TaskName' for $Time daily."
Write-Host "Logs will be written to: $(Join-Path $ProjectRoot 'logs')"
