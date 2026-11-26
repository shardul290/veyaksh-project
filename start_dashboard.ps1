param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$projectPython = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$dashboardFile = Join-Path $projectDirectory 'm6_dashboard\index.html'
$dashboardUrl = "http://127.0.0.1:$Port/?dashboard=live-yolo26"

if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Create the environment first: python -m venv .venv; then .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
if (-not (Get-Content -LiteralPath $dashboardFile -Raw).Contains('id="nav-history"')) {
    throw "This folder contains an older dashboard. Use the updated project folder: $projectDirectory"
}

Write-Host "Project: $projectDirectory"
Write-Host 'Dashboard: live YOLO26 traffic operations'
Write-Host "Open: $dashboardUrl"

$existingResponse = $null
try {
    $existingResponse = Invoke-WebRequest -Uri $dashboardUrl -UseBasicParsing -TimeoutSec 3
} catch {
    # No HTTP response: Uvicorn will report any port or startup error below.
}
if ($existingResponse) {
    if ($existingResponse.Content.Contains('id="nav-history"') -and
        $existingResponse.Content.Contains('/dashboard-assets/dashboard.js')) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        } catch {
            throw "The dashboard server is unavailable. Restart its terminal, then run this launcher again."
        }
        if ($health.playback_analysis -ne 'v1' -or $health.live_pipeline -ne 'v1' -or $health.live_sync -ne 'buffered-v2') {
            throw "An older backend is still running on port $Port. Stop its terminal with Ctrl+C, then run this launcher again to load the live YOLO26 pipeline."
        }
        Write-Host 'The standard dashboard is already running. Open the URL above.'
        Write-Host 'If an old browser tab still shows Sentinel, press Ctrl+Shift+R.'
        return
    }
    throw "Port $Port is serving another dashboard. Stop its terminal with Ctrl+C, then run this launcher again."
}

Push-Location -LiteralPath $projectDirectory
try {
    # --app-dir pins imports to this checkout even when launched elsewhere.
    & $projectPython -m uvicorn m6_backend.main:app --app-dir $projectDirectory --host 127.0.0.1 --port $Port --no-access-log
    if ($LASTEXITCODE -ne 0) {
        throw "Dashboard startup failed (exit code $LASTEXITCODE). Review the error above."
    }
} finally {
    Pop-Location
}
