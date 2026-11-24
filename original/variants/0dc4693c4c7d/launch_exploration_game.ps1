param([int]$Seed = 2304, [switch]$Model, [switch]$Editor)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $projectRoot '.tools\godot\Godot_v4.7.2-stable_win64.exe'
if (-not (Test-Path -LiteralPath $engine)) {
    $found = Get-Command godot -ErrorAction SilentlyContinue
    if (-not $found) { throw 'Godot 4 is missing.' }
    $engine = $found.Source
}
if ($Editor) {
    Start-Process -FilePath $engine -ArgumentList @('--path', ('"' + $projectRoot + '"'), '--editor') -WorkingDirectory $projectRoot
    return
}
if ($Model) {
    throw 'The current open4/world6 rules have no verified model release. An old open3/world5 connection cannot control this game. Manual play remains available.'
}
$gameArgs = @('--difficulty=open4', "--seed=$Seed", '--require-model-context')
$engineArgs = @('--path', ('"' + $projectRoot + '"'), '--') + $gameArgs
# Opens the interactive current-rule game; never starts a model or training service.
Start-Process -FilePath $engine -ArgumentList $engineArgs -WorkingDirectory $projectRoot
