param([string]$PythonPath = '', [string]$Milestone = 'current')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $projectRoot '.tools\godot\Godot_v4.7.2-stable_win64_console.exe'
if (-not (Test-Path -LiteralPath $engine)) { throw 'Godot executable missing.' }
if (-not $PythonPath) {
    $found = Get-Command python.exe -ErrorAction SilentlyContinue
    $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if ($found -and $found.Source -notlike '*WindowsApps*') { $PythonPath = $found.Source }
    elseif (Test-Path -LiteralPath $bundled) { $PythonPath = $bundled }
    else { throw 'Python missing; provide -PythonPath.' }
}
if ($Milestone -ne 'current') { throw 'Only the current clean workspace is verified.' }
$logRoot = Join-Path $projectRoot 'output\current\verification'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$results = [System.Collections.Generic.List[object]]::new()
function Run-Check([string]$Name, [string]$Executable, [string[]]$Arguments) {
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $logPath = Join-Path $logRoot ($Name + '.log')
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $lines = & $Executable @Arguments 2>&1; $nativeExit = $LASTEXITCODE }
    finally { $ErrorActionPreference = $oldPreference }
    $lines | Set-Content -LiteralPath $logPath -Encoding UTF8
    $text = $lines | Out-String
    $engineError = $text -match '(?m)(SCRIPT ERROR:|ERROR:|FAIL failures=|FAILED \()'
    $passed = $nativeExit -eq 0 -and -not $engineError
    $results.Add([pscustomobject]@{ name=$Name; passed=$passed; exit_code=$nativeExit; seconds=[math]::Round($timer.Elapsed.TotalSeconds,3); log=$logPath })
    Write-Output (('{0}: {1} (exit {2})' -f $Name, $(if ($passed) {'PASS'} else {'FAIL'}), $nativeExit))
    if (-not $passed) { Write-Output $text }
}
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'output/runtime_tests') | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
# Keep editor settings inside the project instead of the user profile.
$portableMarker = Join-Path (Split-Path -Parent $engine) '_sc_'
if (-not (Test-Path -LiteralPath $portableMarker)) { New-Item -ItemType File -Path $portableMarker | Out-Null }
Push-Location $projectRoot
try {
    Run-Check 'import' $engine @('--headless','--path','.','--editor','--import','--log-file',(Join-Path $logRoot 'engine-import.log'))
    Run-Check 'python' $PythonPath @('-m','unittest','discover','-s','tests','-v')
    foreach ($suite in @('test_world','test_controller','test_collision','test_collision_physics','test_survival_ui','test_direction_animation','test_rl_environment','test_m8b_game_entry','test_m8x_game_entry','test_model_session','test_survival_v2','test_m8s3_online_entry')) {
        $checkArgs = @('--headless','--path','.','--script', ('godot_tests/' + $suite + '.gd'),'--log-file',(Join-Path $logRoot ($suite + '-engine.log')))
        if ($suite -eq 'test_m8x_game_entry') { $checkArgs += @('--','--difficulty=open3') }
        if ($suite -eq 'test_m8b_game_entry') { $checkArgs += @('--','--difficulty=open2') }
        if ($suite -in @('test_controller','test_survival_ui','test_model_session')) { $checkArgs += @('--','--difficulty=open1') }
        Run-Check $suite $engine $checkArgs
    }
} finally { Pop-Location }
$report = [pscustomobject]@{ created_utc=(Get-Date).ToUniversalTime().ToString('o'); checks=@($results.ToArray()) }
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $logRoot 'checks.json') -Encoding UTF8
if (@($results | Where-Object { -not $_.passed }).Count -gt 0) { exit 1 }
