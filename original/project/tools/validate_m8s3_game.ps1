param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ActorSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$RegistrationSha256,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$Seed = 2304
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $projectRoot '.tools\godot\Godot_v4.7.2-stable_win64_console.exe'
$destination = [IO.Path]::GetFullPath($OutputDirectory)
$allowed = [IO.Path]::GetFullPath((Join-Path $projectRoot 'output')) + [IO.Path]::DirectorySeparatorChar
if (-not $destination.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'Acceptance output must be within project output/.' }
if (Test-Path -LiteralPath $destination) { throw 'Use a fresh acceptance directory; old evidence is retained.' }
$health = Invoke-RestMethod -Uri 'http://127.0.0.1:18081/health' -TimeoutSec 30
if ($health.actor_sha256 -ne $ActorSha256 -or $health.registration_sha256 -ne $RegistrationSha256 -or
    $health.profile -ne 'open4' -or $health.world_version -ne 6 -or
    $health.validation_authorized -ne $true -or $health.numerical_gate_passed -ne $true -or $health.decision_ready -ne $true) {
    throw 'The actual endpoint has not authorized this exact offline-accepted actor.'
}
New-Item -ItemType Directory -Path $destination | Out-Null
$health | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $destination 'health.json') -Encoding UTF8
$arguments = @('--path', ('"{0}"' -f $projectRoot), '--script', 'godot_tests/capture_m8s3_online.gd',
    '--rendering-method', 'gl_compatibility', '--log-file', ('"{0}"' -f (Join-Path $destination 'engine.log')),
    '--', '--difficulty=open4', ('--seed={0}' -f $Seed), '--validate-model', '--require-model-context',
    ('--expected-actor={0}' -f $ActorSha256), ('--expected-registration={0}' -f $RegistrationSha256),
    '--broker-url=http://127.0.0.1:18081', ('--acceptance-output="{0}"' -f $destination))
$process = Start-Process -FilePath $engine -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $destination 'stdout.log') -RedirectStandardError (Join-Path $destination 'stderr.log')
try {
    if (-not $process.WaitForExit(210000)) { throw 'Bounded rendered acceptance exceeded210 seconds.' }
    $process.Refresh()
    $receipt = Get-Content -LiteralPath (Join-Path $destination 'game_receipt.json') -Raw | ConvertFrom-Json
    $engineLog = Get-Content -LiteralPath (Join-Path $destination 'engine.log') -Raw
    if ($process.ExitCode -ne 0 -or $receipt.passed -ne $true -or $receipt.decisions -ne 32 -or $engineLog -match '(?m)(SCRIPT ERROR:|ERROR:)') {
        throw 'Actual rendered acceptance failed; inspect retained evidence.'
    }
    $receipt | ConvertTo-Json -Depth 8
} finally {
    if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
    $process.Dispose()
}
