param([Parameter(Mandatory=$true)][string]$VerificationFile, [string]$BrokerUrl = 'http://127.0.0.1:18080')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'm8x_connection.ps1')
Assert-M8XLocalUrl $BrokerUrl
$projectRoot = Split-Path -Parent $PSScriptRoot
$rulesFile = Join-Path $projectRoot 'configs\m8_exploration_v1.json'
$rulesSha = (Get-FileHash -LiteralPath $rulesFile -Algorithm SHA256).Hash.ToLowerInvariant()
$release = Get-Content -LiteralPath $VerificationFile -Raw -Encoding UTF8 | ConvertFrom-Json
$verificationSha = (Get-FileHash -LiteralPath $VerificationFile -Algorithm SHA256).Hash.ToLowerInvariant()
$health = Invoke-RestMethod -Uri ($BrokerUrl.TrimEnd('/') + '/health') -TimeoutSec 15
Assert-M8XHealthBinding $health $release $verificationSha $rulesSha

# Separate open3 connection; the original open1/M6 connection stays available.
$configPath = Join-Path $projectRoot 'output\agent\gpu_connection_exploration.json'
$connection = [ordered]@{
    enabled = $true; verified_utc = (Get-Date).ToUniversalTime().ToString('o')
    broker_url = $BrokerUrl.TrimEnd('/'); model = $health.model; checkpoint = $health.checkpoint
    actor_sha256 = $health.actor_sha256; quantization = 'BF16 + LoRA'; broker_version = 'm8x.0'
    observation_format = $health.observation_format; schema_encoding_version = $health.schema_encoding_version
    source = 'gpu_ppo_via_netcatty_ssh'; task_goals = $false; scripted_fallback = $false
    survival_gate_passed = $true; release_authorized = $true; evidence_complete = $true; release_phase = 'test'
    decoding = $health.decoding; temperature = 1; transport_context_required = $true
    verification_sha256 = $verificationSha; registration_sha256 = $health.registration_sha256
    verification_file = (Resolve-Path -LiteralPath $VerificationFile).Path
    physical_forward_shape = $health.physical_forward_shape; inference_layout = $health.inference_layout
    encoding_salt = 'm6'; profile = 'open3'; environment_profile = 'open3'; world_version = 5
    environment_version = 'm8_exploration_v1'; balance_sha256 = $rulesSha
    kernel_identity = $health.policy.kernel_identity
    rng_seed_scheme = $health.rng_seed_scheme
}
$null = New-Item -ItemType Directory -Path (Split-Path -Parent $configPath) -Force
$temporaryPath = $configPath + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
try {
    $connection | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryPath -Destination $configPath -Force
} finally {
    if (Test-Path -LiteralPath $temporaryPath) { Remove-Item -LiteralPath $temporaryPath }
}
Write-Output ('Connected verified balanced actor ' + $health.actor_sha256)
