param([int]$Seed = 2304, [switch]$Model, [switch]$Manual, [switch]$Editor, [switch]$SaveGpuPassword)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$credentialPath = Join-Path $env:LOCALAPPDATA 'Meadowmind\gpu-192.0.2.16.dpapi'
$useSavedCredential = -not $Editor -and -not $Manual -and (Test-Path -LiteralPath $credentialPath -PathType Leaf)
if ($SaveGpuPassword -or $useSavedCredential) {
    Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1')
}
if ($SaveGpuPassword) {
    $securePassword = Read-Host 'GPU SSH password for researcher@192.0.2.16' -AsSecureString
    $pendingCredential = "$credentialPath.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        if ($securePassword.Length -eq 0) { throw 'Enter a password to save it.' }
        New-Item -Path (Split-Path -Parent $credentialPath) -ItemType Directory -Force | Out-Null
        ConvertFrom-SecureString $securePassword | Set-Content -LiteralPath $pendingCredential -Encoding ASCII
        Move-Item -LiteralPath $pendingCredential -Destination $credentialPath -Force
        Write-Output 'GPU credential saved for the current Windows user.'
    } finally {
        $securePassword.Dispose()
        if (Test-Path -LiteralPath $pendingCredential) { Remove-Item -LiteralPath $pendingCredential }
    }
    return
}
if ($Model -and $Manual) {
    throw 'Choose either -Model or -Manual.'
}
if (-not $Editor -and -not $Manual) {
    $modelLauncher = Join-Path $PSScriptRoot 'mea_model_game.py'
    if (-not (Test-Path -LiteralPath $modelLauncher -PathType Leaf)) {
        throw "Model launcher not found at $modelLauncher"
    }
    $python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        $python = (Get-Command python -ErrorAction Stop).Source
    }
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $securePassword = $null
    if ($useSavedCredential) {
        try {
            $securePassword = ConvertTo-SecureString (Get-Content -LiteralPath $credentialPath -Raw).Trim()
        } catch {
            throw 'Cannot read the saved GPU credential. Run the launcher with -SaveGpuPassword to update it.'
        }
    } elseif ([Console]::IsInputRedirected) {
        $gpuPassword = [Console]::ReadLine()
    } else {
        $securePassword = Read-Host 'GPU SSH password for researcher@192.0.2.16 (Enter to use an SSH key)' -AsSecureString
    }
    if ($null -ne $securePassword) {
        $secretPointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
        try {
            $gpuPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)
        } finally {
            [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer)
            $securePassword.Dispose()
        }
    }
    try {
        ([string]$gpuPassword) | & $python -X utf8 $modelLauncher --password-stdin --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw 'Could not run the M9 model game. See the message above.' }
    } finally {
        Remove-Variable gpuPassword
    }
    return
}
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
$gameArgs = @('--difficulty=open4', "--seed=$Seed", '--require-model-context')
$engineArgs = @('--path', ('"' + $projectRoot + '"'), '--') + $gameArgs
Start-Process -FilePath $engine -ArgumentList $engineArgs -WorkingDirectory $projectRoot
