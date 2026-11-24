function Get-M8XKernelIdentity($Value) {
    if ($null -ne $Value.kernel_identity) { return $Value.kernel_identity }
    return $Value.policy.kernel_identity
}

function Test-M8XTreeEqual($Left, $Right) {
    if ($null -eq $Left -or $null -eq $Right) { return ($null -eq $Left -and $null -eq $Right) }
    if ($Left -is [array] -or $Right -is [array]) {
        if (-not ($Left -is [array] -and $Right -is [array]) -or $Left.Count -ne $Right.Count) { return $false }
        for ($i=0; $i -lt $Left.Count; $i++) { if (-not (Test-M8XTreeEqual $Left[$i] $Right[$i])) { return $false } }
        return $true
    }
    if ($Left -is [pscustomobject] -or $Right -is [pscustomobject]) {
        if (-not ($Left -is [pscustomobject] -and $Right -is [pscustomobject])) { return $false }
        $leftNames=@($Left.PSObject.Properties.Name | Sort-Object)
        $rightNames=@($Right.PSObject.Properties.Name | Sort-Object)
        if ($leftNames.Count -ne $rightNames.Count) { return $false }
        for ($i=0; $i -lt $leftNames.Count; $i++) {
            if ($leftNames[$i] -cne $rightNames[$i] -or -not (Test-M8XTreeEqual $Left.($leftNames[$i]) $Right.($rightNames[$i]))) { return $false }
        }
        return $true
    }
    return ($Left.GetType() -eq $Right.GetType() -and $Left.Equals($Right))
}

# Pure checks shared by the balanced connector and launcher; no I/O on import.
function Assert-M8XLocalUrl([string]$Value) {
    $endpoint = [uri]$Value
    if (-not $endpoint.IsAbsoluteUri -or $endpoint.Scheme -ne 'http' -or -not $endpoint.IsLoopback -or $endpoint.UserInfo -or $endpoint.Query -or $endpoint.Fragment -or $endpoint.AbsolutePath -ne '/') {
        throw 'Balanced broker must be a local HTTP origin forwarded through Netcatty.'
    }
}

function Test-M8XIdentity($Record, [string]$RulesSha) {
    if ($RulesSha -notmatch '^[a-f0-9]{64}$') { return $false }
    if ($Record.broker_version -ne 'm8x.0' -or $Record.model -ne 'meadowmind-qwen3.5-0.8b-m8x') { return $false }
    if ($Record.release_authorized -ne $true -or $Record.evidence_complete -ne $true -or $Record.release_phase -ne 'test' -or $Record.survival_gate_passed -ne $true) { return $false }
    if ($Record.profile -ne 'open3' -or $Record.environment_profile -ne 'open3' -or $Record.world_version -ne 5 -or $Record.environment_version -ne 'm8_exploration_v1' -or $Record.balance_sha256 -ne $RulesSha) { return $false }
    if ($Record.observation_format -ne 'open_world_exploration_v1' -or $Record.schema_encoding_version -ne 'hierarchical_single_token_exploration_v1' -or $Record.encoding_salt -ne 'm6') { return $false }
    if ($Record.inference_layout -ne 'fixed_batch4_left_pad_v1' -or ($Record.physical_forward_shape -join ',') -ne '4,1536') { return $false }
    if ($Record.temperature -ne 1 -or $Record.decoding -ne 'categorical verb then target; temperature=1; independent episode CUDA generator' -or $Record.transport_context_required -ne $true -or $Record.task_goals -ne $false -or $Record.scripted_fallback -ne $false) { return $false }
    if ($null -eq (Get-M8XKernelIdentity $Record)) { return $false }
    if (-not $Record.checkpoint) { return $false }
    foreach ($key in @('actor_sha256','verification_sha256','registration_sha256')) {
        if ($Record.$key -notmatch '^[a-f0-9]{64}$') { return $false }
    }
    return $true
}

function Test-M8XLaunchBinding($Health, $Connection, [string]$RulesSha) {
    if (-not (Test-M8XIdentity $Health $RulesSha) -or -not (Test-M8XIdentity $Connection $RulesSha)) { return $false }
    if (-not (Test-M8XTreeEqual (Get-M8XKernelIdentity $Health) (Get-M8XKernelIdentity $Connection))) { return $false }
    if ($Health.status -ne 'ok' -or $Health.mode -ne 'model' -or $Health.decision_ready -ne $true) { return $false }
    foreach ($key in @('actor_sha256','verification_sha256','registration_sha256','checkpoint')) {
        if ($Health.$key -ne $Connection.$key) { return $false }
    }
    return $true
}

function Assert-M8XHealthBinding($Health, $Release, [string]$VerificationSha, [string]$RulesSha) {
    if ($Release.protocol -ne 'm8_exploration_reliable_v1' -or $Release.phase -ne 'test' -or $Release.decision -ne 'accept' -or $Release.release_authorized -ne $true -or $Release.evidence_complete -ne $true) { throw 'A complete accepted M8X test release is required.' }
    if (-not (Test-M8XLaunchBinding $Health $Health $RulesSha)) { throw 'Balanced endpoint identity is not ready or differs from local rules.' }
    $checkpoint = ([string]$Release.config.output).TrimEnd('/') + '/' + $Release.selected_checkpoint
    if ($Health.verification_sha256 -ne $VerificationSha -or $Health.registration_sha256 -ne $Release.registration_sha256 -or $Health.actor_sha256 -ne $Release.selected_actor_sha256 -or $Health.checkpoint -ne $checkpoint) { throw 'Endpoint differs from the accepted release.' }
    if ($Release.world_version -ne 5 -or $Release.environment_version -ne 'm8_exploration_v1' -or $Release.balance_sha256 -ne $RulesSha) { throw 'Release differs from local balanced rules.' }
    if ($Health.policy.contract -ne 'open_world_exploration_v1' -or ($Health.policy.physical_forward_shape -join ',') -ne '4,1536') { throw 'Loaded policy representation differs.' }
    if (-not (Test-M8XTreeEqual $Health.policy.kernel_identity $Release.execution_identity.runtime.kernel_identity)) { throw 'Endpoint GPU implementation differs from the accepted result.' }
    $expectedNames = @($Release.source_sha256.PSObject.Properties.Name)
    $actualNames = @($Health.source_sha256.PSObject.Properties.Name)
    $expectedCount = 72
    if ($expectedNames.Count -ne $expectedCount -or $actualNames.Count -ne $expectedCount) { throw 'Balanced frozen source set differs.' }
    foreach ($name in $expectedNames) {
        if ($actualNames -notcontains $name -or $Health.source_sha256.$name -ne $Release.source_sha256.$name) { throw ('Frozen source differs: ' + $name) }
    }
}
