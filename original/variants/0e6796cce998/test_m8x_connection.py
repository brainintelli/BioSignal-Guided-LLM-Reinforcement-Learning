"""Real PowerShell pure binding functions, without HTTP or configuration writes."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from agent.m8x_server import decision_identity


def fixtures(protocol='m8_exploration_reliable_v1'):
    identity=decision_identity('a'*64,'c'*64)
    release=dict(protocol=protocol,phase='test',decision='accept',
        release_authorized=True,evidence_complete=True,config={'output':'/run'},
        selected_checkpoint='checkpoint_0048',selected_actor_sha256='a'*64,
        registration_sha256='b'*64,execution_identity={'runtime':{'kernel_identity':{'fixture':'tested'}}},source_sha256={f'file{i}':str(i) for i in range(72)},
        world_version=5,environment_version='m8_exploration_v1',balance_sha256=identity['balance_sha256'])
    health=dict(identity,mode='model',status='ok',decision_ready=True,
        model='meadowmind-qwen3.5-0.8b-m8x',checkpoint='/run/checkpoint_0048',
        release_authorized=True,evidence_complete=True,release_phase='test',
        survival_gate_passed=True,registration_sha256='b'*64,
        policy=dict(contract='open_world_exploration_v1',physical_forward_shape=[4,1536],kernel_identity={'fixture':'tested'}),
        environment_profile='open3',task_goals=False,scripted_fallback=False,
        transport_context_required=True,source_sha256=release['source_sha256'])
    return health,release


@unittest.skipUnless(shutil.which('powershell'),'Windows PowerShell required')
class ExplorationConnectionTests(unittest.TestCase):
    protocol = 'm8_exploration_reliable_v1'
    def test_real_connector_and_launcher_preserve_legacy_config(self):
        root=Path(__file__).resolve().parents[1]
        health,release=fixtures(self.protocol)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            workspace=Path(tmp);(workspace/'tools').mkdir();(workspace/'configs').mkdir()
            (workspace/'output/agent').mkdir(parents=True)
            engine=workspace/'.tools/godot/Godot_v4.7.2-stable_win64.exe'
            engine.parent.mkdir(parents=True);engine.write_bytes(b'not executed by this test')
            for name in ('m8x_connection.ps1','connect_m8x_model.ps1','launch_exploration_game.ps1'):
                shutil.copy2(root/'tools'/name,workspace/'tools'/name)
            for name in ('m8_exploration_v1.json', 'm8_survival_v2.json'):
                shutil.copy2(root/'configs'/name,workspace/'configs'/name)
            legacy=workspace/'output/agent/gpu_connection.json';legacy.write_bytes(b'legacy sentinel')
            verification=workspace/'release.json';verification.write_text(json.dumps(release),encoding='utf-8')
            import hashlib
            health['verification_sha256']=hashlib.sha256(verification.read_bytes()).hexdigest()
            hp=workspace/'health.json';hp.write_text(json.dumps(health),encoding='utf-8')
            script=workspace/'exercise.ps1'
            script.write_text(r'''param($Root)
$ErrorActionPreference='Stop'
# Keep script-exported Get-FileHash visible across the nested test script scopes.
Import-Module Microsoft.PowerShell.Utility -Global
$global:endpointHealth=Get-Content -LiteralPath (Join-Path $Root 'health.json') -Raw -Encoding UTF8 | ConvertFrom-Json
function Invoke-RestMethod {param($Uri,$TimeoutSec) return $global:endpointHealth}
function Start-Process {param($FilePath,$ArgumentList,$WorkingDirectory) $global:capturedLaunchArgs=$ArgumentList}
$connector=Join-Path $Root 'tools\connect_m8x_model.ps1'
$launcher=Join-Path $Root 'tools\launch_exploration_game.ps1'
$verification=Join-Path $Root 'release.json'
$connection=Join-Path $Root 'output\agent\gpu_connection_exploration.json'
& $launcher -Seed 42
if ($global:capturedLaunchArgs -contains '--model') {throw 'Manual launch enabled model'}
if ($global:capturedLaunchArgs -notcontains '--difficulty=open4') {throw 'Wrong current manual world'}
$blocked=$false
try {& $launcher -Model} catch {$blocked=$true}
if (-not $blocked) {throw 'Unconfigured model launched'}
$global:endpointHealth.balance_sha256='0'*64
$blocked=$false
try {& $connector -VerificationFile $verification} catch {$blocked=$true}
if (-not $blocked -or (Test-Path -LiteralPath $connection)) {throw 'Invalid endpoint wrote config'}
$global:endpointHealth.balance_sha256=(Get-FileHash -LiteralPath (Join-Path $Root 'configs\m8_exploration_v1.json') -Algorithm SHA256).Hash.ToLowerInvariant()
& $connector -VerificationFile $verification
$saved=Get-Content -LiteralPath $connection -Raw -Encoding UTF8 | ConvertFrom-Json
if ($saved.profile -ne 'open3' -or $saved.world_version -ne 5) {throw 'Saved wrong world'}
# The old connector remains auditable, but its old-world release must never
# launch model control in the new current game.
$global:capturedLaunchArgs=$null
$blocked=$false
try {& $launcher -Model -Seed 123} catch {$blocked=$true}
if (-not $blocked -or $null -ne $global:capturedLaunchArgs) {throw 'Old-world model launched current game'}
& $launcher -Seed 123
foreach($arg in @('--difficulty=open4','--seed=123')) {
    if ($global:capturedLaunchArgs -notcontains $arg) {throw ('Missing current manual argument '+$arg)}
}
if ($global:capturedLaunchArgs -contains '--model') {throw 'Old connection silently enabled model control'}
$global:endpointHealth.actor_sha256='d'*64
$global:capturedLaunchArgs=$null
$blocked=$false
try {& $launcher -Model} catch {$blocked=$true}
if (-not $blocked -or $null -ne $global:capturedLaunchArgs) {throw 'Changed actor launched'}
Write-Output 'M8X_CONNECTOR_LAUNCHER_FLOW_PASS'
''',encoding='utf-8')
            result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
                '-File',str(script),str(workspace)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('M8X_CONNECTOR_LAUNCHER_FLOW_PASS',result.stdout)
            self.assertEqual(legacy.read_bytes(),b'legacy sentinel')

    def test_connector_and_launcher_reject_mismatched_releases(self):
        root=Path(__file__).resolve().parents[1]
        health,release=fixtures(self.protocol)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            fixture=Path(tmp)/'fixture.json'
            fixture.write_text(json.dumps(dict(health=health,release=release)),encoding='utf-8')
            script=Path(tmp)/'check.ps1'
            script.write_text(r'''param($Functions,$Fixture)
$ErrorActionPreference='Stop'
. $Functions
$f=Get-Content -LiteralPath $Fixture -Raw -Encoding UTF8 | ConvertFrom-Json
$rulesSha=$f.release.balance_sha256
Assert-M8XHealthBinding $f.health $f.release ('c'*64) $rulesSha
$connection=($f.health | ConvertTo-Json -Depth 12 | ConvertFrom-Json)
if (-not (Test-M8XLaunchBinding $f.health $connection $rulesSha)) {throw 'Valid launcher binding rejected'}
$mutations=@{
    actor_sha256=('d'*64); verification_sha256=('d'*64); registration_sha256=('d'*64)
    release_authorized=$false; evidence_complete=$false; release_phase='selection'; decision_ready=$false
    environment_profile='open1'; profile='open1'; world_version=2; environment_version=2
    balance_sha256=('0'*64); observation_format='open_world_hierarchical_v2'
    physical_forward_shape=@(1,1536); encoding_salt='other'; temperature=0; scripted_fallback=$true
    schema_encoding_version='semantic_single_token_v4'; checkpoint='/wrong'; status='error'
}
foreach($key in $mutations.Keys) {
    $saved=$f.health.$key; $f.health.$key=$mutations[$key]
    $blocked=$false
    try {Assert-M8XHealthBinding $f.health $f.release ('c'*64) $rulesSha} catch {$blocked=$true}
    if (-not $blocked) {throw ('Connector accepted changed '+$key)}
    if (Test-M8XLaunchBinding $f.health $connection $rulesSha) {throw ('Launcher accepted changed '+$key)}
    $f.health.$key=$saved
}
$connection.balance_sha256='0'*64
if (Test-M8XLaunchBinding $f.health $connection $rulesSha) {throw 'Bad saved rules accepted'}
$f.health.source_sha256.PSObject.Properties.Remove('file0')
$blocked=$false
try {Assert-M8XHealthBinding $f.health $f.release ('c'*64) $rulesSha} catch {$blocked=$true}
if (-not $blocked) {throw 'Incomplete source set accepted'}
$f=Get-Content -LiteralPath $Fixture -Raw -Encoding UTF8 | ConvertFrom-Json
$f.release.protocol='unregistered'
$blocked=$false
try {Assert-M8XHealthBinding $f.health $f.release ('c'*64) $rulesSha} catch {$blocked=$true}
if (-not $blocked) {throw 'Unknown protocol accepted'}
$f=Get-Content -LiteralPath $Fixture -Raw -Encoding UTF8 | ConvertFrom-Json
$f.release.phase='selection'
$blocked=$false
try {Assert-M8XHealthBinding $f.health $f.release ('c'*64) $rulesSha} catch {$blocked=$true}
if (-not $blocked) {throw 'Selection-only release accepted'}
$f=Get-Content -LiteralPath $Fixture -Raw -Encoding UTF8 | ConvertFrom-Json
$f.health.policy.kernel_identity.fixture='changed'
$blocked=$false
try {Assert-M8XHealthBinding $f.health $f.release ('c'*64) $rulesSha} catch {$blocked=$true}
if (-not $blocked) {throw 'Different kernel runtime accepted'}
Write-Output 'M8X_CONNECTION_BINDING_PASS'
''',encoding='utf-8')
            result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
                '-File',str(script),str(root/'tools/m8x_connection.ps1'),str(fixture)],
                capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('M8X_CONNECTION_BINDING_PASS',result.stdout)


if __name__=='__main__':unittest.main()
