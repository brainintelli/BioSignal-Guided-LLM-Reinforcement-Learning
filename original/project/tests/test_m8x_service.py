"""Synthetic service lifecycle checks. No remote process is started or stopped."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from agent.m8x_server import decision_identity
from agent.m8x_contract import REWARD_CONFIG, VERSION
from tools import m8x_remote_service as service


class ExplorationServiceTests(unittest.TestCase):
    def test_health_binds_world_and_release(self):
        report=dict(selected_actor_sha256='a'*64,registration_sha256='b'*64,source_sha256={'fixture':'x'},execution_identity={'runtime':{'kernel_identity':{'fixture':True}}})
        health=dict(decision_identity('a'*64,'c'*64),status='ok',decision_ready=True,
            model=service.MODEL_ID,checkpoint='/checkpoint',release_authorized=True,
            evidence_complete=True,release_phase='test',survival_gate_passed=True,
            environment_profile='open3',registration_sha256='b'*64,task_goals=False,
            scripted_fallback=False,transport_context_required=True,source_sha256=report['source_sha256'],policy=dict(contract=VERSION,physical_forward_shape=[4,1536],reward_config=REWARD_CONFIG,kernel_identity={'fixture':True}))
        service.validate_health(health,report,'/checkpoint','c'*64)
        for key,value in [('profile','open1'),('world_version',2),('environment_version',2),
                          ('balance_sha256','0'*64),('actor_sha256','0'*64),
                          ('verification_sha256','0'*64),('decision_ready',False),
                          ('release_authorized',False),('checkpoint','/wrong'),('policy',{}),('source_sha256',{})]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                service.validate_health({**health,key:value},report,'/checkpoint','c'*64)

    def test_other_service_and_reused_pid_are_never_owned(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            record=Path(tmp)/'service.json'
            saved=dict(pid=12,start_ticks='100',executable='/python',command=['/python','-m','agent.m8x_server'])
            record.write_text(json.dumps(saved))
            with patch.object(service,'RECORD',record),patch.object(service,'identity',return_value={**saved,'start_ticks':'101'}):
                with self.assertRaises(RuntimeError):service.owned()
            for module in ('agent.m8_server','agent.m6_serve','agent.m8b_train'):
                record.write_text(json.dumps({**saved,'command':['/python','-m',module]}))
                with self.subTest(module=module),patch.object(service,'RECORD',record),patch.object(service,'identity') as inspect:
                    with self.assertRaises(RuntimeError):service.owned()
                    inspect.assert_not_called()

    def test_reused_created_child_is_not_signalled(self):
        child=MagicMock(pid=555);child.poll.return_value=None
        saved=dict(pid=555,start_ticks='100',executable='/python',command=['/python','-m','agent.m8x_server'])
        with patch.object(service,'identity',return_value={**saved,'start_ticks':'101'}),patch.object(service.os,'killpg',create=True) as signal:
            with self.assertRaises(RuntimeError):service.reclaim_created(child,saved)
            signal.assert_not_called()

    def test_failed_ownership_record_reclaims_only_new_child(self):
        child=MagicMock(pid=555)
        identity=dict(pid=555,start_ticks='100',executable='/python',command=['/python','-m','agent.m8x_server'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp,patch.object(service.subprocess,'Popen',return_value=child),patch.object(service,'identity',return_value=identity),patch.object(service,'atomic_record',side_effect=OSError('disk full')),patch.object(service,'reclaim_created') as reclaim:
            with self.assertRaises(OSError):service.spawn_owned(identity['command'],Path(tmp),{},Path(tmp)/'log',{})
            reclaim.assert_called_once_with(child,identity)

    def test_busy_gpu_blocks_start_without_touching_training(self, protocol='m8_exploration_reliable_v1'):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            root=Path(tmp);project=root/'m8b-project';project.mkdir()
            evidence=root/'m8b-evidence';evidence.mkdir()
            run=root/'m8b-runs/run';run.mkdir(parents=True)
            model=root/'models/model';model.mkdir(parents=True)
            report=dict(protocol=protocol,phase='test',decision='accept',
                release_authorized=True,evidence_complete=True,selected_checkpoint='checkpoint_0048',
                selected_actor_sha256='a'*64,publication_request={'fixture':True},
                config=dict(project=str(project),output=str(run),model=str(model)))
            verification=evidence/'verified.json';verification.write_text(json.dumps(report))
            comparator = 'tools.compare_m8x_survival.build_result'
            with patch.object(service,'ROOT',root),patch.object(service,'owned',return_value=None),patch(comparator,return_value=report),patch('agent.m8x_release.build_result',return_value=report),patch.object(service.subprocess,'check_output',return_value='1780066, training, 45159 MiB'),patch.object(service,'spawn_owned') as spawn,patch.object(service.os,'killpg',create=True) as signal:
                with self.assertRaisesRegex(RuntimeError,'GPU compute work left untouched'):
                    service.run_locked(SimpleNamespace(action='start',verification=str(verification)))
                spawn.assert_not_called();signal.assert_not_called()


    def test_isolated_runtime_overrides_inherited_pythonpath(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            root=Path(tmp);project=root/'m8b-project';project.mkdir()
            overlay=root/'m8b-runtime-candidates/test/site';overlay.mkdir(parents=True)
            with patch.object(service,'ROOT',root),patch.dict(service.os.environ,{'PYTHONPATH':'unrelated'}):
                env=service.isolated_environment(dict(kernel_overlay=str(overlay)),project)
                self.assertEqual(env['PYTHONPATH'],str(overlay)+':'+str(project))
                with self.assertRaises(ValueError):service.isolated_environment(dict(kernel_overlay=str(root)),project)

if __name__=='__main__':unittest.main()
