import copy,json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_session import binding
try:
    from recovery import verify_preoutput_resume
except ImportError:verify_preoutput_resume=None


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verify_preoutput_resume,'verified pre-output recovery missing')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.old=self.root/'old';self.old.mkdir();self.journal=self.root/'launch.json'
        self.b=binding();route=self.root/'route.cfg';route.write_text('frozen');self.b['route_cfg']=str(route)
        (self.old/'route_input.cfg').write_text('frozen')
        self.write(self.journal,dict(binding=self.b,output=str(self.old)))
        self.p=dict(status='failed',stimulus_attempted=False,stimulus_acknowledged=False,cleanup_errors=[],
                    candidate={'channel':640,'electrode':8858},
                    stages=[dict(name='preflight',wall=1000.,mono=100.),dict(name='configure_current_route')])
        self.write(self.old/'progress.json',self.p)
        self.write(self.old/'identity_qc.json',dict(status='passed',chip_id='NEW42',candidate=self.p['candidate']))
        cmd='mea_array_command owned\nmea_array_disconnect_amplifier_from_stimulation 360'
        self.log=[dict(command=cmd,mono=2),dict(command=cmd,error='Error')]
        for i in range(32):self.log.append(dict(command=f'mea_set_stimulation_unit {i} 0 0 0 0 0 0',ack='Ok'))
        self.log.append(dict(command='mea_set_dac 0 512 512',ack='Ok'))
        self.save_log()
    def write(self,path,value):path.write_text(json.dumps(value))
    def save_log(self):(self.old/'api_audit.jsonl').write_text('\n'.join(json.dumps(x) for x in self.log))
    def verify(self):return verify_preoutput_resume(self.journal,self.b,self.root/'stimulus.json')
    def test_exact_pre_download_failure_can_recover_same_candidate(self):
        result=self.verify();self.assertEqual(result['candidate'],{'channel':640,'electrode':8858})
        self.assertEqual(result['original_clock'],{'wall':1000.,'mono':100.})
    def test_no_recovery_if_any_output_download_or_ambiguous_cleanup(self):
        original=copy.deepcopy(self.log)
        for command in ['sequence_send token','mea_set_dac 0 311 512','mea_array_command t\nmea_array_download_wells 0']:
            self.log=original+[dict(command=command,mono=3)];self.save_log()
            with self.assertRaises(RuntimeError):self.verify()
        self.log=original[:-1];self.save_log()
        with self.assertRaises(RuntimeError):self.verify()
    def test_changed_start_chip_route_or_consumed_claim_rejects(self):
        b=copy.deepcopy(self.b)
        for key,value in [('chip_id','other'),('session_started_at','1970-01-01T00:16:00+00:00')]:
            self.b=dict(b,**{key:value})
            with self.assertRaises(RuntimeError):self.verify()
        self.b=b;Path(b['route_cfg']).write_text('changed')
        with self.assertRaises(RuntimeError):self.verify()
        Path(b['route_cfg']).write_text('frozen');(self.root/'stimulus.json').write_text('{}')
        with self.assertRaises(RuntimeError):self.verify()
    def test_failed_baseline_does_not_become_configuration_recovery(self):
        self.p['stages'].append({'name':'baseline_600s'});self.write(self.old/'progress.json',self.p)
        with self.assertRaises(RuntimeError):self.verify()


if __name__=='__main__':unittest.main()
