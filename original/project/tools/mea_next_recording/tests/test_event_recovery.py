import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_session import binding
try:
    from recovery import verify_event_setup_resume
except ImportError:verify_event_setup_resume=None


class EventRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verify_event_setup_resume,'event-format recovery verifier missing')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.b=binding();self.b['route_cfg']=str(self.root/'route.cfg')
        (self.root/'route.cfg').write_text('frozen');(self.root/'route_input.cfg').write_text('frozen')
        self.candidate={'channel':640,'electrode':8858}
        self.write('binding.json',self.b)
        self.p=dict(status='failed',stimulus_attempted=False,stimulus_acknowledged=False,cleanup_errors=[],
                    candidate=self.candidate,stages=[dict(name='preflight',wall=1000.,mono=100.),dict(name='configure_current_route'),dict(name='connection_settling')],
                    verified_preoutput_recovery={'original_clock':{'wall':1000.,'mono':100.}})
        self.write('progress.json',self.p)
        self.write('identity_qc.json',dict(status='passed',chip_id='NEW42',candidate=self.candidate))
        self.write('selected_route.json',dict(**self.candidate,stimulation_unit=1))
        command='sequence_append marker\nsystem_trigger_event 0 1 7001 baseline_no_nonzero_output'
        error="RuntimeError('MaxLab rejected sequence_append marker: Error: Malformed Event description. The event description needs to contain an even number of values, i.e. the same number of keys and values. Your event description was: baseline_no_nonzero_output')"
        self.logs=[dict(command='sequence_new pulse',mono=1),dict(command='sequence_new marker',mono=2),dict(command=command,mono=3),dict(command=command,error=error)]
        for unit in range(32):self.logs.append(dict(command=f'mea_set_stimulation_unit {unit} 0 0 0 0 0 0',ack='Ok'))
        for command in ['mea_set_dac 0 512 512','sequence_delete pulse','sequence_delete marker']:
            self.logs.append(dict(command=command,ack='Ok'))
        self.save_log()
    def write(self,name,data):(self.root/name).write_text(json.dumps(data))
    def save_log(self):(self.root/'api_audit.jsonl').write_text('\n'.join(json.dumps(x) for x in self.logs))
    def verify(self):return verify_event_setup_resume(self.root,self.b,self.root/'stimulus.json')
    def test_exact_unexecuted_event_format_error_can_continue(self):
        self.assertEqual(self.verify()['candidate'],self.candidate)
    def test_unknown_error_missing_sequence_cleanup_or_any_send_rejects(self):
        self.logs[3]['error']="TimeoutError('reply lost')";self.save_log()
        with self.assertRaises(RuntimeError):self.verify()
    def test_no_send_or_cleanup_ambiguity_is_accepted(self):
        self.logs.append(dict(command='sequence_send pulse',mono=7));self.save_log()
        with self.assertRaises(RuntimeError):self.verify()
        self.logs.pop();self.logs.pop();self.save_log()
        with self.assertRaises(RuntimeError):self.verify()
    def test_completed_or_failed_baseline_never_retried(self):
        self.p['stages'].append({'name':'baseline_600s'});self.write('progress.json',self.p)
        with self.assertRaises(RuntimeError):self.verify()


if __name__=='__main__':unittest.main()
