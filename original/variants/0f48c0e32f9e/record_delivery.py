from pathlib import Path
import json
B=Path(__file__).resolve().parent;root=B.parents[1]
checks=json.loads((B/'evidence/delivery_checks.json').read_text());pilot=json.loads((B/'evidence/pilot_qc.json').read_text())
session={'schema':'mea.physical_session.followup.v1','device_host':'192.0.2.21','chip_id':'22554',
 'organoid_session_label':'22554-20261117T032900Z','organoid_biological_id':None,
 'session_started_at':'2026-11-17T03:29:00+00:00','hard_deadline':'2026-11-17T05:29:00+00:00',
 'execution_finished_utc':checks['finished_utc'],'cumulative_biological_seconds':checks['cumulative_biological_seconds'],
 'execution_status':'stopped_after_failed_pilot_signal_stability_qc','baseline_passed':True,'pilot_qc_passed':False,
 'pilot_qc_errors':pilot['errors'],'formal_six_pair_control_executed':False,'new_true_pulses':1,'new_sham_events':1,
 'cumulative_true_pulses':6,'cumulative_sham_events':1,'r2_enabled':False,'realtime_training':False,'further_stimulation_enabled':False,
 'target_phase_amplitude_mv':300,'nominal_phase_amplitude_mv':298.1901,'phase_us':200,'interval_s':45,
 'stim_electrode':10900,'fixed_readout_roi_count':377,'cleanup_errors':[],
 'previous_session_record':str(root/'output/mea_live_22554_20261117/session.json'),'report':str(B/'report.md'),
 'delivery_checks':str(B/'local_delivery_verification.json')}
(B/'session.json').write_text(json.dumps(session,ensure_ascii=False,indent=2),encoding='utf-8')
p=root/'configs/mea_maxone_stimulation_20261117.json';old=p.read_bytes()
backup=B/'parameter_registration_before_update.json'
if not backup.exists():backup.write_bytes(old)
r=json.loads(old);r.update(status='connected_baseline_passed_pilot_qc_stopped',further_stimulation_enabled=False,
 latest_session_record=str(B/'session.json'),latest_report=str(B/'report.md'),latest_followup=session,
 note='Historical first-probe template retained. New connected baseline passed; one sham and one true pilot completed with valid markers, but prespecified signal-stability QC failed in both arms. Six-pair formal control not run. Session stopped at 12:58:37 CST; cumulative 6 true, 1 sham; R2/training off. No automatic continuation.')
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Wrote follow-up session and updated latest registration; old session unchanged.')
