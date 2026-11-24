"""Unarmed stimulation/capture preparation: no device imports or network access."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from .protocol import PROFILE
from .features import digest
from .train import write


def waveforms():
    signals={}
    for site in ('H','W','F'):
        segments=[]
        for frame in (0,200,400):
            if segments:
                segments.append(dict(start_frame=segments[-1]['start_frame']+4,
                                     duration_frames=192,voltage_mv=0.))
            segments.extend([dict(start_frame=frame,duration_frames=4,voltage_mv=200.),
                             dict(start_frame=frame+4,duration_frames=4,voltage_mv=-200.)])
        segments.append(dict(start_frame=408,duration_frames=0,voltage_mv=0.))
        signals[site]=dict(logical_site=site,physical_electrode=None,stimulation=True,profile=asdict(PROFILE),
            pulse_start_frames=[0,200,400],end_frame=408,sampling_hz=20000,segments=segments,
            voltage_units='nominal_mV_not_load_measurement',polarity_requires_verification=True,
            actual_dac_lsb_mv=None,actual_dac_codes=None)
    signals['quiet']=dict(logical_site=None,physical_electrode=None,stimulation=False,
                          pulse_start_frames=[],end_frame=0,segments=[dict(start_frame=0,duration_frames=0,voltage_mv=0.)])
    return signals


def binding_template():
    return dict(schema='mea.prospective.binding.v2',armed=False,planned_chip_id='22021',
                actual_chip_id=None,chip_material=None,organoid_id=None,acquisition_host=None,
                original_session_started_at=None,cumulative_deadline=None,route_sha256=None,
                logical_electrodes={'H':None,'W':None,'F':None},stimulation_units={'H':None,'W':None,'F':None},
                recording_channels=None,actual_dac_lsb_mv=None,polarity_verified=False,
                output_verified=False,recovery_qc_passed=False,profile_sha256=PROFILE.digest,
                hard_limit_seconds=7200,new_stimulation_cutoff_seconds=5700,
                max_nonzero_attempts=30,max_pulses=90,unknown_sends_consume_budget=True)


def validate_binding(binding):
    required=binding_template()
    if set(binding)!=set(required):raise ValueError('Binding fields differ')
    for name in ('actual_chip_id','chip_material','organoid_id','acquisition_host','original_session_started_at',
                 'cumulative_deadline','route_sha256','recording_channels','actual_dac_lsb_mv'):
        if not binding[name]:raise ValueError('Unbound '+name)
    for name in ('polarity_verified','output_verified','recovery_qc_passed'):
        if binding[name] is not True:raise ValueError('Unverified '+name)
    electrodes=binding['logical_electrodes']
    if set(electrodes)!={'H','W','F'} or any(type(v) is not int or v<0 for v in electrodes.values()) or len(set(electrodes.values()))!=3:
        raise ValueError('Three distinct verified electrodes required')
    if binding['profile_sha256']!=PROFILE.digest:raise ValueError('Pulse profile changed')
    raise ValueError('Preparation artifact is intentionally unarmed; a separately verified live controller is required')


def capture_template(channels=1024):
    if type(channels) is not int or not 2<=channels<=1024:raise ValueError('Invalid planned channel count')
    size=channels*20000*2*7200
    return dict(schema='mea.prospective.capture.v2',continuous_raw=True,spikes=True,events=True,
        record_only_spikes=False,device_frame_numbers=True,sampling_hz=20000,gain=512,hpf_hz=300,
        channel_count_for_capacity_only=channels,bytes_per_raw_sample=2,
        uncompressed_bytes_two_hours=size,minimum_free_bytes=math.ceil(size*1.3),
        compressed_size_assumed=False,minimum_cycle_frames=400000,
        feature_window_relative_frames=[240000,260000],feature_roi_excludes='stimulation_and_fixed_bad_channels',
        recovery_checks_after_last_pulse_seconds=[2,10],window_recovery_verified=False,
        acquisition_clock='device_frames',event_identity=['session_id','organoid_id','chip_id','episode_id',
            'transition_id','event_id','pulse_index','profile_sha256','route_sha256'],
        qc=['continuous_frame_coverage','no_dropped_frames','channel_map_matches','pulse_marker_offsets',
            'output_receipt_matches_intent','raw_clipping_and_rail_recovery','noise_floor_recovery',
            'no_overlapping_stimulus_in_state2_window','fresh_snapshot_context'],
        recording_before_stimulation_required=True,stop_receipt_and_file_close_required=True)


def prepare(profile_path,out):
    source=json.loads(Path(profile_path).read_text())
    if source['execution_profile']!=asdict(PROFILE):raise ValueError('Latest source profile mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'waveforms.json',waveforms())
    write(out/'binding.template.json',binding_template())
    write(out/'capture.template.json',capture_template())
    candidates=[dict(electrode_id=e,historical_chip_id='22021',historical_configured=True,
                     current_route_verified=False,selected_for_role=None)
                for e in source['historical_recording']['configured_stimulation_electrodes']]
    write(out/'historical_electrode_candidates.json',candidates)
    write(out/'event.template.json',dict(session_id=None,organoid_id=None,chip_id=None,episode_id=None,
        transition_id=None,event_id=None,profile_sha256=PROFILE.digest,route_sha256=None,logical_site=None,
        physical_electrode=None,commanded_pulse_frames=None,observed_pulse_frames=None,
        output_verified=False,state2_qc_valid=False,raw_file_sha256=None,
        state2_window_frames=None,r1=None,r2_raw=None,training_admitted=False))
    write(out/'readiness.json',dict(preparation_complete=True,hardware_ready=False,armed=False,
        hardware_commands_sent=0,planned_chip_id='22021',historical_candidates=len(candidates),
        candidate_cycle_seconds=20,nonzero_group_duration_ms=20.4,
        missing=['current_chip_material_and_identity','organoid_identity_and_cumulative_start','current_acquisition_host',
                 'three_current_routed_electrodes','dac_and_polarity_verification','continuous_recording_preflight',
                 'load_output_and_recovery_verification','durable_live_budget_controller'],
        latest_h5_not_admitted_for_training=True))
    (out/'README.md').write_text('''# 真机准备包（未武装）

H/W/F 三个逻辑位点的候选波形已生成：每次只选一个位点，标称每相200mV、200μs，3脉冲，起点间隔10ms，总长20.4ms，最后归零。电压时间面积对称不等于已实测电荷平衡。实际DAC步长、极性和负载输出仍需在当前设备核验。

候选电极表来自22021历史文件的10个配置编号，不代表当前有效路由或已选出的三个电极。完成当前基线后，按稳定记录活动和可路由性选三个互不重复位点，固定绑定生命/水分/饱食；不从其它芯片复用物理编号。

采集必须包含连续raw、spikes、events与设备帧号，先开始记录并核验覆盖，再允许任何刺激。刺激通道保留在原始采集中，排除出模型ROI。每个游戏动作保存前后状态、R1、计划/实际输出、逐脉冲事件、state2窗口及模型版本。

候选周期至少20秒，输入窗口为事件后[12,13)秒；窗口恢复尚未在当前样本验证。缺失数据、无回执或预算用尽不记正R2。恢复检查不能仅因神经活动变强或变弱而筛样本。

磁盘按实际通道数、20kHz、每样本2字节估算，另留30%余量；不假定压缩率。1024通道两小时raw约294.9GB，带余量约383.4GB。容量检查失败不启动。

本枚样本累计两小时，95分钟停止新刺激，预留恢复与关停。最多30次非零尝试/90脉冲包含校准和未知发送尝试；这些是候选软件预算，不是实测生物安全剂量。

本包没有发送接口。绑定模板空项、未通过恢复质控和持久剂量控制未完成时，不具备真实闭环运行条件。最新22021文件仅提供参数及时序参考，不具备本轮游戏对应关系。
''',encoding='utf-8')
    from hashlib import sha256
    write(out/'manifest.json',{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    return json.loads((out/'readiness.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.profile,a.output)))
