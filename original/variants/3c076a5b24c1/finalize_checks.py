"""Join the independent replay, checkpoint and statistics receipts and write the audited report."""
from pathlib import Path
import csv
import hashlib
import json
import math
import tarfile

V=Path(__file__).resolve().parent
OUT=V.parent
STUDY=OUT.parent


def read(path):return json.loads(path.read_bytes())


def file_hash(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:return list(csv.DictReader(stream))


def sample_key(r):return r['scene'],int(r['seed']),r['arm'],int(r['world'])


def put(path,content):
    temp=path.with_name(path.name+'.tmp');temp.write_text(content,encoding='utf-8');temp.replace(path)


def crosscheck_recovery(pins, early):
    with tarfile.open(V/'recovered-scenes.tar.gz') as archive:
        sets={}
        for entry in archive:
            if entry.isfile() and entry.name.startswith('pinned-scenes/'):
                _,scene,role,name=entry.name.split('/',3)
                sets.setdefault((scene,role),{})[name]=hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
        assert archive.extractfile('original-scene-check.json').read()==(STUDY/'offline-stress-three-arms/stress/scene-check.json').read_bytes()
    assert len(sets)==6
    for (scene,role),files in sets.items():
        pin=hashlib.sha256(json.dumps(files,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert pin==pins['restored_scene_pins'][scene][role]
    expected={r['original']:r for r in early['items']}; baselines={}; count=0
    with tarfile.open(V/'early-receipts.tar.gz','r|gz') as archive:
        for entry in archive:
            assert entry.isfile() and entry.name in expected
            item=expected[entry.name];raw=archive.extractfile(entry).read()
            assert hashlib.sha256(raw).hexdigest()==item['original_sha256']
            assert baselines.setdefault(item['current'],raw)==raw;count+=1
    assert count==early['archived_occurrences'] and len(baselines)==early['distinct_current_receipts']
    repair=read(OUT/'design-repair.json');old=repair['spec_before'].encode();new=repair['spec_after'].encode()
    with tarfile.open(V/'evaluated-evidence.tar.gz','r|gz') as archive:
        for entry in archive:
            if entry.name in baselines:
                raw=archive.extractfile(entry).read();assert raw.count(new)==1 and raw.replace(new,old)==baselines.pop(entry.name)
    assert not baselines


def check():
    data=read(V/'local-verification.json'); sources=read(V/'source-verification.json'); d=read(OUT/'design.json'); summary=read(OUT/'comparison.json')
    audit=read(V/'native-replay.json');reload=read(V/'policy-reload.json');pins=read(V/'source-recovery.json');early=read(V/'prior-receipt-sample.json'); flags=read(V/'policy-flags.json');ck=read(V/'checkpoint-verification.json');powr=read(V/'paired-power.json')
    native_records=readcsv(V/'native-replay.csv'); native={sample_key(r):r for r in native_records};episode_rows=readcsv(OUT/'episodes.csv'); orig={sample_key(r):r for r in episode_rows};assert len(native)==len(orig)==len(native_records)==len(episode_rows)==data['episode_rows']==18432 and native.keys()==orig.keys()
    assert audit['episodes']==len(native) and audit['csv_sha256']==file_hash(V/'native-replay.csv') and audit['verifier_sha256']==file_hash(V/'replay_actions.py') and audit['source_receipt_sha256']==file_hash(V/'native-sources.json')
    assert audit['primary_and_wait_mismatches']==audit['transition_digest_mismatches']==0 and audit['decisions']==sum(int(r['decisions']) for r in native.values())==data['decisions']
    for k,r in native.items():
        assert all(int(r[m])==int(orig[k][m]) for m in ['decisions','end_tick','survived','cells','drink_wait','eat_wait']) and math.isclose(float(r['r1_recorded']),float(orig[k]['r1']),abs_tol=1e-11,rel_tol=0)
    diffs=[abs(float(r['r1_recorded'])-float(r['r1_native'])) for r in native.values()];assert max(diffs)==audit['max_abs_r1_difference'] and sum(x>1e-10 for x in diffs)==audit['r1_differences_over_1e_10'] and sum(int(r['cut']) for r in native.values())==audit['episodes_crossing_tick60']
    models={(s,a) for s in d['training_seeds'] for a in d['arms']};assert {(r['seed'],r['arm']) for r in reload['records']}==models and len(models)==96==len(reload['records'])
    assert reload['verifier_sha256']==file_hash(V/'reload_trajectories.py') and reload['world']==d['worlds'][0] and reload['action_mismatches']==0
    selected=set();steps=0
    for rec in reload['records']:
        assert rec['checkpoint_state']==d['states'][f"{rec['seed']}/{rec['arm']}"] and rec['verifier_sha256']==reload['verifier_sha256']
        for game in rec['checked']:
            k=game['scene'],rec['seed'],rec['arm'],game['world'];assert k not in selected and game['world']==d['worlds'][0] and game['decisions']==int(orig[k]['decisions']);selected.add(k);steps+=game['decisions']
    assert steps==reload['decisions'] and len(selected)==reload['episodes']==288
    file_index={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (V/'checkpoint-files.sha256').read_text().splitlines()};assert file_hash(V/'checkpoint-files.sha256')==ck['manifest_sha256'] and len(file_index)==ck['files']==576 and ck['training_audit_sha256']==file_hash(OUT/'training-audit.json') and ck['errors']==0
    assert len(flags['records'])==flags['checkpoints']==96 and not any(flags[k] for k in ['r2_enabled','beta','stimulation','realtime_training']) and flags['r2_source']=='disabled'
    for row in flags['records']:assert row['policy_sha256']==file_index[f"models/{row['seed']}/{row['arm']}/checkpoint/policy.json"]
    assert pins['original_scene_check_sha256']==pins['reproduced_file_sha256']==file_hash(STUDY/'offline-stress-three-arms/stress/scene-check.json') and pins['verifier_sha256']==file_hash(V/'recover_scene_pins.py') and pins['pilot_scenarios']==36
    assert early['early_archive_sha256']==file_hash(V/'early-receipts.tar.gz') and early['differences_beyond_spec']==0
    assert sources['recording_manifest_digest']==d['source_sha'] and powr['pilot_sha256']==file_hash(STUDY/'offline-stress-three-arms/stress/paired.csv') and powr['script_sha256']==file_hash(V/'paired_power.py') and powr['outer_replicates']==4000 and powr['inner_draws_per_replicate']==40000
    crosscheck_recovery(pins,early)
    for name in ['supplemental.sha256','recovery.sha256','completion.sha256']:
        for line in (V/name).read_text().splitlines():
            pin,filename=line.split('  ',1);assert file_hash(V/filename)==pin,filename
    all_inputs=['local-verification.json','source-verification.json','checkpoint-verification.json','policy-flags.json','native-replay.json','policy-reload.json','source-recovery.json','prior-receipt-sample.json','paired-power.json','power-crosscheck.json','transfer.sha256','completion.sha256','supplemental.sha256','recovery.sha256']
    report=dict(execution_complete=True,prospective_registration_integrity='incomplete: post-evaluation metadata replacement, registration digest placeholder and legacy execution-plan labels, cyclic condition dispatch',original_receipts_with_independent_spec_only_evidence=early['distinct_current_receipts'],affected_receipts=early['scope_total'],pilot_overlap={'training_seeds':data['pilot_training_seeds_reused'],'evaluation_maps':data['pilot_worlds_reused']},all_primary_contrasts=6,registered_primary_stable_gains=sum(r['stable_gain'] for r in data['all32'] if r['family']=='primary_endpoints'),registered_secondary_stable_gains=sum(r['stable_gain'] for r in data['all32'] if r['family']=='secondary_endpoints'),training_models=len(models),training_steps=data['training_steps'],evaluation_games=len(native),decisions=audit['decisions'],paired_rows=data['paired_rows'],action_digest_mismatches=audit['transition_digest_mismatches'],native_reproduction_endpoint_mismatches=audit['primary_and_wait_mismatches'],policy_reloaded_episodes=len(selected),policy_reloaded_decisions=steps,policy_reloaded_action_mismatches=reload['action_mismatches'],verified_checkpoint_files=ck['files'],r2_contribution=0,stimulation=False,realtime_training=False,posthoc_r1_different_episodes=audit['r1_differences_over_1e_10'],posthoc_r1_abs_max_difference=audit['max_abs_r1_difference'],inputs={f:file_hash(V/f) for f in all_inputs})
    put(V/'integrity-summary.json',json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return data,summary,audit,reload,early,powr,report


def render():
    data,comparison,audit,reload,early,power,receipt=check()
    groups=comparison['groups']; effects={(r['scene'],r['metric'],r['control']):r for r in data['all32']}
    names={'low_health':'低生命','thirst':'濒渴','hunger':'濒饿'}
    def eff(scene,metric,control):
        r=effects[scene,metric,control];scale=100 if metric=='survived' else 1
        return f"{r['delta']*scale:+.2f} [{r['lo']*scale:+.2f}, {r['hi']*scale:+.2f}]"
    def vals(scene,metric):
        gg=[groups[scene][a] for a in ['ordered','masked','shuffled']]
        if metric=='survived':return [f"{v['alive']:,}/{v['n']:,}（{100*v['survived']:.2f}%）" for v in gg]
        return [f"{v[metric]:.2f}" for v in gg]
    def statrow(scene,metric,title):return '| '+' | '.join([names[scene]+' · '+title,*vals(scene,metric),eff(scene,metric,'masked'),eff(scene,metric,'shuffled')])+' |'
    old_power=readcsv(STUDY/'power-simulation/power_summary.csv')
    def oldpu(s,m,t):return float(next(row['power_joint'] for row in old_power if (row['scene'],row['metric'],float(row['beneficial_effect']),int(row['n_training_seeds']),int(row['n_worlds_per_seed']))==(s,m,t,32,64)))
    def pu(s,m,t):return next(x['joint'] for x in power['results'] if (x['scene'],x['metric'],x['target_gain'])==(s,m,t))
    for scene in ['thirst','hunger']:
        for arm in ['masked','ordered','shuffled']:
            event_table=groups[scene][arm]['resource_and_survival'];assert event_table['0_1']==event_table['1_0']==0
    diff_count=audit['r1_differences_over_1e_10'];mx=audit['max_abs_r1_difference'];num_cut=audit['episodes_crossing_tick60']
    report=[
    '# 22021 离线确认性实验结果与完整性核验','',
    '2026 年 11 月 20 日。顺序回放 MEA 信号未达到预先写明的稳定改善标准。低生命开局下，顺序组平均多揭示地图格子。濒渴和濒饿开局下，顺序组的存活率点估计均低于两个对照。六项主要比较的校正区间均包含零。', '',
    '## 登记的主要结果','',
    '每个场景每臂 2,048 局。差值按顺序组减对照计算，存活差以百分点计，探索差以格计。区间用 32 个训练种子和 64 张公共地图交叉配对重采样 40,000 次，双侧置信水平为 99.167%。稳定改善同时要求区间下限大于零，且至少 24 个训练种子的配对均值有利。','',
    '| 场景与终点 | 顺序回放 | 屏蔽 | 两秒块打乱 | 对屏蔽的差值 [区间] | 对打乱的差值 [区间] |','|---|---:|---:|---:|---:|---:|',
    statrow('low_health','cells','新揭示格数'),statrow('thirst','survived','第 60 刻存活'),statrow('hunger','survived','第 60 刻存活'),'',
    '低生命探索相对屏蔽有 25/32 个种子有利，相对打乱有 21/32 个有利。濒渴存活分别有 7/32 和 10/32 个有利，濒饿存活分别有 5/32 和 9/32 个有利。三个主要终点共六项比较，达标数为 0。低生命场景的三个条件均为 2,048/2,048 局存活，该场景的存活率已到上限。探索量取初始到第 60 刻或死亡时的 `discovered.size()` 增量，包含随视野揭示的格子。作为支持性指标，濒渴场景的顺序组比屏蔽组和打乱组分别多揭示 3.18 格和 1.23 格，濒饿场景分别多 3.31 格和 1.02 格，这四个校正区间也均包含零。','',
    '## 补水与进食','',
    '只统计合法动作后，游戏逐刻记录中水分或饱食度实际高于动作前的第一次。先死亡或到第 60 刻仍未成功，等待值均记 60。下表均值以游戏刻计。差值按草案取对照等待减顺序组等待，正值表示顺序组更早补给，区间为双侧 98.75%。','',
    '| 场景与终点 | 顺序回放 | 屏蔽 | 两秒块打乱 | 对屏蔽的差值 [区间] | 对打乱的差值 [区间] |','|---|---:|---:|---:|---:|---:|',statrow('thirst','drink_wait','首次成功补水'),statrow('hunger','eat_wait','首次成功进食'),'',
    '四个等待差值的点估计均指向顺序组略晚补给，四个区间均含零。逐局核对显示，在这批濒渴地图上，至少成功补水一次与存活到第 60 刻完全重合。濒饿地图上，至少成功进食一次与第 60 刻存活也完全重合。本实验测量的是归档信号输入下的游戏行为，能够据此判断的指标是探索、存活和补给时间。主观恐惧与实时湿件反馈效果没有在本轮测量。','',
    '## 训练、冻结与逐动作完整性','',
    '- 从 19 份原始 HDF5 记录重新提取芯片 22021 的 324 通道特征。24 个校准窗、48 个训练窗、4 个训练自举窗和 60 个评估窗与实际执行输入逐项一致。物理训练块 1–4 与评估块 5–6 的帧段互不重叠。32 份训练置换、64 份评估置换均保持试验内两秒块完整。',
    '- 三臂从同一暖启动参数开始。96 个模型各有 4 批 × 12 个决策、8 次 Actor 更新、64 次 Critic 更新。已核对 4,608 条训练记录中的窗口、Actor 实际提示和 Critic 输入。96 个 checkpoint 共 576 个文件与远端 SHA-256 清单一致。',
    f"- 每个种子、场景和条件各 64 局，共 18,432 局、6,144 个三臂配对、288 份冻结回执。原生 Godot 重放 669,715 个已存动作，每一步转移摘要一致。全部逐局存活、死亡或截止刻、新揭示格数和补给等待与原表一致。{num_cut:,} 局的末动作跨越第 60 刻。使用游戏原生边界快照复算后，R1 总回报有 {diff_count:,} 局与原表相差超过 10⁻¹⁰，最大绝对差为 {mx:.8g}。逐局两种 R1 均保存在 `verification/native-replay.csv`。",
    f"- 另行从磁盘重载全部 96 个 checkpoint，在每个场景使用登记清单中第一张地图 `{reload['world']}` 重新采样。288 局、{reload['decisions']:,} 个动作、所用 MEA 窗口和转移摘要全部复现，参数哈希在重采样前后一致。此为事后按固定索引做的模型核验，全量逐动作核验使用上述已存动作。",
    f"- 96 份策略元数据均记载 `r2_enabled=false`、`beta=0`、`stimulation=false`、`realtime_training=false`。训练和评估中 R2 奖励贡献均为零。评估保留了 {data['evaluation_diagnostic_events']:,} 次净补水诊断记录。整轮使用历史输入，未接入实时采集或反馈。",'',
    '## 登记符合情况','',
    '- 书面草案将低生命探索、濒渴存活和濒饿存活列为主要终点。执行 JSON 沿用了旧研究中“三场景均以存活为主”和“6/8 种子”的描述，`registration_sha256` 字段也是占位字符串。本报告使用本地原草案和字节相同的远端草案副本，独立按 24/32 门槛重算。分析脚本修正了两处超出作用域的种子计数引用。三臂执行顺序可追溯为按种子轮换，草案中的随机化执行顺序缺少对应记录。',
    '- 原评估使用同一游戏源的两份副本，跨过第 60 刻时，Python 适配器用逐刻记录补写观察。后续用历史构建脚本恢复原生观察器，并补回每个项目的 39 个 Godot 资源标识文件。恢复后的三种场景、两种角色全部六个源码清单与历史固定哈希一致，四张开发地图的 36 个检查重现了原场景核验文件的完整字节。运行过的适配器已归档，共享客户端已恢复到运行前的 SHA-256。',
    f"- 评估后，执行记录更新了场景核验文件的哈希，并同步改写 18,720 份逐局与冻结回执中的设计标识。原场景核验文件仍在，设计修改前后的摘要可以复算。独立保存的早期副本覆盖 {early['distinct_current_receipts']} 份不同回执，已证实这部分仅有设计标识差异。其余回执缺少覆盖全量的改写前逐文件哈希。因此本轮逐动作和统计结果可复核，揭盲前全量锁定这一登记条件无法追溯验证。",
    '- 64 张评估地图与本轮 128 张训练地图没有交集。相对于功效模拟数据，本轮重复使用全部 64 张地图和原有 8 个训练种子，新增 24 个种子。仅看新增种子的事后敏感性分析，六项主要比较的校正区间仍全部含零。低生命探索对屏蔽为 +6.04 格 [−0.20, +12.32]，对打乱为 +2.80 格 [−1.80, +7.80]。本研究只有一个 MEA 记录样本，训练种子数量反映算法重复。','',
    '## 对原功效模拟的复核','',
    '使用原来的 8 个种子 × 64 张地图，事后重算 32 × 64 设计的计划功效。旧脚本为两个对照分别抽样、各训练种子分别抽地图，并用种子均值的正态近似构造区间。复算在每一级为全部比较共用种子和地图索引，4,000 次外层模拟中的每次都执行 40,000 次登记规定的交叉配对自举。以原数据的成对差值为基础，居中后加上假设效应，再计算两项比较都达标的比例。存活项在差值尺度上平移，估计依赖这一模拟假设。只使用先导数据。','',
    '| 假设的有利效应 | 原脚本联合功效 | 配对复算联合功效 |','|---|---:|---:|',
    f"| 低生命多揭示 5 格 | {oldpu('low_health','cells',5):.3%} | {pu('low_health','cells',5):.2%} |",f"| 低生命多揭示 8 格 | {oldpu('low_health','cells',8):.3%} | {pu('low_health','cells',8):.2%} |",f"| 濒渴存活提高 5 个百分点 | {oldpu('thirst','survived',0.05):.3%} | {pu('thirst','survived',.05):.2%} |",f"| 濒饿存活提高 5 个百分点 | {oldpu('hunger','survived',0.05):.3%} | {pu('hunger','survived',.05):.2%} |",'',
    '原表和登记草案保留在原目录。复算表明，本轮对 5 格或 5 个百分点量级的改善辨别力较低。完整模拟结果含每项和联合功效、蒙特卡洛标准误，另有独立 CPU 模拟供交叉核对。','',
    '## 数据与核验文件','',
    '- [原登记草案](../offline-confirmatory-registration/确认性实验登记.md) · [原功效模拟](../power-simulation/power_summary.csv) · [配对功效复算](verification/paired-power.json)',
    '- [逐局结果](episodes.csv) · [三臂配对结果](paired.csv) · [按原登记重算的效应和新增种子分析](verification/local-verification.json)',
    '- [完整性核验摘要](verification/integrity-summary.json) · [逐刻重放回执](verification/native-replay.json) · [每局原生边界核验与 R1](verification/native-replay.csv) · [冻结模型重载回执](verification/policy-reload.json)',
    '- [评估证据包（元数据修订后）](verification/evaluated-evidence.tar.gz) · [证据清单](verification/SHA256SUMS.evaluated) · [历史场景源核验](verification/source-recovery.json) · [回执改写记录](design-repair.json)', '',
    '训练权重保存在远端 `~/mea-confirmatory-stress-22021/models/<训练种子>/<条件>/checkpoint/`。本地证据包包含训练轨迹、checkpoint 清单和服务器评估后的逐局动作。远端生成的 `final-report.md` 和 `final-integrity.json` 保留原始内容。其通过标志覆盖改写后文件的内部一致性。','']
    put(OUT/'实验结果与完整性核验.md','\n'.join(report))
    print(json.dumps({k:v for k,v in receipt.items() if k!='inputs'},ensure_ascii=False))

if __name__=='__main__':render()
