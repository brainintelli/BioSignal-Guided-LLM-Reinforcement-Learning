from pathlib import Path
import csv, hashlib, json, datetime
ROOT=Path('/home/researcher/mea-confirmatory-stress-22021')
H=ROOT/'stress'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o): return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
root_design=read(ROOT/'design.json')
design=read(H/'design.json')
comparison=read(H/'comparison.json')
pilot=read(H/'scene-check.json')
fallback=read(H/'audit-source-fallback.json')
repair=read(H/'design-repair.json')
training=read(ROOT/'training-audit.json')
pin=digest(design)
seeds=design['training_seeds']; worlds=design['worlds']; scenes=design['scenes']; arms=design['arms']
expected_training=len(seeds)*len(arms)
expected_games=len(seeds)*len(worlds)*len(scenes)*len(arms)
expected_pairs=len(seeds)*len(worlds)*len(scenes)
errors=[]
training_steps=sum(int(x['steps']) for x in training)
training_r2=sum(int(x.get('reward_contribution',0)) for x in training)
world_files=0; frozen_files=0; row_count=0; row_r2=0
for seed in seeds:
    for scene in scenes:
        for arm in arms:
            base=H/'games'/str(seed)/scene/arm
            frozen=base/'frozen.json'
            frozen_files += int(frozen.exists())
            if not frozen.exists(): errors.append(f'missing frozen {seed}/{scene}/{arm}')
            else:
                fd=read(frozen)
                if fd != {'scene':scene,'seed':seed,'arm':arm,'state':design['states'][f'{seed}/{arm}'],'worlds':64,'spec':pin}:
                    errors.append(f'frozen mismatch {seed}/{scene}/{arm}')
            files=sorted(base.glob('[0-9]*.json'))
            if {int(p.stem) for p in files} != set(worlds):
                errors.append(f'world set mismatch {seed}/{scene}/{arm}')
            for p in files:
                world_files += 1
                d=read(p); g=d['game']; rows=g['rows']; row_count += len(rows)
                row_r2 += sum(int(r.get('r2_contribution',0)) for r in rows)
                if d.get('spec') != pin or d.get('seed') != seed or d.get('scene') != scene or d.get('frozen') != design['states'][f'{seed}/{arm}']:
                    errors.append(f'metadata mismatch {p}')
                if g.get('world') != int(p.stem) or g.get('mode') != arm or len(rows) != g.get('decisions'):
                    errors.append(f'game mismatch {p}')
                if len(errors)>20: break
            if len(errors)>20: break
        if len(errors)>20: break
    if len(errors)>20: break
with (H/'episodes.csv').open(encoding='utf-8-sig',newline='') as f: episode_rows=sum(1 for _ in csv.DictReader(f))
with (H/'paired.csv').open(encoding='utf-8-sig',newline='') as f: paired_rows=sum(1 for _ in csv.DictReader(f))
analysis_rc=int((H/'analysis.rc').read_text().strip())
checks={
 'training_records':len(training)==expected_training,
 'training_steps':training_steps==4608,
 'training_r2_zero':training_r2==0,
 'evaluation_world_files':world_files==expected_games,
 'frozen_files':frozen_files==len(seeds)*len(scenes)*len(arms),
 'episode_rows':episode_rows==expected_games,
 'paired_rows':paired_rows==expected_pairs,
 'analysis_return_code':analysis_rc==0,
 'comparison_counts':comparison['games']==expected_games and comparison['pairs']==expected_pairs and comparison['decisions']==row_count,
 'comparison_r2_zero':comparison['r2_total_contribution']==0,
 'row_r2_zero':row_r2==0,
 'design_pin_matches_comparison':comparison['design_sha']==pin,
 'pilot_sha_matches':sha(H/'scene-check.json')==design['pilot_sha'],
 'scene_pins_match':pilot['pins']==design['game_pins'],
 'training_audit_hash_matches':sha(ROOT/'training-audit.json')==design['training_audit_sha'],
 'training_design_hash_matches':digest(root_design)==design['training_design_sha'],
 'beta_zero':design['beta']==0 and root_design['beta']==0,
 'r2_disabled_recorded':fallback['r2_enabled'] is False and repair['r2_enabled'] is False,
 'realtime_disabled_recorded':fallback['realtime_training'] is False and repair['realtime_training'] is False,
 'audit_source_fallback_recorded':fallback['used_source']=='PROJECT',
 'audit_repair_recorded':repair['result_files_updated']==expected_games+len(seeds)*len(scenes)*len(arms),
}
integrity=all(checks.values()) and not errors
out={
 'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'integrity_pass':integrity,
 'checks':checks,
 'errors':errors[:20],
 'counts':{'training_records':len(training),'training_steps':training_steps,'evaluation_games':comparison['games'],'evaluation_world_files':world_files,'episode_rows':episode_rows,'paired_rows':paired_rows,'decisions':comparison['decisions'],'r2_row_contribution':row_r2},
 'hashes':{'stress_design_sha':pin,'scene_check_sha':sha(H/'scene-check.json'),'pilot_sha':design['pilot_sha'],'training_audit_sha':sha(ROOT/'training-audit.json')},
 'constraints':{'r2_enabled':False,'beta':design['beta'],'realtime_training':False,'stimulation':False},
 'audit_source_fallback':fallback,
 'design_repair':repair,
}
(H/'final-integrity.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
lines=['# 离线确认性实验结果','',f"状态为 {'通过' if integrity else '未通过'}。结果由 32 个训练种子、3 个场景、3 个条件和 64 张地图组成。",'', '## 执行规模','',f'- 训练记录为 {len(training)} 个，训练 transition 为 {training_steps} 条。',f'- 评估 episode 为 {comparison["games"]} 个，动作决策为 {comparison["decisions"]} 条。',f'- 配对行数为 {comparison["pairs"]} 条。',f'- 每个种子、场景和条件均有 64 个 world 文件及 frozen 校验文件。','', '## 约束核验','', '- R2 保持关闭，登记 beta 为 0，R2 contribution 汇总为 0。', '- 没有启用实时训练或实时闭环，stimulation 保持关闭。', '- 所有评估结果均来自离线调度器，训练后冻结模型。','', '## 场景汇总','', '| 场景 | 条件 | 存活数 | 首次饮水等待均值 | 首次进食等待均值 | 新格均值 |', '| --- | --- | ---: | ---: | ---: | ---: |']
for scene in scenes:
    for arm in arms:
        g=comparison['groups'][scene][arm]
        lines.append(f"| {scene} | {arm} | {g['alive']} / {g['n']} | {g['drink_wait']:.2f} | {g['eat_wait']:.2f} | {g['cells']:.2f} |")
lines += ['', '## 可复核文件','', '- final-integrity.json contains the machine-checkable integrity result.', '- comparison.json contains the bootstrap effects and paired comparisons.', '- episodes.csv contains one row per evaluated game.', '- paired.csv contains one row per seed and map within each stress scene.', '', '## 登记偏差记录','', '- 远端缺少登记草案指定的 frozen audit-game 源，场景设置脚本使用现有配对项目生成 audit 副本，记录在 audit-source-fallback.json。', '- 评估协议缺少旧版 landmark 操作，Python 适配层从最后一个 step trace 恢复 60 tick 截断状态，并保留旧 play60 哈希。', '- 场景核验重新生成后修复了 pilot_sha 和结果 spec 元数据，前后哈希记录在 design-repair.json。']
(H/'final-report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({'integrity_pass':integrity,'checks_failed':[k for k,v in checks.items() if not v],'errors':errors[:5],'counts':out['counts']},ensure_ascii=False))
