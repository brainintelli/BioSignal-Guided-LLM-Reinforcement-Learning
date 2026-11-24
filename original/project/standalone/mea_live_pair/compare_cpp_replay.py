"""Verify archived PPO artifacts and summarize the eight paired holdout worlds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from livepair.figures import _plot_style
from mea_state2_r2.features import digest

ARMS = {
    'ordered': '实测顺序 + R1/R2',
    'r1': '实测顺序 + R1',
    'shuffled': '训练打乱 + R1/R2',
    'masked': '屏蔽信号 + R1/R2',
}
METRICS = ('r1_sum', 'new_cells', 'final_health', 'final_thirst', 'final_hunger', 'terminated')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def file_hash(path):
    with path.open('rb') as data:
        return hashlib.file_digest(data, 'sha256').hexdigest()


def compare(folder):
    training = folder / 'ppo' / 'run01'
    data = read(folder / 'ppo-data.json')
    common = None
    values = {}
    verified = {}
    for arm in ARMS:
        directory = training / arm
        reg = read(directory / 'registration.json')
        result = read(directory / 'result.json')
        manifest = read(directory / 'checkpoint' / 'manifest.json')
        init = (reg['initial'], reg['config'], reg['training_seed'], reg['eval_seeds'])
        if common is None:
            common = init
        if init != common or reg['source_sha'] != digest(data) or result['initial'] != reg['initial']:
            raise ValueError(f'Initialization, budget or source mismatch in {arm}')
        if result['checkpoint_sha'] != digest(manifest):
            raise ValueError(f'Checkpoint manifest mismatch in {arm}')
        for name, expected in manifest['files'].items():
            if file_hash(directory / 'checkpoint' / name) != expected:
                raise ValueError(f'Checkpoint file mismatch in {arm}/{name}')
        if result['after']['actor'] != manifest['actor_sha256']:
            raise ValueError(f'Actor mismatch in {arm}')
        for component in ('actor', 'critic', 'base', 'reference'):
            changed = result['after'][component] != result['initial'][component]
            if changed != (component in ('actor', 'critic')):
                raise ValueError(f'Unexpected parameter state {arm}/{component}')
        rows = [json.loads(line) for line in (directory/'train.jsonl').read_text(encoding='utf8').splitlines()]
        order = [v['snapshot']['window_index'] for v in rows]
        if sorted(order) != list(range(1,13)) or order != reg['train_order'] or reg['test_order'] != list(range(13,19)):
            raise ValueError(f'Train/test split mismatch in {arm}')
        if arm == 'shuffled' and any(i == w for i,w in enumerate(order,1)):
            raise ValueError('Shuffle left a trial at its original position')
        if arm != 'shuffled' and order != sorted(order):
            raise ValueError(f'Wrong order in {arm}')
        for row in rows:
            obs = row['prepared']['wetware_features']
            expected = [0.]*4 if arm == 'masked' else row['snapshot']['normalized']
            rr = row['reward_record']
            if obs != expected or rr['actual_feedback_to_game'] or not math.isclose(row['reward'],row['r1_score']['reward']+rr['r2_contribution']):
                raise ValueError(f'Observation or reward provenance mismatch in {arm}')
        ev = result['eval']
        if [v['seed'] for v in ev] != reg['eval_seeds']:
            raise ValueError('Game seeds changed')
        for v in ev:
            sample = read(directory / f"eval-{v['seed']}.json")
            if [s['index'] for s in sample] != list(range(13,13+len(sample))) or not math.isclose(sum(s['r1'] for s in sample),v['r1_sum']):
                raise ValueError(f'Holdout trace mismatch in {arm}')
        verified[arm] = dict(optimizer_steps=result['optimizer']['optimizer_steps'], signed_train_r2=result['simulated_r2'],
            total_train_r2_contribution=result['r2_contribution'], checkpoint_files=len(manifest['files']),
            train_windows=order, clock='C++ host milliseconds')
        values[arm] = ev
    paired = {}
    for arm in ('r1','shuffled','masked'):
        diffs=[a['r1_sum']-b['r1_sum'] for a,b in zip(values['ordered'],values[arm])]
        paired[arm]=dict(mean=mean(diffs),positive=sum(v>1e-9 for v in diffs),negative=sum(v< -1e-9 for v in diffs),
                         tied=sum(abs(v)<=1e-9 for v in diffs),by_seed=diffs)
    result = dict(means={a:{k:mean(v[k] for v in rs) for k in METRICS} for a,rs in values.items()},
                  ordered_minus_control_r1=paired,verified=verified,by_seed=values)
    (folder/'training-comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    with _plot_style():
        fig=Figure(figsize=(9.5,4.2),dpi=145,layout='constrained')
        a,b=fig.subplots(1,2)
        for ax,key,title in [(a,'r1_sum','游戏原生回报 R1'),(b,'new_cells','新增探索格数')]:
            for i,seed in enumerate(common[3]):
                ax.plot(range(4),[values[arm][i][key] for arm in ARMS],color='#8a9dab',alpha=.65,
                        linewidth=.9,marker='.',label='同一游戏种子' if i==0 else None)
            ax.plot(range(4),[result['means'][arm][key] for arm in ARMS],color='#145b79',marker='o',
                    linewidth=2.2,label='8 个种子的均值')
            ax.set_xticks(range(4), ['实测\nR1/R2','实测\n仅 R1','打乱\nR1/R2','屏蔽\nR1/R2'])
            ax.set_title(title)
            ax.grid(axis='y',alpha=.22)
        a.legend(fontsize=8)
        fig.suptitle('芯片 28080 · 两轮留存数据上的游戏表现')
        FigureCanvasAgg(fig).print_png(str(folder/'ppo-holdout.png'))
    print(json.dumps({k:result[k] for k in ('means','ordered_minus_control_r1','verified')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    compare(parser.parse_args().folder)

