from pathlib import Path
from shutil import copy2
w=Path(__file__).parent
snap=w/'source_snapshot';snap.mkdir(exist_ok=True)
for name in ['report.md','paired_effects.csv','coverage.csv']:
 copy2(Path('output/mea_r2_beta_sweep_20261120')/name,snap/name)
def upd(file,changes):
 p=w/file;s=p.read_text(encoding='utf8')
 for old,new in changes:
  assert old in s,old
  s=s.replace(old,new)
 p.write_text(s,encoding='utf8')
upd('main.py',[
 ('类器官神经活动记录对语言模型游戏探索的影响','类器官神经信号对语言模型游戏探索的影响'),
 ('零输出记录组多发现 2.16 格','当轮停发刺激但仍记录放电的零输出组多发现 2.16 格'),
 ("h('3.2　得分与三种读法'",'''para('角色由 Qwen3.5-0.8B 控制，这是一个约八亿参数的语言模型。负责选择动作的部分称为策略网络（actor），估计往后能拿多少分的部分称为价值网络（critic）。训练使用近端策略优化（proximal policy optimization，PPO）：角色先在游戏中试走，再根据实际得分小幅调整选择概率 [5]。到测试时停止调参。动作选择和价值估计都收到相同的神经观察，屏蔽时两处同时置零。')
h('3.2　得分与三种读法' '' '.strip()'''.replace("' '' '.strip()'","'"))] if False else [
 ('类器官神经活动记录对语言模型游戏探索的影响','类器官神经信号对语言模型游戏探索的影响'),
 ('零输出记录组多发现 2.16 格','当轮停发刺激但仍记录放电的零输出组多发现 2.16 格'),
 ("h('3.2　得分与三种读法',small=True)","para('角色由 Qwen3.5-0.8B 控制，这是一个约八亿参数的语言模型。策略网络（actor）负责选动作，价值网络（critic）估计往后能拿多少分。训练使用近端策略优化（proximal policy optimization，PPO）：角色先在游戏中试走，再根据实际得分小幅调整选择概率 [5]。两部分都收到神经观察，屏蔽时两处同时置零。')\nh('3.2　得分与三种读法',small=True)"),
 ('要了解不同生物样本间的差异，还需实际更换类器官','要了解不同生物样本间的差异，还需实际更换类器官'),
 ('各占一行。','各占一行。测试局数均按最终评估口径计，训练和先导评估另计。'),
 ('双侧 95% 置信区间。确认性实验','双侧 95% 置信区间（confidence interval），用来描述数据对平均差大小的约束。确认性实验')])
upd('layout.py',[("sty('Title',23","sty('Title',20"),("if style in ['Normal','BodyQuote']:p.alignment=Align.JUSTIFY","if style=='Normal':p.alignment=Align.JUSTIFY\n if style=='BodyQuote':p.alignment=Align.LEFT"),('类器官神经活动记录对语言模型游戏探索的影响','类器官神经信号对语言模型游戏探索的影响')])
upd('results.py',[('（附表 4）','（表 4）'),(" h('6.2　较大的实验能看清多小的差异'"," para('等时比较数据　'+R+'paired_effects.csv','SourcePath')\n h('6.2　较大的实验能看清多小的差异'")])
upd('fig.py',[("t(730,17,'平均差和区间（格）',19,mut,True)","t(730,17,'平均差和区间（格）',18,mut,True);t(992,17,'正向数',16,mut,True,'rm')")])
upd('data.py',[("for r in read(R+'paired_effects.csv'):","for r in read(W/'source_snapshot/paired_effects.csv'):"),
 ("with (O/'data_table.csv').open",'''for scene in sorted({r['scene'] for r in rb}):
 for arm in ['ordered','masked','shuffled']:
  selected=[v for v in rb if v['arm']==arm and v['scene']==scene]
  assert len(selected)==2048
  for metric in ['cells','survived','drink_wait','eat_wait']:
   if metric=='drink_wait' and scene!='thirst' or metric=='eat_wait' and scene!='hunger':continue
   value=st.mean(float(v[metric]) for v in selected)
   add('B 各组原始均值',scene+' '+arm,metric,value*(100 if metric=='survived' else 1),B+'episodes.csv',{'cells':'格','survived':'%'}.get(metric,'刻'),seeds=32,maps=64,note='每组每场景 2048 局')
for c,d,l,u in [('刺激后 − 刺激前',-.006883,-.074936,.059320),('刺激后 − 无信号',-.009394,-.098805,.079071)]:
 add('28080 单点独立训练',c,'R1',d,H,'游戏分',l,u,95,8,note='共同前后上下文评估，区别于六决策实验')
for r in read(W/'source_snapshot/coverage.csv'):
 add('A β 扫描快照',f"{r['chip']} {r['arm']} β={r['beta']}",'完整种子',r['complete_seeds'],R+'coverage.csv','个',seeds=r['expected_seeds'],maps=48,note='该条件已核验局数 '+r['complete_worlds'])
with (O/'data_table.csv').open''')])
upd('end.py',[('依 11 月 20 日 19:08 的阶段报告，16 个条件中 14 个完成，核验 11,328 局。','依 11 月 20 日 19:28 的阶段报告，16 个条件中 15 个完成，核验 11,520 局。22554 的 β=0.10 尚缺打乱组，故不估计该档的组间差。'),
 ('差值为对照减顺序，正数表示顺序组较早。','差值为对照减顺序，正数表示顺序组较早。'),
 ('数据总表 data_table.csv 的每行给出实验','数据总表 data_table.csv 的每行给出实验'),
 ('近端策略优化（proximal policy optimization，PPO）根据一次次动作的得分微调选择概率，同时限制每次调整的幅度 [5]。Actor 负责选动作，Critic 估计当前局面的后续得分。','策略网络与价值网络分别更新。下表解释这些参数怎样影响每次参数调整。'),
 ("para('行为核验从原始放电记录一直追溯到游戏终点。","para('行为核验从原始放电记录一直追溯到游戏终点。")])
