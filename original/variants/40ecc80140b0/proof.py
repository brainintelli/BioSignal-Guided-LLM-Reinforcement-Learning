from pathlib import Path
w=Path(__file__).parent
edits={'results.py':[('规律刺激后少发现 0.81 格 [−3.18, +0.91]','规律刺激后少发现 0.813 格，差值区间为 [−3.182, +0.906]'),('随机刺激后多发现 0.20 格 [−1.50, +1.68]','随机刺激后多发现 0.195 格 [−1.495, +1.682]'),('在低生命开局，三组在第 60 刻均有','低生命开局把生命设为 5，保留水分和饱食各 95 点。三组在第 60 刻均有'),('压力场景直接考查及时补给的能力。','另两种开局直接考查及时补给：角色都从 25 点生命出发，缺水时水分仅 1.6 点，缺食时饱食仅 4 点。')],'main.py':[('H5 是保存原始电压和事件的文件格式。','H、W、F 是三个位点的代号；H5 是保存电压和事件的文件格式。')],'end.py':[("f('new_cells',2)","f('new_cells',3)")]}
for n,changes in edits.items():
 p=w/n;s=p.read_text(encoding='utf8')
 for a,b in changes:
  assert a in s;s=s.replace(a,b)
 p.write_text(s,encoding='utf8')
