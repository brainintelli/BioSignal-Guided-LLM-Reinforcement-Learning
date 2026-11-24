from pathlib import Path
w=Path(__file__).parent
p=w/'end.py';s=p.read_text(encoding='utf8')
s=s[:s.index(" para('下面两段直接摘自")]+''' h('系统提示词原文',small=True)
 para('下面是 22021 首枚样本、训练种子 102083199、零输出组第一批第一步的完整系统提示。它规定动作的意思和世界规则。下一页给出同一步的完整局面输入及实际选择。')
 z=json.loads(Path(src+'train-1.jsonl').open(encoding='utf8').readline())
 messages=z['prepared']['messages']
 para(messages[0]['content'],'BodyQuote')
 para('提示词来源　'+src+'train-1.jsonl（第 1 行 prepared.messages[0].content）','SourcePath')
 h('附录 C 续　一次决策的完整输入与实际动作',True)
 para('角色读到的用户消息如下。附近可见的地标、背包和身体状态来自游戏，最后一行来自同一时步的电极记录。')
 para(messages[1]['content'],'BodyQuote')
 h('原始记录中的选择',small=True)
 para('该步选中单格移动（move），目标标识为 tile_49_48，动作历时 1 刻，游戏得分约为 0.009775。该条轨迹用于说明提示与动作怎样对应，动作频率的比较采用全部留存对局（图 4）。')
 para('提示与动作来源　'+src+'train-1.jsonl（第 1 行 prepared.messages[1].content、action、duration、reward）。编码器：standalone/mea_state2_r2/mea_state2_r2/adapter.py。','SourcePath')
'''
p.write_text(s,encoding='utf8')
p=w/'main.py';s=p.read_text(encoding='utf8').replace("A2+'本轮采集刺激结果.md',K,H,Q,J]","A2+'本轮采集刺激结果.md',B+'verification/source-verification.json',K,'output/22554_full_20261119_1224/采集记录.md',H,'output/mea_full_28080_20261117_1406/REPORT.md',Q,J]")
p.write_text(s,encoding='utf8')
