"""Reproduce the survival curve from frozen Netcatty evaluation records.

Uses ReportLab's LinePlot and exports PDF/SVG plus a PyMuPDF-rendered PNG.
No model inference, training data edits, or GPU use.
"""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import fitz
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'evaluation_data.json').read_text(encoding='utf-8'))
records = sorted(data['records'], key=lambda row: row['update'])
worlds = {(e['seed'], e['replicate']) for e in records[0]['episodes']}
rows = []
for record in records:
    episodes = record['episodes']
    assert len(episodes) == 16
    assert {(e['seed'], e['replicate']) for e in episodes} == worlds
    assert all(e['replicate'] == 0 for e in episodes)
    assert all(0 <= e['survival_ticks'] <= 1800 for e in episodes)
    assert all(e['terminal'] != e['alive_at_horizon'] for e in episodes)
    assert all(not e['alive_at_horizon'] or e['survival_ticks'] == 1800 for e in episodes)
    mean_ticks = sum(e['survival_ticks'] for e in episodes) / 16
    successes = sum(e['alive_at_horizon'] for e in episodes)
    assert mean_ticks == record['summary']['rmst']
    assert successes == record['summary']['successes']
    rows.append(dict(update=record['update'], fresh_training_decisions=record['update'] * 2048,
                     mean_survival_ticks=mean_ticks, mean_survival_days=mean_ticks / 180,
                     survivors_at_day10=successes, success_percent=successes / 16 * 100,
                     worlds=16, horizon_ticks=1800, source_sha256=record['file_sha256']))
assert len({r['update'] for r in rows}) == len(rows)
with (ROOT / 'survival_curve.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

latest, previous = rows[-1], rows[-2]
best = max(rows, key=lambda r: (r['survivors_at_day10'], r['mean_survival_ticks'], -r['update']))
summary = dict(latest=latest, previous=previous, best=best,
               observed_beijing=datetime.fromisoformat(data['observed_utc']).astimezone(
                   timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
               evaluated_checkpoints=len(rows), unique_worlds=16,
               interpretation='Mean min(lifetime,1800 ticks), not uncensored full-lifetime mean;16 repeated selection worlds, not100 final test worlds')
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNBold', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
ink, muted = HexColor('#172C3A'), HexColor('#586B79')
teal, blue, red = HexColor('#087F8C'), HexColor('#4169A8'), HexColor('#B84040')
grid, rule = HexColor('#E4EAF0'), HexColor('#AAB6BF')
W, H = 1120, 930
d = Drawing(W, H)
d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))


def text(x, y, value, size=12, color=ink, bold=False, anchor='start'):
    d.add(String(x, y, value, fontName='CNBold' if bold else 'CN', fontSize=size,
                 fillColor=color, textAnchor=anchor))


text(55, 884, 'SUPPLY3  /  平均存活时间随训练轮次变化', 25, bold=True)
text(55, 855, f"节点5 · 同一组16个筛选世界 · 数据截至 {summary['observed_beijing']} 北京时间", 12, muted)
d.add(Line(55, 834, 1065, 834, strokeColor=grid))

for x, label, value, foot in [
    (55, '最新平均存活时间', f"{latest['mean_survival_days']:.2f} 游戏天", f"{latest['mean_survival_ticks']:.0f} tick · 第{latest['update']}轮"),
    (400, '最新10天存活率', f"{latest['success_percent']:.1f}%", f"{latest['survivors_at_day10']} / 16 局存活"),
    (745, '相较上次评估', f"{latest['mean_survival_days'] - previous['mean_survival_days']:+.2f} 天", f"第{previous['update']}轮：{previous['mean_survival_days']:.2f}天"),
]:
    text(x, 809, label, 11, muted)
    text(x, 776, value, 25, red if x == 745 else ink, bold=True)
    text(x, 754, foot, 11, muted)


def panel(y, height, values, ymax, ysteps, color):
    x, width = 88, 930
    for v in ysteps:
        yy = y + v / ymax * height
        d.add(Line(x, yy, x + width, yy, strokeColor=grid, strokeWidth=0.7))
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = x, y, width, height
    lp.data = [[(row['update'], value) for row, value in zip(rows, values)]]
    lp.lines[0].strokeColor = color
    lp.lines[0].strokeWidth = 2.3
    marker = makeMarker('FilledCircle')
    marker.size, marker.fillColor, marker.strokeColor = 5.3, white, color
    marker.strokeWidth = 1.4
    lp.lines[0].symbol = marker
    for axis in (lp.xValueAxis, lp.yValueAxis):
        axis.strokeColor = rule
        axis.strokeWidth = 0.7
        axis.labels.fontName = 'CN'
        axis.labels.fontSize = 10
        axis.labels.fillColor = muted
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax = 0, 160
    lp.xValueAxis.valueSteps = list(range(0, 161, 16))
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = 0, ymax
    lp.yValueAxis.valueSteps = ysteps
    d.add(lp)
    px = x + latest['update'] / 160 * width
    py = y + values[-1] / ymax * height
    d.add(Circle(px, py, 4.5, fillColor=red, strokeColor=white, strokeWidth=1.3))
    return x, width, px, py


text(55, 718, '01  平均存活时间', 15, bold=True)
text(1065, 718, '单位：游戏天；1天 = 180 tick', 11, muted, anchor='end')
x, width, px, py = panel(413, 264, [r['mean_survival_days'] for r in rows], 10.5, [0, 2, 4, 6, 8, 10], teal)
limit_y = 413 + 10 / 10.5 * 264
d.add(Line(x, limit_y, x + width, limit_y, strokeColor=rule, strokeWidth=1, strokeDashArray=[5, 4]))
text(1018, 687, '10天评估上限：存活到此处即截断记录', 10, muted, anchor='end')
text(px - 10, py - 23, f"{latest['mean_survival_days']:.2f}天", 12, red, True, 'end')
for update in (48, 128):
    xx = x + update / 160 * width
    d.add(Circle(xx, limit_y, 4, fillColor=teal, strokeColor=white, strokeWidth=1))
    text(xx, limit_y - 23, f'{update}轮：10天', 10, teal, anchor='middle')
text(553, 374, '训练轮次（每轮新增2048次决策）', 11, muted, anchor='middle')

text(55, 337, '02  活到第10天的比例', 15, bold=True)
text(1065, 337, '辅助观察：平均值接近10天，不代表每局都能存活', 11, muted, anchor='end')
x, width, px, py = panel(125, 170, [r['success_percent'] for r in rows], 105, [0, 25, 50, 75, 100], blue)
threshold_y = 125 + 93.75 / 105 * 170
d.add(Line(x, threshold_y, x + width, threshold_y, strokeColor=rule, strokeWidth=1, strokeDashArray=[5, 4]))
text(1018, 307, '筛选门槛 ≥15/16（93.75%）', 10, muted, anchor='end')
text(px - 10, py - 21, f"{latest['success_percent']:.1f}%", 12, red, True, 'end')
text(553, 87, '训练轮次', 11, muted, anchor='middle')
d.add(Line(55, 67, 1065, 67, strokeColor=grid))
text(55, 45, '口径：16局的 min(实际存活tick,1800) 均值；幸存者统一记为10天，并非完整寿命均值。', 10, muted)
text(55, 26, '第0轮为上一轮最佳模型暖启动。折线连接已完成评估点，未平滑；此图不是最终100局验收。', 10, muted)

pdf_path = ROOT / 'survival_curve.pdf'
renderPDF.drawToFile(d, str(pdf_path))
renderSVG.drawToFile(d, str(ROOT / 'survival_curve.svg'))
with fitz.open(pdf_path) as document:
    document[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(ROOT / 'survival_curve.png')
print(json.dumps(summary, ensure_ascii=False, indent=2))
