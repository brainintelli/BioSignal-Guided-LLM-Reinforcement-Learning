# 四向角色图集 v1

2026-11-04，内置 image_gen 生成，透明 RGBA 原图 1254×1254，未经裁切、重采样或手工绘画替换。项目资产：`hero_directions_v1.png`；角色身份和画风参考既有 `world_atlas.png` 第0格。

图集4列×4行。行顺序：向下正面、向左侧面、向右侧面、向上背面；每行4个步态。上行方向显示背包，完全不画面部。脚底基线从原图alpha实测后在渲染时对齐，物理圆与碰撞语义不变。

最终提示词：

> Use case: stylized-concept. Create a production-ready transparent sprite sheet for a cute bright hand-painted top-down 2D survival game. Reference image: use ONLY the little green-hooded adventurer in the TOP LEFT cell of the provided world atlas as the character identity/style reference. Ignore all trees/objects. Same child explorer, oversized sage-green hood with two-leaf emblem, chestnut hair, cream tunic, brown boots, small golden-brown camping backpack. Output exactly 4 columns by 4 rows evenly spaced uniform cells, 16 full-body poses, no labels or text or grid lines. Each sprite centered in its cell, same scale, ground contact/feet baseline identical per row, generous transparent padding, no ground shadow. Row1: facing DOWN toward viewer, 4 walking cycle phases (neutral contact, left foot forward/right back, passing, right foot forward/left back). Row2: facing LEFT, true left side view, same four walking cycle phases. Row3: facing RIGHT, true right side view, same four walking cycle phases. Row4: facing UP away from viewer, true back view with backpack prominently visible and NO face or eyes visible, same four walking phases. Strong clear consistent silhouette; realistic direction change for backpack and hood, do not rotate a front-facing drawing. Mild arm swing and alternating legs, body and head stable and consistent across frames. Cute painted 2D game art matching reference, warm gentle sunlight, clean anti-aliased edges, fully transparent alpha background throughout. All sixteen sprites fit within their own equal cells; no overlap.

运动状态只从`actor_position_px()`的实际位移更新；停住、等待、暂停不会推进步态，停步保留最后朝向。摄像机跟随物理位置，窗口大小决定绘制范围。PC视觉目标为60fps，尚未做设备Profiler性能承诺；图集在ready加载一次，步态不创建节点。移动触摸输入仍沿现有控制链路，未新增手机布局或真机验收。
