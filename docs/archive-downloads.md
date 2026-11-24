# Training files and evaluation receipts · 训练与评估归档

The [GitHub Release](https://github.com/brainintelli/BioSignal-Guided-LLM-Reinforcement-Learning/releases/tag/research-artifacts-v1) distributes the screened training and evaluation collections. Download just the collection you need. The public copies of the original 12 archives total 63,932,043,498 bytes in 245 volumes. Raw electrode recordings are indexed separately in the [H5 access guide](data-access.md).

## Download and verify · 下载与校验

From the public repository root, use Python 3.9 or later. Run the first command to list the collections. The next two each download and verify one collection. Existing verified parts are reused.

```sh
python docs/archives/restore.py
python docs/archives/restore.py receipts --download --output .
python docs/archives/restore.py checkpoints --download --output restored-archives
```

只想查看训练日志和逐局决策时，选择单卷 `receipts`。若要载入论文确认性实验的模型，选择 `checkpoints` 的 24 卷。脚本按清单核验每卷和重组文件的 SHA-256；中断后可重跑同一命令。也可以在发布页下载所需文件到 `downloaded-parts`，然后用同一命令省略 `--download`。确认性模型还需要[固定版本的 Qwen 基座](project-contents.md)。

The [public manifest](archives/manifest.json) gives the original and distributed file identities. For independent checks, see the [per-volume SHA-256 list](archives/SHA256SUMS) and [member-level change register](archives/member-changes.csv).

The [original export indexes](https://github.com/brainintelli/BioSignal-Guided-LLM-Reinforcement-Learning/releases/download/research-artifacts-v1/training-provenance-index.zip) list 88,494 original archive members and 45,572 references to files in earlier backups. The ZIP includes the original checksum program, model indexes and export notes. Its SHA-256 is `f03488406f837defee9ddc2ca100dfe41f833f7ed7bd7128c1c803a303dcc7da`.

| Collection key | Contents / 内容 | Volumes | ZIP bytes |
|---|---|---:|---:|
| `receipts` | Per-game evaluation receipts, training batches, update logs, registrations and archived programs / 逐局评估回执、逐批训练轨迹、更新日志、登记与程序 | 1 | 157,781,100 |
| `checkpoints` | 96 confirmatory model checkpoints, including optimizer and random states / 确认性实验 96 个模型及优化器和随机状态 | 24 | 6,312,952,030 |
| `parameters` | Historical policy adapters, critics and saved optimizer states / 历史策略参数、价值网络和已留存的优化器状态 | 140 | 37,556,732,119 |
| `history-logs` | Training, decision and evaluation logs / 训练、决策和评估日志 | 10 | 2,480,892,441 |
| `six-group-a` | Six-arm and reward-weight runs, node A / 节点甲的六组实验与奖励权重扫描 | 28 | 7,389,951,941 |
| `six-group-b` | Six-arm and reward-weight runs, node B / 节点乙的六组实验与奖励权重扫描 | 36 | 9,614,564,624 |
| `history-supplement` | Additional historical records and model components / 历史补充记录及模型组件 | 1 | 214,888,470 |
| `project-source` | Historical game, training and analysis source / 历史游戏、训练及分析源码 | 1 | 13,562,460 |
| `supplementary-source` | Supplementary experiments and figure generation / 补充实验与绘图源码 | 1 | 1,920,686 |
| `runtime-a` | Node A runtime source snapshot / 节点甲运行时源码 | 1 | 2,789,233 |
| `runtime-b` | Node B runtime source snapshot / 节点乙运行时源码 | 1 | 3,760,555 |
| `warm-start` | Project-specific warm starts / 项目暖启动参数 | 1 | 182,247,839 |

## What can be checked · 可复核的范围

The confirmatory collection contains 32 training seeds in each of three input conditions. Each of the 96 models includes its policy adapter, critic, optimizer and saved random state. The project trained parameter-efficient adapters on a frozen, upstream Qwen3.5-0.8B base. The [base-model inventory](../data/upstream/base-model.json) pins the upstream revision and files.

The public receipts preserve the original bytes of all 18,720 game receipts, 384 training trajectory files and 96 update logs. For the adverse-start evaluation, [the documented receipt replay](data-access.md#training-evidence-and-parameters) reconstructs 18,432 games and 669,715 recorded decisions using 19 H5 recordings. The rebuilt endpoint CSV matches the paper’s archived table. The manuscript discusses the historical design-identifier changes and pre-registration deviations in Sections 5 and 8 and Appendix B.

Across the 12 public archives, all 3,028 PyTorch and Safetensors members retain their original bytes. Site-specific configuration and log metadata have a public counterpart in the change register. Original paths or initialization fields referenced by a historical manifest can have different public hashes. Compare an unchanged weight by its original SHA-256 and a mapped metadata file by its distributed SHA-256. Project software, model artifacts, study data and upstream dependencies have [scope-specific terms](../LICENSE-TRAINING).

历史训练快照的保存内容随实验阶段而异，有的初始优化器尚未生成 Adam 矩，有的暖启动快照未保存 RNG 状态。确认性实验的 96 个模型均保存了优化器和随机状态。历史模型按研究阶段、种子和文件哈希定位，跨归档有重复内容。

The original exports of three earlier studies (ablation, autonomous and state2) recorded 224 checkpoint manifests whose references include 672 weight files absent from these 12 collections. The file-level [historical checkpoint index](archives/historical-checkpoint-availability.csv) gives the expected SHA-256 and size, totaling 14,714,729,344 bytes. These three studies are separate from the 96 confirmatory models.

三组早期实验导出时，部分权重只记录了大小和哈希。上述索引逐项列出 12 个归档中缺少原文件的引用，便于和其他备份核对。确认性实验 96 个模型的策略适配器、价值网络和优化器状态均有实物可供核验。
