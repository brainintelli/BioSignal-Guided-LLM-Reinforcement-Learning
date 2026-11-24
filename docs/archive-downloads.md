# Checkpoint and archive downloads · 模型与归档下载

The paper, game source, paired outcomes and raw MEA download instructions are [available here without signing in](https://github.com/brainintelli/Organoid-Replay-Arena). The trained checkpoints, optimizer states, original training receipts and historical packages are distributed from a separate **private** GitHub repository. Sign into the GitHub account invited to that repository. Access requests can be sent to [Brainintelli Research Lab](mailto:research@brainintelli.com).

论文、游戏源码、逐局派生数据和原始 MEA 下载说明可以直接从公开仓库获取。模型 checkpoint、优化器状态、原始训练回执和历史大包存放在另一个**私有**仓库。请使用获邀的 GitHub 账号登录。需要访问归档时可联系 [Brainintelli Research Lab](mailto:research@brainintelli.com)。

| Collection / 内容 | Download / 下载 |
|---|---|
| Model checkpoints, optimizer states and historical packages / 模型参数、优化器与历史归档 | [Private release with split volumes / 私有分卷下载页](https://github.com/brainintelli/organoid-replay-survival/releases/tag/model-training-archives-v1) |
| How to select, verify and reassemble a package / 如何选择、校验并重组一个包 | [Manifest and reconstruction instructions / 清单及重组方法](https://github.com/brainintelli/organoid-replay-survival/blob/main/docs/archive-downloads/README.md) |
| Original training and per-game receipts / 原始训练和逐局评估回执 | [Private receipts release / 私有回执下载页](https://github.com/brainintelli/organoid-replay-survival/releases/tag/study-evidence-v1) |
| Original H5 recordings / 原始 H5 | [Public data access and SHA-256 instructions / 获取方式及校验说明](data-access.md) |

Download every part of the desired package from the same release. Use the supplied manifest to check each part, assemble in its numbered order and verify the whole-file SHA-256 before extraction. The archive manifest distinguishes intact upstream bytes from packages rebuilt to exclude restricted members. The receipts ZIP has SHA-256 `248907bc759199690d475dd8c6ccbac2920f30df21316cb944a6ad7a1fa523f2`.

从同一版本下载目标归档的所有分卷，按照清单逐卷核对 SHA-256，按编号顺序重组，整包校验通过后再解压。请留意清单中哪些包保留了原始字节，哪些包因排除受限内容而重新打包。未获邀或尚未登录时，私有下载链接会返回 404。
