# 入库说明与版本对应关系

入库日期：2026-11-20（Asia/Shanghai）。本目录汇总此前已经完成的数学核查、实验结果、代码和后续实验方案。原仓库的论文、模型实现、检查点和实验结果保留在原路径；新增研究材料集中放在本目录。

## 证据来源

| 路径 | 本次保存内容 | 正确解释 |
|---|---|---|
| `stage1/` | 19 个文件，包含原执行清单 | 2026-11-18 的数学核查、NumPy 实现和数值控制历史快照 |
| `stage2/` | 364 个文件，包含原执行清单 | 2026-11-20 的 Torch CPU 检查、独立重建 Burgers FNO、NS 数值控制、CPU 计时、完整实验记录和扩展方案 |
| `reviewed_manuscript/onesided-delta-physics-icml-v2.pdf` | 用户提供的审阅输入 PDF | 上述补充分析所依据的论文版本；与仓库当前 `paper_draft/main.pdf` 内容不同 |
| `README.md`、`FINDINGS.md` | 本次整理的入口与结论摘要 | 帮助读者区分研究对象、已完成实验和待执行工作 |
| `import_provenance.json`、`import_verification.json` | 入库来源、校验和与本次验证记录 | 记录新增材料与原仓库的对应关系 |

两个阶段的研究文件逐字节保存，原执行清单和原文件哈希不变。部分历史 CSV 使用 CRLF 换行，归档目录保留这些原始字节以保持校验一致。未重复存入相同内容的 ZIP、顶层图表副本、运行缓存或日志。`ns_controls/preliminary_dt005/` 是正式清单排除的预跑输出；说明为何降低参考解误差的 `stage2/ns_controls/results/preliminary_numerical_floor_note.json` 仍然保留。

## 旧报告中的“尚未获得”应如何理解

`stage1/` 中“当前环境未安装 PyTorch”“只做语法检查”等文字记录了第一阶段的执行条件。第二阶段已经完成 168 个 Torch CPU 解析检查，并训练、评测了三种子 Burgers FNO。使用当前实现时请从 `stage2/` 的入口开始。

`stage2/` 中“未提供原模型代码、检查点或 Aitken 实现”等文字，记录了这些实验执行时尚未获得本 GitHub 仓库的情况。仓库地址在实验完成后提供，本次已确认原仓库包含以下相关资产：

- [原模型与求解器实现](../../fno_dual/)。
- [原 Recast/Aitken 实验脚本](../../run_ffno_equation_recast_3seed.py)。
- [75-epoch NS 单种子敏感性实验](../../results/ffno_ns64_budget75_seed0/)。
- [原实验结果目录](../../results/)及[重新导出的轨迹数据](../../data/trajectories/)。

这些资产的发现使下一步原检查点统一比较具备了可核查的入口。它不改变此前实验的来源：新增 Burgers 数据来自独立重建模型，新增 NS 数据来自数值参考推进器。原权重、normalizer、测试轨迹、时间步及 Recast 变体之间的完整适配和重新评测，仍需单独执行。不能把新增数值直接填入原 F-FNO 的结果表，也不能把原单种子 75-epoch NS 实验称作三种子实验。

本次上传没有运行模型训练、原检查点迁移比较或 GPU 基准。上传时只重新运行了解析实现检查和评测器集成检查，结果见 `import_verification.json`。

## Recast 与闭式隐式修正

早期讨论中提出的外部隐式方程为

\[
y=\Phi_\theta(u)+h(\nu-\nu_0)\Delta y.
\]

在相同周期离散 Laplacian 下，它可以通过一次网络计算和 Fourier 除法求解。补充代码中的 `implicit` 就是这个基线。

审阅版本和本仓库脚本中的输入型 Recast 则把校正后的状态传入网络：

\[
y=\Phi_\theta\!\left(u+h(\nu-\nu_0)\Delta y\right).
\]

这里网络输入随迭代改变，通常不能把网络输出缓存为常量。因此，新增闭式隐式结果是一个独立而必要的比较方法；它不是对现有输入型 Recast/Aitken 的代数等价替换，不能继承其准确率，也不能仅凭外部隐式方程宣布原网络调用统计错误。详细分析见 [FINDINGS.md](FINDINGS.md) 和两个阶段的完整报告。

## 路径和复现

原 `runner/*results*/summary.json` 中少数 `provenance.path` 字段是实验执行时的工作区绝对路径。它们作为历史元数据保存，不是克隆后的输入路径。复现应使用 `stage2/README.txt` 和 `stage2/runner/README.txt` 中的相对配置入口，并将新的运行结果写到新的输出目录，以免覆盖已归档证据。

在仓库根目录运行 `python verify_artifacts.py` 可以核对本次提交的完整 SHA-256 清单。这个校验不执行模型，也不将研究结果认证为独立复现或同行评审结论。
