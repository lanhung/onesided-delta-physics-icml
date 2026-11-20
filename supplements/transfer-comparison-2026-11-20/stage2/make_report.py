"""Assemble the Chinese execution report from recorded results; runs no experiment."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABEL = {
    "frozen": "Frozen FNO", "delta": "Δ-Physics", "explicit": "显式修正",
    "lie_pre": "Lie 前置", "lie_post": "Lie 后置", "strang": "Strang",
    "implicit": "闭式隐式修正",
}
METHODS = ["frozen", "delta", "lie_pre", "lie_post", "strang", "implicit", "explicit"]


def table(headers, rows):
    return "\n".join([" | ".join(headers), " | ".join(["---"]*len(headers))]
                     + [" | ".join(map(str, row)) for row in rows])


def fmt(x, digits=6):
    return "失败，未给均值" if x is None else f"{x:.{digits}f}"


def main():
    agg = json.loads((ROOT / "burgers_fno/results/aggregate_summary.json").read_text())
    by = {(r["ratio"], r["method"]): r for r in agg}
    selected = {m: by[2., m] for m in METHODS}
    main_rows = []
    for m in METHODS:
        r = selected[m]
        av = ("失败，未给均值" if r["avg50_mean_over_training_seeds"] is None else
              f'{r["avg50_mean_over_training_seeds"]:.6f} ± {r["avg50_sample_sd_over_training_seeds"]:.6f}')
        st = ("失败，未给均值" if r["step50_mean_over_training_seeds"] is None else
              f'{r["step50_mean_over_training_seeds"]:.6f} ± {r["step50_sample_sd_over_training_seeds"]:.6f}')
        main_rows.append([LABEL[m], av, st, f'{r["failed_ics_total_over_seed_ic_pairs"]}/24'])
    main_table = table(["方法", "Avg-50：均值 ± 训练种子SD", "Step-50：均值 ± 训练种子SD", "失败 seed–IC 对"], main_rows)
    scan_table = table(["黏度比", *[LABEL[m] for m in METHODS]], [
        [f"{ratio:g}", *[fmt(by[ratio, m]["avg50_mean_over_training_seeds"]) for m in METHODS]]
        for ratio in [1., 1.25, 1.5, 1.75, 2., 3., 4.]])
    ablation = json.loads((ROOT / "burgers_fno/budget_ablation/aggregate_summary.json").read_text())
    ab = {(r["fixed_checkpoint_epoch"], r["method"]): r for r in ablation}
    budget_table = table(["方法", "epoch 15", "epoch 30", "epoch 50", "epoch 75"], [
        [LABEL[m], *[fmt(ab[e, m]["avg50_mean"]) for e in [15, 30, 50, 75]]]
        for m in METHODS])
    b1 = json.loads((ROOT / "cpu_benchmark_threads1/benchmark.json").read_text())
    b4 = json.loads((ROOT / "cpu_benchmark/benchmark.json").read_text())
    timing_rows = []
    for r in b1["summary"]:
        if r["batch_size"] != 8:
            continue
        timing_rows.append([LABEL.get(r["method"], r["method"]),
            f'{r["median_seconds_per_50step_rollout"]*1000:.3f}',
            f'{r["iqr_seconds"]*1000:.3f}', f'{r["avg50_rel_l2"]:.8g}',
            f'{r["step50_rel_l2"]:.8g}'])
    timing_table = table(["方法", "50步批量耗时中位数/ms", "IQR/ms", "Avg-50", "Step-50"], timing_rows)
    timing_by = {(r["batch_size"], r["method"]): r for r in b1["summary"]}
    direct_t = timing_by[8, "rk4_dt0.002"]["median_seconds_per_50step_rollout"]
    delta_factor = timing_by[8, "delta"]["median_seconds_per_50step_rollout"]/direct_t
    strang_factor = timing_by[8, "strang"]["median_seconds_per_50step_rollout"]/direct_t
    thread_table = table(["方法", "Torch1线程 / FFT1worker：ms", "Torch4线程 / FFT1worker：ms"], [
        [LABEL.get(m, m),
         f'{next(r for r in b1["summary"] if r["batch_size"]==8 and r["method"]==m)["median_seconds_per_50step_rollout"]*1000:.3f}',
         f'{next(r for r in b4["summary"] if r["batch_size"]==8 and r["method"]==m)["median_seconds_per_50step_rollout"]*1000:.3f}']
        for m in ["frozen", "delta", "lie_pre", "lie_post", "strang", "implicit", "rk4_dt0.002"]])
    with (ROOT / "ns_controls/results/summary.csv").open() as f:
        ns = list(csv.DictReader(f))
    nby = {(int(r["grid"]), float(r["ratio"]), r["method"]): r for r in ns}
    ns_rows = []
    for m in METHODS:
        vals = []
        for ratio in [1.5, 2., 4.]:
            r = nby[128, ratio, m]
            vals.append("3/3 失败" if int(r["failure_count"]) else f'{float(r["avg50_rell2_mean"]):.8g}')
        ns_rows.append(["Frozen 数值参考流" if m == "frozen" else LABEL[m], *vals])
    ns_table = table(["方法", "黏度比1.5", "黏度比2", "黏度比4"], ns_rows)

    report = f"""Δ-Physics 补充实验：执行结果、证据范围与下一阶段方案
实际执行日期：2026年11月20日（Asia/Shanghai）
配套目录：delta_physics_stage2/

一、已完成工作的结论

本次已经执行闭式隐式修正、Lie 前置/后置、Strang、显式修正与 Δ-Physics 的统一比较，并新增真实的三训练种子、75-epoch Burgers FNO 实验；同一训练轨迹的 epoch 15/30/50/75 消融；单线程与四线程 CPU 实测；以及 64×64/128×128 Navier–Stokes 数值参考流控制。

目前最有决策价值的结果是：
1. 在独立重建的 Burgers FNO 上，Δ-Physics 显著优于不修正的 Frozen，但 Strang、Lie 与闭式隐式修正都构成实质竞争。黏度比2时，Strang 的 Avg-50 在三个训练种子上都优于 Δ，逐种子相对下降平均28.40%；Step-50 的跨种子均值则是 Δ 略低，不能宣称 Strang 全面支配。
2. 闭式隐式修正已获得真实误差与耗时，而不再是待测空白。它每个物理步只调用一次网络，单线程小网格 CPU 成本与其余单次调用方法接近。
3. 同一 CPU、float32、单线程、batch=8、黏度比2的64点 Burgers 上，优化后的频域 RK4 比 Δ/Strang 更快且相对同一半离散参考解更准确。这一配置没有显示神经方法对直接求解的推理优势。
4. 精确度很高的 NS 数值参考推进器上，Strang 明显优于加性方法；这说明经典分裂比较必要，不能据此替代原论文 F-FNO 上的实际结果。

证据边界：本次 Burgers 模型是依据可见描述独立实现、重新生成源数据并训练的 FNO，属于 independently_reconstructed_neural。没有获得原论文的 NS/F-FNO 源代码、原始75-epoch权重、源归一化器及原始测试轨迹，因此没有把这些结果标为原论文复现。NS 控制没有神经网络。CUDA 不可用，GPU 实验没有执行。

完成规模：
- Burgers 主实验：3训练种子 × 7黏度比 × 7方法 = 147组合；每组合8个测试初值，50步；1176条轨迹，58800行逐步指标。
- 训练预算诊断：3训练种子 × 4固定epoch × 7方法 = 84组合；672条轨迹，33600行逐步指标。75-epoch快照与主实验重合，不能再算独立重复。
- NS 数值控制：2网格 × 3黏度比 × 7方法 × 3初值 = 126条轨迹；6300行逐步指标。
- CPU benchmark：每种线程设置18个方法×batch配置，每配置3次完整预热、10次完整正式计时，共180条正式计时记录；两种线程设置分别保留。

二、Recast 公式必须按当前 PDF 区分

实际核对的是 onesided-delta-physics-icml-v2.pdf，14页，SHA256：
ca6f5da8564eb7287d14808be5633a09ff5495996a98e118696471219ecdc473

该版本 Appendix C 将固定点写作：
    y = Φθ(u + h(ν−ν0)Δy)。
未知量 y 位于网络输入内部。每次迭代输入改变，不能把 Φθ 当作与 y 无关的常量缓存。文中采用 vector Aitken relaxation；缺少实际原始实现时，不能随意换成另一种迭代器并复用论文误差/调用次数。

本次补充的闭式隐式方法则定义为：
    y = Φθ(u) + h(ν−ν0)Δy，
    Fourier(y)(k) = Fourier(Φθ(u))(k) / [1+h(ν−ν0)|k|²]。
它是不同的、有必要比较的单次调用基线；不是该 v2 输入型 Recast 的闭式求解。因此，旧表 Recast Avg-50=0.17859 不能赋给本方法，原8.81次调用也不能仅凭外置式推导宣布是重复计算错误。

原稿针对特定迭代实现的调用次数差异仍需原始代码验证；但一旦加入一调用的 Lie/Strang/隐式方法，约9倍调用数差异就不足以单独论证 Δ 是最有效的物理修正形式。

三、统一方法定义与实现

记 h 为冻结网络所学的一整个物理时间步，B=(ν−ν0)Δ，Sν=exp(hνΔ)。本实验只对 ν≥ν0 运行耗散参数主比较。

方法 | 物理状态的一步映射 | 每步网络调用
--- | --- | ---
Frozen | Φθ(u) | 1
Δ-Physics | Φθ(u)+(Sν−Sν0)u | 1
Explicit | Φθ(u)+hBu | 1
Lie 前置 | Φθ(exp(hB)u) | 1
Lie 后置 | exp(hB)Φθ(u) | 1
Strang | exp(hB/2)Φθ(exp(hB/2)u) | 1
闭式隐式 | (I−hB)^−1 Φθ(u) | 1

Strang 中间仍为网络完整的一步 h，不需要、也没有虚构 h/2 的神经网络。FFT 使用物理波数2πn/L；输入、输出及扩散都按物理状态解释。网络若预测增量，则先按训练时固定归一化还原完整状态；不对坐标通道做扩散，不把物理参数直接作用在未经正确还原的增量上。所有方法共用固定源统计，没有根据目标轨迹重新拟合归一化。

共168个解析 CPU 检查覆盖 float32/64、一维/二维、非单位长度、长方形和奇数网格、多batch与多通道、源参数恒等性、单模态解析响应、一次调用计数、输入不变和负方向保护。最大解析相对误差：float32为4.12078e−7，float64为4.84040e−15。低黏度加性传播子差的有界性控制误差为2.64220e−16；它不证明真实神经 rollout 在该方向稳定。

四、Burgers 真实神经实验的设计

方程：u_t + (u²/2)_x = νu_xx，周期单位区间；64点；ν0=0.01；h=0.02；50步至T=1。
初值：四个随机正余弦模态，系数按模态编号衰减，去均值后 RMS=0.5。
数据：24条源训练轨迹、6条源验证轨迹，每条20 transitions，共480训练样本；8条独立测试初值在所有方法和黏度中配对复用。
种子：训练种子0/1/2；训练初值1000–1023，验证初值2000–2005，测试初值3000–3007。三个训练种子共用同一源数据集，只改变初始化/训练随机顺序。
模型：标准一维 FNO 独立实现，width32、modes16、四个 Fourier+pointwise 层、GELU、128宽输出头。输入归一化状态与坐标，输出归一化增量，再解码为物理状态。74209个参数张量元素；因 Fourier 权重是复数，实参数自由度为139745。
训练：75 epochs，Adam，lr=1e−3，weight decay=1e−6，batch32；预先固定 StepLR，每25 epochs乘0.5。仅以源验证集一步 RelL2 选择 checkpoint。这个具体实现和调度不是对原论文未提供代码的逐项复现。
全部目标数据在三个模型训练完成后生成，没有用于选择权重、调参或调节修正系数。三个源验证最优 checkpoint 都在 epoch75，对应一步验证 RelL2 为0.00262163、0.00253448、0.00221408。末端曲线仍下降，应称75-epoch模型，不能据此宣布充分收敛。
计算：网络float32、Fourier权重complex64，真值和指标float64。RK4真值在64点半离散方程上进行时间细化；目标步长0.00025对0.000125的所有黏度、时间和初值最大相对差≤1.053e−10。这个检查不建立连续PDE的空间收敛性。

指标：对每个训练种子s、测试初值i、输出步t计算 e_sit=||预测−真值||_L2/||真值||_L2；Avg-50先对8初值×50步平均，Step-50先对8初值末步平均，再报告三个训练种子的均值与sample SD（分母n−1）。时间点与初值不是额外的训练种子。SD不是95%置信区间，本次没有据3个种子作显著性宣称。
改进百分比在同一训练种子内先计算，再跨种子平均；因此它可能不同于两个跨种子均值直接相除所得的百分比。
逐步记录还包括绝对L2、参考L2、谱功率相对误差PSRE和均值漂移。在强耗散导致参考解变小时，需结合这些量解读相对误差。
失败：预先固定为非有限值或max|u|>10^4。失败后保持NaN及首次失败步；只要完整均值所需样本缺失，就返回null，不删除失败轨迹后算“幸存者均值”。

五、Burgers 主结果

黏度比2，三训练种子，8个配对测试初值：

{main_table}

Δ 相对 Frozen 的 Avg-50 逐种子改善平均92.29%。Strang 相对 Δ 的逐种子改善分别28.68%、32.72%、23.81%，平均28.40%；隐式修正相对 Δ 平均改善24.18%。这些是本次独立重建数据，不能替换原论文中 NS 或 Burgers 的数字。

Step-50的Δ均值0.030129略低于Strang的0.030860；逐种子的排序也不同。完整曲线显示，Strang的优势主要体现在部分前中段滚动，而终点差异较小且不一致。论文应同时保留轨迹平均、末步、失败率与曲线，避免用一个有利指标宣布全面胜出。

全部黏度的 Avg-50 跨种子均值：

{scan_table}

在黏度比1时，全部七个方法与 Frozen 逐元素一致，构成不改变源域预测的检查。显式方法在1.25时24个seed–IC对均保持有限；在1.5、1.75、2、3、4时全部24/24失败。这是指定h、网格和模型的实测失败，不是对所有显式修正情形的普遍否定。

六、训练预算消融

同一次75-epoch训练的固定epoch15/30/50/75权重快照，在黏度比2上执行同样的测试。它不是分别重新训练的四组实验，也不是各预算内择优结果；这些相关快照不能被解释成12个独立训练种子。

{budget_table}

Strang相对Δ的Avg-50逐种子改善均值依次为19.36%、20.77%、23.51%、28.40%。源域验证一步误差的跨种子均值依次为0.00390756、0.00305105、0.00265165、0.00245673。这支持在本设置中加强源训练后，经典组合仍是必要且很强的对照；不支持把原NS/F-FNO的排名直接外推出来。

七、真实计算效率：直接求解器的比较改变了论证重点

硬件标识：AMD EPYC 9V74 80-Core Processor；容器可见logical CPU数9。Python3.12.14，Torch2.7.1+cpu，NumPy2.3.5，SciPy1.17.0；CUDA不可用。
基准：训练seed0、64点、黏度比2、相同h=0.02、50个物理输出、batch1和8；3次完整预热、10次完整重复，每轮打乱方法顺序。所有其它实验进程已经结束后才运行计时。
计入：初值复制、源归一化、物理修正、全部50个物理时刻的输出计算、最终状态保留。
不计入：训练、权重/数据载入、真值生成、指标计算、逐步失败检查、全轨迹堆叠保存。每种方法按同样的最终状态保留策略计时，RK4并未省略50个输出时刻的物理场计算。
每种网络方法由实际hook计得50次batch网络forward。直接RK4没有网络调用。显式修正在独立的全轨迹精度预检中失败，故保留失败记录并不报告成功推理耗时。
直接求解器：float32/complex64频域状态、SciPy rFFT、同一保守Burgers非线性、同一2/3非线性滤波、Nyquist一阶导数处理与全波数Laplacian。内部dt=0.002、0.001、0.0005；50步输出时间完全一致。

主计时表采用Torch1线程与SciPyFFT1worker。batch=8，每一行的准确率来自这个seed0的完整轨迹预检，不是三种子主结果均值：

{timing_table}

RK4内部dt=0.002的中位时间为0.042534s，Δ为0.069266s，Strang为0.069727s；RK4相对两者分别约{delta_factor:.2f}倍、{strang_factor:.2f}倍快，其Q3也低于两者Q1。RK4的Avg-50相对误差3.48084e−7，而Δ/Strang约2.10083e−2/1.49828e−2。

这证明本次小网格CPU配置尚未显示神经方法相对直接求解的优势。结果限于具体硬件、代码、精度、batch及问题规模，不是算法复杂度界，不代表GPU、NS或高分辨率的速度关系。训练成本没有计入；也没有借本结果推断其摊销收益。

线程敏感性，batch=8中位时间：

{thread_table}

四线程FNO结果具有更宽的IQR，且小模型多线程开销明显；保留该组作为敏感性结果，用单线程比较作为主结论。不同一调用包装器的计时区间相互重叠，不宣称某个方法有可靠的小幅速度优势。

float32直接RK4的更小内部步长没有单调降低观测误差，结果与舍入误差累积相容；不能把这三档计时数据当作严格收敛阶证据。独立的float64真值细化检查承担数值参考可信度验证。频域float32 RK4与原float64物理状态RK4的单输出步交叉核对最大相对差为1.97861e−7。

八、64×64 / 128×128 Navier–Stokes 数值参考控制

这部分有意使用高精度数值源黏度流映射代替网络，用来隔离修正方式本身的误差。三个seeds是初值seeds，而不是训练seeds。ν0=0.001、h=0.05、50步至T=2.5，黏度比1.5/2/4；两个网格从同一连续低模态Fourier初值采样。
强迫为0.1[sin(2π(x+y))+cos(2π(x−y))]；单位方形周期边界。源和目标采用去混叠频谱RK4；非线性输入/输出采用严格2/3滤波，扩散保留全Laplacian，输出没有偷偷做额外全场投影。

128×128的Avg-50，三个初值均值：

{ns_table}

64×64在黏度比2下，Δ=0.0073231870、Lie后置=0.0014361369、Strang=0.0000134112、隐式=0.0018825113，与128×128的对应值非常接近。它支持接近精确的源推进器下Strang是强竞争方法；不能据此说明神经模型必然具有同等改善比例。

全矩阵精度检查：目标dt减半最大相对差1.82933e−12；源黏度完整rollout的dt减半最大差1.76571e−11；各方法实际源推进器输入探测的最大差8.29003e−9，全部低于预先设置1e−7阈值。预跑源dt=0.005在显式路径未达到该阈值，统一减至0.0025后重跑全部矩阵；依据是数值精度而非方法名次，说明单独归档。

64与128目标解共同频带，在已保存的0/1/10/25/50步及三个初值上，黏度比1.5/2/4的最大相对差分别8.79827e−9、9.27321e−10、1.44710e−12。限制操作使用正确FFT归一化并去除粗网格Nyquist轴。初值最高模态仅4，在64点已经得到充分分辨；这是分辨率一致性控制，不是困难高分辨率任务。没有256网格锚点，因此也没有宣称128空间收敛已获证明。

显式方法首次失败步：
网格 | 黏度比1.5 | 黏度比2 | 黏度比4
--- | --- | --- | ---
64×64 | 50步内均未触发 | 35/35/34 | 19/19/19
128×128 | 23/23/23 | 17/17/17 | 12/12/12

在非线性滤波截止以外，显式方法的线性放大因子为g(q)=R4(−ν0qδt)^n−h(ν−ν0)q。黏度比1.5时最高模式|g|由64网格约2.004增至128网格约8.085，解释高频失稳在高分辨率更早显现的机制。64网格暂时未超过失败阈值不是稳定性证明；放大因子分析也不精确预测实际失败时刻。

九、统一评测与交叉核对

runner/evaluate.py要求显式声明证据类别、h、黏度、空间长度/网格、物理通道、dtype、唯一轨迹ID，以及权重/归一化器/数据的SHA256。固定网格测试数据格式是[T+1,B,C,*space]。真实数据不齐时拒绝生成带“original_neural”标识的结果。
该入口已实际加载本次Burgers训练seed0，在黏度比2、8初值、全部7方法上执行50步。独立适配器得到的全部rollout与训练脚本逐元素相同，包括Explicit失败后的NaN位置；Avg-50/Step-50标量最大差5.55112e−17；每种方法实际forward为50。
参考真值保留float64，神经推理保留float32和complex64 Fourier权重。不能用全模型.to(dtype=float32)将复数权重虚部丢掉。相对误差零分母一律记null并保留绝对误差；包括0/0也不擅自填0。原始Recast/Aitken未实现时明确拒绝，不用外置implicit替代。

十、下一阶段：按决策问题扩展

优先级0：接入原始NS/F-FNO。
需要模型定义及物理包装代码、原75-epoch checkpoint（至少一个，主结论需3训练seed）、源归一化器、测试初值/目标轨迹或可重生的准确求解器配置、原Recast/vector Aitken实现。先复现原Frozen/Δ数值，再在同一权重/轨迹上运行本次7方法及真实Recast。计划黏度比1/1.25/1.5/1.75/2/3/4，主50步，延长100/200步。保存一整条轨迹的误差、绝对范数、谱、失败步，不只保留终点。

优先级1：将两种高分辨率问题分开。
A. 原生128模型：在128源黏度数据上训练3个模型，然后冻结并改变黏度。增加具有更宽实际能量带的初值；选若干轨迹以256网格作空间锚点，并做独立时间细化。
B. 64→128零重训：在模型结构允许时将原64冻结权重用于128，保留源统计。先报告ν=ν0时的纯分辨率误差，再报告黏度迁移附加误差；不能把它与A合并为同一种实验。网格改变后的FFT模式、padding、坐标构造必须按实际架构核对。
扩展batch1/8/32、CPU/GPU（有相应硬件才执行）、相同物理时长和输出频率；比较相同误差目标下的直接求解成本。预先检查目标真值的高频尾部及细化差，避免只把低模态平滑问题放到更大数组上。

优先级2：更换耗散算子，先做hyperviscous NS。
使用固定算子−(−Δ)^2，频率符号−|k|^4，训练新的源κ0模型后冻结，再改变κ/κ0=1/1.25/1.5/2。κ单位是长度^4/时间，不能与普通ν的数值直接视为同一黏度。真值采用经过验证的指数/隐式积分，或证明稳定且细化充分的RK4。当前Laplacian包装器没有直接实现该扩展，需要明确替换算子接口并进行单模态验证。
若进一步研究普通Laplacian→hyperviscosity的算子形状变化，应单独定义Ltarget−Lsource；该差的谱可能并非全为耗散，不得沿用单侧扩散的假设。

优先级3：反应–扩散。
建议双组分Gray–Scott，先固定反应参数、共同改变两个扩散系数，再只改变一个组分。每个新PDE训练自己的源模型后冻结；保持反应项与源归一化不变。报告分组分误差、形态长度尺度、相位敏感的场误差、负浓度比例。当前包装器不含分组分扩散，必须扩展并验证；若加入正值投影，应对所有方法采用公开一致策略，不能隐藏裁剪。

后续：KdV与Kuramoto–Sivashinsky。
KdV检验解析传播器的相位变化，报告谱相位和不做平移对齐的场误差；可另外报告对齐误差但不能替代主误差。酉的线性流不自动保证加性传播子差的整体能量稳定。
KS包含增长与衰减模态，分开短时误差、预测视界与长时统计，并检查隐式分母、分裂放大因子及非线性作用。原负半定耗散命题不可直接沿用。两项都属于计划，尚未执行。

方法选择规则：优先以原模型上配对seed的Avg-50/Step-50、长时失败率和实际延迟共同判断。若指标互有胜负，保留条件性结论；不根据单个有利指标宣布唯一最优方法。没有目标训练数据仍可作冻结迁移研究，但不能据此免除与经典数值方法的比较。

十一、对论文定位的建议

本次结果使“继续强化单一加性公式并以9倍网络调用优势作为主贡献”的路线风险增大。更可检验的问题是：解析物理传播器应怎样与存在误差的冻结时间推进器组合，才能获得可靠的参数迁移？

保留Δ作为重要的一调用基线，同时将Lie、Strang、闭式隐式与原始输入型Recast纳入统一比较。解释为什么精确源流时Strang非常强，而在不完美网络上其优势只在部分轨迹指标上体现。Lie后置同样将当前原始状态送入网络，所以“避免预先改变网络输入”不是Δ相对所有分裂法的独占优势。
局部误差的阶数论证必须区分解析流的截断误差与冻结网络误差；本次NS控制只在固定h比较大小，没有新建时间步收敛阶实验，不能把它写成已实测二阶收敛。
对低黏度方向，传播子差有界不等于整个神经rollout稳定，也不等于真正的逆向热算子。将单侧策略表述为在所测设置下由经验稳定性支持的策略，保留其适用范围。

十二、文件、复现和证据索引

本报告数值全部由配套实际输出汇总，既有论文信息只用于公式核对与后续接入。图和表不含预期或虚构实验值。
- burgers_fno/results/aggregate_summary.json/.csv：主表，49行跨训练种子聚合。
- burgers_fno/results/per_seed_summary.csv：147行独立模型种子×黏度×方法结果。
- burgers_fno/results/per_step_metrics.csv：58800行，含失败状态、绝对/相对误差及谱指标。
- burgers_fno/results/seed_0/、seed_1/、seed_2/：实际权重、源验证曲线、完整预测、训练元数据。
- burgers_fno/results/source_data.npz、truth_r*.npz、normalization.json、locked_config.json：完整数据与设置。
- burgers_fno/budget_ablation/：固定epoch预算诊断。
- cpu_benchmark_threads1/、cpu_benchmark/：计时原始重复、准确率、运行版本和各自实际执行脚本快照。
- ns_controls/results/：正式数值参考控制、收敛验证、50步指标、五个时刻的场快照、显式失稳诊断。
- runner/：统一入口、数据约定、真实重建模型适配器、交叉核对记录。
- shared/、validate_torch_cpu.py、torch_cpu_checks.json：修正器实现与解析验证。
- figures/：可直接导出的PNG/PDF图；每张图标识证据类别。
- next_experiments.json：未执行的后续矩阵。
- execution_manifest.json：包内容、SHA256及验证记录。
- README.txt、requirements-stage2.txt：完整运行命令和已用环境版本。

交付包保留正式输出、代码、权重与完整数据；排除可重生的缓存、运行日志和不达数值门槛的预跑完整副本，但保留修改数值步长的原因说明。原论文PDF只以文件名与SHA引用，不在新实验包中重复嵌入。

所有本次训练与计算已结束；未启动后台长时任务。后续原论文NS比较需要上述原始资产，不能以本次独立重建结果代替。
"""
    (ROOT / "delta_physics_stage2_report_zh.txt").write_text(report, encoding="utf-8")
    exports = ROOT / "tables"
    exports.mkdir(exist_ok=True)
    for source, name in [
        ("burgers_fno/results/aggregate_summary.csv", "burgers_fno_results.csv"),
        ("burgers_fno/budget_ablation/aggregate_summary.csv", "burgers_fixed_epoch_results.csv"),
        ("ns_controls/results/summary.csv", "ns_numerical_control_results.csv"),
    ]:
        (exports / name).write_bytes((ROOT / source).read_bytes())
    all_timing = []
    for thread_count, folder in [(1, "cpu_benchmark_threads1"), (4, "cpu_benchmark")]:
        with (ROOT / folder / "summary.csv").open() as f:
            for row in csv.DictReader(f):
                all_timing.append({"torch_threads": thread_count, "scipy_fft_workers": 1, **row})
    with (exports / "burgers_cpu_timing.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_timing[0]))
        writer.writeheader()
        writer.writerows(all_timing)
    print("Wrote report and four table exports from recorded outputs.")


if __name__ == "__main__":
    main()
