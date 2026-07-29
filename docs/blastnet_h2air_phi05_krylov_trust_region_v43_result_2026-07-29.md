# BLASTNet H2-air phi=0.5：四方向 Krylov warm start 的 v42/v43 结果

> 日期：2026-07-29  
> 证据角色：已经打开的数据上的机制诊断（post-open mechanism development）  
> 最终状态：`VALIDATED_POST_OPEN_BOUNDED_NEGATIVE_V43`  
> 突破状态：`algorithm_breakthrough=false`

## 这次真正检验了什么

当前主线不是直接训练一个更大的神经网络，而是先回答一个更基础、也更能决定后续算力是否值得投入的问题：

**在 Direct-K3 解之后，只允许使用同一条 straight-ray 逆问题产生的四个连续 CGLS 增量，并把它们按四个标量重新加权，是否能够同时达到或优于 Direct-K4 的场误差、梯度误差和 curved observation 误差门限？**

若这个四方向子空间里确实存在稳定通过三重门的系数，那么才有理由训练一个只看部署时可见 observation 的小模型来预测四个系数，把它变成 warm initializer。反过来，如果经过冻结范围内的强搜索仍找不到通过者，那么继续训练“系数预测器”只是在学习一个本身没有被证明有用的目标。

这次因此做了两层互补诊断：

1. v42 直接在精确 curved observation 目标上做受约束搜索，检验最直观的 oracle 优化是否能工作。
2. v43 在 v42 暴露出数值成本问题后，改用精确 JVP 构造局部 observation 模型，在固定四方向空间内执行有界 trust-region 搜索，并对每个候选重新做精确 curved forward 评分。

这两次都没有训练神经网络。这里检验的是“可学习目标是否存在足够的开发空间”，不是网络表达能力。

## v42：实际运行了，但计算预算耗尽，所以结论只能是 inconclusive

v42 先对每个快照执行了 323 个确定性 screen 点。四个快照均没有 screen 候选通过完整三重门。随后它以 screen 中最好的点作为起点，调用精确 curved-objective SLSQP，希望直接在四维系数空间中找到满足场与梯度约束、同时降低 observation 误差的候选。

实际运行结果是：每个快照的第一个 SLSQP 起点就耗尽了剩余预算。四个快照合计使用了 2800 次 curved objective forward 和 1508 次 reverse VJP，但四次优化都没有成功收敛到可判定解。独立重放完整复算了 screen、目标和门限，最大数值差为 0，因此这不是记录错误。

v42 的正确结论不是“四方向基底失败”，而是：

`POST_OPEN_CONSTRAINED_ORACLE_SEARCH_INCONCLUSIVE`

原因很具体：直接对 curved forward 进行带硬约束、带门限的 SLSQP，在当前目标的局部非光滑性和高调用成本下不是合适的数值实现。它能说明原方法的搜索器不可用，不能说明候选在数学上不存在。v42 正式运行约 1186.77 秒，独立重放约 1212.47 秒。这笔算力换来的主要信息，是及时停止继续堆叠同类 direct-autograd SLSQP。

## 为什么改成 v43

v43 保留了科学问题和候选空间，只替换造成 v42 卡死的优化数值路径：

- 初始点仍是 v42 的最佳确定性 screen 候选。
- 四个方向仍是同一 Direct-K3 后的四个连续 straight-ray CGLS 增量。
- 系数盒仍固定为 `[-2, 3]^4`。
- 场误差与梯度误差仍用精确二次形式约束，没有放宽门。
- 每个 trust-region 中心只计算一次精确 curved prediction，并对四个方向分别计算精确 `torch.func.jvp`。
- 在这个局部 affine observation 模型上解析构造 QCQP，再由 SLSQP 求四维 trust-region 子问题。
- 候选必须回到精确 curved forward 上重算；只有实际误差下降且实际/预测下降比不小于 0.1 才接受。
- 每个快照最多 8 个外层迭代，并设置独立的 F、JVP、VJP 硬预算。

这样做的理由不是让门变容易，而是把大量 reverse-mode 目标优化替换成可审计的局部方向导数，再用精确 forward 决定是否接受。v43 正式运行约 121.09 秒，独立重放约 118.78 秒，相比 v42 快约一个数量级，也没有再出现预算耗尽。

## v43 的逐快照结果

下表中的三个 ratio 都以同一快照的 Direct-K4 为 1。冻结门要求三者都不超过 1.01。小于 1 表示优于 Direct-K4，大于 1 表示更差。

| 快照 | 接受步数 | 最佳场 ratio | 最佳梯度 ratio | 最佳 observation ratio | curved 账本 | 完整门 |
|---|---:|---:|---:|---:|---:|---|
| S1 | 0 | 0.979208 | 1.010000 | 1.020448 | 8 F + 24 JVP + 0 VJP | 未通过 |
| S2 | 3 | 0.984410 | 1.010000 | 1.055484 | 5 F + 32 JVP + 0 VJP | 未通过 |
| S3 | 1 | 0.980775 | 1.010000 | 1.022709 | 9 F + 28 JVP + 0 VJP | 未通过 |
| S4 | 1 | 0.978627 | 1.010000 | 1.057592 | 9 F + 28 JVP + 0 VJP | 未通过 |

四个最佳候选的原始误差中位数为：

- field relative-L2：0.959613
- gradient relative-L2：0.980764
- observation relative-L2：0.313639

对应 Direct-K4 中位数为：

- field relative-L2：0.979420
- gradient relative-L2：0.971054
- observation relative-L2：0.301608

这说明搜索确实找到了“场值误差更低”的候选，但它们都被推到了梯度门的边界附近，curved observation 仍比 Direct-K4 差。四个快照的 observation ratio 全部大于 1.01，范围约为 1.0204 到 1.0576。也就是说，问题不是搜索完全没有移动；问题是移动后出现了稳定的场/梯度/观测权衡，完整三重门始终没有同时成立。

正式 v43 总成本为：

`31 F + 112 JVP + 0 VJP`

这个账本只是机制搜索成本，不能与部署算法的 A/A^T 调用账直接混写，也不能由此主张速度提升。

## 独立复算做了什么

独立 validator 没有信任正式 runner 写出的“通过/失败”标签，而是重新完成以下工作：

- 重新构造四个 CGLS 方向和所有二次约束量。
- 重新执行每一步 curved forward 与 JVP 轨迹。
- 重新求 trust-region 内部问题并重放接受/拒绝状态。
- 重新计算每个候选的场、梯度、observation 指标。
- 重新计算完整三重门和总调用账。
- 检查正式结果在复算前后没有被修改。

独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_KRYLOV_JVP_TRUST_REGION_ORACLE_V43`

逐行最大数值差、联合指标最大数值差和选中场最大绝对差均为 0。由此可以确认：v43 的“0/4 快照通过”是可复算结果，不是 runner 自己给自己的判决。

这里仍有明确边界：独立 validator 重放的是同一个冻结代理物理问题，它没有证明真实 BOST 光路、实验标定、噪声模型和组内数据上的端到端独立性。

## 得出的科学结论

在冻结的四方向 Krylov 子空间、系数盒、三重门和 v43 有界搜索范围内，没有找到任何完整通过候选。因此：

**不授权为这个固定四方向基底训练 observation-only 系数预测网络。**

这是一条有用的负结果，因为它阻止了后续把大量时间花在“预测四个系数”的网络规模、损失函数和调参上。v43 已经说明，即使 oracle 可以看见目标真值并使用精确 JVP，在四个快照上仍然遇到一致的 observation 失配。仅仅把 oracle 换成 FNO、DeepONet 或 MLP，不会自动消除候选空间本身的权衡。

但这不等于数学上的不可行性证明。v43 只排除了：

- 当前四个连续 CGLS 增量张成的固定四维空间；
- 当前 `[-2, 3]^4` 系数范围；
- 当前预注册 trust-region 迭代、接受规则和预算；
- 当前已经打开的 BLASTNet phi=0.5 四快照机制诊断。

它没有排除更换方向生成机制、增加真正携带 curved-observation 信息的新方向、改变状态表示，或在真实实验数据上重新定义等精度门。下一条值得投入的路线必须改变“基底能表达什么”，而不是在同一个四维盒里继续加大搜索器或训练器。

## 可以和不可以写进论文的内容

目前可以写：

- 直接 curved-objective SLSQP 在冻结预算内耗尽，v42 结论为 inconclusive。
- JVP trust-region 把同一机制诊断从约 20 分钟级缩短到约 2 分钟级，并完成独立逐步重放。
- v43 在四个快照上得到 0/4 完整门通过，所有 observation ratio 均超过 1.01。
- 结果支持停止固定四方向系数预测支线，转向新的方向/基底设计。

目前不可以写：

- 找到了可部署的新 warm-start 算法。
- 在同精度下实现了重建加速。
- 方法优于 Direct-K4、FNO、DeepONet 或其他神经算子。
- 已证明跨工况泛化、真实 BOST 有效或论文已经成功。
- 已证明四方向空间中数学上不存在可行解。

因此本轮没有突破性算法进展，`algorithm_breakthrough=false`。真正的产出是一个经过正式运行和独立复算的、能改变研究决策的负结果：固定四方向系数预测不再值得作为近期主线继续训练。
