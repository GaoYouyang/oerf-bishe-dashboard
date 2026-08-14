# v144 局部可辨识性诊断：当前 155D 邻域假设失败

> 日期：2026-08-14  
> 范围：五条已开封 PoolFire straight-ray 三维代理轨迹，5/7/9/12 个活动相机  
> 判决：`FAIL_LOCAL_OBSERVABLE_IDENTIFIABILITY_V144`  
> 边界：`algorithm_breakthrough=false`，不授权 GPU、神经训练、物理重放、资源门、外部门或真实 BOST 声明

## 先说人话

v143 已经证明：把原始系数换成更物理的 Riesz-action 坐标，也不能用一个跨轨迹共享线性模型稳定预测。v144 继续回答更关键的问题：会不会全局线性关系不存在，但相近的部署可见观测/几何输入，仍对应相近的 action 目标，因此一个小型局部非线性模型就够了？

我在看结果前固定了同一组 20 个哨兵、155 维 deployment-visible 特征、结构键内白化、邻居数量与距离规则。跨轨迹 kNN 只通过 `1/20`；即使把候选邻居限制到同一条轨迹，这个只用于诊断、不能部署的上限也只有 `8/20`。两种方法的五条轨迹 p90 都没有通过 `<=0.35` 门。

所以，问题不是“只差一张算卡”或“只差一个小 MLP”。当前 155D 逐方向局部度量没有把目标组织成稳定邻域；GPU 只会更快拟合一个尚未被证实可辨识的表示。这个结果关闭当前局部邻域假设，但不证明所有非线性模型或全局条件表示都失败。

## 实际结果

| 方法 | 哨兵通过 | 误差中位数 | 误差 p90-higher | 最坏误差 | 余弦中位数 | 轨迹尾部通过 |
|---|---:|---:|---:|---:|---:|---:|
| 跨轨迹 kNN | `1/20` | `0.60845` | `0.84100` | `0.90479` | `0.79359` | `0/5` |
| 同轨迹诊断 kNN | `8/20` | `0.44161` | `0.62566` | `0.63569` | `0.89720` | `0/5` |
| 结构均值 control | `0/20` | `0.91391` | `0.95657` | `0.96484` | `0.40587` | 不适用 |

冻结门是每个哨兵尺度不变 L2 误差 `<=0.45`、余弦 `>=0.90`，每条轨迹 p90-higher `<=0.35`，并要求 `20/20` 与 `5/5` 全过。

跨轨迹 kNN 在 5/7/9/12 相机组的通过数分别是 `0/5、1/5、0/5、0/5`。同轨迹诊断在四种相机数量中都是 `2/5`，说明失败并不只集中于一个相机数量。

## 为什么同轨迹诊断也重要

同轨迹 kNN 不是可部署算法，因为它允许查询同一条已开封轨迹中的其他时刻，只用于判断“局部连续性是否至少存在”。它的邻域半径确实更小：中位数从跨轨迹的 `1.10587` 降到 `0.69976`；误差也下降，但仍只有 `8/20`，五条轨迹尾部全部失败。

因此，跨轨迹失败不能只归因于工况域偏移。当前逐方向 155D 特征本身缺少能稳定决定 action 的非局部样本状态或全局相机集合上下文。

## 独立复算与透明审计

第二实现独立重建结构键、155D 特征、白化统计、邻居顺序、预测、指标和判决。所有整数邻居数组逐项相同，浮点数组最大绝对差为 `8.88e-16`，科学判决完全一致，正式结果树在验证前后不变。

初始独立报告仍被机械标成 `INCONCLUSIVE`，原因不是科学数组不同，而是验证器要求嵌套 JSON 中的轨迹尾部浮点数逐字相等；两套实现的最大差只有 `1.11e-16`。随后进行透明标注的结果后容差审计：不改邻居、预测、目标、阈值或判决顺序，只用结果无关的 `1e-12` 比较轨迹尾部，并保持数组 `1e-9` 门。审计通过，恢复由两套数组共同支持的负判决。

## 对“要不要租算卡”的直接结论

现在不租。正式运行与独立复算都是几秒级 CPU 诊断，新增精确调用为 `+0A/+0A^T`，训练参数为 `0`。当前瓶颈是表示的可辨识性，不是训练吞吐。

只有物理上不同、结果前冻结的全局 residual / camera-set 状态表示先证明目标可辨识，再冻结最小模型；若 CPU 原型随后显示训练吞吐成为瓶颈，才有理由租 GPU。

## 证据边界

- 这是已开封 PoolFire straight-ray 代理上的 post-open 机制诊断；
- 没有运行完整物理 replay，也没有得到可用 warm initializer；
- 没有 matched-accuracy 调用减少、wall time、RSS 或外部泛化结果；
- 不证明全局条件模型、所有非线性模型或数学上的不可辨识性；
- 不能写成算法突破、SOTA、真实 BOST 或论文成功。

## English checkpoint

v144 asks whether v143 failed only because the target is globally nonlinear while remaining locally identifiable. The experiment freezes the same 20 sentinels, a 155-dimensional deployment-visible feature, structural-key-local whitening, and a deterministic eight-neighbor rule before reading target outcomes. Complete-trajectory cross-trajectory kNN passes only `1/20` sentinels. A same-trajectory diagnostic, which is not a deployable model and serves only as a local-continuity upper bound, passes `8/20`. All five trajectory p90-higher gates fail for both methods. The structural-mean control passes `0/20`.

Independent code rebuilds structural keys, features, whitening, neighbor order, predictions, metrics, and decisions. Integer neighbor arrays match exactly and the maximum floating-point array difference is `8.88e-16`. The initial mechanical status remained inconclusive only because nested trajectory-tail JSON values were required to be exactly equal; their maximum difference was `1.11e-16`. A transparently labeled post-result tolerance audit changes no neighbor, prediction, target, threshold, or decision order and confirms the same scientific verdict: `FAIL_LOCAL_OBSERVABLE_IDENTIFIABILITY_V144`.

This closes the frozen local-neighborhood hypothesis for the current 155D per-direction representation. It does not reject every nonlinear or globally conditioned model. No physical replay, neural training, GPU rental, resource gate, external-generalization claim, curved-ray result, or real-BOST claim is authorized. `algorithm_breakthrough=false`.
