# PSU B0 可观测残差风险门：development 结论与 fresh 预注册

更新日期：2026-07-16

当前状态：`FRESH_OPENED_AFTER_FROZEN_COMMIT_CANDIDATE_PASS_SYNTHETIC_ONLY`

## 1. 为什么需要这一层

首轮正谱预条件器在熟悉分布上相对 validation-selected inverse-Sobolev
获得约 4% 的平均场误差增益，但在联合越界条件下三种随机种子全部失败。
active-view exact fallback 能阻止 4 至 5 视角越界，却不能识别仍处于 6 至
9 视角范围内的陌生流场形态。

因此下一候选不再继续盲目扩大网络，而是回答一个更具体的问题：

> 只利用部署时可获得的残差、精确伴随梯度、视角掩码和候选方向，能否在
> 不增加前向/伴随调用的情况下，识别学习预条件器可能伤害重建的样本？

这个问题与 learned preconditioner、selective prediction 和 conformal risk
control 的交叉有关。已有工作已经表明神经网络可以参与构造预条件器，但
这并不自动解决有限视角逆问题中的分布外可靠性：

- [Learning Preconditioners for Conjugate Gradient PDE Solvers](https://proceedings.mlr.press/v202/li23e.html)
- [DeepONet Based Preconditioning Strategies for Solving Parametric Linear Systems](https://epubs.siam.org/doi/10.1137/24M162861X)
- [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/)
- [On Learned Operator Correction in Inverse Problems](https://epubs.siam.org/doi/10.1137/20M1338460)

## 2. 候选算法

暂称 **Observable Conformal Residual-Risk Gate，OCRRG**。

1. 第一步仍计算原问题的精确伴随梯度。
2. 同时构造学习正谱方向和固定 inverse-Sobolev 方向。这一步只有轻量网络
   和 FFT，不调用 BOST 前向算子。
3. 提取 16 个无真值特征：active-view fraction、白化残差统计、两个像移分量
   的比例与相关性、梯度频谱质心/高频占比/各向异性、两个方向的夹角与相对
   修正量、学习谱增益跨度及 controller 系数尺度。
4. 标准化 ridge 模型预测“学习路径相对 Sobolev 的场误差增益”。
5. 使用每个冻结模型种子独立的 one-sided split-conformal overprediction
   quantile，形成预测增益下界。
6. 同时满足视角硬支撑、特征距离阈值和下界阈值时，整条四步求解使用学习
   方向；否则整条求解精确回退至 Sobolev。

部署调用预算仍为每个样本 `4F + 4A^T`。风险判断不需要先完整运行两种重建。

## 3. development-only 结果

数据使用真实 PSU support-view 几何和 32³ 解析反应流形态代理；没有使用 PSU
实验像移值，解析形态也不是 CFD。102 个独立场被拆分为：

- risk train：48 个场；
- risk validation：24 个场；
- risk calibration：30 个场；
- 每个场分别评估三个已经冻结的谱预条件器 checkpoint。

直接使用学习器时：

- train harm over 1%：20.8%；
- validation harm over 1%：16.7%；
- calibration harm over 1%：13.3%。

采用冻结风险门后：

- validation coverage：52.8%，mean selected gain：2.07%，harm：0%；
- calibration coverage：52.2%，mean selected gain：1.62%，harm：0%；
- 三种模型种子在 validation 和 calibration 上都保持非零覆盖且没有超过
  1% 的已接受伤害。

这些仍然只是 development 结果。它们只授权冻结 fresh 协议，不授权宣布
算法优越。

## 4. 已主动移除的潜在泄漏

最初的开发特征包含三项绝对 `sigma` 统计。合成数据中的 `sigma` 由每个样本
的 clean signal RMS 构造，可能隐含真值尺度。正式冻结前已经删除这三项，
改为 16 特征版本。删除后 calibration coverage 从 47.8% 上升至 52.2%，
harm 仍为 0%，因此没有保留这一不必要的信息通道。

真实实验迁移仍要求通过独立 flow-off repeats 估计 camera noise；不能把
合成相对噪声公式直接当作实验噪声标定。

## 5. fresh 集在冻结提交后打开

配置：
`demo_t16_operator/configs/psu_b0_residual_risk_fresh_prereg_v1.json`

七组各 24 个场：

1. familiar-family support IID；
2. 完全未进入 development 的 `vortex_pair` / `multi_plume`；
3. 更强 row-column correlation、view bias 与 signal-dependent noise；
4. 未见形态和强相关噪声联合越界，但仍有 6 至 9 个 active views；
5. 3 至 5 视角 geometry OOD；
6. 未见形态、强相关噪声和 3 至 5 视角 joint OOD；
7. QMC8 exact-operator control。

fresh support splits 强制保留非零 candidate coverage，不能靠全部回退通过；
3 至 5 视角组则要求 candidate coverage 精确为零并与 Sobolev 指标一致。
三种模型种子至少两种必须通过所有对应门。

冻结配置、checkpoint 哈希、development 风险模型与本文件先在 Git 提交
`cd5d4a0ffa997eb1d4f3820d0bc5e0aa626287df` 中提交并推送，随后才执行一次
fresh audit。三种冻结模型种子全部通过事先声明的合成候选门；完整判决见：

- [fresh 合成压力测试判决](psu_b0_residual_risk_fresh_result_2026-07-16.md)
- [公开结果 JSON](psu_b0_residual_risk_fresh_public_summary.json)

结果仍保留 4 条 accepted `>1% harm` 记录，集中在两个 6-view 样本，因此
状态只能是 synthetic-only candidate pass，不能提升为算法优越性或实验结论。

## 6. 证据边界

- development 和 fresh 都使用解析形态代理，不是实验场或 CFD；
- conformal 的交换性保证不能被外推到任意 OOD；
- 即使 fresh 通过，也只得到“真实 PSU 几何上的合成压力测试候选”；
- 与 FNO、DeepONet、FCG-NO 或 NeRIF 的公平优越性比较尚未进行；
- development rotation 40 和 final audit 均未打开；
- 私有 checkpoint、逐样本特征、mask、volume 和本地路径不进入 Pages。

## 7. 复现

development：

```bash
PYTHONPATH=. .venv/bin/python \
  site_tools/run_psu_b0_residual_risk_development.py \
  --root . \
  --config demo_t16_operator/configs/psu_b0_residual_risk_development_v1.json \
  --view-root <PRIVATE_PSU_VIEW_ROOT> \
  --checkpoint-dir <PRIVATE_CHECKPOINT_DIR> \
  --source-private-report <PRIVATE_SOURCE_REPORT> \
  --device mps
```

fresh 命令已经在预注册文件及其冻结 hash 提交后执行一次。后续验证器只读取
冻结私有报告并重建公开聚合，不得用重复 fresh 运行挑选更好随机结果。
