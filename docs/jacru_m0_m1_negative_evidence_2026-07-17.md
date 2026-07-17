# JACRU M0-M1 负证据判决：为什么主线转向“物理底座 + 学习残差”

> 更新时间：2026-07-17  
> 证据等级：E1，独立解析梯度 renderer 的开放 development 诊断  
> 当前判决：**NO-GO / REVISE**。不允许声称 JACRU 优于 CGLS、Huber-PDHG、FNO、
> DeepONet、NeRIF 或任何实验方法。

## 先讲结论

第一版 JACRU 试图同时从观测中优化三维场、界面、跳跃强度与 camera bias。它在当前小型
独立 renderer 考卷上明显输给经典基线。随后发现，最漂亮的界面 F1 不是算法从观测中学到
的：固定 `x` 方向的初始平面在读取观测前就对两个单界面样本得到 `F1@1dx = 1.000`，最终
反而降到 `0.974`。因此所有“界面增益”已经撤回。

修复尺度与初始化后，场误差从 `1.988` 降到 `0.769`，说明实现诊断有效，但仍比
Huber-PDHG 的 `0.480` 差 60.18%。M1 再把 24 对物理调用拆成 18 对 CGLS 底座与 6 对
JACRU 残差更新，场误差降到 `0.495`：比纯 CGLS 的 `0.499` 好 0.78%，却仍比
Huber-PDHG 差 3.11%，H1 还差 15.74%。这不是方法胜出，但它给出一个清楚的路线信号：

> **不要让神经网络或逐样本 Adam 重做经典反演已经会做的部分；让稳定物理求解器负责主场，
> 跨样本算子只学习经典底座在界面、相机变化和模型失配下的结构化残差。**

## 考卷怎样避免最窄的 inverse crime

观测端调用 `analytic_bost_phantoms.py` 的连续解析梯度并沿射线积分；逆问题端使用独立的
有限差分、三线性插值体素算子 `PSUB0VoxelGradientOperator`。两侧共享射线几何，这是同一
实验必须共享的量，但不共享生成观测的离散前向链。

当前开发考卷为：

- `12 x 12 x 12` 三维体素；
- 3 台相机，每台 `6 x 6` 射线，共 108 条射线、216 个位移分量；
- 每条射线的解析 renderer 使用 24 个积分样本；
- 两个 development seed：101、211；
- 两类形态：平滑无界面、单界面；
- 1% 相对观测噪声与 2% camera-wise bias；
- 每个方法严格使用 24 次 forward 与 24 次 reverse/adjoint；
- 体场真值只进入最终评分，不进入任何重建 API。

这里仍然只有四个开发样本，不能用来判断总体泛化；它的用途是尽早淘汰明显不成立的机制。

## 四轮结果放在同一张表里

| 轮次 | 候选 | 平均 field relative-L2 | H1 相对误差 | measured reprojection | 对最强经典场基线 | 判决 |
|---|---|---:|---:|---:|---:|---|
| M0 | JACRU + bias | 1.987836 | 3.174674 | 1.913683 | 比 Huber 差 314.03% | NO-GO |
| M0.1 | 修复尺度、随机平面、关闭初始 gate | 0.769049 | 1.138376 | 0.408105 | 比 Huber 差 60.18% | NO-GO |
| M1 | CGLS-18 + JACRU-6，无 bias | 0.495027 | 1.011757 | 0.010213 | 比 CGLS 好 0.78%，比 Huber 差 3.11% | NO-GO |
| 经典基线 | CGLS-24 | 0.498911 | 1.022353 | 0.003739 | 数据拟合最好，场不如 Huber | baseline |
| 经典基线 | Huber-PDHG-24 | **0.480119** | **0.874186** | 0.114924 | 当前场与 H1 最强 | champion |

所有数值来自同一批开发 case；M0.1 和 M1 已经看到 M0 的结果，属于 post-open diagnosis，
不能当作独立确认。完整机器可读摘要位于：

- [M0 摘要](../demo_t16_operator/results/jacru_m0_synthetic_gate_public/summary.json)
- [初始化审计](../demo_t16_operator/results/jacru_m0_initialization_audit_public/summary.json)
- [M0.1 摘要](../demo_t16_operator/results/jacru_m0_1_diagnostic_public/summary.json)
- [M1 摘要](../demo_t16_operator/results/jacru_m1_cgls_residual_diagnostic_public/summary.json)
- [跨报告 validator](../demo_t16_operator/results/jacru_m0_m1_evidence_validation.json)

## 那个“79.68% 界面 ASSD 增益”为什么作废

M0 的自动 gate 曾显示单界面 ASSD 改善 79.68%、F1 绝对提高 0.444。初始化审计随后做了一件
很简单、也很致命的检查：在任何观测进入算法之前直接给初始 level set 打分。

| 检查 | 结果 |
|---|---:|
| 两个单界面 case 的初始 `F1@1dx` | 1.000 |
| 最终 JACRU 平均 `F1@1dx` | 0.973958 |
| 最终相对初始变化 | -0.026042 |
| 128 个随机平面平均 F1 | 约 0.40 |
| 固定平面在随机平面分布中的百分位 | 100% |
| 平滑场初始假阳性率 | 100% |

生成器的界面恰好近似沿 `x` 方向，而 JACRU 默认初始化也是 `x + 0.05y - 0.03z = 0`。
因此这里不是普通随机波动，而是 data-free geometry alignment。修复规则已经冻结：

1. 初始化方向必须随机化或由严格 train-only 规则产生；
2. deployment gate 必须从阈值以下开始；
3. 界面指标必须报告“最终减初始”，不能只报最终绝对值；
4. 无界面场要报告假阳性；
5. fresh case 不得为提高界面分数选择初始化。

## M1 的 0.78% 为什么仍有价值

`0.78%` 远小于论文方法门，而且没有赢 Huber-PDHG，所以不能写进摘要当性能贡献。它的价值
在于比较了两种优化职责分配：

- M0/M0.1：候选负责从近乎空白状态优化整个三维场；
- M1：CGLS 先做 18 对稳定物理更新，候选只拿剩余 6 对调用修正残差。

M1 几乎恢复 CGLS 的场误差和重投影，而 M0.1 仍明显发散。这支持继续测试“经典底座 + 小残差”
的工程路线，但并不证明残差必须由神经网络、level set 或 JACRU 表示来学习。

## M2 要改变什么

M2 不再是逐样本拟合参数，而是一个真正跨样本训练的算子：

```text
y, geometry
  -> fixed-budget CGLS base x0
  -> per-view adjoint lift of y - A x0
  -> permutation-invariant geometry/set encoder
  -> bounded support-limited residual delta x
  -> x0 + gate * delta x
  -> optional fixed data-consistency correction
```

必须同时满足：

1. **零初始化回退：**训练前输出逐位等于 CGLS；失败可退回底座。
2. **不读取真值：**模型 forward API 不接受 truth、family label 或 interface mask。
3. **跨几何而非固定布局：**相机顺序置换不改变输出；pose 和 active mask 明确进入模型。
4. **残差有界：**修正受 support、尺度与 gate 限制，避免 OOD 时覆盖整个物理解。
5. **两种损失：**field/H1 监督只在 train 使用；data consistency 对所有训练样本可计算。
6. **强基线：**至少比较 CGLS、Huber-PDHG、参数匹配 3D CNN、FNO 和 DeepONet。
7. **同预算账本：**报告 reconstruction F/A 调用、训练 FLOPs/时间、参数量、峰值内存和推理时间。
8. **独立 renderer：**训练、development、OOD、fresh 均继续使用解析观测端与离散逆端分离。

## 下一轮可证伪门

M2 只有同时达到下列开发门，才允许冻结后打开一次 fresh：

| 门 | 最低要求 |
|---|---|
| 平均场误差 | 相对最强经典基线至少改善 5% |
| H1 / front | H1 至少改善 3%，且单/双界面 F1 不下降 |
| 多种子 | 至少 3 个训练 seed，paired gain 的方向一致 |
| OOD | camera count、pose、noise、bias 与 morphology OOD 均不得出现系统性反转 |
| 尾部风险 | `>1%` field harm 不超过 5%，worst case 不低于 -5% |
| 重投影 | 不得比 CGLS base 恶化超过 10%；若恶化，必须有清楚 Pareto 图而非隐藏 |
| 消融 | 去掉 geometry、set aggregation、zero-init 或 residual input 后，关键优势应消失 |
| 成本 | 参数、训练时间与 F/A 调用完整披露；不能把离线求解成本藏在 input preprocessing |

若 M2 只赢 CGLS、不赢 Huber-PDHG，结论仍是 NO-GO；若只赢固定几何、不赢 pose OOD，
不能声称通用算子；若只在一个 seed 上成功，不打开 fresh。

## 现在可以和不可以对师兄说什么

可以说：

- 已建立不共享离散出题链的最小三维 BOST 考卷；
- 经典 CGLS 和 Huber-PDHG 在相同 24 对物理调用下可重放；
- 逐样本全场 JACRU 已被证伪，固定界面初始化泄漏已定位；
- CGLS 底座加小残差比从零联合优化稳定，下一步转向跨样本残差算子；
- 当前 Mac 足够做 `12^3` 机制筛选，后续再为大网格租 GPU。

不可以说：

- JACRU 已经找准界面；
- M1 已优于现有算法；
- 当前结果证明能迁移到 OERF、NeRIF、TDBOST 或真实三维折射率场；
- 低重投影误差等价于三维场准确；
- M2 已有原创性或高水平论文资格。

## 最需要师兄确认的四个现实问题

1. 组内可提供的训练对象是折射率/密度三维真值、CFD 场，还是只有多视角位移？
2. 真实 forward 是直线弱偏折、有限孔径，还是包含迭代 ray bending？可否暴露 `A` 与 `A^T`？
3. 最终论文最重视 field、front、held-out reprojection、时间分辨率还是速度？
4. 相机 pose、标定不确定性、背景帧 covariance 与连续时间戳能否随数据一起提供？

这些答案决定 M2 是做三维空间残差算子、有限孔径 mismatch corrector，还是转成四维时序预测。

