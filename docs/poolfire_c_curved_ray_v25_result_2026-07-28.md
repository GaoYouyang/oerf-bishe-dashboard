# v25：曲折光线物理压力测试

## 一句话结论

冻结的 `Observable Reduced Warm K1` 在三条 PoolFire 轨迹、五个结果前固定的
曲率强度上，始终是通过同一 field / gradient / observation 兼容门的最便宜方法。
最高档 `beta=0.002` 也通过，但这只是受控曲折光线 proxy 的积极鲁棒性证据：

```text
PASS_CONTROLLED_CURVED_RAY_PROXY_STRESS_V25
algorithm_breakthrough=false
real_BOST=false
paper_success=false
```

## 为什么做这一步

v24 的三轨迹结果仍建立在直线射线算子上。真实折射率场会使光路随场弯曲；
如果 warm start 的优势只在训练和反演共享同一个线性正演时存在，它就很可能在
真实 BOST 迁移时消失。

这次没有重训模型，也没有在看到结果后改门槛。我从师兄提供的 BOS 模拟工具中
只提取了各向同性 eikonal 光线方程这一物理思想，重新写了独立实现：

```text
dr/ds = t
dt/ds = (I - t t^T) grad(n) / n
n = 1 + beta * (rho_ROI - mean(rho_ROI))
```

输出使用最终方向变化除以 `beta`，使 `beta -> 0` 时回到现有线性
密度梯度投影尺度。私有 notebook、路径、数据和代码均未复制到公开仓库。

## 冻结实验

- 三条 post-open fit-morphology 轨迹：`p14-s05`、`p33-s01`、`p58-s03`。
- 每条 101 帧，共 303 帧。
- `beta = 0, 1e-4, 5e-4, 1e-3, 2e-3`。
- 正式积分 192 步；帧 0、50、100 用 96 步独立做步长收敛对照。
- 数值门：96/192 步相对 L2 最坏不得超过 0.5%。
- 每个 `trajectory × beta` 都比较八个冻结方法。
- 全 101 帧和奇数 50 帧必须同时通过；任一严格更便宜的方法通过，主方法就算被支配。
- 本轮不重新测 wall/RSS，不能升级 v24 的资源结论。

## 压力是否真实存在

最高曲率档相对 `beta=0` 线性极限的 observation 变化为：

| 轨迹 | p50 | p90 | 最坏 | 96/192 步最坏差 |
|---|---:|---:|---:|---:|
| p14-s05 | 2.253% | 3.113% | 5.307% | 0.080% |
| p33-s01 | 10.428% | 14.188% | 17.707% | 0.372% |
| p58-s03 | 3.758% | 5.705% | 8.257% | 0.187% |

p33 的 p90 已改变 14.188%，因此这不是把几乎等于直线模型的数据重新跑一遍。
同时三条积分收敛误差都小于 0.5%，最大的 p33 也只有 0.372%。方向归一误差
最坏为浮点舍入量级。

`beta` 是人为控制的无量纲压力参数，尚未由波长、气体组分和 Gladstone-Dale
常数标定。因此上表证明的是数值与形态压力，不是实验中的真实折射强度。

## 八个方法的结果

最高曲率档上，三条轨迹共同的判决如下：

| 方法 | 101 帧完整调用 | 三条轨迹是否都通过 |
|---|---:|---|
| normalized BP | 202 | 否 |
| Zero CGLS K1 | 202 | 否 |
| **Observable Reduced Warm K1** | **354** | **是** |
| Full parent Warm K1 | 404 | 是 |
| Zero CGLS K2 | 404 | 否 |
| Geometry PCGLS K2 | 404 | 否 |
| Zero CGLS K3 | 606 | 否 |
| Zero CGLS K4 reference | 808 | 是 |

主方法在三条轨迹的全 101 帧和奇数 50 帧上均为：

```text
joint matched fraction = 100%
joint harm fraction = 0
```

相对 Zero-K4 的 p90 比值为：

| 轨迹 | field | gradient | observation |
|---|---:|---:|---:|
| p14-s05 | 1.0006 | 1.0044 | 0.9952 |
| p33-s01 | 1.0013 | 1.0020 | 0.9929 |
| p58-s03 | 1.0028 | 1.0023 | 0.9985 |

这些比例不是说主方法逐项更准，而是它在冻结的单侧非劣包络内，用
`202A + 152A^T = 354` 次调用达到与 `404A + 404A^T = 808` 的 Zero-K4
兼容的结果。完整父模型需要 404 次，主方法少 50 次精确伴随。

## 独立核验

正式 runner 完成后，新写的独立判决器没有导入正式 runner，而是重新读取冻结
协议和结果，逐项复算：

```text
3 trajectories × 5 beta levels × 8 arms = 120 decisions
PASS_INDEPENDENT_DECISION_RECOMPUTATION_CURVED_RAY_STRESS_V25
```

它重新核对了数值门、两套帧集、每个方法的 `A/A^T` 账、最便宜兼容方法、
严格支配关系、五个曲率档汇总和最高通过档。代码测试还包括常场零偏折、仿射场
对独立 Gauss-Legendre 参考、小 beta 极限、步长收敛和越界 fail-closed。

实现过程中发现一个必须公开的数值问题：如果 detector rays 恰好落在
cell-centred trilinear gradient 的导数不连续结点上，小 beta 极限会不稳定。
正式几何使用高分辨率 `32×32×64` 场和由 block mean 得到的
`16×16×32` detector 坐标，避开了这个人工对齐；相关回归测试已固定。

## 是否成功

**成功的部分：**

1. 在不重训、不改门和不换方法的条件下，主方法经受了比直线 forward 更强的
   场依赖非线性观测压力。
2. 三条轨迹、五个强度档逐条通过，没有靠平均值掩盖失败。
3. 202/404 次的便宜经典方法没有通过，354 次的主方法是最便宜兼容方法。
4. 数值收敛、调用账和 120 个判决均有机器可复核证据。

**没有成功或尚未证明的部分：**

1. 没有相机内外参、背景图像渲染、位移提取、光流误差和实验噪声。
2. `rho -> n` 的组分、波长和 Gladstone-Dale 标定没有闭合。
3. 三条轨迹是已打开的同一公开 PoolFire 数据集形态压力，不是 official test。
4. 本轮没有重新测曲折正演下的端到端 wall 和 RSS。
5. 没有真实 BOST、跨数据集泛化或论文级成功。

因此这是一项**真实的正向科学增量**，但不是突破性进展。它排除了一个重要失败
解释：“优势只在完全共享的直线线性 forward 下存在。”它还没有排除更真实的
相机成像、标定和噪声会使优势消失。

## 下一道决定性实验

下一次能升级论文结论的工作，不是继续增加 synthetic `beta`，而是把冻结方法
接入组内真实 BOST 链：

1. 用真实波长、组分和参考状态闭合 `rho/T/Yk -> n`。
2. 接入相机标定、背景平面、flow-off/flow-on 图像和位移提取。
3. 用重复采集与标定不确定度定义真实“同精度”。
4. 原样比较 Zero-K4、父模型、Reduced Warm K1 的完整调用、wall 和全流程 RSS。

只有这一门通过，才有资格把当前结果从“受控物理 proxy 鲁棒性”升级为
“真实 BOST 重建加速证据”。

## 一级来源

- [Unified Deflection Estimation and Error Analysis for Background-Oriented Schlieren](https://arxiv.org/abs/2607.15567)
- [Neural Deflection Fields for Sparse-View Tomographic Background-Oriented Schlieren](https://arxiv.org/abs/2409.19971)
- [Neural refractive index field: Unlocking the Potential of Background-oriented Schlieren Tomography](https://arxiv.org/abs/2409.14722)
