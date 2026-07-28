# Direct-Field Warm Start v29：实际做了什么，成功了什么，失败了什么

更新时间：2026-07-28

## 先给结论

这轮不是又写了一份计划，而是完成了一个真实训练、五折跨轨迹评估、开放验证、
独立复算和 fresh-process 资源测试的算法闭环。

当前最准确的结论是：

```text
阶段性关键进展 = true
正式算法突破 = false
```

成功的部分：

- 10,524 参数的因果三维 CNN 从部署可见的相邻两帧均衡 BP 生成场初值；
- 初值经过只看观测的标量校正，再接完全未修改的 CGLS K1；
- 五条 PoolFire fit 轨迹做 leave-one-trajectory-out，五折三项 p90 全部优于
  Zero-CGLS K3；
- 在此前已经打开、但没有进入本次训练的 p14-s01 上，三项 p90 再次全部优于
  Zero-K3，100 帧的 field / gradient / observation harm 都是 0；
- truth-free 部署代码实际执行 `200A + 201A^T`，对照为
  `300A + 300A^T`，完整算子调用减少 33.17%。

失败的部分：

- 当前 Mac 上的直线小代理算子太便宜，CNN 开销超过少算的算子开销；
- CPU compute wall 慢 2.59-2.91 倍，fresh-process wall 慢 1.26-1.38 倍，
  peak RSS 高 1.28-1.34 倍；
- Direct-K1 没有在全部指标上击败 Zero-K4，尤其 observation 仍可能落后；
- 尚未完成 fresh trajectory、非线性曲线光路、真实 BOST 和组内实验迁移。

因此这不是“论文已经成功”，而是一条首次同时取得**重建精度信号**和
**真实执行调用数下降**、但仍被端到端资源门挡住的候选方法。

## 为什么改做 Direct-Field

v28 的 reduced/dual 曲线方案相对 Zero 有信号，但独立审计发现它在 7/7 个样本上
伤害 Full Parent 的梯度精度。增加 Krylov 内步只能继续压观测残差，无法找回丢失的
三维近零空间结构。

所以这轮不再为旧表示补丁，而是建立同规模的强对照：

```text
上一帧 observation -> A^T -> geometry equalizer ----\
                                                       -> 3D CNN -> field proposal
当前帧 observation -> A^T -> geometry equalizer ----/

field proposal
  -> observable-only scalar line search
  -> strict arbitrary-start CGLS K1
  -> reconstructed field
```

这个对照回答一个重要问题：收益究竟来自特殊的 dual/reduced 表示，还是来自更一般的
“学一个好的三维场初值”。如果 Direct-Field 本身更稳，就不能把旧 dual 表示包装成
必要创新。

## 模型和训练合同

模型是 `DirectFieldTemporalCNN`：

- 输入：上一帧均衡 BP、当前帧均衡 BP、固定物理 base、几何 support；
- 输出：有界的三维场修正；
- 参数量：10,524；
- 因果：当前帧只使用当前及过去的观测；
- 训练目标：field、voxel-gradient 和 differentiable observation 三项；
- 训练课程固定为 `30 + 15 + 15` epochs；
- 部署不读取 truth；
- 后处理只使用观测可计算的标量线搜索；
- refinement 是未修改的 CGLS K1。

训练后没有用更大网络挽救结果。五个 LOTO fold 和 p14 开放验证都使用同一结构、
同一训练课程和同一 refinement。

## 五折跨轨迹结果

五条 fit 轨迹各自轮流作为 held-out，模型只能在另外四条轨迹上拟合
`alpha`、correction cap 和网络参数。共重新训练五次，评估 500 帧。

聚合 p90：

| 指标 | Direct-Field K1 | Zero-CGLS K3 | 相对下降 |
|---|---:|---:|---:|
| field relative-L2 | 0.572526 | 0.679131 | 15.70% |
| gradient relative-L2 | 0.806682 | 0.916389 | 11.97% |
| observation relative-L2 | 0.370540 | 0.380479 | 2.61% |

逐轨迹三项 p90 都通过。唯一的尾部问题出现在 p45-s05：
observation 有 12/100 帧略差于 Zero-K3；其余四折三项逐帧 harm 都为 0。
所以结论是“五折 p90 稳定通过”，不是“500 帧逐帧全部支配”。

## p14 开放验证

p14-s01 的角色是 `model_selection_validation`。它在旧实验中已经被打开，所以不能
冒充 untouched test；但 v29.3 的当前 runner 先在五条 fit 上完成
`alpha / cap / training / checkpoint`，保存 checkpoint 后才读取 p14。

100 帧 p90：

| 指标 | Direct-Field K1 | Zero-CGLS K3 | 相对下降 |
|---|---:|---:|---:|
| field relative-L2 | 0.484033 | 0.633080 | 23.54% |
| gradient relative-L2 | 0.703886 | 0.835100 | 15.71% |
| observation relative-L2 | 0.353301 | 0.371543 | 4.91% |

三项逐帧 harm 都是 `0/100`。

随后新增独立 validator，不导入正式 runner 的数据准备、deployment、Zero-CGLS 或
metric 函数，完成：

1. 从五条 fit 重新计算固定 `alpha` 和 correction cap；
2. 重载 checkpoint，保持 cap、轴顺序、float32/float64 和 101 帧因果条件；
3. 重新实现 truth-free Direct-K1、Zero-K3、Zero-K4；
4. 重新计算 field、物理坐标 gradient 和 observation 指标；
5. 再核对每一次 `A/A^T`。

独立复算状态：

```text
PASS_INDEPENDENT_RECOMPUTATION_OPENED_VALIDATION_V29_3
maximum metric difference = 9.54e-9
```

该差异来自 CNN 从单帧 batch 改成 7 帧 batch 后的 float32 舍入，不改变任何指标、
harm 或判决。

## 调用数为什么真的下降

对 100 个因果帧对，部署路径实际执行：

```text
101 x A^T y              当前帧及启动历史 BP
100 x A x0               观测标量校正
100 x A^T residual       CGLS K1
100 x A normal           CGLS K1
```

因此是：

```text
Direct-Field K1 = 200A + 201A^T = 401 complete calls
Zero-CGLS K3    = 300A + 300A^T = 600 complete calls
reduction       = 33.17%
```

五折累计执行账为 `2005` 对 `3000`，包含每条轨迹的一次启动 `A^T`。这不是从公式
推测出的理论账，而是计数 operator 的实际回执。

## 为什么现在仍不能说“加速”

完整调用减少并不等于当前机器更快。对五个 fold、每臂三次 fresh process 的 CPU
streaming benchmark：

| 资源指标 | Direct / Zero-K3 | 判决 |
|---|---:|---|
| compute wall | 2.59-2.91x | FAIL |
| fresh-process wall | 1.26-1.38x | FAIL |
| peak RSS | 1.28-1.34x | FAIL |

MPS batch 也没有通过。原因不是算子账错误，而是当前 `16x16x32` straight-ray
NumPy 算子极便宜，Python、CNN、张量转换和进程启动成本占主导。

测得的量级表明，单次物理算子至少要比当前 proxy 贵约 5.8 倍，Direct 才可能达到
wall parity；贵约 8.3 倍，才可能达到 10% wall speedup。真实曲线 ray tracing、
JVP/VJP 或实验 BOST 可能进入这个区间，但现在没有证据，不能提前写成成功。

## Zero-K3 与 Zero-K4 的边界

K3 不是为了挑一个容易打的基线。在这组逆问题里继续压观测会开始放大不可观测方向：
五折聚合时 Zero-K4 的 observation p90 从 `0.380479` 降到 `0.343041`，但
gradient p90 从 `0.916389` 恶化到 `0.961550`。

Direct-K1 相对 K3 同时改善三项，说明它在少迭代下取得更平衡的场恢复；但它没有在
全部指标上击败 K4，尤其 observation 不是最小。因此允许写“优于冻结的
early-stopped Zero-K3 平衡基线”，不允许写“全面击败 CGLS”。

## 是否是突破性进展

本轮有一个真实的阶段性关键进展：

```text
跨轨迹 p90 稳定改善
+ 已打开验证再次通过
+ 独立复算一致
+ 实际完整算子调用减少 33.17%
```

但正式 `algorithm_breakthrough` 仍为 false，因为四个关键门还没有关闭：

1. p14 不是 untouched/fresh；
2. 当前 wall/RSS 明确失败；
3. v29 尚未进入非线性曲线光路 inverse；
4. 尚无真实 BOST、噪声重复测量和标定不确定度。

真正能把它推进成论文贡献的下一实验，不是再增加网页或重复 straight-ray：

```text
同一 Direct-Field proposal
  -> nonlinear curved F(x), J(x)v, J(x)^T w
  -> 与 Zero 和 Full Parent 做 matched accuracy
  -> fresh-process wall + whole-pipeline RSS
  -> 再接组内真实 BOST 定义“实验同精度”
```

如果它在昂贵曲线算子下仍守住 field / gradient / observation，同时把调用下降变成
真实 wall 优势，这条路线才接近正式突破；如果失败，就应把结果写成“learned warm
start 的精度-成本边界”，而不是虚构成功。

## 可复核材料

- 脱敏机器摘要：`docs/poolfire_c_direct_field_v29_public_summary.json`
- 结果图：`assets/poolfire_c_direct_field_v29.png`
- 模型核心：`learning_labs/poolfire_c_direct_field_warm_v29.py`
- truth-free 部署：`learning_labs/poolfire_c_direct_field_deployment_v29_2.py`
- 五折独立验证：`site_tools/validate_poolfire_c_direct_field_curriculum_v29_1.py`
- p14 独立验证：`site_tools/validate_poolfire_c_direct_field_opened_validation_v29_3.py`
- 脱敏构建器：
  `site_tools/build_poolfire_c_direct_field_v29_public_artifacts.py`

公开页面不包含 checkpoint、原始数据、私有路径、私有哈希或组内材料。
