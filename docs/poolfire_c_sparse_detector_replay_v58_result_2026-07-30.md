# v56-v58：固定几何稀疏探测器空间重放

> 当前判决：**公开固定几何 straight-ray 代理上的重大机理正结果**。它已经通过独立
> 数值复算和 170 个 fresh process 的资源门，但还不是算子学习结果、外部几何结果、
> 真实 BOST 结果或论文成功。`algorithm_breakthrough=false`。

## 1. 为什么转到这条路线

此前冻结的 PoolFire `w16d2` dual CNN 在 BLASTNet 外部数据族上留下了小幅迁移
信号，却没有同时追平 Zero-CGLS K4 的 field、gradient 和 observation。继续扩大
同类 CNN 缺乏依据，因此 v56 先检查一个更基础的问题：

> 对固定几何的线性算子 \(A\)，测量空间是否存在足够紧凑、又不牺牲 CGLS 结果的
> 物理表示？

全局低秩路线失败了。rank 64 只覆盖 \(15.29\%\) 的
\(\lVert A\rVert_F^2\)，随机可实现观测的投影残差中位数为 `0.920678`。这说明把
所有测量压进一个很小的稠密全局 basis 不合适。

## 2. 真正有效的结构

固定线性几何下定义

\[
B = A A^\top .
\]

零初值 CGLS 的每个迭代场都落在 \(\mathrm{Range}(A^\top)\) 内。令
\(x_k=A^\top z_k\)，就可以在探测器空间对 \(B\) 运行等价的 conjugate-residual
递推，四步结束后只做一次精确 \(A^\top\)：

\[
r_0=y,\quad z_0=0,\quad p_0=r_0,\quad q_0=Br_0,
\]

\[
\alpha_k=\frac{r_k^\top Br_k}{q_k^\top q_k},\quad
z_{k+1}=z_k+\alpha_kp_k,\quad
r_{k+1}=r_k-\alpha_kq_k,
\]

\[
\beta_k=\frac{r_{k+1}^\top Br_{k+1}}{r_k^\top Br_k},\quad
p_{k+1}=r_{k+1}+\beta_kp_k,\quad
q_{k+1}=Br_{k+1}+\beta_kq_k .
\]

最后取 \(x_4=A^\top z_4\)。因此在线调用账从

```text
Zero-CGLS K4:     4 A + 4 A^T
detector replay:  4 B + 1 A^T
```

变成理论上少 `87.5%` 的完整物理算子调用。四个 \(B\) 乘法不是免费的，所以最终
是否加速必须由端到端实测决定，不能只看这张调用账。

## 3. 意外但关键的几何事实

当前冻结 straight-ray 几何的 \(B\) 是 `2072 × 2072`。按结果前冻结的纯浮点
舍入阈值

\[
64\epsilon_{\mathrm{float64}}\max(1,\lVert B\rVert_\infty)
\]

去掉数值零后：

```text
nonzeros                         110,688
density                          2.5782%
row nnz min / median / max       31 / 49 / 97
dense in-memory storage          34,345,472 bytes
CSR in-memory storage             1,336,548 bytes
storage reduction                   96.1085%
discarded relative Frobenius      4.11e-18
```

这不是后验按重建结果调阈值；阈值只由几何矩阵尺度和 float64 机器精度决定。

## 4. 独立数值复算

另一套 validator 没有导入正式 v58 core 或 runner。它重新读取几何、独立构造 CSR、
重新运行五条已打开 fit 轨迹的全部 505 帧，并在验证前后检查封存输入不变。

```text
CSR data / indices / indptr        逐元素完全一致
最大 field relative difference     4.3442e-16
最大 residual relative difference  1.4357e-15
stored frame/trajectory/summary 差  0
```

因此“稀疏重放得到同一个 K4 场”不是正式实现自己证明自己。

## 5. Fresh-process 资源结果

资源门使用五条轨迹、两种 arm、17 次确定性乱序重复，共 `170` 个全新子进程。每个
子进程完整处理一条 101 帧轨迹；线程数固定为 1，峰值 RSS 在进程级测量。

```text
轨迹等权 fresh end-to-end wall ratio   0.824652
最慢轨迹的 wall-ratio 中位数            0.846643
baseline peak RSS p90-higher           108.33 MiB
candidate peak RSS p90-higher          108.42 MiB
RSS p90 ratio                            1.000865
```

换句话说，在当前 Mac CPU 和该公开代理上：

- 核心计算典型快约 `33%`；
- 包含 Python 启动和数据装载的端到端典型快约 `17.5%`；
- 五条轨迹的中位 wall ratio 全部低于 `0.85`；
- 两臂峰值内存 p90 基本持平。

第一次资源运行曾错误地把 85 对独立进程的逐对 RSS 最大比当成总体内存门，因此
机械 FAIL。该结果被保留，没有复用；随后先冻结“按两 arm 各自 p90-higher 作比”的
仓库统一口径，再完整重跑全部 170 个新进程，得到上述 PASS。

## 6. 现在能说什么

可以说：

1. 固定 straight-ray 几何下，标准零初值 CGLS K4 可以在探测器空间被数值精确地
   重放。
2. 当前 BOST 代理几何的 \(AA^\top\) 具有强稀疏性，CSR 比 dense 表示少
   `96.11%` 内存。
3. 在五条已打开 PoolFire 轨迹的 505 帧上，稀疏重放保持机器精度等价。
4. 在 170 个 fresh process 上，端到端 wall 典型下降约 `17.5%`，RSS p90 基本
   不变。

现在不能说：

- 这是新的神经算子或 learned warm start；
- 已经跨几何、跨相机或跨 forward 泛化；
- 已经适用于曲折光线、变化标定或非线性 NeRIF；
- 已经在真实 BOST 位移和真实实验噪声上成立；
- 已经证明全球首创、SOTA 或顶刊论文成功。

## 7. 对论文前景的诚实判断

这是目前少数同时越过“等价结果、少完整调用、fresh wall、RSS”四道门的结果，
所以论文前景比 v55 后明显更乐观。但它单独仍不够顶刊，原因有四个：

1. **代数核心是经典的。** CGLS 在 \(AA^\top\) 探测器空间诱导出的 CR
   递推关系不是新发现；它不是标准 CGNE，二者的迭代系数不同。
2. **当前只有一个固定几何。** 若换相机/视角后 \(B\) 不再稀疏或不再加速，贡献会
   降级为特例工程优化。
3. **离线 setup 尚未完全摊销。** 黑盒构造 \(B\) 需要 `2072 A + 2072 A^T`；
   仅按完整调用计算的 break-even 是约 592 帧，而本次公开评估是 505 帧。真实高速
   BOST 长序列可能容易越过，但必须实测。
4. **真实物理链未闭合。** 仍缺组内相机标定、真实位移、噪声、曲折光线和真实
   solver。

最有论文价值的升级不是把经典等价关系重新命名，而是：

> 把 exact sparse \(B\) 作为固定物理核心和强 classical control，只让学习模块预测
> 几何变化、曲折光线或模型失配导致的小型对称/半正定修正，并保留 fail-closed 的
> exact fallback。

下一道决定性证据是用不同 BLASTNet 相机/几何重新构造 \(B\)，不复用当前稀疏
pattern，检查“稀疏、精确、加速”能否一起保住。它通过后，才有理由把当前结果升级为
可泛化的 physics-structured operator-compression 方法；再接组内真实 BOST，才有
资格评估高水平期刊。

## 8. 公开材料

- [机器可读摘要](poolfire_c_sparse_detector_replay_v58_public_summary.json)
- [结果总图](../assets/poolfire_c_sparse_detector_replay_v58.png)
