# v59 外部坐标几何：代数迁移通过，5 帧资源迁移失败

> 日期：2026-07-30  
> 正式裁决：`MIXED_EXTERNAL_COORDINATE_GEOMETRY_RESULT_V59`  
> 突破标记：`algorithm_breakthrough=false`

## 1. 这次真正问了什么

v58.1 已经证明：在固定 PoolFire 三正交视角 straight-ray 代理中，稀疏
\(B=AA^\top\) 可以在探测器空间精确重放 zero-start CGLS K4，并减少完整
\(A/A^\top\) 调用。

v59 不复用 PoolFire 的 \(B\)，也不复用它的稀疏位置。我们从另一套公开
BLASTNet H2-air 数据的物理坐标重新构造 \(A\)、\(B=AA^\top\) 和 CSR，然后问：

1. 稀疏结构和 CGLS 重放是否仍成立？
2. 在一个只有 5 个 snapshot 的全新进程中，wall 和 RSS 是否仍有优势？

外部坐标跨度为

```text
x = 0.0503226 m
y = 0.0309677 m
z = 0.0157460 m
```

它与 PoolFire 的尺度和长宽比明显不同，但视图拓扑仍是同样的三个正交轴向投影。
所以这是 **coordinate-scale / aspect-ratio transfer**，不是任意相机布局迁移。

## 2. 先把算法名称说准确

当前 detector-space 递推是 CGLS 在 \(B=AA^\top\) 空间诱导出的
**Conjugate Residual (CR)**：

\[
\alpha_k =
\frac{r_k^\top B r_k}{(Bp_k)^\top(Bp_k)}.
\]

它不是标准 CGNE。CGNE 对 \(Bz=y\) 使用的系数是
\((r_k^\top r_k)/(p_k^\top Bp_k)\)，一般不会逐步重放同一个 CGLS 迭代。
小型随机矩阵对照中，CR lift 与 CGLS 的最大差为 \(3.0\times10^{-16}\)，
而标准 CGNE 与同一 CGLS 迭代的最大差为 \(0.379\)。

## 3. 代数和稀疏结构：通过

独立 validator 没有导入正式 v59 runner，也没有导入 v57/v58 replay core。它从
外部坐标重新构造 \(A\) 和 \(AA^\top\)，重建 CSR，并重新计算全部 5 个 observation。

| 检查 | 结果 |
|---|---:|
| 最大 field 相对差 | `4.30e-15` |
| 最大 residual 相对差 | `6.03e-15` |
| dense \(B\) 最大绝对差 | `5.68e-14`，在冻结舍入界内 |
| CSR indices / indptr | 完全一致 |
| 非零元比例 | `2.5782%` |
| CSR 存储下降 | `96.1085%` |
| 封存输入 | 验证前后不变 |

因此可以说：

> 在相同三轴视图拓扑、但不同物理坐标尺度和长宽比下，稀疏 detector-space CR
> 仍能以机器精度重放 zero-start CGLS K4。

不能把它扩大成 arbitrary-camera geometry transfer。

## 4. Fresh-process 资源门：严格失败

资源门使用

```text
17 repetitions × 2 arms = 34 fresh processes
```

每个进程都重新读取坐标、构造几何并处理 5 个 snapshot。候选还要加载 CSR \(B\)。
判决结果如下：

| 指标 | candidate / baseline | 冻结门 | 判决 |
|---|---:|---:|---|
| 核心计算 wall p50 | `0.809684` | 诊断 | 核心快约 19.0% |
| worker lifetime p50 | `1.110023` | 诊断 | 慢约 11.0% |
| outer process wall p50 | `1.023894` | `<=0.90` | **FAIL** |
| outer process wall p90 | `1.122534` | 诊断 | 尾部更慢 |
| peak RSS p90-higher | `1.153122` | `<=1.05` | **FAIL** |

RSS 的绝对量为：

```text
Zero-CGLS K4       36.53125 MiB
sparse replay      42.12500 MiB
```

这不是测量误差范围内的擦边失败：wall 和 RSS 两道正式门都没有通过。

## 5. 为什么核心更快，端到端却更慢

17 次运行的每臂中位数显示：

| 分量 | Zero-CGLS K4 | sparse replay |
|---|---:|---:|
| load | `6.08 ms` | `8.05 ms` |
| 5 帧 compute | `4.94 ms` | `4.15 ms` |
| worker lifetime | `11.12 ms` | `12.35 ms` |
| outer process | `80.51 ms` | `81.97 ms` |

稀疏重放在 5 帧计算中省下约 `0.80 ms`，但加载 \(B\) 多花约 `1.97 ms`，并多占
约 `5.6 MiB` 的 p90 RSS。五帧太短，固定成本没有被摊薄。

这支持“batch length 是资源收益的必要条件”，但目前没有正式证明外部几何在长序列
下一定通过。不能循环复制这 5 帧后就把结果写成跨轨迹泛化。

## 6. 对研究路线的真实影响

### 保留下来的东西

- \(B=AA^\top\) 的稀疏模式不是 PoolFire 坐标数值的偶然特例；
- 在当前三轴拓扑内，coordinate scale / aspect ratio 改变后仍可精确重放；
- 核心计算的确变快，不是单纯把 \(A/A^\top\) 改名隐藏。

### 没有成立的东西

- 5 帧外部 bundle 的端到端加速；
- RSS no-harm；
- 任意相机布局、真实 BOST、曲折光线或标定扰动迁移；
- 算子学习结果；
- 全局首创、SOTA、算法突破或论文成功。

### 下一步

1. 结果前冻结一个 **batch-length amortization** 实验，只研究部署资源曲线，不把重复
   snapshot 当作新物理数据；
2. 构造真正不同的视角拓扑，检查 \(AA^\top\) 的 block sparsity 是否仍存在；
3. exact sparse \(B\) 只作为固定物理核心；学习模块只处理几何变化或曲折光线造成的
   小修正，并设置精确回退；
4. 最终必须接组内真实 BOST 标定与长序列，重新测精度、wall、RSS 和 setup 摊销。

## 7. 一句话结论

> v59 把“单一 PoolFire 坐标特例”升级为“相同三轴拓扑内的外部坐标代数迁移”，
> 但同时证明 5 帧短序列没有部署资源优势。它让论文边界更准确，却没有把当前工作
> 推成顶刊或算法突破。

