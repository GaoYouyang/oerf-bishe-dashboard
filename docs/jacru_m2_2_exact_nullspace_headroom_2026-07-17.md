# JACRU-M2.2：exact null-space oracle 找到可行 headroom，但还不是算法成功

**实验日期：** 2026-07-17  
**状态：** `M2_2_EXACT_NULLSPACE_HEADROOM_FOUND_ORACLE_ONLY`  
**数据状态：** 已打开的 M2-T0 synthetic train / development / exploratory-OOD  
**部署算法：** `false`  
**Fresh / final 授权：** `false`  
**唯一新增授权：** `continue_matrix_free_projection_research=true`

## 一句话结论

在 12³ toy 上，把 learned correction 精确分解为 approximate inverse operator 的 row-space
与 numerical-null-space 分量后，可以同时恢复 CGLS-24 的内部重投影，并保留绝大多数三维场
收益。JACRU oracle 相对 CGLS-24 的 development / exploratory-OOD field gain 为
`45.28% / 37.54%`，measured reprojection ratio 均为 `1.00000000000001` 左右。

这证明 **目标在当前离散算子的解空间里存在**，不证明已经有可部署方法，更不证明对真实光学
forward 有效。下一轮的研究问题从“网络能否学到场先验”收缩为“怎样用有限次 matrix-free
调用逼近这个 affine/null-space projection”。

## 1. Oracle 到底做了什么

对每个冻结几何，用 CPU float64 组装 active-support dense matrix `A`，只做一次完整 SVD。
令 `x_ref` 为 CGLS-24，网络原始预测为 `x_net`，修正为

```text
delta = x_net - x_ref
delta_row  = P_row(A) delta
delta_null = delta - delta_row
x_oracle   = x_ref + delta_null
```

由于 `A delta_null = 0`，所以 `A x_oracle = A x_ref`。这个投影不读取 truth、family label 或
界面 mask；truth 只在投影完成后进入评分。

它仍然是 oracle，原因有三：

1. dense matrix 与 SVD 在真实百万级 rays / voxels 上不可直接部署；
2. SVD setup 没有伪装进重建调用预算，也没有速度主张；
3. `ker(A_voxel)` 只是有限差分、三线性插值近似逆算子的零空间，不是连续有限孔径光学过程的
   真零空间。

## 2. 为什么这次投影与 M2.1 不同

M2.1 的有限步 near-null filter 是 `(I-tau A^T A)^k delta`。对病态算子，小奇异值方向衰减很慢，
11 步后 `||A delta_k|| / ||y-Ax0||` 仍为 2--3，重投影比 matched CGLS-24 差 42--46 倍。

M2.2 用 SVD 一次性计算精确 row-space projector，直接消除所有数值可观测分量。因此它不是
“继续多跑几步 Landweber”，而是回答 Landweber 是否在逼近一个值得逼近的目标。

## 3. 算子维度与可辨识性

12 个 development / exploratory-OOD 几何全部得到：

| 量 | 结果 |
|---|---:|
| measurement dimension | 150 |
| active voxel dimension | 1,000 |
| numerical rank (`rtol=1e-10`) | 150 |
| numerical nullity lower bound | **850** |
| largest singular value | 2.891--3.031 |
| smallest retained singular value | 0.303--0.331 |
| 最大内部投影残差 | `2.62e-15` |
| 最大可见 null correction fraction | `3.69e-15` |

这不是说真实 BOST 一定有 85% 零空间；只是当前小考卷中 1,000 个 active unknown 只有 150 个
标量观测。它解释了为什么监督网络能显著改善 field，却同时产生巨大投影冲突：网络主要在巨大
欠定空间中放入形态先验，但其中约 40% correction norm 仍有 row-space 分量。

## 4. 主要结果

### 4.1 JACRU-M2

| Split | CGLS-24 field | raw network field | exact-null field | field gain vs CGLS-24 | H1 gain | raw gain retention | reproj ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 0.669907 | 0.366905 | 0.377104 | **45.28%** | **43.75%** | 97.67% | 1.000000000000012 |
| exploratory OOD | 0.662346 | 0.449677 | 0.417439 | **37.54%** | **40.19%** | 143.06% | 1.000000000000012 |

OOD retention 超过 100% 不是超物理现象：原始 correction 的 row-space 部分在 OOD 上恰好还
损害了 field；精确删除它后，field 比 raw network 更好。

### 4.2 Pooled CNN

| Split | exact-null field gain | H1 gain | raw gain retention | reproj ratio |
|---|---:|---:|---:|---:|
| development | **44.24%** | **42.90%** | 92.01% | 1.000000000000010 |
| exploratory OOD | **37.38%** | **39.57%** | 124.35% | 1.000000000000013 |

CNN 与 JACRU 仍然非常接近。这一正信号授权的是“通用 learned residual + affine projection”研究，
不是 JACRU 集合编码器优越性。

### 4.3 correction 能量在哪里

| 模型 | Split | null correction norm / total | row correction norm / total |
|---|---|---:|---:|
| JACRU | development | 0.913 | 0.401 |
| JACRU | OOD | 0.903 | 0.426 |
| CNN | development | 0.915 | 0.397 |
| CNN | OOD | 0.899 | 0.433 |

row 与 null 正交，因此两项 norm ratio 不按线性相加为 1，而满足平方和约为 1。大部分 correction
energy 位于 numerical null space，说明有限调用投影若能有效去掉约 40% 的 row norm，理论上可
保留主体 learned prior。

## 5. 由此得到的 M2.3 最小算法

精确 projector 为

```text
P_row = A^T (A A^T)^dagger A
delta_null = delta - P_row delta
```

下一候选不需要先改网络，而是在 measurement space 解：

```text
b = A delta
(A A^T + lambda I) z = b
x_hat = x_ref + delta - A^T z
```

使用固定步数 PCG/LSQR 时，`A A^T v` 每步只需一次 `A^T` 和一次 `A`。算 `b` 与最终
`A^T z` 各一次，因此 k 步投影总计 `(k+1)F + (k+1)A`。这比对体素场盲目做 Landweber 更
直接地逼近 row-space removal。

### 必须有的对照

1. CGLS / Huber 在相同总 F/A 调用；
2. `x_ref +` base-only measurement-space CG，不加 learned correction；
3. JACRU 与 pooled CNN 接受同一 projector；
4. unpreconditioned CG、Jacobi、固定 low-rank、geometry-conditioned preconditioner；
5. exact SVD 只作上界，不进入方法排名或 runtime 图；
6. zero / shuffled residual 与 norm-matched random correction，检查网络是否只记住 morphology。

### M2.3 的硬门

- 在固定 `k` 下，measured reprojection 不高于同预算 CGLS 的 `1.10x / 1.15x`；
- 至少保留 exact oracle field gain 的 50%，且相对最强匹配经典 field 仍改善 5% / 2%；
- H1、harm rate、worst case 与三模型 seed 同时通过；
- JACRU 若不能在 camera-count / pose / mask OOD 上显著超过 pooled CNN，删除 JACRU 优越性主张；
- 独立 renderer clean reprojection 必须过门，否则说明只服从 approximate `A`；
- 当前 opened T0 只能开发 M2.3，不能产生 confirmatory 结论。

## 6. 原创性边界

零空间网络、data consistency 与 unrolling 都不是新概念：

- [Deep Null Space Learning](https://arxiv.org/abs/1806.06137)
- [Learned Primal-Dual](https://arxiv.org/abs/1707.06474)
- [MoDL](https://arxiv.org/abs/1712.02862)
- [Learned Operator Correction](https://doi.org/10.1137/20M1338460)

可能形成论文贡献的窄组合是：**有限孔径 BOST 的可变相机集合 residual prior +
geometry-conditioned matrix-free measurement-space projector + approximate/independent forward 双域审计 +
OOD fail-closed 风险门。** 这个组合尚未由本轮证明新颖，只是被证据选出来的检索与实现方向。

## 7. 可复现入口

- [冻结配置](../demo_t16_operator/configs/jacru_m2_2_exact_nullspace_oracle_postopen_v1.json)
- [机器摘要](../demo_t16_operator/results/jacru_m2_2_exact_nullspace_oracle_postopen_public/summary.json)
- [180 行 oracle 指标](../demo_t16_operator/results/jacru_m2_2_exact_nullspace_oracle_postopen_public/metric_rows.csv)
- [30 行 CGLS-24 参考](../demo_t16_operator/results/jacru_m2_2_exact_nullspace_oracle_postopen_public/reference_rows.csv)
- [诊断图 PDF](../demo_t16_operator/results/jacru_m2_2_exact_nullspace_oracle_postopen_public/diagnostic.pdf)
- [oracle 模块](../demo_t16_operator/jacru_m2_exact_nullspace_oracle.py)
- [runner](../site_tools/run_jacru_m2_2_exact_nullspace_oracle.py)

```bash
PYTHONPATH=. .venv/bin/python \
  site_tools/run_jacru_m2_2_exact_nullspace_oracle.py
```

准确结论是：**当前合成离散算子中存在可保留 learned field gain 的 data-consistent null-space
解；把这个 oracle 变成有限调用、跨几何、跨 forward 仍有效的算法，才是下一阶段。**
