# JACRU-M2.1 判决：普通数据一致性步无法把 learned prior 拉回同预算 CGLS

**实验日期：** 2026-07-17  
**状态：** `M2_1_POSTOPEN_DATA_CONSISTENCY_NO_GO`  
**数据状态：** 已经打开的 M2-T0 train / development / exploratory-OOD  
**Fresh / final 授权：** `false`  
**方法主张授权：** `false`

## 一句话结论

JACRU-M2 和 pooled CNN 的三维场收益并未被 11 步物理校正完全抹掉，但普通 Landweber
校正收敛得远远不够快：在同为 `24F/24A` 的预算下，JACRU measured-pullback 仍比
CGLS-24 的逐 case 重投影误差差 `43.12x / 41.95x`（development / exploratory OOD）。
因此 M2.1 仍是 NO-GO；下一轮若继续，只能测试更强的 matrix-free Krylov / affine projection
机制，不能再把“多跑几步梯度下降”包装成创新。

## 1. 为什么第一版比较被作废

最初的 post-open 脚本把网络路径 `CGLS-12 + feature pair + k correction pairs` 与固定
CGLS-13 比较。这个比较不公平：网络在 `k=11` 时已经使用 `24F/24A`，经典方法却只有
`13F/13A`。第一版原样保留在
`jacru_m2_1_data_consistency_postopen_unmatched_v1_public`，只作为流程错误记录，不进入结论。

修订版在第一次执行前冻结，并为每个 `k in {0,1,3,5,11}` 增加三类相同总预算对照：

1. `CGLS-(13+k)`；
2. `Huber-PDHG-(13+k)`；
3. CGLS-12 后追加 `(k+1)` 步 base-only Landweber。

第三类对照尤其重要：若“网络 + correction”只和它相当，收益来自额外物理迭代而不是 learned
residual。所有网络与经典对照仍使用同一 approximate voxel-FD/trilinear inverse operator；
dense SVD norm setup 独立报告，不计入重建速度或调用预算。

## 2. 两条 truth-free 校正路径

令 `x0` 为 CGLS-12，网络修正为 `delta`，`H=A^T A`。

### A. measured pullback

```text
x_(j+1) = support * [x_j + tau A^T(y - A x_j)]
```

它直接下降 `||Ax-y||²`。这就是标准 Landweber；重投影下降是数学预期，不能算网络贡献。

### B. base near-null spectral filter

```text
delta_(j+1) = support * [delta_j - tau A^T A delta_j]
x_j = x0 + delta_j
```

有限步得到 `(I-tau H)^k delta`，只会按奇异值大小衰减可观测分量。只有无限步极限才可能接近
`P_ker(A) delta`，所以本报告明确称它为 **near-null spectral filter**，不是精确零空间投影。

runner 使用 `tau=0.98/L_geometry`，并在代码接口中强制 `0 < tau < 2/L`。每步由实际算子
counter 验证为一对 forward/adjoint。

## 3. 匹配预算结果

### 3.1 24F/24A 的最终比较

| 方法 | Dev field | Dev H1 | Dev measured reproj | OOD field | OOD H1 | OOD measured reproj |
|---|---:|---:|---:|---:|---:|---:|
| CGLS-24 | 0.669907 | 1.192056 | **0.000813** | 0.662346 | 1.215725 | **0.000904** |
| Huber-PDHG-24 | 0.615806 | 1.016369 | 0.133577 | 0.617116 | 1.021299 | 0.132547 |
| base-only Landweber-24 | 0.663419 | 1.181259 | 0.009203 | 0.659267 | 1.208213 | 0.007534 |
| JACRU + measured-11 | **0.342376** | **0.605925** | 0.031799 | **0.398214** | **0.679620** | 0.034803 |
| JACRU + near-null-11 | **0.341742** | **0.605502** | 0.033292 | **0.398040** | **0.679212** | 0.035756 |
| CNN + measured-11 | 0.349255 | 0.615291 | 0.032265 | 0.398521 | 0.683730 | 0.036845 |
| CNN + near-null-11 | 0.348601 | 0.614804 | 0.033756 | 0.398347 | 0.683443 | 0.037764 |

JACRU measured-11 相对同预算最强经典 field 基线仍改善 `45.34% / 35.68%`，相对 matched
base-only Landweber 改善 `49.44% / 39.98%`。这证明场收益不是简单多跑 11 步造成的。
但其重投影相对 matched CGLS 为 `43.12x / 41.95x`，离硬门 `1.10x / 1.15x` 相差近两个
数量级；零个候选点通过完整 gate。

### 3.2 near-null 路径没有完成它的任务

11 步后，`||A delta_k|| / ||y-Ax0||` 的均值仍为：

| 模型 | Development | Exploratory OOD |
|---|---:|---:|
| JACRU-M2 | 2.282 | 3.189 |
| pooled CNN | 2.305 | 3.388 |

未来值得进入预注册的门槛是 `<=0.10`。当前不是“差一点”，而是有限步谱滤波仍留下比 CGLS-12
数据残差大 2--3 倍的可观测修正。

## 4. 这轮真正学到了什么

1. learned correction 中确实有大量 truth-space 有用分量；否则 matched field/H1 不会持续领先。
2. 这些分量和强可观测误差混在一起，普通固定步 Landweber 无法在 24 对调用内分离。
3. measured pullback 与 near-null filter 的轨迹几乎重合，说明当前 11 步还远未进入可解释的
   null-space 极限。
4. JACRU 与更小 CNN 继续接近；可变相机集合结构的独特价值仍未建立。
5. approximate inverse operator 的零空间不等于真实有限孔径光学 forward 的零空间。即使未来
   内部投影通过，也必须在独立 renderer 与真实 held-out image consistency 上再审计。

## 5. 下一轮 M2.2 的合法问题

下一轮不再扫描更多 Landweber 步，而先做两级 headroom：

1. **exact dense null-space oracle，仅限 12³ toy。**用 SVD 计算
   `x_ref + P_ker(A)(x_net-x_ref)`，判断“场收益与同算子投影一致性”是否在数学上能同时存在。
   这是 oracle，不是部署算法。
2. **matrix-free Krylov / warm-start CGLS 近似。**若 exact oracle 保留至少 25% 原始 field gain，
   再用固定迭代、匹配总调用预算的 LSQR/CGLS 近似 row-space removal，并与 CGLS/Huber、
   base-only Krylov、公平参数 CNN 全面对照。

若 exact oracle 都抹掉场收益，监督 residual 路线应停止。若 oracle 可行、matrix-free 近似不可行，
论文问题才变成“怎样以有限调用近似 BOST 几何的 affine/null-space projection”。这可能是算法贡献，
但零空间网络或数据一致性本身已有明确先例，不能作为原创标题。

相关先例：

- [Deep Null Space Learning for Inverse Problems](https://arxiv.org/abs/1806.06137)
- [Learned Primal-Dual Reconstruction](https://arxiv.org/abs/1707.06474)
- [MoDL](https://arxiv.org/abs/1712.02862)
- [Learned Operator Correction](https://doi.org/10.1137/20M1338460)

## 6. 可复现入口

- [v1.1 冻结配置](../demo_t16_operator/configs/jacru_m2_1_matched_data_consistency_postopen_v1_1.json)
- [机器摘要](../demo_t16_operator/results/jacru_m2_1_matched_data_consistency_postopen_public/summary.json)
- [1,620 行 learned 轨迹](../demo_t16_operator/results/jacru_m2_1_matched_data_consistency_postopen_public/metric_rows.csv)
- [450 行 matched 基线](../demo_t16_operator/results/jacru_m2_1_matched_data_consistency_postopen_public/matched_baseline_rows.csv)
- [诊断图 PDF](../demo_t16_operator/results/jacru_m2_1_matched_data_consistency_postopen_public/diagnostic.pdf)
- [runner](../site_tools/run_jacru_m2_1_data_consistency_diagnostic.py)
- [data-consistency 实现](../demo_t16_operator/jacru_m2_data_consistency.py)

```bash
PYTHONPATH=. .venv/bin/python \
  site_tools/run_jacru_m2_1_data_consistency_diagnostic.py
```

本轮最重要的结论仍是负的：**网络先验有场信息，但当前有限调用机制无法把它变成可信的物理
重建。**
