# PoolFire C v10.5 fit-only 留一频谱分配结果

## 一句话结论

fit-selected DCT 分配不是“天然更先进”：在 15 个轨迹与秩组合中，均匀低频
DCT 赢 10 个，fit-selected 赢 5 个；后者的 5 次胜利全部发生在 rank 384。

## 比较方法

对五条冻结 fit 轨迹逐条留一。每个 fold 只用其余四条轨迹的 K3 dual correction
选 DCT atom，再在 held-out 轨迹上与同秩均匀 DCT 比较。两边都经过完全相同的
`exact A^T -> observable alpha -> strict CGLS K1` 壳，参考是 zero-start CGLS K4
teacher。指标是三项 teacher deficiency 最大值 `q` 的 higher-p90，越低越好。

| rank | selected 胜 | uniform 胜 | `selected - uniform` 中位 |
|---:|---:|---:|---:|
| 96 | 0/5 | 5/5 | +0.09364 |
| 216 | 0/5 | 5/5 | +0.03233 |
| 384 | 5/5 | 0/5 | -0.00300 |

## 对下一步的影响

1. rank 96/216 不再把 fit-selected 当首选，它们目前被简单均匀低频控制稳定压过。
2. rank 384 保留 selected 与 uniform 两条臂，因为 selected 在五个 held-out fold
   都略好。
3. 不改已经冻结的 p14 12-arm 矩阵，让独立 validator 如实报告两个 family。
4. 这不是算法突破。它只是排除了“自适应选 atom 一定更好”的错误直觉。

## 证据边界

本轮只读取五条 fit observation，并由同一冻结 straight-ray operator 构造 K3/K4
证书；没有读取 p14 truth、p22 stopping validation、fresh 或 test。当前结果仍是
未封存的探索性 fit-only 诊断，正式引用前还要在 clean committed HEAD 下生成
不可覆盖 result bundle。

机器可读数据见
[`poolfire_c_dual_spectral_fit_loto_v10_5_public_summary.json`](poolfire_c_dual_spectral_fit_loto_v10_5_public_summary.json)。
