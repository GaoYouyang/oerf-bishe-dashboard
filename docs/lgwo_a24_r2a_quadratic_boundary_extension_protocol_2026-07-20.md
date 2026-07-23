# LGWO-A24 R2-A 二次型边界扩展协议

状态：`FROZEN_BEFORE_FIRST_BOUNDARY_EXTENSION_RUN`

## 为什么必须扩一次边界

第一轮 28 点 pilot 只有 `(lambda_l2=1e-2, lambda_h1=1e-4)` 在 `k20/k24` 形成预设 signal，且它正好是
mixed Sobolev 网格的最大点。独立复算确认 24/24 validation cases 的 field 与 gradient 都为正收益，但这是
opened validation 上的边界现象，不能直接冻结后去 test：我们还不知道更强正则会继续改善、出现峰值，还是开始过平滑；
也不知道主要贡献来自 L2、H1 或二者组合。

## 冻结设计

仍只运行相同的 24 个 validation cases，不加载任何 test/OOD：

- CGLS：1 条；
- L2-only：`1e-2, 3e-2, 1e-1, 3e-1, 1`；
- H1-only：`1e-4, 3e-4, 1e-3, 3e-3, 1e-2`；
- mixed Sobolev：`lambda_l2={3e-3,1e-2,3e-2,1e-1}` 与
  `lambda_h1={3e-5,1e-4,3e-4,1e-3}` 的完整 4×4 factorial。

共 27 个配置。每个配置仍严格 `24F/24A^T`，评分 checkpoint、指标、support、spacing、门和 evaluator
与第一轮完全相同。第一轮最佳点在 factorial 中原样复测，作为跨 run 稳定性哨兵。

## 只允许得到的结论

这轮最多可以：

1. 找到 validation optimum 是否被两侧较差配置括住；
2. 描述 L2-only、H1-only 与 mixed 的平均路径差异；
3. 冻结一个单一 `(lambda_l2, lambda_h1, k)` 供新 seed D1/D2 设计；
4. 若所有更强点持续改善而最优仍在边界，则判 `SCALE_NOT_BRACKETED`，不得进入 test。

即使出现很大的 mean gain，也不能称为算法优越性、泛化、真实 BOST、论文结果或突破。

## 后续 formal gate

只有 optimum 被括住后，才新建与当前 seed/mask/noise 不重叠的 D1 tune 和 D2 lockbox。D2 运行前冻结：

- 单一配置和 checkpoint；
- IID 与 family-OOD 双主分区；
- joint geometry+noise OOD 与 exact-operator control；
- phantom-cluster bootstrap、逐 cluster p10/worst/harm；
- simultaneous field/gradient non-inferiority 与至少一项实质改善；
- active/held-out residual、调用、wall time、RSS 和完整失败表。

