# v236：Case 7 留一 rig 尾差子空间归因否定固定全局低秩修补

## 结论

v235 已经在前瞻 Case 7 上否定固定 Direct Low64 K11：它虽然通过绝对精度门，但相对合格 K16 reference 只有 `330/546` 个匹配单元和 `0/13` 个完整 rig。v236 不改 K11、不换未打开工况，也不训练模型，而是在这个已开封负结果上回答一个更窄的问题：K16 与 K11 之间缺失的物理尾差，是否至少能被一个跨相机 rig 迁移的 64 维空间装下？

结果是否定的。每次完整留出一个 rig，用其余 12 个 rig 的 `504` 个尾差场建立空间基，再以真值可见的最优投影系数重建留出 rig。rank 64 的全局相对残差 `p50/p90/worst` 仍为 `0.645458 / 0.731692 / 0.805609`，13 个留出 rig 全部失败。rank 16、32 也都是 `0/13`；原固定 Low64 控制更差，为 `0.953580 / 0.974420 / 0.990718`。

| 尾差空间 | 完整 rig | 全局 p50 | 全局 p90 | 全局 worst |
| --- | ---: | ---: | ---: | ---: |
| 固定 Low64 控制 | `0/13` | `0.953580` | `0.974420` | `0.990718` |
| 留一 rig rank 16 | `0/13` | `0.670064` | `0.755728` | `0.836898` |
| 留一 rig rank 32 | `0/13` | `0.659817` | `0.748166` | `0.824446` |
| 留一 rig rank 64 primary | `0/13` | `0.645458` | `0.731692` | `0.805609` |

门槛在看留一 rig 结果前冻结：每个 rig 的全帧与后期帧都必须达到 `p90 <= sqrt(0.1)=0.316228`、`worst <= 0.5`。rank 64 最差 rig 的全帧 p90/worst 为 `0.788572/0.805609`，后期帧 p90/worst 为 `0.628163/0.630998`，不是擦线失败。

## 为什么全体 SVD 看起来低秩，留一 rig 却失败

在冻结 v236 主门前，我们已经公开披露开封后观察：把 13 个 rig 全部混在一起做 SVD，90%、95%、99% 总能量只需要 rank `18/24/74`；相邻帧尾差方向也很平滑。但这不能证明跨 rig 迁移，因为被测试 rig 自己已经参与了那个联合空间。

留一 rig 后，结论反转。每个 rig 都贡献了不同的主导物理方向；其他 12 个 rig 的 rank-64 空间只能保留留出尾差约一半左右的能量。换句话说，“联合数据低秩”是一个会泄漏目标 rig 的漂亮假象，不是可部署的共享修正基。

失败的时间结构仍有物理信息：0–24 帧所有 rig 都匹配，frame 25 有 8 个 rig 失败，26–41 帧则 13 个 rig 全部失败；但固定 Low64 只保留尾差能量中位数约 `9.11%`。这说明晚期反应场演化与相机几何共同改变了所需方向，不能靠给现有 K11 壳附加一个固定全局低秩场修补解决。

## 独立验证

正式实现用 `504x504` 样本 Gram 特征分解，并显式重建物理基；独立程序不导入正式 runner，对 13 个 `504x8192` 训练矩阵逐一做直接 economy SVD。逐单元相对残差最大差为 `6.66e-15`，汇总最大差为 `5.77e-15`，离散判决完全一致，输入与输出封存不变。

最终科学判决是 `FAIL_CASE7_LORO_TAIL_SUBSPACE_CAPACITY_V236`。关闭的是固定全局低秩尾差修补解释，不是整个 C 路线，也不证明所有 geometry-conditioned、局部或非线性物理机制不可能。不得用更大 rank、CNN/FNO/UNO 或 GPU 把这次容量失败包装成成功，也没有授权 wall/RSS 或未打开外门。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v236: leave-one-rig Case 7 tail attribution rejects a fixed global low-rank repair

## Conclusion

v235 prospectively rejected fixed Direct Low64 K11 on Case 7. It passes the absolute-accuracy gate but matches the qualified K16 reference in only `330/546` cells and `0/13` complete rigs. v236 changes neither K11 nor the condition and trains no model. It asks a narrower post-open question: can the missing physical tail between K16 and K11 at least fit in a 64-dimensional spatial subspace that transfers across camera rigs?

The answer is no. Each fold holds out one complete rig, builds a spatial basis from `504` tail fields in the other twelve rigs, and uses truth-aware optimal projection coefficients on the held-out rig. At rank 64, global relative-residual `p50/p90/worst` remains `0.645458 / 0.731692 / 0.805609`, and all 13 held-out rigs fail. Ranks 16 and 32 also reach `0/13`; the original fixed Low64 control is worse at `0.953580 / 0.974420 / 0.990718`.

| Tail space | Complete rigs | Global p50 | Global p90 | Global worst |
| --- | ---: | ---: | ---: | ---: |
| Fixed Low64 control | `0/13` | `0.953580` | `0.974420` | `0.990718` |
| Leave-one-rig rank 16 | `0/13` | `0.670064` | `0.755728` | `0.836898` |
| Leave-one-rig rank 32 | `0/13` | `0.659817` | `0.748166` | `0.824446` |
| Leave-one-rig rank 64 primary | `0/13` | `0.645458` | `0.731692` | `0.805609` |

The gate was frozen before any leave-one-rig result: every rig had to satisfy all-frame and late-frame `p90 <= sqrt(0.1)=0.316228` and `worst <= 0.5`. The worst rank-64 rig reaches all-frame p90/worst `0.788572/0.805609` and late-frame p90/worst `0.628163/0.630998`, so this is not a borderline miss.

## Why the joint SVD looked compact

Before freezing the v236 primary, the disclosed post-open exploration showed that an SVD mixing all 13 rigs needs only ranks `18/24/74` for 90/95/99 percent total energy, and adjacent-frame tail directions are smooth. That does not establish rig transfer because the tested rig already contributes to the joint space.

Holding out a rig reverses the conclusion. Each rig contributes different dominant physical directions; a rank-64 space from the other twelve rigs retains only roughly half of the held-out correction energy. The compact joint spectrum is therefore a target-rig-leaking appearance, not a deployable shared correction basis.

The chronology remains informative: every rig matches through frames 0–24, eight rigs fail at frame 25, and all thirteen fail throughout frames 26–41. Yet fixed Low64 retains only about `9.11%` median tail energy. Late reacting-flow evolution and camera geometry jointly change the needed directions, so attaching one fixed global low-rank field repair to the current K11 shell is not supported.

## Independent validation

The formal implementation uses a `504x504` sample-Gram eigendecomposition and explicitly reconstructs physical bases. The independent program imports no formal runner and directly computes an economy SVD for each of the thirteen `504x8192` training matrices. Maximum per-cell residual and summary differences are `6.66e-15` and `5.77e-15`, with identical discrete decisions and unchanged sealed inputs and outputs.

The final scientific decision is `FAIL_CASE7_LORO_TAIL_SUBSPACE_CAPACITY_V236`. This closes the fixed global low-rank tail-repair explanation, not the full C route, and it does not prove all geometry-conditioned, local, or nonlinear physical mechanisms impossible. Increasing rank or using CNN/FNO/UNO or GPU rescue is not authorized; neither wall/RSS nor an unopened external gate is authorized.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
