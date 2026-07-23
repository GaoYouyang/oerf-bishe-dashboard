# JACRU-M2.9 fixed learned warm-start CGLS 冻结协议

**冻结日期：** 2026-07-20
**冻结状态：** `FROZEN_BEFORE_FIRST_M2_9_EXECUTION_OPENED_SYNTHETIC_ONLY`
**证据范围：** 已打开的 M2-T0 train / development / exploratory-OOD
**Fresh / 真实 OERF：** 不打开
**突破监测：** 尚无算法突破

## 1. 为什么做这一支

GCT-KMix 已严格否决“只在零初值 CGLS checkpoint 凸包里选权重”：即使 truth 事后选择，
development/OOD 的 field-gain 乐观上界也只有 `2.3334% / 1.9649%`，低于冻结的 `5%` 门。
M2.2 则在 12 立方 toy 中表明，learned proposal 的离散 numerical-null 分量存在明显 field
headroom。M2.9 因此不再学习 selector，而测试一个更直接的问题：

> 固定 learned field 作非零初值后，CGLS 能否保留它带入的离散 exact-kernel 先验，同时用同一个
> `A/A^T` 在严格预算内修正可观测分量？

这不是新网络。JACRU、pooled CNN、grid DeepONet 与 pooled FNO 的 12 个模型已经在 reconstruction
评分前保存为无 pickle NPZ；manifest 文件 SHA-256 为
`a0fffd545d4cadb6ca75564f98efbe03c67a6a5bc02b40301df2158883f87636`。评分 runner 只能加载这些
确切字节，不能临场重训。

## 2. 算法与不变量

记 support-restricted 线性算子为 `A_S = A P_S`，固定 learned 初值为 `x_0`。warm-start CGLS
只产生

```text
x_k = x_0 + q_{k-1}(A_S^T A_S) A_S^T (y - A_S x_0).
```

因此在精确算术中，`x_k-x_0` 属于 `range(A_S^T)`。对离散 kernel 的正交投影 `P_ker`：

```text
P_ker x_k = P_ker x_0.
```

也就是说，CGLS 可以改正 learned 初值的 row-space 部分，但不会凭空删除它的离散 exact-kernel
分量。小奇异值 near-null 分量会随 CGLS 多项式改变，协议不声称严格保留 near-null。这个结论
只针对冻结线性离散算子和固定 support，不等于真实有限孔径、弯曲光路的连续光学零空间。
float64 运行仍要用 retrospective dense SVD 检查 numerical drift；SVD 不进入算法。

算法 API 不接受 truth、family、field metric 或 test mask。learned prediction 在 warm CGLS 前
固定，不能边看轨迹边用 truth 改网络或 checkpoint。

## 3. 调用预算

M2-T0 的 learned proposal 不是免费初值。每个 case 先支付：

```text
CGLS-12 base                  12 F / 12 A^T
terminal residual projection  1 F
camera-grouped lift                1 grouped A^T
feature preparation total     13 F / 13 A^T
```

`grouped A^T` 已包含在 13 个 adjoint calls 中，但它不宣称与一个 pooled adjoint flop 等价。
`k=0` 是原始固定 learned field，只作必要控制，总成本为 `13F/13A^T`。一旦 `k>0`，非零初值
必须另付 `1 F` 计算 `A x_0`。warm CGLS 的第 `k` 个 checkpoint 因而是：

```text
candidate total = (14+k) F / (13+k) A^T, plus neural inference
paired budget   = 14+k
```

每个 `k>0` 同时报告三种经典控制：

1. componentwise CGLS `13+k` 步，逐项不超过候选的 F/A^T；
2. pair-rounded CGLS `14+k` 步，多给经典法一个 adjoint，是偏强对照；
3. pair-rounded Huber-PDHG `14+k` 步。

另跑与 warm CGLS refinement 完全相同调用的 M2.4 affine-CGNE 控制。两者进入同一个仿射 Krylov
空间，但 warm CGLS 在该空间直接最小化 data residual；CGNE 用作机制消融，不参与替候选选步。
所有方法额外评分 forward 都单独记录，不算优化预算。训练、特征准备、网络推理、物理迭代和
retrospective SVD 必须分栏报告。

## 4. 数据与选择

- train：只训练四个冻结 M2-T0 proposal；
- checkpoint fit 只用 32 个 train cases 与 base seeds `2101/2129/2153` 的 6 个 early-stop cases；
- route-selection development 只含 base seeds `2113/2141/2161` 的 6 cases / 3 geometry clusters；
- 在 `k=0..10` 中为每个 architecture 选一个全局固定 checkpoint；
- exploratory-OOD：只能评分 development 已选 checkpoint，不能反向改 `k`；
- 同一 geometry 下的多个 family 不冒充独立 rig；结果是 opened synthetic mechanism evidence。

field/H1 的保守参考是同 case、同 budget 下 componentwise CGLS、pair-rounded CGLS 与 Huber-PDHG
中 field 更好的一个；这个 per-case truth oracle 只属于 evaluator。主 measured/independent-clean
reprojection ratio 始终除以 pair-rounded CGLS，同时另报 componentwise CGLS 分母，避免换分母制造胜利。

## 5. 结果前冻结门

development 必须同时满足：

1. mean field gain `>=5%`；
2. mean H1 gain `>=3%`；
3. measured 与 independent-clean reprojection ratio 都 `<=1.10`；
4. `>1%` field harm rate `<=5%`，worst field gain `>=-5%`；
5. 三个 model seed 的 mean field gain 全部为正；
6. breakdown rate 为 0，CGLS recurrence residual closure `<=1e-12`，retrospective exact-kernel
   component relative drift `<=1e-10`。

若存在多个 development 合格点，先取 mean field gain 最大者，再取 paired budget 较小者，再取
较小 `k`。exploratory-OOD 对这个固定点要求 field/H1 gain `>=2% / >=0%`，两个 reprojection
ratio `<=1.15`，并保持同样 harm、worst、seed、breakdown 与 drift 门。

## 6. 判决语言

- 没有 development 合格点：`M2_9_FIXED_WARM_CGLS_DEVELOPMENT_NO_GO`；
- development 过门而 OOD 失败：`M2_9_FIXED_WARM_CGLS_OOD_NO_GO`；
- 两者都过：`M2_9_FIXED_WARM_CGLS_OPENED_MECHANISM_PASS_NO_FRESH`。

即使第三种状态出现，也只能说“固定 learned warm start 在当前 opened synthetic 离散算子中通过
预注册机制门”。不能说原创算法、真实 BOST、跨 rig 泛化、NeRIF/TDBOST/DeepONet/FNO
优越性或论文成功。它至多授权起草一个使用外部 forward/真实 OERF 数据的新预注册实验。

## 7. 停止规则

- 任一源文件/hash、split、预算或门槛漂移：fail closed；
- 任一模型用 OOD 选 epoch/checkpoint：fail closed；
- 任何候选只靠 pooled mean 过门但逐 case harm 超门：NO-GO；
- retrospective SVD setup 不得进入算法或效率排名；
- 本轮不训练新的 selector、risk gate 或 null-space network，不打开 fresh/final。
