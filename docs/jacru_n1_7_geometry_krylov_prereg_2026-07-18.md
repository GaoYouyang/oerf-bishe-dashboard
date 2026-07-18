# N1.7 几何条件化 Krylov 表示门：预注册

- 冻结日期：2026-07-18
- 阶段：`POSTOPEN_REPRESENTATION_ORACLE_ONLY`
- 数据：只重用已经打开的 synthetic fit / calibration / development
- OOD / fresh / final / 真实 BOST：全部保持关闭
- learner：本轮不存在，也不训练

## 1. 要证伪的问题

N1.6 已经否掉跨几何共享的 measurement PCA 表示和 ridge predictor，但 exact mismatch oracle
仍保留约 8.62% 的 field headroom。N1.7 只问一个更窄的问题：

> 由当前低阶几何算子现场生成的四维 measurement correction space，能否在相同低阶物理调用
> 预算下保留足够的有限步三维重建收益？

如果表示 oracle 不过门，本轮停止，不训练系数网络，不打开 confirmation。

## 2. 部署可见表示

先运行 CGLS-12 得到 `x_w`，再显式计算 `p_w=A x_w`。令：

```text
d = component damping correction
r = (y - p_w) / signal_scale
P = reconstruction support projection
K(v) = A P A^T v
B = orth([d, r, K(d), K(r)])
correction(c) = d + B c
```

`B` 使用 float64、固定顺序、两遍 modified Gram-Schmidt；近共线方向按 `1e-10` 删除。构造
函数只接收 `d/r/A/A^T/P`，签名中没有 truth、exact mismatch、family label 或高阶 output。

这里必须是 `A P A^T`，因为实际 CGLS 在每次伴随之后应用 support。使用裸 `AA^T` 会让
basis 与求解器可达方向不一致。

## 3. 公平调用预算

| 阶段 | N1.7 candidate | component damping | low CGLS-24 |
|---|---:|---:|---:|
| warm solve | 12F / 12A^T | 12F / 12A^T | 24F / 24A^T |
| warm projection | 1F | 1F | final projection 1F |
| Krylov probes | 2F / 2A^T | 0 | 0 |
| refine | 10F / 10A^T | 12F / 12A^T | 0 |
| **总计** | **25F / 24A^T** | **25F / 24A^T** | **25F / 24A^T** |

总调用数由 warm、projection、probe、refine 的实际 operator counter 差值相加，不允许只从配置
抄写。两个 probe 顺序执行，不用 batch 规避调用计数。

## 4. 系数边界

每个 case 的系数 L2 半径只由部署可见尺度决定：

```text
rho = min(0.50 ||d||, max(4.0 ||r||, 0.10 ||d||))
||c|| <= rho
```

fresh exact mismatch 不参与搜索半径。该边界允许在 warm residual 较大时扩大修正，但最多只能
改变 damping anchor 的 50% L2 尺度。

## 5. 三种 oracle 的角色

1. `measurement_projection_oracle`：把 evaluator-only exact residual `epsilon-d` 投影到 `B`，
   再经过固定 CGLS-10。它是主 representation gate；通过只允许预注册下一轮 learner。
2. `adjoint_projection_oracle`：最小化当前 `A^T` 诱导残差，只作机制对照。
3. `truth_conditioned_finite_k_oracle_search`：在同一 L2 球内，以真实 field relative-L2 为
   目标，每个系数都完整重跑 CGLS-10。固定 zero / measurement / adjoint 三个起点，每个起点
   Powell 最多 96 次函数评估。它不保证全局最优，且所有搜索调用单列为 evaluator cost。

finite-K oracle 即使通过，也不是可部署算法；若 measurement projection 不通过，只能说明可能
存在“学习型正则器”支线，不能把它写成 operator correction 成功。

另外固定两个隔离对照：N1.6 已冻结的 global PCA rank-4 affine oracle 支付同样的两次 probe
tax 并只 refine 10 步，用来判断收益是否真的来自 per-case operator conditioning；exact mismatch
再增加一条不付 probe tax、refine 12 步的同总预算路线，用来量化“表示 probe 占掉两步 CGLS”
本身造成的损失。70% retention 的分母始终是 exact-mismatch refine-10，而不是 N1.6 的旧数字。

## 6. 冻结门

主 candidate 是 `measurement_projection_oracle`。development 上必须同时满足：

- mean field gain vs matched low CGLS-24 `>=5%`；
- mean H1 gain `>=3%`；
- mean field gain vs component damping `>=1%`，且两个 field family 各自都必须为正；
- mean field gain vs high-order teacher beta=.75 `>=0`；
- worst case gain vs low `>=0`，worst geometry gain `>=0`；
- 相对 low 和 damping 的 >1% harm rate 都为 0；
- support-projected evaluator mean adjoint-residual gain vs damping `>=50%`；
- 至少保留 exact mismatch oracle mean field gain 的 70%；
- 实际部署账本为 25F/24A^T、0 high-order F/A^T；
- 每个 case basis rank `>=2`，正交缺陷 `<=1e-10`。

## 7. 预先冻结的判决

- 主 representation gate 全过：`REPRESENTATION_ELIGIBLE_LEARNER_NOT_YET_TESTED`。
- 主门失败但 finite-K diagnostic 全过：`FIELD_ORACLE_ONLY_REFRAME_OR_STOP`。
- 两者都失败：`REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER`。

任何状态都禁止声称 learned gain、真实 BOST 泛化、论文算法成功或优于 DeepONet/FNO/NeRIF/
TDBOST。development 已打开，本轮只能形成下一假设，不能作为独立 confirmation。
