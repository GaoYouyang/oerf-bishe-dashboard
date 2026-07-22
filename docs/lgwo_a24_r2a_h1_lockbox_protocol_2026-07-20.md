# LGWO-A24 R2-A H1 新 seed synthetic lockbox 协议

状态：`FROZEN_BEFORE_FIRST_SYNTHETIC_LOCKBOX_RUN`

## 1. 候选已经冻结

opened validation 的边界扩展显示，L2-only 没有 signal，收益主要由 H1 项产生。在满足 mean active/held-out
safeguard 的点中，`lambda_h1=1e-3, k=20` 的 field/gradient max-min 最强，因此本 lockbox 只运行：

- baseline：unregularized fully reorthogonalized CGLS；
- candidate：同一个 solver 壳上的 H1 增广算子，`lambda_l2=0, lambda_h1=1e-3`；
- 两者都从零场出发，最多 `20F/20A^T`，保存 `k={4,8,12,16,20}`；
- test 时禁止改 lambda、checkpoint、support、spacing、whitening、门或指标。

这不是“我们的新算法”。H1-Tikhonov 是经典基线。本轮只判断它是否足够稳定，值得作为后续 hybrid/learned
路径必须击败的基线。

## 2. 新 seed 与 cluster

所有 field、mask、noise seeds 与 R0/pilot 分离。每个 phantom seed 是一个 cluster，同一个体场配 4 个不同
mask/noise replicate；bootstrap 只重采样 cluster，并把其 4 个 replicate 一起带走。
两个 primary split 的对应 cluster index 共享同一组 4 个 mask，因此四个 primary endpoint 在每个
bootstrap draw 中共用同一组 mask-block 索引，同时保留 split 内 field/gradient 与 split 间几何难度的相关性。
simultaneous lower bound 使用 studentized cluster-bootstrap max-T，而不是四条各自独立的 percentile interval。

| split | role | clusters × replicates | morphology | views | noise | truth operator |
|---|---|---:|---|---:|---:|---|
| lockbox_iid | primary | 24×4 | plume / wavy_front | 6–8 | 1.5–5.5% | qmc32 |
| lockbox_family_ood | primary | 24×4 | thin/double front | 6–8 | 2–6% | qmc32 |
| lockbox_joint_ood | secondary OOD | 24×4 | thin/double front | 4–5 | 8–12.5% | qmc32 |
| no-mismatch stress | secondary diagnostic | 12×4 | plume / wavy_front | 6–8 | 2–6% | qmc8 |

两个 primary split 共享同一 mask bank，目的是把几何模式配对，只改变 morphology/field/noise；它们不能当作
完全独立 rig。所有结论仍限于同一 PSU public geometry 与同一 analytic generator family。
每个 case 最多使用 8/9 个视角，强制至少留出 1 个视角给 held-out B；不得将全视角 case 的空集残差当作 0。
开盲前将 84 个 phantom seed 与 6 个唯一 mask/noise seed 展开为 90 个整数，对 606 个 tracked JSON 做了
exact integer-value 扫描，排除本 config 后的旧命中文件数为 0；runner 只保证本 lockbox
内 field range 不重叠、noise seed 各 split 唯一与预注册 mask seed 共享，不把“全局历史 seed 新鲜性”写成运行时已证明。

## 3. 主要门

在 IID 与 family-OOD 两个 primary split 中分别要求：

1. 两个 split × 两个 endpoint 共四项使用 cluster-bootstrap max-T simultaneous 95% lower bound，均不低于 0；
2. 每个 primary split 至少一个 endpoint 的 simultaneous lower bound 达到 1%；
3. 直接由 raw mean error 计算的 aggregate field/gradient gain 均不低于 0，且至少一项达到 1%；
4. candidate mean point 不被任何预算内 CGLS checkpoint 支配，且至少支配一个 CGLS checkpoint；
5. cluster p10 field/gradient gain 均不低于 -1%，worst cluster 均不低于 -2%；
6. cluster 与单 case 中任一 endpoint 低于 -2% 的 rate 均不超过 10%，worst case 不低于 -5%；
7. mean active residual ratio ≤1.05，cluster p90 ≤1.10，worst case ≤1.15；
8. mean held-out-B ratio ≤1.05，worst case ≤1.10。

joint-OOD 同时要求 case-relative mean 与 raw aggregate mean 均非负、cluster p10 ≥-2%、worst cluster/case ≥-5%、
cluster/case harm-rate 均≤10%，active mean/p90/worst ≤1.10/1.20/1.35，held-out mean/worst ≤1.10/1.30。
front F1 和图像观感不能挽救主要门失败。
no-mismatch stress 的 field/noise 没有与 IID 成对，因此它只是一个非配对的 qmc8 诊断面板，不能单独识别
qmc32→qmc8 mismatch 因果；事先不将它设为决策门，也不得事后用它挑选子集。

## 4. 执行与封存

- 执行身份固定为本 config、本地 PSU geometry audit root、Apple MPS 与唯一输出目录；CLI 不提供 override；
- 每条 solver path 必须精确 `20F/20A^T`；generation、solver、evaluator 的 forward 成本分开记录；
- 进程在任何仓库本地模块 import 前先捕获 bootstrap HEAD，导入后只允许继续使用该 commit；写完图表后和
  rename 前再次核对 HEAD、tracked-clean 与所有绑定 hash；
- 任何后段失败会把已写产物改名为 `partial_*`，根目录权威状态只保留 `INVALID_*` failure。

## 5. 状态解释

- `INVALID_*`：基础设施、hash、调用账、finite、residual consistency 或矩形失败，禁止科学解释；
- `VALID_*_NO_GO_*`：合法负结果，不得选子集挽救；
- `VALID_R2A_SYNTHETIC_LOCKBOX_PARETO_SIGNAL_AUTHORIZE_HYBRID_COMPARISON_ONLY`：
  只说明经典 H1 路径在当前新 seed synthetic contract 下稳定，允许下一步做 hybrid/TV/有界学习比较。

即便通过，也仍然不是：真实 BOST 结果、未见 rig 泛化、新算法、NeRIF/TDBOST 优越性、论文成功或突破。
