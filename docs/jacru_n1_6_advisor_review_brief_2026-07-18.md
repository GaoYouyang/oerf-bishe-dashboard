# 给何远哲师兄：N1.6 判决与 N1.7 最小研究提案

- 日期：2026-07-18
- 希望师兄审核：**物理问题是否真实、算子接口是否可得、N1.7 是否值得进入真实数据阶段**

## 1. 我现在没有在做什么

- 没有宣称新算法已经优于 FNO、DeepONet、NeRIF 或 TDBOST。
- 没有把 synthetic mean gain 写成真实 BOST 成功。
- 没有继续扩大课题组方向；主线只保留三维 BOST 的 forward mismatch 与 operator correction。
- 没有在已经打开的 development 上事后重调 N1.6。

## 2. N1.6 做了什么

目标是不用 test truth、fresh exact mismatch 或高阶 forward，在低阶 BOST 重建中估计一个
measurement correction。候选只读 geometry、measured observation、CGLS-12 warm field 与其
projection；输出先留在 measurement space，再经过当前几何的 `A^T`。

预算完全匹配：候选和 CGLS-24 都是 `25F/24A^T`；候选部署的高阶 F/A^T 为 `0/0`。

## 3. 一次性 opened-development 判决

6 个 geometry clusters、12 个 paired fields：

| 指标 | 结果 | 预注册要求 |
|---|---:|---:|
| field gain vs CGLS-24 | `+3.539%` | `>=5%` |
| H1 gain | `+8.242%` | `>=3%` |
| worst field gain | `+0.167%` | `>=-1%` |
| field gain vs component damping | `-0.184%` | `>=+0.5%` |
| takeover | `50%` | `>=75%` |
| high-order deployment calls | `0/0` | `0/0` |

总判决：`POSTOPEN_NO_GO_NO_CONFIRMATION_ROUTE`。

## 4. 失败分解

- exact mismatch oracle：field `+8.616%`，worst `+2.203%`；说明 model correction 有物理
  headroom，但 oracle 不可部署。
- shared rank-4 adjoint oracle：field `+4.985%`；表示空间已丢掉明显幅度，并有相对 damping
  的尾部伤害。
- raw ridge predictor：伴随残差相对 damping `-25.357%`；预测方向没有迁移。
- fail-closed：阻止了明显坏预测，但 50% case 回退，最终仍输给 damping。

我的理解：问题不只是网络容量，首先是跨几何共享的静态 PCA basis 不随射线路径和 `A/A^T`
旋转。

## 5. N1.7 最小提案

暂名 KCRC，先只做 representation gate：

```text
x_w = CGLS12(A, y)
r   = y - A x_w
B_z = [damping, r, A A^T damping, A A^T r]
delta_y = damping + B_z c
```

- `B_z` 由当前 geometry 的 `A/A^T` 和 measured residual 生成，不用 global PCA。
- 先让 evaluator 求最优 `c`，检查这个可部署 basis 的 oracle ceiling。
- 两次 `AA^T` probe + CGLS-10 refine，使总预算仍为 `25F/24A^T`。
- 只有 oracle 同时超过 damping、高阶教师、5% field 和逐 geometry tail，才训练有界小网络预测
  `c`。
- learner 输出 measurement 系数，不直接输出三维场；训练目标穿过有限步 CGLS，以 field/H1 为主。

## 6. 请师兄帮我判断的六个问题

1. **真实主导误差：**当前系统最值得先建模的是 finite aperture、ray bending、camera calibration，
   还是 optical-flow bias？能否按影响排序？
2. **算子接口：**现有代码能否 matrix-free 调用严格配对的 `A(v)` / `A^T(u)`？一次调用大概多贵？
3. **高低保真配对：**能否对同一 field/geometry 运行 `G_L/G_H`，并记录 f-number、焦面、焦距、
   背景距离和 calibration perturbation？
4. **真实数据 packet：**能否给一小段原始 reference/distorted images、camera parameters、
   displacement、mask/confidence 和当前 reconstruction output？
5. **独立验证：**能否永久留一台 camera、一个 session 或一个 f-number，不参与训练、选模和停止？
6. **论文目标：**师兄更认可“有限孔径/弯曲光线的快速 operator correction”，还是“通用的
   geometry-conditioned correction layer”？我倾向前者作物理主问题，后者作方法结构。

## 7. 希望师兄给出的最小回复

```text
主导 mismatch 优先级：
A/A^T 可调用：是 / 否 / 需要改接口
G_L/G_H paired data：有 / 可生成 / 暂无
可永久留出的 audit evidence：
N1.7 representation gate：继续 / 修改 / 停止
最需要改的数学或物理假设：
```

完整复现、oracle 表和证据边界：
[N1.6 严格判决](jacru_n1_6_adjoint_low_rank_no_go_2026-07-18.md)。
