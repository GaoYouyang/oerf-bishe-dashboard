# 给何远哲师兄：N1.7 表示门判决与下一步请教

- 状态：`REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER`
- 数据：opened synthetic development，6 geometry / 12 paired fields
- learner、真实 BOST、OOD/fresh/final：均未打开

## 我做了什么

用当前低阶算子构造每个 case 自己的 measurement correction frame：

```text
B = orth([d, r, A P A^T d, A P A^T r])
correction = d + Bc
```

其中 `d` 是 component damping，`r` 是 CGLS-12 warm residual，`P` 是重建 support。basis 不读
truth 或 family。两次 probe 加 CGLS-10 refine，与 CGLS-24 和 damping-refine12 都严格匹配
`25F/24A^T`。

## 冻结结果

| 指标 | measurement oracle | 要求 |
|---|---:|---:|
| field gain vs CGLS-24 | `+4.828%` | `>=5%` |
| H1 gain | `+11.076%` | `>=3%` |
| field gain vs damping | `+1.193%` | `>=1%` |
| worst case / worst geometry | `+1.673% / +2.716%` | 都 `>=0` |
| smooth / interface gain vs damping | `+0.247% / +2.140%` | 各自 `>0` |
| support-adjoint gain vs damping | `+16.281%` | `>=50%` |
| exact-refine10 headroom retention | `56.717%` | `>=70%` |

17 项门过 14 项，但三个核心门失败，所以没有训练系数网络。

finite-K truth search 找到 `+5.560%` field，但用了额外 `33,780F/33,780A^T` evaluator calls，
retention 仍只有 `65.313%`，36 个 Powell 起点只有 5 个报告收敛。它只能说明当前 span 里还有
一点 solver-aware 方向，不能叫可部署算法。

## 我目前的判断

- exact mismatch refine-10 仍有 `+8.513%`，物理 headroom 没消失。
- per-case frame 比 frozen global PCA rank-4 只高约 `0.197` 个 field 百分点，geometry
  conditioning 有帮助但不够。
- 12/12 measurement coefficients 都撞到冻结 trust radius，所以失败不能只归因于 basis，也
  可能有保守边界过紧；不能在同一 development 上放宽后重判。
- 下一步优先考虑同 2-probe 预算的 camera-block / ray-coordinate modulation，而不是换大 MLP。

## 事后半径审计：它改变机制判断，不改变正式判决

为区分 span 与安全边界，我把三个 visible radius 系数统一放大四倍，并把 Powell `maxfev` 从
96 提高到 384；数据、basis、17 项门和预算都不变。measurement projection 不再触边，结果为
field `+5.556%`、retention `65.264%`、adjoint gain `28.364%`，仍只过 15/17。truth-conditioned
finite-K 为 field `+6.186%`、retention `72.669%`，只剩 adjoint 门失败，过 16/17；但 development
搜索用了 `74,010F/74,010A^T`，不可部署。

因此半径确实重要，但“只预测更大系数”仍不够。下一步应在新 split 上比较 current Krylov-4、
camera/ray hybrid-4 与 camera-block rank-6，再决定是否训练 learner。完整审计见
[N1.7-D 半径敏感性](jacru_n1_7_radius_sensitivity_audit_2026-07-18.md)。

## 想请师兄确认

1. `A/A^T` 是否能严格配对并显式应用当前 support？一次调用和一组 RHS 的真实成本是多少？
2. 真实 mismatch 优先级是 finite aperture、ray bending、camera calibration 还是 optical-flow
   bias？哪一个最值得做论文主问题？
3. camera index、ray coordinate、mask/confidence 是否可作为部署可见 basis 输入？
4. 能否给同一 field/geometry 的 `G_L/G_H` paired projection，或一个最小真实失败 packet？
5. 能否永久留一台 camera、一个 session 或一个 f-number，只做最终审计？
6. 您更认可“快速物理 forward correction”，还是“solver-aware regularizer”？如果 field-optimal
   correction 不再逼近真实 mismatch，我会主动改名，不混淆物理含义。
7. 预注册的 `P A^T gain >=50%` 是否应保留为 forward correction 的物理门，还是以 held-out
   reprojection / field / flame-front 作为下一轮主终点？

完整数字、复现和边界见 [N1.7 严格判决](jacru_n1_7_geometry_krylov_no_go_2026-07-18.md)。
