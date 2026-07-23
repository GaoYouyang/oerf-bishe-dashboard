# N1.7 几何条件化 Krylov 表示门：opened-development NO-GO

- 日期：2026-07-18
- 状态：`REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER`
- 预注册提交：`4c2d1a9`
- 全量运行代码提交：`01ae0a8`
- 证据：E1 synthetic、已经打开的 development 机制筛选
- learner / OOD / fresh / final / 真实 BOST：全部未打开

> 机器状态名描述的是冻结的“rank-4 basis + visible trust radius + 17 项门”协议，不是否定所有
> Krylov 表示。事后四倍半径审计见 [N1.7-D 半径敏感性](jacru_n1_7_radius_sensitivity_audit_2026-07-18.md)。

## 一句话判决

由当前 `A P A^T` 现场生成的四维 correction space 比冻结 global PCA 略好，但没有保留足够的
exact-mismatch 重建 headroom。主 measurement projection oracle 的 field gain 为 `4.8281%`，
低于冻结 `5%` 门；support-projected adjoint gain 只有 `16.2811%`，低于 `50%`；exact oracle
headroom retention 为 `56.7173%`，低于 `70%`。因此停止在 representation gate，不训练 learner，
也不打开 confirmation。

## 1. 这次真正测试了什么

对每个 case 先运行 CGLS-12，并计算一次 warm projection。令：

```text
d = component damping correction
r = (y - A x_warm) / signal_scale
P = reconstruction support projection
K(v) = A P A^T v
B = orth([d, r, K(d), K(r)])
correction(c) = d + B c
```

`B` 只读 measured observation、warm state、damping、当前低阶 `A/A^T` 与 support。它不接收
truth、exact mismatch、field family 或 high-order output。两个 normal probes 顺序执行，实际账本
为 2F/2A^T；float64 两遍 modified Gram-Schmidt 在全部 development case 都保留 rank 4，最大
正交缺陷 `4.44e-16`。

系数 L2 半径只由部署可见尺度决定：

```text
rho = min(0.50 ||d||, max(4.0 ||r||, 0.10 ||d||))
```

## 2. 预算没有偷换

| 路线 | warm / projection | probes | refine | 总低阶调用 | 高阶调用 |
|---|---:|---:|---:|---:|---:|
| matched low CGLS-24 | - | - | 24 + final projection | 25F / 24A^T | 0 |
| component damping | 12F/12A^T + 1F | 0 | 12F/12A^T | 25F / 24A^T | 0 |
| N1.7 correction space | 12F/12A^T + 1F | 2F/2A^T | 10F/10A^T | 25F / 24A^T | 0 |

每条 case row 都保存 warm、projection、setup、refine 的分段实际 counter。finite-K 真值搜索的额外
调用另记为 evaluator cost，不塞进 25F/24A^T。

## 3. 主 representation gate

opened development 含 6 个 geometry clusters、每个 geometry 两种 paired field family，共 12 个
case。独立统计单位是 geometry cluster，不是 12 个独立 rigs。

| 冻结检查 | measurement projection oracle | 门 | 判决 |
|---|---:|---:|---|
| mean field gain vs CGLS-24 | `+4.8281%` | `>=5%` | fail |
| mean H1 gain vs CGLS-24 | `+11.0763%` | `>=3%` | pass |
| mean field gain vs damping | `+1.1932%` | `>=1%` | pass |
| mean field gain vs high-order teacher .75 | `+0.0930%` | `>=0` | pass |
| worst case field gain vs low | `+1.6733%` | `>=0` | pass |
| worst geometry gain vs low | `+2.7159%` | `>=0` | pass |
| >1% harm vs low / damping | `0 / 0` | 都为 0 | pass |
| single-interface / smooth gain vs damping | `+2.1399% / +0.2465%` | 各自 `>0` | pass |
| support-projected adjoint gain vs damping | `+16.2811%` | `>=50%` | fail |
| exact-refine10 headroom retention | `56.7173%` | `>=70%` | fail |
| low / high-order deployment calls | `25F/24A^T` / `0` | matched / `0` | pass |

17 项检查通过 14 项，三项失败。不能把它表述为“基本通过”，因为失败的正是主 field 幅度、
solver-visible mismatch 与 recoverable headroom 三个核心门。

## 4. 对照把瓶颈放在哪里

| 路线 | field vs low | field vs damping | adjoint gain vs damping | exact headroom retention |
|---|---:|---:|---:|---:|
| component damping refine-12 | `+3.7201%` | `0` | `0` | `43.7006%` |
| probe schedule + damping, c=0 | `+3.7284%` | `+0.0080%` | `0` | `43.7984%` |
| frozen global PCA rank-4 + probe tax | `+4.6313%` | `+0.9932%` | `+28.8736%` | `54.4056%` |
| N1.7 measurement projection | `+4.8281%` | `+1.1932%` | `+16.2811%` | `56.7173%` |
| N1.7 adjoint projection | `+4.2561%` | `+0.5677%` | `+17.0659%` | `49.9982%` |
| finite-K truth search | `+5.5599%` | `+2.0196%` | `+11.7282%` | `65.3135%` |
| high-order teacher beta=.75 | `+4.7341%` | `+1.1013%` | `+17.6843%` | `55.6132%` |
| exact mismatch refine-10 | `+8.5126%` | `+5.1901%` | `+100%` | `100%` |
| exact mismatch refine-12 | `+8.6164%` | `+5.3059%` | `+100%` | `101.2196%` |

结论有四层：

1. **probe tax 不是主因。** exact refine-10 与 refine-12 只差约 0.104 个百分点；`c=0` 的
   candidate schedule 也几乎复现 damping。
2. **geometry conditioning 有帮助但很小。** N1.7 measurement oracle 比 frozen global PCA
   高约 0.197 个 field 百分点，不足以跨门。
3. **当前四维 span 没抓住主要 mismatch。** development measurement / support-adjoint target
   residual ratio 平均仍为 `0.8538 / 0.8293`。
4. **trust region 也是瓶颈，但不是唯一瓶颈。** 12/12 development measurement coefficients 都
   撞到冻结半径。独立事后审计把半径放大四倍后，measurement field 升到 `+5.556%`，却仍只保留
   `65.264%` exact headroom、adjoint gain 为 `28.364%`；正式判决不变。

## 5. finite-K 为什么仍不是成功

truth-conditioned finite-K search 在 field 上达到 `+5.5599%`，逐 family 与逐 geometry 都为正，
看起来跨过 5% 门；但它仍失败两项冻结机制门：adjoint gain 只有 `11.7282%`，exact headroom
retention 只有 `65.3135%`。

更重要的是，它让 evaluator 看 truth，并在 development 上额外运行 `33,780F / 33,780A^T`。
36 个 Powell 起点只有 5 个在 96 次函数评估内报告收敛；所以这个值只是“确定性有界搜索找到的
最好点”，不是已证明的全局 oracle ceiling，更不是 25F/24A^T 可部署算法。

## 6. 下一步不应直接训练 learner

本轮主 representation gate 已失败，按预注册停止 learner。下一步应分两层：

1. **N1.7-D 只读分解已完成：**四倍半径下 measurement projection 不再触边，field 跨过 5%，
   但 retention 与 adjoint 仍失败；finite-K 为 `+6.186%`、`72.669%` retention，仍只过 16/17，
   且读取 truth。它只生成新假设，不改变本轮 NO-GO。
2. **N1.8 新表示预注册：**优先测试同预算的 camera/ray hybrid 或 camera-block modulation，例如
   两个 centered global modes 加两个 per-case Krylov modes，且把 global mean 计入总 correction
   norm。目标是捕获 mismatch 随 view 和界面位置变化的结构，而不是增加 MLP 宽度。
3. **若 field-optimal 与 mismatch-optimal 持续分离：**把支线明确改名为 solver-aware learned
   regularizer；不能继续叫 forward-model correction。
4. **新 learner 只能在新 geometry/session split 上预注册。** 当前 6 个 development clusters
   已经打开，不能用于选择新 dictionary、半径或停止规则后再称独立证据。

## 7. 给何远哲师兄的最小问题

1. 现有 BOST/NeRIF 代码能否暴露严格配对、带 support 的 matrix-free `A(v)` / `A^T(u)`？
2. camera index、ray coordinate、mask/confidence 能否在不暴露 truth 的情况下进入 correction basis？
3. 真实主导 mismatch 更可能是 finite aperture、ray bending、calibration drift 还是 optical-flow
   bias？能否提供同一 field/geometry 的低高保真 paired projection？
4. 是否能永久留出一个 camera、session 或 f-number，完全不参与 dictionary、半径、learner 和
   stopping 的选择？
5. 若组内更关心 field/front/held-out reprojection 中哪一个，请明确主终点；当前 synthetic
   field-L2 不能代替真实实验终点。

## 8. 复现与完整性

```bash
cd oerf-bishe-dashboard

.venv/bin/python site_tools/run_jacru_n1_7_geometry_krylov_oracle.py \
  --config demo_t16_operator/configs/jacru_n1_7_geometry_krylov_postopen_v1.json \
  --output-dir demo_t16_operator/results/jacru_n1_7_geometry_krylov_postopen_full1

cd demo_t16_operator/results/jacru_n1_7_geometry_krylov_postopen_full1
shasum -a 256 -c checksums.sha256
```

12 个登记文件全部通过 SHA-256 检查。结果包含机器摘要、逐 case 指标、逐 family/geometry 聚合、
basis diagnostics、truth diagnostics、finite-K 搜索账本、配置快照、provenance、PNG/PDF 图和
checksums。旧 summary 的 `finite_k_evaluator_total_*` 实际只指 development；后续审计包已补充分区
与整包账本。33 个 N1.5/N1.6/N1.7 聚焦回归测试通过；N1.7 单元与 smoke 为 15/15 通过。

## 9. 允许和禁止的表述

**允许：**“在 opened synthetic surrogate 上，per-case `A P A^T` frame 比 frozen global PCA
略有提高，但主 representation gate 为 NO-GO。”

**禁止：**“N1.7 已经是新算法”“finite-K +5.56% 已证明优于 FNO/DeepONet/NeRIF/TDBOST”
“真实 BOST 泛化成功”“放宽半径后一定能过门”。
