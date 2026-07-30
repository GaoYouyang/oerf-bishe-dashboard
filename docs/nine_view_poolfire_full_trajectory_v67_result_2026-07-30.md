# v67.1：九视角 PoolFire 全轨迹 Warm Refinement 独立复算结果

日期：2026-07-30  
正式状态：`PASS_POOLFIRE_FULL_TRAJECTORY_COMPATIBILITY_V67`  
独立验证：`PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_FULL_TRAJECTORY_V67`

## 一句话结论

固定的 q8-K1 与 q4-K2 没有只在五个好看的时间点上有效。它们在五条公开
PoolFire 形态轨迹的全部 505 帧、三档已知九视角 straight-ray 几何上都通过了
结果前冻结的 field、完整 gradient、内部 gradient 和 observation 联合门：

```text
5 trajectories × 101 frames × 3 geometries = 1,515 cells / arm
q8 + exact A^T + CGLS K1                 1,515 / 1,515 PASS
q4 + exact A^T + CGLS K2                 1,515 / 1,515 PASS
formal and independently replayed atoms 27,270
```

这是真正推进科学判断的全轨迹代理正结果，因此下一步 fresh-process wall/RSS
资源门已经获准运行。但当前仍没有真实速度、整条流水线内存、独立反应流族、
曲折光线、真实相机或组内 BOST 证据：

```text
fresh_resource_stage_c_authorized=true
fresh_wall_speedup=false
whole_pipeline_peak_memory_result=false
independent_public_family_transfer=false
neural_operator_result=false
curved_ray_transfer=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```

## 1. 这次实际检验了什么

### 固定候选

q8-K1：

```text
observation
 -> fixed q8 geometry-compressed detector proposal
 -> one exact known-geometry A^T lift
 -> unchanged exact CGLS K1
```

q4-K2 只把 q8 换成 q4，并运行两步未修改 CGLS。

这两个候选从 v65、v66.1 原样带入，没有看完 505 帧后再调 rank、挑帧或改门。
q8-K1 每帧使用 `2A+2A^T`，q4-K2 使用 `3A+3A^T`；Zero-K4 使用
`4A+4A^T`。因此理论精确算子对分别减少 50% 和 25%。每个候选还有四次
cheap factor forward 和四次 cheap factor adjoint；这些动作及 factor setup 没有
被冒充为零成本，必须在 Stage C 一起计入 wall/RSS。

### 数据和几何

- 三维网格：`32×16×16`。
- 五条已经开封的公开 PoolFire CFD 形态轨迹，每条 101 帧。
- 三档已知九视角 straight-ray 几何扰动。
- 每个 arm 共 1,515 个单元，18 个 arm 共 27,270 个正式原子。
- p14 validation 只按事前协议分别选择 dual-ridge 和 direct-field-ridge 的
  lambda；两者都选到 `0.01`，但没有共享选择结果。
- 两条 test trajectory 仍未打开。

### 16 个对照

对照不是只放一个容易击败的 Zero-K4，而是包括：

- Zero-CGLS K1/K2/K3/K4；
- scaled exact backprojection K0/K1/K2；
- geometry-only Jacobi-PCGLS K2/K3/K4；
- 两种解析 A0 proposal；
- leave-one-trajectory-out dual-ridge K1/K2；
- leave-one-trajectory-out direct-field ridge K1/K2。

所有对照都用相同 1,515 个单元评分；完整 A/A^T 账逐调用点记录，模型训练、
状态大小和 cheap actions 另列，不能混进“精确调用减少”。

## 2. 主结果

### q8-K1：主候选

| 门 | field | gradient | interior gradient | observation |
|---|---:|---:|---:|---:|
| 最坏 harm / Zero-K4 | 1.002008 | 0.999111 | 1.001399 | 0.929281 |
| 最坏比值 / 同调用 Zero-K2 | 0.910587 | 0.986495 | 0.953952 | 0.644986 |
| 全 1,515 单元 p90 | 0.722928 | 0.934536 | 0.768525 | 0.408457 |
| 全 1,515 单元 worst | 0.777967 | 0.951077 | 0.889755 | 0.428706 |

结果前冻结的 harm 上限是 `1.01`，同调用 Zero-K2 比值上限是 `1.0`。q8
四项都通过，而且 15 个 trajectory-by-geometry 层每层都是 `101/101 PASS`。

这里需要诚实解释两个略高于 1 的数：

- 最坏 field harm 为 `1.002008`，即某一个单元相对 Zero-K4 约差 0.20%；
- 最坏 interior-gradient harm 为 `1.001399`，即某一个单元约差 0.14%。

它们通过的是预先给出的 1% compatibility envelope，而不是“每个指标每个样本
都严格更优”。与此同时，q8 在相同 `2A+2A^T` 预算下对 Zero-K2 的四项最坏
比值全部小于 1，说明收益不是单纯多算了物理算子。

### q4-K2：次候选

| 门 | field | gradient | interior gradient | observation |
|---|---:|---:|---:|---:|
| 最坏 harm / Zero-K4 | 0.970006 | 0.993170 | 0.984406 | 0.841856 |
| 最坏比值 / 同调用 Zero-K3 | 0.927389 | 0.987028 | 0.954109 | 0.701032 |
| 全 1,515 单元 p90 | 0.698465 | 0.927374 | 0.746269 | 0.356938 |
| 全 1,515 单元 worst | 0.759164 | 0.944692 | 0.857028 | 0.374593 |

q4 精度余量更大，但只减少 25% 精确调用，因此仍是次候选和资源门中的支持性
对照，不替代 q8 的主问题。

## 3. 强对照告诉了我们什么

全 1,515 单元的 p90/worst 如下：

| 方法 | exact A+A^T | field | gradient | interior gradient | observation |
|---|---:|---:|---:|---:|---:|
| q8-K1 | 2+2 | 0.722928 / 0.777967 | 0.934536 / 0.951077 | 0.768525 / 0.889755 | 0.408457 / 0.428706 |
| LOTO direct-field K1 | 2+1 | 0.756683 / 0.805103 | 0.952355 / 0.977582 | 0.868074 / 0.996389 | 0.519215 / 0.568143 |
| LOTO dual-ridge K1 | 2+2 | 0.756683 / 0.805103 | 0.952355 / 0.977582 | 0.868074 / 0.996389 | 0.519215 / 0.568143 |
| Zero-K2 | 2+2 | 0.866955 / 0.886492 | 0.958773 / 0.969731 | 0.846943 / 0.933054 | 0.700315 / 0.731873 |
| Zero-K4 | 4+4 | 0.733873 / 0.786342 | 0.937464 / 0.954292 | 0.778323 / 0.900146 | 0.457765 / 0.479837 |

q8 不是靠一个弱基线通过。它在汇总尾部上同时优于 Zero-K2、LOTO dual-ridge、
LOTO direct-field ridge 和 Zero-K4。更严格的逐单元 Pareto 检查中，没有任何
一个同等或更低 exact A/A^T 预算的 control 在任一单元同时不劣于 q8 的 field、
完整 gradient、内部 gradient 和 observation；最大四指标共同支配比例为 0。

q4 有 9/1,515 个单元被 A0-A0lift control 四指标共同支配，但没有 control 全局
支配 q4，也没有 control 支配全部 15 层的 p90-higher 和 worst。这个局部弱点
保留披露，不影响 q8 主候选的 Stage B 判决。

## 4. 为什么独立复算可信

正式 runner 先完成 27,270 个原子并只发布 pending 结果。另一个 detached、
tracked-clean 的程序随后：

1. 从六条原始 rho 重新执行固定轴变换、ROI、block mean 和规范化；
2. 独立生成 observation、q4/q8、Zero、BP、PCGLS、A0 和 LOTO 两族；
3. 在真正的 forward/adjoint 调用点重新累计 selection、base 和 LOTO 阶段总账；
4. 重算全部逐帧误差、p90-higher、worst、harm、dominance 和最终 Stage B 门；
5. 确认正式 payload、raw 输入和 pair 输入前后不变；
6. 最后才原子写入 `VALIDATED_READY`。

复算结果：

```text
formal atoms / unique cell IDs          27,270 / 27,270
duplicate cell IDs                      0
independently compared row values       534,795
maximum row absolute difference         6.43e-10
independently compared result values    14,136
maximum result absolute difference      2.55e-12
maximum selection absolute difference   2.27e-13
actual execution-ledger difference      0
formal/raw/pair inputs unchanged         true
```

验证器仍共享冻结的体素梯度算子和三线性 stencil，所以不能写成端到端物理完全
独立。结果前第二轮只读红队给出 P0=0、P1=0；残余 P2 是逐臂调用包装粒度、
LOTO capability isolation 和 validator 原始行数断言仍可进一步加强。本轮另做
独立外部行数/唯一 ID 核对，确认正式文件恰好 27,270 行且没有重复。

## 5. 成功在哪里，尚未成功在哪里

### 已成功

1. 排除了“只挑五个好帧”的解释：全部 505 帧、三档几何均通过。
2. 排除了“只是多调用物理算子”的解释：q8 逐单元击败同 `2A+2A^T` Zero-K2。
3. 排除了“普通线性回归就够了”的简单解释：q8 的全局尾部优于独立选择
   lambda 的 LOTO dual/direct ridge。
4. 排除了“某个便宜 control 全局支配候选”：q8 的四指标共同支配计数为 0。
5. 给 fresh 资源实验提供了足够坚实的精度前提。

### 尚未成功

1. 50% 只是 exact A/A^T 理论减少，不是端到端速度。
2. factor setup、八次 cheap factor actions、Python 调度和缓存都可能吃掉收益。
3. 当前 q8 是固定 geometry-compressed factor，不是神经网络或 neural operator。
4. 五条轨迹属于同一个 PoolFire CFD 家族，不能称 broad generalization。
5. straight-ray、已知无噪声几何不等于 pinhole、曲折光线、位移提取和真实标定。
6. 没有组内位移图、重复测量噪声和认可基线，不能称真实 BOST。
7. 没有全球原创性证明，也没有论文成功。

因此本轮应称为：

> **经过独立 raw-rho 复算的公开 CFD straight-ray 全轨迹兼容性正结果。**

不应称为：

> 神经算子突破、真实 BOST 加速、跨数据集泛化或顶刊论文完成。

## 6. 下一唯一科学门

下一步不再调 rank、挑帧或训练大网络，而是只比较 q8-K1 与 Zero-K4：

1. factor setup 必须放进 fresh worker；
2. 串行随机相邻完整区组至少 11 次；
3. 逐次记录完整 A/A^T、cheap actions、worker-self 和 process-tree RSS；
4. 要求 outer-wall p50 `<=0.90`、p90 `<=1.05`；
5. 任一轨迹不得慢超过 5%；
6. RSS p90 ratio `<=1.05`，并测 whole-pipeline peak RSS。

只有这个 Stage C 通过，才冻结一个独立公开反应流族做一次外门；外门也通过后，
才有理由把固定 q8 baseline 扩成最小 observation/geometry-conditioned
rank/coefficient predictor，并申请组内真实 BOST 数据。

## 公开附件

- 机器摘要：`docs/nine_view_poolfire_full_trajectory_v67_public_summary.json`
- 结果图：`assets/nine_view_poolfire_full_trajectory_v67.png`

