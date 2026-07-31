# v75 外部精度门结果：固定 PoolFire q8 因子不能原样跨反应流族

## 一句话结论

预先选定的 BLASTNet vitiated H2-air Case 3 已完成一次性正式开封和独立复算。
固定 loaded q8-K1 候选在 `75` 个外部单元中只有 `5` 个通过完整合同，所以
v75 的科学判决是：

```text
FAIL_VITIATED_H2AIR_CASE3_EXTERNAL_ACCURACY_SENTINEL_V75
```

资源阶段没有启动，Case 4/6 也不能用来替换这次负结果：

```text
external_accuracy_pass=false
resource_stage_authorized=false
external_resource_result=false
algorithm_breakthrough=false
paper_success=false
```

## 这次比较了什么

Case 3 在读取网格或密度数值前，按“至少 25 个有序快照、所需 rho 字节数最小、
case id 最小”的固定规则选出。正式开封后，25 帧完整 DNS 密度场确定性插值到
`32x16x16`，再在三档冻结的九视角 straight-ray 几何下形成：

```text
25 frames x 3 geometries = 75 cells
```

三条臂保持结果前合同不变：

| 角色 | 方法 | 精确调用 |
|---|---|---:|
| 候选 | loaded q8 detector CR → exact A^T lift → CGLS K1 | `2A+2A^T` |
| 同调用对照 | Zero-CGLS K2 | `2A+2A^T` |
| 精度参考 | Zero-CGLS K4 | `4A+4A^T` |

每个单元必须同时满足：

1. field、full-gradient、interior-gradient、observation 相对 Zero-K4 的
   harm ratio 都不超过 `1.01`；
2. 同四项误差都不高于相同 `2A+2A^T` 成本的 Zero-K2；
3. `75/75` 缺一不可。

## 正式且独立复算后的数字

### 1. 完整合同只有 5/75 通过

| 九视角几何 | 完整合同通过 | 全部 Zero-K4 harm 门通过 |
|---|---:|---:|
| F12+ | `1/25` | `24/25` |
| F15+ | `2/25` | `25/25` |
| F30+ | `2/25` | `25/25` |
| **合计** | **`5/75`** | **`74/75`** |

这不是“整体崩溃”，但按事先冻结的全单元合同仍然是明确 FAIL。

### 2. 真正卡住的是同成本内梯度

| 指标 | 相对 Zero-K4 的 1.01 门 | 相对 Zero-K2 的 1.00 门 |
|---|---:|---:|
| field | `75/75` | `75/75` |
| full-gradient | `75/75` | `46/75` |
| interior-gradient | `74/75` | `5/75` |
| observation | `75/75` | `75/75` |

失败组合进一步拆开是：

```text
40 cells   只在 interior-gradient / Zero-K2 上失败
29 cells   同时在 full-gradient 与 interior-gradient / Zero-K2 上失败
 1 cell    interior-gradient 同时越过 Zero-K4 与 Zero-K2 门
 5 cells   全部门通过
```

也就是共有 `70/75` 个单元没有通过 interior-gradient / Zero-K2 门。

候选相对 Zero-K2 的 p50 / p90 / worst 为：

| 指标 | p50 | p90 | worst |
|---|---:|---:|---:|
| field | `0.96946` | `0.98317` | `0.99237` |
| full-gradient | `0.99844` | `1.00252` | `1.00309` |
| interior-gradient | `1.01501` | `1.02677` | `1.04685` |
| observation | `0.62754` | `0.68970` | `0.73877` |

因此候选在 field 和 observation 上稳定优于同调用 K2，full-gradient 很接近，
但 interior-gradient 中位仍差约 `1.50%`，高尾差到 `4.69%`。固定 PoolFire
q8 因子把低频场和观测残差迁移得不错，却没有保住新反应流形态的局部梯度结构。

候选的 boundary-shell gradient absolute error / full-gradient absolute error
中位为 `0.94879`。这说明 full-gradient 大部分被边界壳误差主导，单看全梯度
会掩盖内部梯度的系统性失败；保留独立的 interior-gradient 门是必要的。

5 个完整通过单元只出现在 frame index `0` 和 `2`，从 index `3` 到 `24`
没有完整通过单元。这提示固定因子可能只兼容早期流场形态，但当前一条 Case 3
不足以把原因严格归结为某种火焰动力学机制。

### 3. 相对 K4 的信号仍值得保留，但不能改判决

相对四步 Zero-K4，候选四项 p50 harm ratio 为：

```text
field                 0.99618
full-gradient         0.99973
interior-gradient     0.99953
observation           0.92368
```

`74/75` 个单元同时守住四项 `1.01` harm 门；唯一越线单元的
interior-gradient harm 为 `1.01097`。这说明“以一半精确调用逼近 K4”在外部
形态上仍有很强信号，但完整 v75 还要求不被同调用 K2 支配，所以不能把 `74/75`
重新包装成 PASS。

## 独立验证覆盖了什么

独立 validator：

- 重新哈希全部 29 个原始文件；
- 不导入正式 runner、正式预处理 helper 或正式高层 solver；
- 自己实现 factor CR、warm CGLS 与 zero CGLS recurrence；
- 重新生成 25 个真值、75 个 observation 和三条臂；
- 重新核对每单元及整档几何的实际 `A/A^T` 调用账；
- 验证正式输出和原始输入在复算前后均未改变。

正式与独立复算的最大绝对/相对指标差为：

```text
2.13e-14 / 4.71e-16
```

两套程序仍共享冻结的低层几何与 exact operator kernel，所以
`end_to_end_physics_independence_proven=false`；这项边界必须保留。

## 这项负结果关闭什么

v75 关闭的是：

> 将 PoolFire 上编译的固定 loaded q8 factor 不做 observation-adaptive
> 修正，原样迁移到另一个反应流族，并声称满足完整同成本精度合同。

它没有证明：

- 所有 neural operator 或 learned warm start 都无法跨族；
- `2A+2A^T` 预算下不存在可观测自适应的候选；
- 曲线光线、噪声、相机标定或真实 BOST 一定失败；
- 当前结果已经形成论文成功或算法突破。

## 下一步为什么不是下载 Case 4 重跑同一个候选

协议在结果前已经规定：Case 3 失败后，Case 4/6 不能替换这次外部负结果。
继续用同一个固定 factor 挑另一个工况，只会变成结果后选样。

下一条有价值的门应先在已经开封的 Case 3 上回答：

> 在同一 `2A+2A^T` 精确预算内，现有 warm-CGLS 已经生成的
> `h=A^Tz`、`Ah`、`n=A^T(y-Ah)`、`An` 所张成的可观测二维子空间，是否存在
> 能同时修复 interior-gradient 且不损害 field / observation 的系数？

先做 truth-aware oracle feasibility，只判断表示有没有 headroom；若没有，
直接关闭该二维表示。只有 oracle 有 headroom，才允许训练最小
observation-only 系数预测器，并把 Case 3 永久作为 development，另行冻结
未打开工况作新算法的一次性外部门。无论新方法以后是否成功，都不能抹掉 v75
这次 `5/75 FAIL`。

## 公开产物

- 脱敏机器摘要：
  `docs/nine_view_vitiated_h2air_external_accuracy_v75_public_summary.json`
- 结果图：
  `assets/nine_view_vitiated_h2air_external_accuracy_v75.png`
