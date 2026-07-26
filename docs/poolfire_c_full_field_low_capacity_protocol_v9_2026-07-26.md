# PoolFire C 路线 v9：低容量完整三维初值协议

> 当前状态：`FROZEN_BEFORE_V9_IMPLEMENTATION_AND_RESULTS`  
> 当前结论：协议已冻结，数字实验尚未运行；`algorithm_breakthrough=false`。

## 1. 这一步到底在做什么

师兄给出的主线没有改变：

> 用算子学习产生三维重建的 warm start，在最终精度相同的条件下，减少重建所需的
> 完整 forward/adjoint 调用、端到端时间和内存。

前面的 v6、v7.1、v8 依次排除了三条过于简单的路线：

1. raw `BP=A^T y` 的质心有明显工况偏差；
2. 固定几何均衡 `D^{-1}A^T y` 只能局部改善；
3. 8 个部署可见特征、15 个参数的质心校准仍只通过 3/6 条已见轨迹。

所以 v9 不再猜“火焰中心应该往哪里平移”，而是直接学习完整的
`16×16×32` 三维 warm field。不过它仍从很小、可解释、能失败的模型开始，不直接上
3D U-Net、FNO、UNO 或 DeepONet。

机器可读协议：
[poolfire_c_full_field_low_capacity_audit_v9.json](../learning_labs/protocols/poolfire_c_full_field_low_capacity_audit_v9.json)

## 2. 一个必须修正的证据边界

`p=22kw_size=01` 在 v8 这一轮确实没有被读取，但它早期已经用于 classical
refinement-depth 和半收敛诊断。因此：

- 它是**历史已打开的 development 数据**；
- 它不能恢复为 fresh stopping validation；
- v9 明确禁止再用它选择模型、正则、步数或救失败结果。

v9 在只查看公开 metadata、没有下载 payload 的条件下，预先指定
`p=45kw_size=03` 为一次性 fresh proxy holdout。这个工况的 power=45 和 size=03
都分别在开发数据里见过，但组合尚未使用。它不是 untouched test，也不能被写成
unseen-power、unseen-size 或真实 BOST 泛化。

只有五条 fit 的 outer-LOTO 和已经见过的 p14 veto 全部过门、模型和报告模板完全锁定
之后，才允许获取并打开 `p45-s03`。失败后不得更换 holdout。

## 3. 主候选的物理和数学形式

记：

- `y`：三视角、2072 个分量的部署可见 observation；
- `G(x)=x-mean(x)`：去掉不可观测的常数 gauge；
- `q=G(A^T y)`：raw backprojection；
- `e=G(Wq)`：v7.1 已冻结的 geometry-equalized BP，`W` 只由
  `diag(A^T A)` 决定；
- `x4=CGLS_K4(0,y)`：零初值 CGLS 第 4 步的 observation-only teacher。

主模型不重新学习 `x4` 的全部大结构，而只学习 equalized BP 还缺少的残差：

```text
target = (x4 - e) / sr

x0 = G(
  e + sr * sum_j theta_j * phi_j(q / sq, e / se)
)
```

`sq`、`se`、`sr` 都只由当前 fold 的训练轨迹计算。不得按 heldout 轨迹或单帧真值
重新归一化。

`phi` 是两个输入通道各自的 7 个十字邻域位置：

```text
center, -x, +x, -y, +y, -z, +z
```

一共只有 `2×7=14` 个 float64 权重，没有 bias、坐标通道、时间通道或工况标签。
边界采用事前固定的 reflect 规则。预测后再次去均值。

这比“8 个二阶差分系数”略宽，但仍非常小；它允许正负方向具有不同响应，可以检查
相机几何和边界是否造成方向性偏差。与此同时：

- `lambda -> infinity` 时，修正项归零，模型严格退回 `e`；
- observation 整体乘正数时，输出按同一比例缩放；
- 推理不增加新的 `A` 或 `A^T`；
- 14 个权重本身只有 112 bytes，但最终内存账必须同时包含 scaler、geometry state、
  模型文件和 inference workspace，不能只报权重。

## 4. 为什么 target 选 K4 residual

直接拟合完整 `x4` 会浪费低容量模型去重复 BP 已经恢复的大尺度结构。拟合
`x4-e` 有三个好处：

1. 问题变成“equalized BP 还缺什么局部修正”，物理解释更清楚；
2. 强正则极限有明确的保守回退；
3. 如果残差没有跨轨迹规律，模型会诚实失败，而不是靠平均场制造好看的 L2。

但 K4 只是训练 teacher。最终 field 和 gradient 仍对独立高分辨率
rho-forward 生成的 coarse proxy truth 评分，observation residual 仍对 `y` 评分。
不能用 coarse inverse 的 `A x_true` 重新生成 observation，否则会产生 inverse crime。

## 5. 事前冻结的强对照

| 方法 | 可训练参数 | 作用 |
|---|---:|---|
| Raw BP identity | 0 | 判断完全不学习时的下界 |
| Equalized BP identity | 0 | 检查几何均衡本身 |
| Raw BP observable line search | 0 | 强幅值校准对照 |
| Equalized BP observable line search | 0 | 强预条件方向对照 |
| DCT low-mode residual | 44 | 非周期、低频传递对照 |
| Observation-to-K4 dual ridge | 数据依赖 | 高容量线性 direct baseline |
| **Cross14 residual ridge** | **14** | 唯一主候选 |
| Zero-CGLS K3 | 0 | 与主候选 K2 同为 6 次完整算子调用 |
| Zero-CGLS K4 | 0 | 最终精度参照，8 次完整调用 |
| Geometry-PCGLS K3 | 0 | 几何预条件同成本对照 |
| Normalized-BP + CGLS K2 | 0 | 可观测线搜索 warm-start 对照 |

DCT 使用正交 DCT-II，不使用会让火焰从一侧“绕回”另一侧的周期 FFT。只允许
`0<=kx<=2, 0<=ky<=2, 0<=kz<=4` 且去掉 DC 的 44 个低模增益；看到结果后不得换
cutoff。

主候选是事前指定的唯一判决对象。即使某个 control 数字更好，也只能用于解释下一步，
不能把它临时改名为“v9 主算法”。

## 6. 训练和防泄漏结构

五条 fit trajectory 做完整的 nested leave-one-trajectory-out：

1. 外层每次整条留出一条轨迹；
2. 内层只在其余四条中的训练轨迹计算 `sq/se/sr`、拟合参数并选择 lambda；
3. lambda 固定为 `{1e-6, 1e-4, 1e-2, 1, 1e2}`；
4. 用 inner trajectory 等权误差和 one-standard-error rule 选择更强正则；
5. 若 lambda 落在网格边缘，只报告，不能扩网格重跑；
6. 505 或 606 帧绝不能当作 505 或 606 个独立样本。

实现必须拆成三个进程：

- fit worker：只收到训练 BP/observation 和训练 K4 teacher；
- deployment worker：只收到冻结模型和 heldout observation/BP；
- score worker：必须等预测原子发布后，才收到预测、heldout K4、proxy truth 和
  observation。

不能复用旧 classical runner，因为旧 runner 在同一进程里同时持有 truth 和
initializer。

## 7. 完整调用账

主候选固定做两步 refinement：

| 阶段 | `A` | `A^T` |
|---|---:|---:|
| 计算 raw/equalized BP | 0 | 1 |
| 对非零 warm field 建立初始 residual | 1 | 0 |
| 两步 CGLS refinement | 2 | 2 |
| **总计** | **3** | **3** |

因此主候选 K2 是 6 次完整算子调用；Zero-K4 是 `4A+4A^T=8` 次。只有精度、
harm、wall、RSS 和 fresh holdout 全部过门后，才能说“在该公开 proxy 上实现 25%
调用减少”。现在还不能这样写。

Zero-K3 也是 6 次调用，所以它是非常重要的同预算对照：如果 Cross14-K2 不能在
field/gradient 上稳定优于 Zero-K3，学习模型就没有存在价值。

## 8. 一次实验怎样才算过

每条完整 trajectory 都分别检查：

- field relative-L2；
- gradient relative-L2；
- observation residual relative-L2；
- joint matched-frame fraction `>=90%`；
- 总 harm 和每项 harm `<=5%`；
- p50、p90(higher)、worst；
- 一体素边界壳层和内部区域误差；
- 精确 `A/A^T` 调用；
- trajectory-amortized wall time；
- fresh-process whole-pipeline peak RSS。

主候选还必须：

1. 五个 outer heldout 和 p14 mandatory veto 全部通过；
2. 每条轨迹的 field/gradient p90 都不差于同成本 Zero-K3；
3. field 或 gradient 至少一项相对 Zero-K3 改善 2%；
4. 相对 Zero-K4 的轨迹等权 wall 中位数至少下降 10%，任何轨迹不得慢 5% 以上；
5. 每条轨迹 fresh RSS p90 不超过 Zero-K4 的 1.05 倍；
6. 锁定后的一次性 `p45-s03` 也按原规则通过。

只要任意一条失败，v9 就正式 FAIL。不得用 DCT、p22、test、扩 lambda 网格、改 K、
换 holdout 或加大网络来救结果。

## 9. 通过与失败分别意味着什么

**如果失败：**

- 说明 14 参数局部共享核不足以近似跨工况 K4 residual；
- controls 可以帮助区分问题来自局部性、频率表示、幅值还是模型容量；
- 仍不能直接训练 FNO/UNO/DeepONet；
- 负结果本身能阻止后续几周浪费在错误表示上。

**如果通过：**

- 只证明公开 PoolFire straight-ray morphology proxy 上出现可信 headroom；
- 只授权另行预注册一个小型 BP-conditioned 3D U-Net sentinel；
- 仍不是 neural operator 优势、真实 BOST 泛化、论文成功或算法突破；
- 仍需一次性 test、组内真实 BOST 小样例、相机/折射率量纲闭合和独立复算。

## 10. 接下来按什么顺序写代码

1. 先提交本协议，确保协议 commit 严格早于实现 commit；
2. 实现 exact cross-shift、fold-train-only RMS 和 14 参数 Gram ridge；
3. 写常数场、边界、gauge、正比例齐次和 lambda 极限测试；
4. 实现 fit/deployment/score 三进程和原子 prediction 发布；
5. 实现所有 controls 和 exact-K2 fresh-process runner；
6. 先跑五条 outer LOTO 与 p14 veto；
7. 独立 validator 不导入正式 fit/predict/score helper，逐数组重算；
8. 只有 development gate 通过，才锁模并获取 `p45-s03`；
9. 结果同步到网页和学习日志，继续保留所有负面声明边界。

这一步的价值不是“模型终于赢了”，而是把下一次真正的数值实验变成了一个有明确
物理问题、强对照、成本账、失败动作和新鲜证据边界的可审查实验。
