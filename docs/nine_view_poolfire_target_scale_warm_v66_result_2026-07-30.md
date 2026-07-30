# v66.1：目标尺寸 PoolFire 形态代理上的低秩 Warm Start 兼容性通过

## 结论先说

这轮得到了一项可信的正结果：

> 在 `32×16×16` 三维网格、九视角已知 straight-ray 几何和五条已经开封的
> 公开 PoolFire 形态轨迹上，standalone q8/q4 低秩算子只负责生成 detector-space
> 起点，再经过精确 `A_g^T` 提升与未修改 CGLS K1/K2；两个候选均在预先固定的
> 75 个场景-轨迹-帧单元上通过 field、gradient、observation 三项兼容性门。

这使五条轨迹合计 505 帧的 Stage B 获得执行资格。它仍不是 wall/RSS 加速、算子学习、
曲折光线迁移、真实 BOST、广泛泛化或论文成功。

## 为什么做这一步

v63 已证明：低秩近似不能直接冒充真实已知几何算子。v65 在 `8×8×8` 小预言机
上进一步证明了一个更合理的角色分工：

```text
y
 -> cheap detector-space CR with standalone low-rank Atilde
 -> dual proposal z
 -> exact lift x0 = A_g^T z
 -> unchanged exact A_g-CGLS
```

v66.1 回答的是：这个现象扩大到毕业设计真正使用的 `32×16×16` 网格，并换成
公开反应流 CFD 形态以后，是否仍然成立。

## 冻结实验

- 三维网格：`32×16×16`，一体素零边界。
- 探测器：每视角 `32×16`，共九视角、两个梯度分量。
- 几何：三档有限源距、elevation、roll、焦距、主点和目标偏移扰动。
- 数据：五条已经开封的公开 PoolFire fit trajectory。
- 每条只取帧 `0 / 25 / 50 / 75 / 100`。
- 每个候选共有 `3×5×5=75` 个单元。
- 七条臂共 `525` 个正式原子，轨迹等权汇总。
- q8 是一次 randomized-SVD 构造；q4 是同一次构造的不可变前四阶前缀。
- 历史三视角 observation 不参与；每帧都由精确九视角 `A_g` 重新生成观测。

## 结果

| 候选 | 精确成本 | cheap factor 动作 | 逐单元通过 | 相对 Zero-K4 最坏 field / gradient / observation |
|---|---:|---:|---:|---:|
| q8 + exact lift + K1 | `2A + 2A^T` | `4 forward + 4 adjoint` | `75/75` | `0.995624 / 0.999003 / 0.925687` |
| q4 + exact lift + K2 | `3A + 3A^T` | `4 forward + 4 adjoint` | `75/75` | `0.970006 / 0.991862 / 0.797811` |

相对相同精确调用预算的普通零初始化：

| 候选 / 对照 | 最坏 field | 最坏 gradient | 最坏 observation |
|---|---:|---:|---:|
| q8 / Zero-K2 | `0.902464` | `0.984080` | `0.634459` |
| q4 / Zero-K3 | `0.927389` | `0.985671` | `0.679828` |

所有比值小于 1。q4 有两个单元被同预算 A0 warm control 在三项误差上同时压过，
但没有一个相同或更低精确预算的 control 能在全部 75 个单元上全局支配 q4 或
q8。因此结论不是“每一帧都优于每一种方法”，而是两个固定候选均守住冻结门，
且没有被更便宜的经典对照整体淘汰。

## 低秩近似本身并不需要非常精确

q8 与 q4 最坏单块相对 Frobenius 残差分别是 `0.1782` 和 `0.3842`。q4 的近似
并不精细，却仍能提供 Zero-CGLS 前几步没有找到的有用 detector-space 方向。
这再次支持 v65 的物理解读：

> 近似算子适合提议方向，不适合替代冻结的 forward；noise-free straight-ray
> proxy 下的精确 `A_g^T` 与未修改 CGLS 负责把起点拉回该离散算子的约束下。

## 为什么这次结果可以信

第一次 v66 runner 虽然数值为正，但红队先后找到了五类 P1 证据链缺口：独立程序
没有锁死正式 commit/source closure，`gauge_truth` 没有从原始 `rho` 重建，
调用账仍可依赖声明值，Stage B 的发布接口存在绕过完整复算的路径，且验证文件与
最终 READY 的原子发布次序不够严格。旧授权已在发布前撤销并原样归档，不能作为
这次结论的依据。

修正后的 validator 从已提交、detached、tracked-clean 的 checkout 运行，并恢复
了与正式 runner 相同的 Python、NumPy、PyTorch 数值环境。它完成了以下检查：

- 从五条公开 PoolFire 原始 `rho` 独立重做翻转、固定 ROI、`2×2×2`
  block-mean 和全局均值规范化，逐值比较五个 `gauge_truth`。
- 共重建 `505` 帧、`4,136,960` 个真值数，最大绝对差为 `0`，mismatch 为 `0`。
- 原始 bundle、pair 身份和正式结果在验证前后保持不变；validation/test truth
  均未打开。
- 525 个原子的完整笛卡尔积、类别和在实际调用点累加的逐臂调用账全部精确匹配。
- q8/q4 六套因子诊断分别重算，q4 不继承 q8 的残差数字。
- 正式 runner 始终保持 `Stage B=false`；只有完整验证通过后，validator 才以
  no-clobber、READY-last 的顺序一次发布验证结果。
- control dominance 和 trajectory-equal 汇总进入正式判决。

独立验证器不导入正式 runner、v66 factor core、v63 geometry core 或 v62 analytic
core；它从冻结几何和随机种子重新实现射线、行块、randomized-SVD、q4 前缀、
factor action、A0/Ag、CGLS、指标、聚合和判决。复算结果：

```text
525 atoms                          exact membership PASS
505 raw-rho frames                 exact reconstruction PASS
4,136,960 raw-truth numbers        max absolute drift 0
6,150 row numbers                  max absolute drift 8.88e-14
1,062 q4/q8 factor numbers         max absolute drift 1.85e-14
2,366 aggregate/decision numbers   max absolute drift 8.33e-16
all mismatches                     0
```

验证器仍共享最低层的体素梯度算子和三线性 stencil，因此
`end_to_end_physics_independence_proven=false`，这一边界没有隐藏。

## 当前允许与不允许的说法

允许：

- 目标尺寸、九视角、已知 straight-ray 几何上的 Stage A compatibility 通过。
- 五条已经开封 PoolFire 形态轨迹的 25 帧抽样上，q8/q4 候选均为 `75/75`。
- 在冻结抽样上，精确调用对相对 Zero-K4 理论减少 50% / 25%。
- 五条轨迹合计 505 帧的 Stage B 已获准执行。

不允许：

```text
full_trajectory_result=false
fresh_wall_speedup=false
whole_pipeline_peak_memory_result=false
operator_learning_result=false
curved_ray_transfer=false
camera_calibration_result=false
real_bost=false
broad_generalization=false
algorithm_breakthrough=false
paper_success=false
```

## 下一项直接改变论文判断的实验

下一步不是改 rank、换帧或上大网络，而是把完全不变的两条候选与所有 controls
扩到五条轨迹的全部 505 帧，并加入 geometry-only Jacobi-PCGLS、验证集选择的
dual-ridge 和 sparse/streaming exact `A_g` 对照。只有全轨迹 matched accuracy
继续成立，才运行 fresh-process wall 与 whole-pipeline peak RSS；只有资源门也
成立，才有资格进入独立公开反应流族。
