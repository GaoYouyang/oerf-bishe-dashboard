# 公开 Phantom D0.5 可微前向门禁结果

> 日期：2026-07-23
>
> 主结论：`D0_5_DIFFERENTIABLE_FORWARD_PASS_IMPLEMENTATION_ONLY`
>
> 证据包一致性与异构复跑：`VALID_PASS_D0_5_IMPLEMENTATION_ONLY`，88 / 88
>
> 突破监测：**没有突破**

## 一句话解释

我们现在有了一条能够在 Mac 的 CPU 和 Apple MPS 上反向传播的直线薄射线算子。它在固定射线、固定 ROI 和固定积分规则下，能以接近机器精度复现上一轮 NumPy/SciPy 的值重放；对体素场的有限差分、JVP 与 VJP 也互相一致。

这只证明“训练时用的尺子和上一轮诊断尺子接在同一处”，不证明重建正确、模型优于 FNO/DeepONet、可迁移到 OERF，或已经有论文结果。

## 为什么先做这一步

D0 v1 虽然能用公开 ground truth 和 CGLS-TV 场重放观测，但实现是 `numpy.gradient + scipy.ndimage.map_coordinates`，没有 autograd。若直接把网络接上去，网络可能在优化一个轴序错误、边界错误或梯度错误的 forward；损失下降也不能说明任何物理问题得到解决。

D0.5 因此只回答三个实现问题：

1. PyTorch 与冻结的 NumPy/SciPy forward 是否逐值一致？
2. 对固定射线、只改变三维场时，自动微分是否与有限差分一致？
3. JVP/VJP 是否满足对偶点积，ROI 外射线是否严格给出零输出和零场梯度？

## 先冻结，再运行

第一次运行前写入协议：

- 协议：`learning_labs/open_nir_bos_d0_5_torch_forward_protocol.json`
- SHA-256：`80df1e5943e7664c6619487e02852a219625d83184d444b87adad163d51d71b3`
- CPU `float64` 是主科学门。
- MPS `float32` 只检查小型合成矩阵的设备兼容性；公开场 MPS 反传另设前置门。
- 公开 Phantom 使用发布的 CGLS-TV 体场、12 个视角、每视角 64 条固定射线和 128 个积分点，只做值一致性。
- 梯度门使用可公开重建的非立方合成场，不导出公开仓库里的体数据、图像或掩膜。
- 看过结果后不得放宽阈值；失败项必须保留。

## clean-room 实现包含什么

`learning_labs/open_nir_bos_d0_5_torch_forward.py` 没有导入或执行作者代码，显式实现：

- `field[x,y,z]` 的 cell-centred 网格，间距为 `2h/N`，不是 `2h/(N-1)`；
- 与 `np.gradient(..., edge_order=2)` 相同的内部中心差分和两端二阶单边差分；
- 与 SciPy `order=1, mode="nearest"` 对应的边界复制三线性插值；
- ray--AABB slab 相交、盒内起点、平行轴和严格 `far > near`；
- midpoint quadrature；输出是积分后的 `[gx, gy, gz]`，不与射线方向点乘；
- 无效射线先把积分长度改为有限的零，再采样和遮罩，避免 `NaN * 0` 污染反向传播；
- 分块射线计算，便于在 Mac 内存内运行。

## 正式结果

### CPU `float64` 主门

| 指标 | 实测 | 冻结上限 | 判定 |
|---|---:|---:|---|
| 合成场 Torch/SciPy relative-L2 | `4.5822e-16` | `1e-11` | PASS |
| 合成场最大绝对误差 | `1.3323e-15` | `1e-10` | PASS |
| 仿射场解析积分最大误差 | `4.4409e-16` | `1e-11` | PASS |
| 线性缺陷 relative error | `1.0941e-16` | `1e-12` | PASS |
| 中心有限差分 vs JVP | `2.4533e-10` | `5e-8` | PASS |
| JVP vs 同算子作用于方向 | `0` | `1e-11` | PASS |
| JVP/VJP 对偶缺陷 | `1.9707e-15` | `1e-11` | PASS |
| ROI 外场梯度最大值 | `0` | `1e-14` | PASS |

### 公开 Phantom 值重放

| 项目 | 数值 |
|---|---:|
| 视角 / 射线 | `12 / 768` |
| AABB 内有效射线 | `754` |
| midpoint 积分点 | `128` |
| PyTorch/SciPy relative-L2 | `3.0585e-16` |
| 最大绝对误差 | `7.1054e-15` |

这里比较的是两个实现对**同一个 CGLS-TV 场**的前向输出，不是预测对 ground truth 的重建误差。

### Apple MPS `float32` 兼容门

第一次 v0 运行保留了一个真实工程失败：统计器试图在 MPS 上直接转成 `float64`，Apple 后端拒绝执行。修复只把顺序改为“先搬到 CPU，再转 `float64` 计算审计统计”，没有改变协议或数值阈值；v0 结果目录没有删除。

修复后的 v1：

| 指标 | 实测 | 冻结上限 | 判定 |
|---|---:|---:|---|
| 值 relative-L2 | `9.1566e-8` | `5e-5` | PASS |
| 值最大绝对误差 | `2.3842e-7` | `5e-5` | PASS |
| 有限差分 vs JVP | `1.8590e-4` | `5e-3` | PASS |
| JVP exact-direction | `0` | `1e-4` | PASS |
| JVP/VJP 对偶缺陷 | `4.9790e-7` | `1e-4` | PASS |
| ROI 外场梯度最大值 | `0` | `1e-6` | PASS |

因此可以写“synthetic MPS compatibility PASS”，但还不能用它授权公开 Phantom 的 MPS 训练。下一步先用公开 CGLS-TV 场的小批固定 rays 做 CPU32--MPS32 值桥接、一次反传、梯度有限性和内存检查；通过后才打开 MPS 烟测。CPU `float64` 的单 Phantom 实现烟测不受这一设备门影响。

## 证据包复核与异构复跑做了什么

`learning_labs/validate_open_nir_bos_d0_5_torch_gate.py` 完成 88 / 88：

- 重算协议锁、结果文件 SHA-256、16 个逐指标阈值和 PASS/FAIL；
- 从 `comparison_statistics.csv` 重算 relative-L2；
- 重新读取外部公开仓库并复跑 768 条公开射线；
- 使用不同随机种子、`5 x 4 x 6` 非立方场、不同射线和 19 个积分点做异构值探针；
- 对另一组 `3 x 3 x 3` 场执行 `torch.autograd.gradcheck`；
- 确认所有越权声明仍为 `false`；
- 确认第一次 MPS 错误结果仍保留。

它直接导入被审的 `project_field`，公开重放也调用同一 runner helper。因此 88 / 88 的准确名称是“包一致性检查与异构复跑”，不是第二套独立实现。它能发现哈希、阈值、结果漂移和 autograd 自洽性问题，但不能独立证明真实 BOS 的有限孔径、光流、曲光线和标定物理已经正确。

发布覆盖层另行绑定私有证据 commit `4ec192fdd80d4ef7a666c54c4147df31140f559f`、tree、核心 forward/runner/validator/两组测试的 SHA-256，以及 v0、v1、validation 三个结果目录的精确文件集合。核心 Python 文件仍不进入公开 Pages；公开层只给出哈希与证据边界。

## 独立审计指出的残余风险

两个只读审计支线没有否定本协议下的 PASS，但给出了下一轮必须看见的限制：

1. 当前梯度合同只对**固定 rays、固定 geometry、改变 voxel field**成立；AABB 分支不光滑，不能宣称相机位姿或曲光线轨迹已有可靠导数。
2. 公开值门只用了 768 条固定射线。完整 6,144 条射线、逐视角尾部和 CPU32--CPU64--MPS32 三角桥接仍应在正式训练基准前补充。
3. D0.5 检查 voxel-field 的 JVP/VJP；Fourier MLP 的参数梯度要在 B1 烟测里再用多方向有限差分检查。
4. 单个公开 Phantom 仍然只有一个独立三维函数。无论切多少 rays，都不能把 instance fitting 变成 operator learning。
5. 与 SciPy 一致只说明两个实现一致，不证明当前直线 ray、resize、光流和 ROI 假设等于真实实验系统。
6. `float32` 中心差分步长不能照搬 CPU64 的 `1e-6`。一次附加诊断中，CPU32 使用该步长得到 `0.1254` 的差分误差，而 MPS 按预注册 `1e-3` 步长为 `1.859e-4`；这是数值步长敏感性，不是 MPS 胜过 CPU 的物理结论。

## 现在允许做什么

允许：在同一个已打开的公开 Phantom 上做 **CPU `float64` matched-budget 优化烟测**，只用于找优化故障和筛掉明显无效参数化。MPS 只能在公开场 mini-batch 前向/反向门通过后加入。

建议三臂：

| 臂 | 角色 | 需要回答的问题 |
|---|---|---|
| B0 low-resolution voxel | 经典可微控制 | 同样 A/A^T 调用下，直接体素能走到哪里？ |
| B1 Fourier MLP | 稳定隐式场基线 | 连续表示是否只改善视觉平滑，还是也改善 `delta-n` 与梯度？ |
| B2 Fourier base + bounded residual | 研究候选 | 残差能否在不伤害尾部和 ROI 外行为时带来增益？ |

三臂必须共享：

- 同一组 opened rays、相同观测值和同一 forward；
- 事前选定的唯一 primary budget，例如 A/A^T 调用数，而不是同时声称 step、参数、wall 都严格相同；
- GT 只用于最后评分，不参与调超参数；
- 主指标为 `delta n` field relative-L2，辅以 gradient/front、reprojection、逐视角尾部、wall、内存和调用数；
- 任何 B2 失败都精确退回 B1，失败证据保留。

不允许：写“新算法”“重建成功”“优于 DeepONet/FNO/NeRIF”“泛化”“OERF 可用”“论文成功”或“突破”。

## 为什么现在还不能直接比 FNO/DeepONet

FNO/DeepONet 学的是函数到函数的映射，需要多个独立输入场、几何或 session。当前公开数据只有一个 Phantom，默认 validation/test 还复用了 train pose 和像素资产。现在强行训练 FNO，最多是在同一个函数上记忆 rays，不是算子泛化。

真正的 operator-learning 比较要等至少一项到位：

- 多个独立 CFD/phantom fields，按 field/run 分组切分；
- 何远哲师兄提供真实 session、geometry 和 callable；
- 明确任务是 warm start、preconditioner、bounded correction 还是完整 inverse operator；
- 冻结未见 field、未见 rig 和 corruption 的分层测试集。

## 给何远哲师兄确认的六个问题

1. 组内 forward 接口接受 voxel field、连续坐标场还是别的表示？
2. 当前主链是 straight ray、curved ray，还是二者都有；残差定义在哪一层？
3. 能否调用 `A(x)`、`A^T(y)`、JVP 或 VJP；调用成本如何计数？
4. geometry、ROI、标定和折射率/密度单位怎样定义？
5. 最希望解决的真实痛点是速度、噪声、少视角、标定漂移、折射误差补偿还是 4D 一致性？
6. 可以提供多少个独立 field/session，哪些能作训练、开发、未见测试？

在这些答案到位前，D0.5 只继续支持公开单 Phantom 的实现烟测，不替实验室制造结论。

## 复现命令

```bash
.venv/bin/python -m pytest \
  learning_labs/test_open_nir_bos_d0_5_torch_forward.py \
  learning_labs/test_validate_open_nir_bos_d0_5_torch_gate.py -q

.venv/bin/python -m learning_labs.run_open_nir_bos_d0_5_torch_gate \
  --external-repo /path/to/Neural-Implicit-Reconstruction-for-BOS \
  --output learning_labs/results/open_nir_bos_d0_5_torch_forward_v1

.venv/bin/python -m learning_labs.validate_open_nir_bos_d0_5_torch_gate \
  --external-repo /path/to/Neural-Implicit-Reconstruction-for-BOS
```

## 证据入口

- 正式 v1 摘要：`learning_labs/results/open_nir_bos_d0_5_torch_forward_v1/summary.json`
- 逐指标门：`learning_labs/results/open_nir_bos_d0_5_torch_forward_v1/metrics.csv`
- 聚合充分统计：`learning_labs/results/open_nir_bos_d0_5_torch_forward_v1/comparison_statistics.csv`
- 包一致性与异构复跑：`learning_labs/results/open_nir_bos_d0_5_torch_forward_v1_validation/validation.json`
- 当前机器授权覆盖层：`docs/open_nir_bos_d0_5_training_authorization_overlay.json`
- 保留的第一次 MPS 错误：`learning_labs/results/open_nir_bos_d0_5_torch_forward_v0/summary.json`
- 可视化：`learning_labs/results/open_nir_bos_d0_5_torch_forward_v1/d0_5_gate.png`

## 最终证据边界

**这轮的真实增量是：一条通过 CPU 主门与公开值重放、并在 Apple MPS 小型合成矩阵上通过兼容门的可微薄射线实现；证据包一致性与异构复跑为 88 / 88。**

**这轮没有得到：新的三维重建算法、真实 OERF 结果、跨场/跨 rig 泛化、对 DeepONet/FNO/NeRIF 的优势、高质量论文结论或突破。**
