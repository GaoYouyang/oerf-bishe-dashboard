# D0.6 单公开 Phantom 等预算参数化筛选设计

> 文档状态：`FROZEN_POSTAUDIT_REPAIR_BEFORE_RUNNER_AND_EXECUTION`
> 设计编号：`D0.6_SINGLE_FIELD_MATCHED_BUDGET_SMOKE`
> 日期：`2026-07-23`
> 冻结机器协议：`learning_labs/open_nir_bos_d0_6_matched_budget_protocol.json`
> 协议 SHA-256：`28025859be3177bccebddc4741b3e574181a8a9416eb8bf5fb8a0915532270dd`
> 冻结输入身份 SHA-256：`df7806ab6193876ea1e4b4c3186324dd01d48e7f22291cecbc91f26b4b857688`
> 最高可能证据等级：单个已开封公开 Phantom 上的参数化筛选证据
> 本文不是运行记录；本文创建时没有启动训练、没有生成结果，也没有授权任何算法、重建、泛化或论文主张。

## 0. 研究问题与止损边界

D0.6 只回答一个窄问题：在**同一个已开封公开 Phantom、同一组直线射线、同一数据拆分和同一主计算预算**下，以下三种折射率场参数化中，哪一种更值得进入下一阶段的多场、跨几何验证？

- `S0_VOXEL`：低分辨率体素控制组；
- `S1_FOURIER`：固定 Fourier 编码的 MLP 基线；
- `S2_BOUNDED_RESIDUAL`：与 `S1` 共用基础表示、后 30 步只训练有界残差头的候选。

本设计是**单场逆问题拟合/参数化筛选**，不是函数到函数映射学习，因此不是算子学习实验。公开 Phantom 已在 D0/D0.5 中开封；即使 `S2` 通过，也只能得到“值得进入新数据验证”的候选，不构成 fresh、真实 BOST、OERF、泛化、优于现有论文或可投稿成功的证据。

第一版冻结设计保存在 Git commit `01ce64d`。独立审计在任何训练前发现 LR step-4 fit-union forward 漏算预算、G0/G3 分支冲突和 prefix 证据不足；本版是事前 v1.1 修复。没有沿用旧绿灯，也没有产生训练结果。

## 1. 机器常量表

下面的 JSON 是可读设计镜像；当前唯一机器常量源已经冻结为 `learning_labs/open_nir_bos_d0_6_matched_budget_protocol.json`，输入文件、射线、拆分和 batch 身份另由 `learning_labs/open_nir_bos_d0_6_input_identity.json` 绑定。执行器必须逐项读取或逐项断言，不得在看到结果后修改。当前仍未实现 runner、角色隔离、机械 dry-run 或正式训练，因此机器协议中的所有执行授权保持 false。

```json
{
  "schema_version": "oerf-open-nir-bos-d0-6-matched-budget-design-1.1",
  "design_id": "D0.6_SINGLE_FIELD_MATCHED_BUDGET_SMOKE",
  "status": "FROZEN_POSTAUDIT_REPAIR_BEFORE_RUNNER_AND_EXECUTION",
  "source_release": {
    "repository": "https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS",
    "commit": "a385cce83d88df24ed05dccfd6fde20e124f5604",
    "dataset": "Phantom 1/140x294x140",
    "opened_public_phantom_count": 1,
    "external_raw_assets_may_be_exported": false
  },
  "device": "cpu",
  "dtype": "float64",
  "deterministic_algorithms_required": true,
  "technical_seeds": [20260731, 20260732, 20260733],
  "views": 12,
  "selected_rays_per_view": 512,
  "split_per_view": {"fit": 400, "dev": 56, "audit": 56},
  "totals": {"selected": 6144, "fit": 4800, "dev": 672, "audit": 672},
  "field_shape_xyz": [27, 53, 27],
  "roi_half_size_xyz": [0.9524, 2.0, 0.95274],
  "camera_pose_translation_scale": 0.00054421,
  "outside_aabb_policy": "zero",
  "loss_ray_policy": "aabb_valid_rays_only",
  "primary_quadrature_steps": 128,
  "post_seal_quadrature_diagnostic_steps": 256,
  "fit_batch": {"views": 12, "rays_per_view": 40, "total_rays": 480},
  "fixed_batches_per_cycle": 10,
  "training_cycles": 11,
  "training_updates": 110,
  "lr_candidates": [0.01, 0.003, 0.001, 0.0003],
  "lr_trial_updates_per_candidate": 4,
  "lr_near_tie_relative_tolerance": 0.005,
  "optimizer": {
    "name": "Adam",
    "betas": [0.9, 0.99],
    "eps": 1e-12,
    "weight_decay": 0.0,
    "gradient_clipping": false
  },
  "parameters": {
    "S0_VOXEL": 31875,
    "S1_FOURIER": 31873,
    "S2_BOUNDED_RESIDUAL": 31970,
    "maximum_pairwise_difference": 97,
    "maximum_pairwise_relative_difference_using_minimum_denominator": 0.0030433282088287894
  },
  "budget": {
    "unit": "RQWU_matched_projection_work",
    "definition": "quadrature_steps * (forward_ray_count + vjp_ray_count)",
    "lr_screen_forward_calls": 24,
    "lr_screen_vjp_calls": 16,
    "lr_screen_forward_rays": 18048,
    "lr_screen_vjp_rays": 7680,
    "training_forward_calls": 110,
    "training_vjp_calls": 110,
    "training_forward_rays": 52800,
    "training_vjp_rays": 52800,
    "total_forward_calls_per_arm_seed": 134,
    "total_vjp_calls_per_arm_seed": 126,
    "total_ray_evaluations_per_arm_seed": 131328,
    "primary_rqwu_per_arm_seed": 16809984,
    "final_scoring_is_outside_primary_budget": true
  },
  "s2_schedule": {"base_updates": 80, "residual_only_updates": 30},
  "s2_rho_multiplier": 0.25,
  "s2_rho_floor": 0.001,
  "b0_minimum_fit_mse_reduction_vs_zero": 0.20,
  "b1_maximum_median_field_ratio_vs_b0": 1.00,
  "b1_maximum_each_seed_field_ratio_vs_b0": 1.05,
  "b1_maximum_secondary_median_ratio_vs_b0": 1.02,
  "b1_maximum_audit_median_ratio_vs_b0": 1.02,
  "b1_maximum_audit_p90_ratio_vs_b0": 1.02,
  "b1_maximum_audit_worst_ratio_vs_b0": 1.05,
  "b2_minimum_median_field_gain_vs_b1": 0.05,
  "b2_minimum_secondary_gain": 0.02,
  "b2_maximum_audit_median_ratio_vs_b1": 1.02,
  "b2_maximum_audit_p90_ratio_vs_b1": 1.02,
  "b2_maximum_audit_worst_ratio_vs_b1": 1.05,
  "b2_minimum_improving_seed_count": 2,
  "b2_maximum_per_seed_field_harm": 0.05
}
```

## 2. 固定数据合同与拆分

### 2.1 输入与允许读取项

`SPLIT_BROKER` 是唯一允许读取以下完整公开输入的预处理角色：

1. 固定提交 `a385cce...` 中的 `transforms_train.json`；
2. 12 个训练视角对应的 16-bit RGB 位移观测图；
3. 用于选择二维有效像素的 `mask2D.mat`；
4. D0 已验证的相机约定、坐标轴顺序 `XYZ`、ROI 与直线射线生成规则；
5. 本节生成并冻结的 6,144 个 `(view_id, flat_pixel_id)` 及 `fit/dev/audit` 标签。

broker 的只读挂载必须只包含上述 14 个 allowlisted 文件，不能把整个外部仓库或 GT/CGLS/3D support 路径挂进去。它按冻结索引解码后，必须生成互不重叠的私有 `fit/dev/audit` shard、逐 shard SHA-256 和 split manifest，然后退出。`TRAINER` 启动时只能挂载 fit/dev shard 与 split manifest；原始公开图像目录和 audit shard 都不得出现在其挂载命名空间中。训练角色也不得读取三维真值、CGLS-TV 结果或其任何派生统计，详细规则见第 8 节。

### 2.2 复用 D0 的 12 x 512 射线

必须复用 D0 v1 的相同射线选择：随机种子 `20260723`，每个视角从 `mask2D > 0.5` 的像素中无放回抽取 512 个像素，随后按 `flat_pixel_id` 升序保存。12 个选择数组的 SHA-256 必须等于：

```text
view_01 c183428006343c4d2d3fa9674e4eb9a5f1fc16980652f819afbf8267496dc0a0
view_02 1b7fcabea1b91c2acc36e6257aade07d6f05f382bbc9bb01ec54d8cdedcef212
view_03 3f808c8b4c05564d4eb60fbe3150a49415f02978711572bae13160c1b9120d2d
view_04 7adcad1c3b8324da62e3967022ba33f8aa5a56807d4e2f2b2e8acc184ba966df
view_05 225ae034f197ffb718fd648ff1f7bdfcc28f976f6b7c281468da3e1f761f4407
view_06 fc4fe8511ad2e1e440a72e064e9ca3f18714e1bc1f76fccba22589d1200d5cd9
view_07 7a00f6944a013e50a5c4d5b35bbdfe55a48aa9dda4db850fc6395bd7fa64c5a3
view_08 518a293c798d9a34f3ad7285bd8409816c49ee7b3622b94091ba5747f42ad29f
view_09 955e2a6f244029b70919def516edcac300289506e011b77d09c89cb18eccf45c
view_10 1eb6879b976d14114673fd5817ed57a10538b6929f6578f43b64dc7653b72db1
view_11 e07e6f11b52f667b8a922242993c415c1756c56a824427e0bb16369a2472fd7f
view_12 9e2408dd2b88eb9c3e589a4ab25c55827f1cca6cf073ee5d938bc61ac8ac3d78
```

任一 SHA 不符，状态立即置为 `D0_6_INPUT_IDENTITY_MISMATCH_NO_RUN`，不得用“近似相同”的抽样继续。

### 2.3 确定性 hash 拆分

只创建一次 RNG，然后按视角 1 至 12 顺序推进；不得每个视角重新播种：

```text
rng = numpy.random.default_rng(20260723)
for view_id in 1..12:
    valid = flatnonzero(mask2D[:, :, view_id-1] > 0.5)
    selected_512 = sort(rng.choice(valid, 512, replace=False))
    key = UTF8("D0.6_SINGLE_FIELD_MATCHED_BUDGET_SMOKE|view=%02d|flat=%d")
    rank_key = SHA256(key).hexdigest()
    order = sort(selected_512, key=(rank_key, flat_pixel_id))
    fit   = order[0:400]
    dev   = order[400:456]
    audit = order[456:512]
```

- `fit`：优化与最终 fit 指标；
- `dev`：只用于四选一 LR；不得早停、选 checkpoint 或调整结构；
- `audit`：由 broker 写入独立 sealed shard；在每个 arm/seed 的 LR、结构和第 110 步 checkpoint 全部封存前不得挂载给 TRAINER，也不可调用 forward 计算其损失。

拆分后必须输出每个视角、每个 split 的有序 `(view_id, flat_pixel_id)` SHA-256。三个技术种子共享完全相同的射线与拆分；种子不得改变数据、批次或评估集。

### 2.4 固定 fit 批次

每个视角的 400 个 fit 像素保持第 2.3 节的 hash 顺序。第 `b in [0, 9]` 个 batch 从每个视角取 `fit[40*b:40*(b+1)]`，拼成 480 条射线；视角升序、视角内 hash 顺序不变。十个 batch 构成一轮，重复 11 轮，共 110 次更新，不 shuffle、不丢弃、不补采样。

LR 筛选的四步固定使用 `batch_00` 至 `batch_03`。所有 LR、arm 和 seed 的数据次序完全一致。

## 3. 共同物理前向与目标函数

三条 arm 必须共享同一前向实现，不得给神经参数化另写一个有利的投影器：

- 场轴序：`(x, y, z)`；射线与观测均为 `XYZ` 三分量；
- ROI 半尺寸：`(0.9524, 2.0, 0.95274)`；
- 场栅格：`27 x 53 x 27`；
- 相机平移缩放：`0.00054421`；
- AABB 外射线输出严格为零，并要求对场参数梯度也严格为零；
- 场物理梯度使用 D0.5 的二阶边界/中心有限差分；
- 梯度在射线中点以三线性方式采样；
- 主积分固定 128 个 midpoint，不得按 arm 改变；
- 优化损失是在 AABB 有效射线、全部三个 XYZ 分量上的均方误差；
- 不拟合全局比例、偏置、符号或轴置换，不加入 TV、PDE、Sobolev 或其他正则项。

写成公式，令 `A_128[n](r)` 为固定直线射线前向，`y_r` 为公开观测：

```text
L(D; theta) = mean_{r in D and valid(r), c in {x,y,z}}
              (A_128[n_theta](r)_c - y_{r,c})^2
```

若某个 batch 没有有效射线、观测/预测/损失/梯度出现非有限数，或 AABB 外零输出/零梯度不成立，该 arm/seed 立即失败；禁止用 clipping、删样本、换 loss、降学习率或重跑 seed 补救。

## 4. 三种参数化与精确参数量

### 4.1 共同坐标与边界包络

D0.5 把栅格值视为 cell-center 值。对轴长 `N_i`、ROI 半尺寸 `h_i`，第 `j` 个中心为：

```text
x_i(j) = -h_i + (j + 0.5) * (2 h_i / N_i)
xi_i(j) = x_i(j) / (h_i * (1 - 1/N_i))
```

因此最外层中心映射到 `xi_i = -1/+1`。神经场共同使用解析边界包络：

```text
e(x) = product_i cos(pi * xi_i / 2)^2
```

它使 27 x 53 x 27 栅格的六个最外平面严格为零。所有 arm 输出同一个栅格，再进入第 3 节的前向；不得让连续 MLP 绕过共同栅格获得额外积分分辨率。

### 4.2 `S0_VOXEL`

- 完整栅格：`27 x 53 x 27`；
- 六个最外平面固定为零，不属于参数；
- 内部自由体素：`25 x 51 x 25 = 31,875`；
- 参数初值全部为零；
- 内部体素写入 raw 栅格、边界补零后，再与 `S1/S2` 完全一样乘共同 envelope `e(x)`，然后进入共同前向。

精确参数量：`31,875`。

### 4.3 `S1_FOURIER`

固定 39 维编码按以下顺序构造，频带 `k=0,...,5`：

```text
gamma(xi) = concat(
  [xi_x, xi_y, xi_z],
  for k in 0..5:
    [sin(2^k*pi*xi_x), cos(2^k*pi*xi_x),
     sin(2^k*pi*xi_y), cos(2^k*pi*xi_y),
     sin(2^k*pi*xi_z), cos(2^k*pi*xi_z)]
)
```

网络为 `39 -> 96 -> 96 -> 96 -> 96 -> 1`，四个隐藏层均用 SiLU，输出层无线性后处理：

```text
base(x) = MLP(gamma(xi(x)))
n(x) = e(x) * base(x)
```

参数计数：

```text
(39*96 + 96) + 3*(96*96 + 96) + (96*1 + 1)
= 3,840 + 27,936 + 97
= 31,873
```

初始化固定为：隐藏层权重 Xavier uniform（gain `1.0`），隐藏偏置为零，97 个输出头参数为零，从而初始场严格为零。每个 seed 只决定一次隐藏层初始化；LR trial 和正式 110 步都从该 seed 对应的同一初态副本重置。

### 4.4 `S2_BOUNDED_RESIDUAL`

`S2` 与 `S1` 具有逐元素相同的 Fourier 编码、四层 96 宽 trunk 和 base 输出头；另加一个从最后 96 维隐藏特征到标量的残差头，共 97 个参数：

```text
base(x) = head_base(trunk(gamma(xi(x))))
residual(x) = head_residual(trunk(gamma(xi(x))))
n(x) = e(x) * (base(x) + rho * tanh(residual(x)))
```

总参数量：`31,873 + 97 = 31,970`。残差头权重和偏置初始化为零。

训练状态机固定为：

1. `step 1..80`：残差头 `requires_grad=False` 且不进入 optimizer；残差分支强制关闭，仅训练 trunk + base；
2. `step 80 seal`：冻结 trunk + base，并在完整 27 x 53 x 27 栅格上计算 `q = P95(|e*base|)`；
3. 固定 `rho = 0.25 * max(q, 1e-3)`，此后不可更新；
4. `step 81` 前丢弃 base optimizer，新建 step=0 的 residual-only Adam；`step 81..110` 只训练 97 参数残差头，trunk + base 保持只读；
5. 若残差分支关闭，`S2` 必须逐点退化为同 seed、同 step 的 `S1` base。

为了证明前 80 步比较公平，`S1` 与 `S2` 对同一 seed 必须复用同一份 trunk/base 初值、同一 LR、同一 batch 顺序。LR trial 也必须逐候选一致。正式前 80 步每一步都要记录并比较：batch SHA、loss 的 float64 十六进制表示、trunk/base gradient L2 与 max、更新后 trunk/base 参数 SHA、Adam optimizer-state SHA。step 0 和 step 80 再做完整逐字节哈希。任一记录不相等则标记 `D0_6_B1_B2_PREFIX_IDENTITY_FAILURE`，`S2` 不得进入判决。

三者最小参数量为 31,873，最大为 31,970；最大差 97，按最小值归一为 `97 / 31,873 = 0.3043%`。因此参数规模差异不作为胜负解释，但三者参数量仍须原样报告。

## 5. 学习率选择与固定训练

### 5.1 LR 四选一

对每个 `arm x seed` 独立执行四个候选：`1e-2, 3e-3, 1e-3, 3e-4`。每个候选都从该 arm/seed 的同一零场初态重置，只运行四个固定 fit batch；在 step 4 checkpoint 上，先额外做一次 1,920-ray `batch_00..03` 并集 forward，再做一次完整 672-ray dev forward。四个训练时 loss 来自四个不同 checkpoint，禁止用它们的平均代替 step-4 fit-union loss。

候选 LR 合格当且仅当：

```text
all values finite
AND L_fit_union_at_step4_checkpoint < L_fit_union_zero_on_same_1920_rays
AND L_dev_after_step4 <= L_dev_zero
```

从合格候选中选 dev MSE 最小者。若候选 dev MSE 不超过最小值的 `1.005` 倍，则视为 0.5% 近似并列，并列中选择数值更小的 LR。若没有合格候选，该 arm/seed 状态为 `D0_6_NO_ELIGIBLE_LR`；禁止追加 LR、增加步数或人工挑选。

`S1` 和 `S2` 的前 80 步必须相同，因此同 seed 的 LR 选择结果必须相同；不相同意味着实现或身份控制失败，不得把差异解释为模型效果。

### 5.2 正式 110 步

选定 LR 后必须再次从同一初态重置优化器和模型，使用 Adam：`beta=(0.9,0.99)`、`eps=1e-12`、`weight_decay=0`。固定执行 110 步，不早停、不保存“最佳 dev”作为最终模型、不做梯度裁剪、不按结果改变批次。

三个 seed 是数值稳定性重复，不是三个独立物理样本。不得挑最好 seed；报告中必须保留全部 seed，并以配对 seed 进行 arm 间比较。

## 6. RQWU 等预算账本

定义一个 Ray-Quadrature Work Unit。它只匹配物理投影采样工作量，因此准确名称是 **matched projection-work budget**，不是端到端 FLOP 等预算：

```text
RQWU = quadrature_steps * (forward_ray_count + VJP_ray_count)
```

这里的 VJP 是通过同一物理前向做的反向传播。对 `S1/S2`，参数化组合是非线性的，因此不得把参数 VJP 夸称为独立实现的物理 `A^T`；结果同时报告 `forward_calls` 与 `VJP_calls`。

每个 `arm x seed` 的主预算严格为：

| 阶段 | forward calls | VJP calls | forward rays | VJP rays | ray evaluations |
|---|---:|---:|---:|---:|---:|
| LR：4 候选 x 4 个 480-ray 更新 | 16 | 16 | 7,680 | 7,680 | 15,360 |
| LR：4 次 step-4 的 1,920-ray fit-union forward | 4 | 0 | 7,680 | 0 | 7,680 |
| LR：4 次 672-ray dev forward | 4 | 0 | 2,688 | 0 | 2,688 |
| 正式训练：110 个 480-ray 更新 | 110 | 110 | 52,800 | 52,800 | 105,600 |
| **合计** | **134** | **126** | **70,848** | **60,480** | **131,328** |

```text
primary_RQWU = 128 * 131,328 = 16,809,984
```

三条 arm 不得超出此预算。最终 checkpoint 的 fit/dev/audit 评分、GT scorer、CGLS-TV anchor 评分和 128 -> 256 诊断不进入优化预算，但必须单独记录它们的 forward calls、RQWU、wall time 与内存，不得混入“免费计算”。

除 RQWU 外，每个 arm/seed 必须另报：总参数、各阶段可训练参数、forward/VJP 次数、wall-clock、进程峰值 RSS、是否发生 clipping（本设计应为 0）、非有限数计数和重跑次数（本设计应为 0）。wall-clock 和内存是硬件描述，不是公平性的替代定义。

## 7. 封存、开封与四角色执行

执行必须分为四个互斥进程角色，输出和挂载不可互相越权：

1. `SPLIT_BROKER`：唯一可读取完整观测图、manifest 与 `mask2D` 的角色；按冻结索引写出 disjoint fit/dev/audit shard 和哈希后退出；
2. `TRAINER`：只挂载 fit/dev shard；训练 3 arms x 3 seeds；原始图像、audit shard 与三维真值路径必须不可见；
3. `AUDITOR`：在九个 checkpoint、LR 决定、结构配置和 SHA-256 manifest 全部封存后，才挂载 audit shard，仅计算重投影指标和 256 点诊断；不得改变 checkpoint；
4. `GT_SCORER`：在 `AUDITOR` 输出也封存后，才允许读取三维 GT 与 CGLS-TV anchor，计算第 9 节的场指标；不得反向写入训练配置。

封存 manifest 至少包含：源码提交与 dirty 状态、协议 SHA、外部提交、全部输入 SHA、数据 split SHA、环境、设备/dtype、每个初态 SHA、选中 LR、step-80 与 step-110 checkpoint SHA、优化器状态 SHA、预算计数和角色开始/结束时间。

正式执行必须使 TRAINER 的挂载白名单、原始图像不可见、audit shard 不可见和 GT 路径不可见成为 fail-closed 断言，而不只是文字约定。任何越权挂载或 shard 哈希不符先记为 `D0_6_SPLIT_BROKER_ISOLATION_FAILURE`；任何 checkpoint、超参数或前处理在 GT 开封后变化，都使整次执行成为 `D0_6_GT_FIREWALL_BREACH_INVALID_RUN`。旧失败包必须保留，不能覆盖重跑。

## 8. GT firewall

### 8.1 训练期禁止项

`TRAINER` 进程及其父进程不得挂载原始公开观测目录或 sealed audit shard；在九个 checkpoint 全部封存前也不得读取、映射、stat 后解析或从缓存恢复：

- `n_GroundTruth.mat`；
- `flowcglsTV.mat`；
- `3Dmask.mat` 或其他三维 support/truth mask；
- 由以上文件生成的降采样场、梯度、边缘、前沿、直方图、归一化常量或摘要；
- 任何曾使用 GT 选择的 LR、seed、训练步数、频带、宽度、残差上界、裁剪区间或轴/符号变换。

`mask2D.mat` 只作为二维有效像素选择器允许读取，不得把它表述为三维 GT。

### 8.2 后开封也禁止的调参

GT 开封后仍禁止：

- 按 GT 选择 seed 或 checkpoint；
- 重新选择 LR、训练步数、Fourier 频带或 `rho`；
- 对输出做全局比例、偏置、符号、轴置换、配准或刚体对齐；
- 做值域 clipping、去噪或只为指标服务的后处理；
- 删除失败 seed、失败视角或不利指标；
- 把 CGLS-TV 用作任何 arm 的初始化、教师、残差目标或训练正则。

`R_CGLS_TV` 只是在全部结果封存后由 `GT_SCORER` 计算的外部参考锚点，不参与 `S0/S1/S2` 排名，也不分享其计算预算。

## 9. 后开封指标

### 9.1 共同评分栅格

所有场指标在共同的 `27 x 53 x 27` cell-center 栅格上计算。`GT_SCORER` 复用 D0 已声明的公开 GT 物理预处理，然后只在共同物理中心做固定三线性采样；CGLS-TV 也用同一采样器。采样器、源场预处理与输入 SHA 必须在 GT 开封前由源码哈希封存，禁止按结果更换插值方法。

### 9.2 必报主指标

对每个 arm/seed 原样报告：

1. **field relative-L2**（主排名指标）：`||n_hat-n_gt||2 / ||n_gt||2`；不拟合 scale/bias；
2. **physical-gradient relative-L2**：三个物理轴梯度拼接后的相对 L2；间距沿用共同栅格；
3. **top-10% gradient-front symmetric Chamfer**；
4. **top-10% gradient-front symmetric Hausdorff95**；
5. fit/dev/audit 重投影 relative-L2；
6. 12 个 audit 视角的 relative-L2：完整数组、median、p90、worst value 与 worst `view_id`；
7. `S2` 修正项 `|e*rho*tanh(residual)|` 的 p90、最大值，以及相对 `|e*base|` p90 的比值；
8. 128 -> 256 audit 预测相对漂移；该值只作积分诊断，不反馈训练或选择；
9. 参数、RQWU、forward/VJP、wall time、峰值内存、clipping、非有限数与失败状态；
10. `R_CGLS_TV` 在同一 scorer 下的指标，单列为外部 anchor，明确排除于胜负判决。

重投影 relative-L2 对每个集合或视角定义为有效射线三分量拼接后的 `||prediction-observation||2 / ||observation||2`。12-view p90 固定使用 NumPy `quantile(..., 0.9, method="linear")`；worst 并列时取较小 `view_id`。

### 9.3 前沿集合的确定性定义

令 `N=27*53*27`，分别对预测和 GT 的物理梯度模长排序。每个集合精确取 `ceil(0.10*N)` 个体素；排序键为 `(-gradient_magnitude, x_index, y_index, z_index)`，从而平值也确定。使用物理中心坐标计算：

```text
Chamfer_sym = 0.5 * (mean_{p in P} min_{q in Q} ||p-q||2
                   + mean_{q in Q} min_{p in P} ||q-p||2)

Hausdorff95_sym = max(
  P95({min_{q in Q} ||p-q||2 : p in P}),
  P95({min_{p in P} ||q-p||2 : q in Q})
)
```

距离以 ROI 的物理单位报告，并附除以 ROI 对角线长度的无量纲版本；判决使用未归一化值的 arm 比值，两者比值应一致。

## 10. 顺序判决门

所有 arm 比较都使用相同 seed 的配对结果。定义误差越小越好，配对比值 `ratio_M_vs_R(seed)=error_M(seed)/error_R(seed)`；“median”指三个配对比值的中位数，而不是先平均误差再相除。

### Gate 0：逐 arm 运行有效性

先独立计算 `arm_valid[S0]`、`arm_valid[S1]`、`arm_valid[S2]`。任一 arm/seed 出现以下情况即为该 arm 无效：输入/源码身份不符、无合格 LR、预算超支、非有限数、读取越权、checkpoint 不完整、重跑覆盖、AABB 外零约束失败。`S2` 还要求 B1/B2 prefix identity 成立。

三 seed 不完整的 arm 不得参与需要它的配对比较，也不得用剩余 seed 代替。G1 只要求 `S0` valid；G2 要求 `S1` 与 `S2` 都 valid；`S2` invalid 或未过 G2 都进入 G3；`S1` invalid 时 G3 明确回退 `S0`。`S0` invalid 则整个 D0.6 停止。

### Gate 1：`S0_VOXEL` 最低可解性门

对每个 seed 计算：

```text
fit_reduction(seed) = 1 - L_fit_step110(seed) / L_fit_zero
```

若三 seed 的 `fit_reduction` 中位数小于 `0.20`，或 `S0` 任一 seed 无效，整个 D0.6 停止，状态为：

```text
D0_6_FORWARD_SCALE_OR_GRADIENT_DIAGNOSIS_REQUIRED
```

此时不得看 GT 后继续挑模型；下一步只允许检查观测尺度、坐标/轴、物理梯度、forward/VJP 和优化实现。

### Gate 2：`S2` 相对 `S1` 的候选门

`S2_BOUNDED_RESIDUAL` 仅当以下条件**全部**成立才通过：

1. `S1`、`S2` 三个 seed 全部有效，且 prefix identity 通过；
2. field relative-L2 配对比值中位数 `<= 0.95`，即至少 5% 改善；
3. 在 gradient relative-L2、front Chamfer、front Hausdorff95 三项中，至少一项配对比值中位数 `<= 0.98`；其余两项均须 `<= 1.00`，即不得变差；
4. audit per-view median 配对比值中位数 `<= 1.02`；
5. audit per-view p90 配对比值中位数 `<= 1.02`；
6. audit worst-view error 配对比值中位数 `<= 1.05`；
7. 至少两个配对 seed 的 field relative-L2 严格优于 `S1`；
8. 每个 seed 的 field 比值都 `<= 1.05`，不允许某 seed 伤害超过 5%。

通过状态：

```text
D0_6_S2_SCREEN_GO_SINGLE_OPENED_FIELD
```

它只授权把 `S2` 带到 fresh 多场/跨几何实验，不授权“新算法有效”或“优于 S1”的一般性结论。

### Gate 3：`S2` 失败后的 `S1`/`S0` 回退

若 Gate 2 失败，先记录每一条失败条件，状态标记为 `D0_6_S2_NO_GO_SINGLE_OPENED_FIELD`，然后检查 `S1`：

- 若 `S1` 三 seed 有效，并同时满足：field 配对比值中位数 `<= 1.00`、每 seed field 比值 `<= 1.05`、gradient/Chamfer/Hausdorff95 三项中位配对比值都 `<= 1.02`、audit median/p90 中位配对比值都 `<= 1.02`、audit worst 中位配对比值 `<= 1.05`，才选择 `S1_FOURIER` 作为下一阶段简洁神经基线：`D0_6_SELECT_S1_AFTER_S2_NO_GO`；
- 若 `S1` 无效，或任一 S1 non-harm 门失败，选择 `S0_VOXEL`：`D0_6_SELECT_S0_B1_HARM_OR_INVALID`；
- 不得因为 `S2` 接近阈值而修改 5%/2%/2%/5% 门，也不得临时把某个 secondary 指标移除。

若 Gate 2 通过，不再用 Gate 3 推翻 `S2`；若 Gate 1 未通过，Gate 2/3 均不得执行。

## 11. 失败分支与允许的下一步

| 失败状态 | 含义 | 唯一允许的下一步 |
|---|---|---|
| `D0_6_INPUT_IDENTITY_MISMATCH_NO_RUN` | 外部提交、射线或输入 SHA 不符 | 恢复固定输入；不训练 |
| `D0_6_SPLIT_BROKER_ISOLATION_FAILURE` | shard 不互斥、哈希不符，或 TRAINER 看得到 raw/audit 路径 | 停止；修挂载隔离与 broker；旧包保留 |
| `D0_6_NO_ELIGIBLE_LR` | 四个 LR 均未通过四步 fit/dev 门 | 保留失败；另立协议检查尺度/优化，不追加 LR |
| `D0_6_B1_B2_PREFIX_IDENTITY_FAILURE` | B1/B2 前 80 步不再是同一基础轨迹 | 修实现与状态复制；旧包作废且保留 |
| `D0_6_GT_FIREWALL_BREACH_INVALID_RUN` | GT/CGLS 或派生量提前泄漏 | 整次无效；建立隔离后以新协议重做 |
| `D0_6_FORWARD_SCALE_OR_GRADIENT_DIAGNOSIS_REQUIRED` | S0 连零场拟合都无法降低 20% | 只诊断尺度、XYZ、forward/VJP；停止模型比较 |
| `D0_6_S2_NO_GO_SINGLE_OPENED_FIELD` | S2 未同时满足全部价值/尾部/稳定性门 | 按 Gate 3 回退 S1 或 S0；不得包装为正结果 |
| `D0_6_RUNTIME_OR_BUDGET_INVALID` | 非有限数、OOM、预算超支、重跑或缺件 | 保留首个失败包；不得静默删样本或改预算 |

任何失败都必须保留首个输出目录、异常、已完成计数和输入/源码身份。后续修复必须使用新版本号和新目录，不覆盖旧包。

## 12. 最小输出合同

未来执行器每次运行至少应生成以下**自有、无外部原始资产**文件：

```text
protocol.json
input_manifest.json
split_manifest.json
broker_manifest.json
private_shard_manifest.json
environment.json
budget_ledger.csv
lr_trials.csv
training_trace.csv
checkpoint_manifest.json
reprojection_metrics.csv
field_metrics_postopen.csv
front_metrics_postopen.csv
per_view_audit.csv
decision.json
summary.json
checksums.sha256
README.md
```

`decision.json` 必须逐条列出 Gate 0/1/2/3 的布尔值、实际值、阈值、失败原因和最终状态；不能只给一个总 `pass`。输出不得包含外部 PNG、MAT、NPY/NPZ、原始观测、GT、CGLS-TV 体数据或可逆恢复它们的数组。

## 13. 明确非主张

无论 D0.6 结果如何，以下字段均固定为 `false`：

```json
{
  "operator_learning": false,
  "new_algorithm": false,
  "reconstruction_success": false,
  "real_bost": false,
  "oerf_data": false,
  "cross_field_generalization": false,
  "cross_geometry_generalization": false,
  "robustness_to_noise_or_calibration_shift": false,
  "better_than_deeponet_fno_nerif_or_nirp": false,
  "production_readiness": false,
  "paper_success": false,
  "breakthrough": false
}
```

允许写出的最强句式只有：

> 在固定预算、固定公开 Phantom 和固定开封后评分规则下，某参数化通过/未通过 D0.6 单场筛选门，因此被选择/不被选择进入 fresh 多场与跨几何验证。

不得写成“提出了更优算法”“完成三维重建”“具有泛化性”“超过 DeepONet/FNO/NeRIF/NIRP”或“达到论文水平”。真正的算子学习至少需要多个训练场、未见场/未见几何测试、严格基线复现和独立物理数据合同；D0.6 不提供这些证据。

## 14. 正式运行前的冻结清单

本文已经完成设计、机器协议与输入身份冻结，但没有完成 runner 或执行授权。任何正式实验开始前，仍须完成：

1. **已完成：**独立 JSON 协议、SHA-256 锁、14 个 preseal 输入、3 个 postseal 体数据、12 组 selection/split hash 和 10 个 batch hash；
2. 固定 split broker、runner/scorer/firewall 的源码提交与 SHA-256；
3. 生成执行期 split manifest，再次核对每视角恰为 `400/56/56`；
4. 用不读取 GT 的单元测试验证 shard 互斥、raw/audit 不可挂载、零场初值、边界零、LR tie-break、B1/B2 prefix identity、参数梯度和角色访问拒绝；
5. 先做非正式最小 dry-run，只验证执行器机械正确性；dry-run 不得复用为正式结果；
6. 只有 runner source binding、角色隔离和 dry-run 都通过，才另发授权覆盖层并创建正式 CPU64 结果目录。

本设计未授权 MPS 上的 `S1/S2` 参数训练。M0 只覆盖固定 public-field voxel 梯度小烟测；Fourier MLP/有界残差若要上 MPS，必须另过参数梯度与至少十步内存门。
