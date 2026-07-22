# LGWO-A24 R2-A 二次型正则化尺度 pilot 协议

状态：`AMENDED_BEFORE_FIRST_TRAJECTORY_AFTER_FAIL_CLOSED_IDENTITY_PATH_FIX`

> 首次启动在读取几何之前因 identity manifest 路径不存在而 fail-closed；耗时约 0.0003 秒，生成科学行数为 0。
> 修订只把路径改为 R0 已使用的 `public_data_manifests/psu_support_geometry_identity_v1.json`，没有改变方法、
> lambda、checkpoint、分区、指标或门槛。失败记录保留在
> `demo_t16_operator/results/lgwo_a24_r2a_quadratic_regularization_pilot_failed_identity_path_v1/`。

> 第二次启动完成几何与 synthetic split 构造后，在第一条 CDLS trajectory 前发现
> `SingleCaseWeightedOperator` 没有公开 base 已有的 `ray_count`，再次 fail-closed，科学行数仍为 0。
> 修订只增加 `self.ray_count = int(base_operator.ray_count)` 及回归断言；方法和实验参数没有变化。失败记录保留在
> `demo_t16_operator/results/lgwo_a24_r2a_quadratic_regularization_pilot_failed_missing_ray_count_v1/`。

## 1. 这一轮只回答什么

R0 已证明当前 `k=1...24` CGLS 路径不存在可学的 field-optimal 早停标签，并观察到 field-L2 下降而
gradient-L2 上升。R2-A pilot 只问：把算子改成

```text
[W A; sqrt(lambda_l2) I; sqrt(lambda_h1) D]
```

后，在仍然严格使用 `24F/24A^T` 的条件下，L2、H1 或混合 Sobolev 二次型是否至少在已打开的 validation
均值曲线上产生值得正式复核的 field-gradient 数值尺度。

它不是算法比较，不是新鲜检验，也不授权学习模块。

## 2. 数据隔离

- 只运行源配置的全部 24 个 `validation` cases。
- `test_iid`、`test_family_ood` 和四个其他 test/control 分区在本 pilot 中禁止加载和评分。
- 所有源分区、R0 结果和机制图在本协议前都已经打开，因此即使出现大幅均值收益也只能称为
  `post-open validation-only scale signal`。
- 真实 PSU 位移、真实三维真值和 OERF 数据均不使用。

## 3. 算子与预算

- `W A`、support、射线几何、whitening、初值和 fully reorthogonalized CDLS 壳与 R0 相同。
- `D` 是物理 spacing 下的 z/y/x 前向差分，只连接 support 内部的 active-active 体素；无周期环绕，
  不跨 support 边界。
- 每个配置每个 case 都运行到 24 步，并必须精确记为 `24F/24A^T`。
- `I/D/D^T` 是本地计算，不计入物理投影调用，但其行数、wall time 和内存必须单列。
- 只评分 `k={4,8,12,16,20,24}`；不得为某个方法追加 checkpoint。
- evaluator 另用一次 batched raw forward，算法账本与 evaluator 账本分开。

## 4. 冻结网格

- unregularized CGLS：1 个配置；
- L2：`1e-6` 到 `1e-2` 的 9 点；
- H1：`1e-9` 到 `1e-5` 的 9 点；
- mixed Sobolev：9 对 `(lambda_l2, lambda_h1)`，固定比例路径，不做二维自由搜索。

总计 28 个配置、24 cases、6 个评分 checkpoint，即 4,032 个评分行。完整网格无论好坏都必须保存，
不得只留下最佳曲线。

## 5. pilot 信号而非成功门

对 validation-case mean curve，一个候选点只有同时满足下列条件才标为 `pilot_signal`：

1. 不被任何六个 CGLS checkpoint 在 field/gradient 两项同时支配；
2. 至少支配一个 CGLS checkpoint；
3. 相对同一 `k` 的 CGLS，field 和 gradient mean gain 均非负；
4. 两者至少一项 mean gain 达到 1%；
5. active measured residual 与 held-out-B mean ratio 均不超过 1.05。

这个标记只允许设计另一个冻结的 R2-A formal screen。不能写成新 Pareto 算法，因为：

- 网格、规则和数据来自同一个已打开 development 世界；
- 24 cases 不是未见 rig，也没有 cluster-level 推断；
- lambda 和 checkpoint 搜索尚未经过独立 lockbox；
- quadratic path 尚未与 TV/Huber、hybrid projected Tikhonov 或组内强基线比较。

## 6. fail-closed 条件

任一非有限值、breakdown、checkpoint 缺失、调用账不等于 `24F/24A^T`、data residual 重算不一致、
tracked source 未提交或输出目录已存在，都使 pilot 状态为 `INVALID`，不得解释科学结果。

## 7. 后续顺序

1. 若 28 个配置均无 signal：缩小问题，转向 hybrid projected H1 或 TV/Huber；不训练网络。
2. 若有少量 signal：围绕最窄的一段 lambda 冻结单一 `(family, lambda, k)`，另建新 seed 的 D1 tune 与
   D2 lockbox，并做 phantom-cluster bootstrap 和 OOD 尾部门。
3. 只有 formal screen 产生稳定新 Pareto 点，才允许研究 observable strength controller；否则继续关闭神经模块。

## 8. 禁止措辞

- “提出了新的 BOST 重建算法”；
- “在真实反应流上有效”；
- “优于 DeepONet/FNO/NeRIF/TDBOST”；
- “证明了泛化”；
- “取得论文突破”。
