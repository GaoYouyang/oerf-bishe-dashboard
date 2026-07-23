# PSU 公开数据到 N2 合同的接口演习

日期：2026-07-18
机器状态：`PUBLIC_PSU_INTERFACE_REHEARSAL_ONLY_N2_BLOCKED`
决策：`GO_INTERFACE_REHEARSAL_STOP_N2_ALGORITHM_CLAIMS`

## 1. 这次到底做了什么

这不是重建实验，也没有训练新网络。我们把 PSU 70-view 开放 BOST 数据的论文事实、公开归档清单、
几何审计、九视角 B0 `A/A^T` 接口和永久留出协议逐字段映射到 OERF N2 真实数据合同。生成器只允许
五种证据状态：

- `PUBLIC_SUPPORTED`：公开一级来源或已审计聚合证据直接支持；
- `PUBLIC_NEGATIVE`：公开证据明确说明该条件不存在；
- `LOCAL_VERIFICATION_REQUIRED`：上游资料看起来存在，但必须绑定实际文件和 checksum；
- `MISSING`：当前材料没有；
- `FORBIDDEN_TO_INFER`：即使看起来相近，也禁止替实验室或论文猜测。

机器报告有 16 个字段组：6 个公开支持、2 个公开负证据、3 个需要本地绑定、2 个缺失、3 个禁止推断。
七个正式 N2 门仍全部为 `false`，所有训练、开 audit、成功和公开原始数据授权也全部为 `false`。

## 2. 公开 PSU 能证明什么

| 能证明的接口事实 | 当前证据 |
|---|---|
| 公开实验有 7 台相机、10 个旋转 run、70 个视角 | 论文与 held-out protocol 一致 |
| rotation-40 位移载荷为 `2160 x 2560`、像素位移和作者 mask 语义可读 | cell-payload 审计 |
| 九个 support views 的 B0 线性 forward/adjoint 可在 CPU64/MPS32 运行 | interface audit |
| 当前最大 CPU64 点积相对误差低于 `5e-16`，MPS32 低于 `1e-7` | 两个网格 profile 均过各自接口门 |
| 三个 support rotations、一个 development rotation、六个 sealed audit rotations 的政策可机械覆盖 70 views | held-out protocol |
| held-out active-vector relative L2 是合法的图像空间一致性终点 | 但不是唯一三维真值 |

这足以做 loader、row order、mask、ray/support、`A/A^T`、cost ledger 和 audit-lock 的接口演习。

## 3. 它为什么还不能证明有限孔径算法成功

一级来源同时给出两个容易被混淆的事实：

1. 论文报告每次 test 实际采集了 2000 张 flow-off 和 2000 张 flow-on 图像；
2. 当前公开压缩包清单只暴露每个 camera-rotation 的平均图像或复合容器，未暴露可逐帧核验的独立
   temporal repeats。公开 inventory 因此仍是每固定条件 `0` 个可访问独立 flow-off repeats。

“实验中采过”不等于“我们当前拿到了逐帧数据”。70 个 camera-rotation views 也不是 70 个时间重复。

论文还报告 105/200 mm 镜头使用 `f/32`，85 mm 镜头使用 `f/22`，但焦距、相机位置和 optical
channel 同时变化。这不是同一光路、同一几何下只改变孔径的干净干预，不能用来断言 residual 差异由
finite aperture 单独造成。

其他硬缺口是：

- 没有可核验 acquisition timestamp 和 session identity；
- 没有每个 N2 sensor 的 calibration version、timestamp、optical-channel binding 和 confidence；
- 没有独立三维场真值；
- 没有已物化的 N2 split member manifest 与可复算 digest；
- 没有项目级本机训练、论文、图和再分发权限合同。

所以公开 PSU 是接口考场，不是 OERF 真实有限孔径算法的成绩单。

```text
接口演习 = 数据能读 + ray 能建 + A/A^T 能跑 + 数值检查通过
算法证据 = 独立物理对照 + 永久留出 + 合法终点 + 重复统计 + 成本账本
```

## 4. 一级来源暴露出的真实算法瓶颈

[Molnar 等人的开放数据论文](https://arxiv.org/html/2508.17120)报告，其 NIRT data loss 使用
cone-ray operator；估计 coefficient of variation 为 `8.5%`，需要约 `8000` points per pixel。
这给出一个比“再堆更大的 DeepONet/FNO”更具体的研究问题：

> 能否在不改变 cone-ray 期望、不偷看真实场、保留可审计伴随和误差条的前提下，显著减少每像素
> aperture samples，并让有限预算三维重建更稳定？

它与已经关闭的 v5y/v6a 不同。v5y/v6a 试图用确定性 geometry-conditioned kernel 直接替换完整
operator；v6a 虽有平均 `8.08%` 改善，却只赢 6/12 rigs、最坏恶化 `13.69%`，因此已停止容量升级。

下一候选只允许作为**预注册假设**：用可解析积分的低阶 aperture surrogate 做 control variate，
再用与 pilot 独立的高保真残差样本校正，使 forward estimator 保持无偏；learner 只分配样本或预测
control-variate 系数，不能输出最终三维场或删除 residual correction。

最小比较必须包含：uniform MC、Hammersley/QMC、thin-ray、冻结 v6a deterministic surrogate 和
control-variate hybrid；按相同 ray-sample 或 wall-time 预算报告 forward relative-L2、gradient cosine、
逐 rig p95/最坏伤害、`A/A^T` 点积、F/A 调用、峰值内存和重建 field/H1。没有 fresh geometry 和
真实 N2 合同时，只能写 synthetic cost/variance 结果。

## 5. 向何远哲师兄要的最小材料

| 最小材料 | 为什么要 |
|---|---|
| 一个完整真实 session manifest：run/session/condition/geometry、时间、单位、来源 | 让数据身份可追踪 |
| 每个固定条件至少 50 张未平均 flow-off/reference repeats | 当前工程门，用于噪声与慢漂移；不是普适统计定理 |
| 每台相机的 calibration version、pixel/ray、mask、confidence、optical channel | 绑定物理 forward 并发现 row-order 错误 |
| 同一 optical channel、同一 geometry 的至少两档 f-number 或 focus | 把有限孔径从镜头/位置/标定混杂中隔离出来 |
| 可调用的 `A/A^T` 或 `JVP/VJP` 与一个 tiny reference case | 做伴随、单位、support 和 matched-cost 审计 |
| 永久留出的 camera/session/geometry manifest 与 SHA-256 | 防止把已打开视角重新命名为 fresh |
| 独立 endpoint：phantom、CFD/PIV/pressure 或已知几何量 | 区分三维场精度与图像一致性 |
| 保存、训练、组会、论文文字、图、公开派生量和 raw data 的分别权限 | 避免组内材料进入公开 Git |

## 6. 可复现入口

```bash
.venv/bin/python site_tools/build_psu_n2_contract_rehearsal.py \
  --output docs/psu_n2_contract_rehearsal_public_summary.json
.venv/bin/python -m pytest -q site_tools/test_build_psu_n2_contract_rehearsal.py
```

- [字段级机器报告](psu_n2_contract_rehearsal_public_summary.json)
- [论文原文事实审计](psu_primary_source_fact_audit_2026-07-18.json)
- [N2 完整合同](oerf_n2_physical_mismatch_data_contract_2026-07-18.md)
- [师兄一页确认单](oerf_n2_advisor_intake_brief_2026-07-18.md)

## 7. 结论边界

本轮真实增量是“公开数据可支持的接口证据”和“只能由 OERF 真实交付补齐的物理证据”已经由机器分开。
它没有产生模型分数、三维重建、真实提升或论文成功。下一次科学 Go 必须来自一个冻结 primary mismatch、
一个不可回看的 audit unit，以及不依赖已打开 v5y/v6a rig 的新预注册实验。
