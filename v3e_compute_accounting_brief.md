# T16 v3e 跨架构算力账本与下一轮比较合同

## 一句话判决

**五种架构的成本测量框架已经跑通，但跨架构确认性 superiority 仍然关闭。** 本页是成本轴；后续 v3f 已补当前 DeepONet/FNO 的样本等权 24→240 epochs development frontier，U-Net、F-Adapter 与未来自研方法仍未补齐。

这一步的价值不是宣布谁最好，而是防止把“参数少”“单点误差低”或“训练 epoch 相同”误写成公平。

## 测量协议

- 数据接口：同一 K=6、8×16×16、42-channel ridge/ray/mask/angle/support/coordinate 输入。
- 架构：ridge residual 3D U-Net、FNO、DeepONet、历史 ray-set operator、frozen-FNO zero-init ray-set adapter。
- 设备：本机 Apple Silicon MPS，FP32。
- 重复：每个模型 3 个全新 Python worker；总计 15 个独立进程。
- 推理：batch=1，8 次 warm-up，40 次同步计时。
- 训练：batch=12，3 次 warm-up，20 次同步计时；包含 field、gradient、BOST reprojection、boundary loss、backward 和 AdamW step。
- 内存：采样 MPS allocated/driver memory，同时记录独立进程 peak RSS。
- 权重：仅做成本测量的确定性随机初始化；不读取 checkpoint，不用 validation/test error 排成本名次。

## 当前五模型成本

| 方法 | 总参数 | 可训练参数 | FLOPs-v1 / forward | 推理 p50, B=1 | 完整训练步 p50, B=12 | 观测 MPS 训练分配峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 3D U-Net | 94,193 | 94,193 | 84.87 M | 1.16 ms | 18.40 ms | 33.16 MiB |
| Residual FNO | 44,203 | 44,203 | 14.72 M | 1.49 ms | 8.54 ms | 38.34 MiB |
| Residual DeepONet | 49,313 | 49,313 | 13.66 M | 0.77 ms | 3.01 ms | 35.08 MiB |
| Ray-set operator | 45,973 | 45,973 | 41.88 M | 2.07 ms | 95.06 ms | 100.26 MiB |
| Frozen-FNO zero-init adapter | 49,191 | 4,988 | 54.79 M | 2.99 ms | 94.17 ms | 74.74 MiB |

表中时间是修正 validation 聚合后重新运行的三个 fresh workers 中位数。DeepONet 完整训练步的 worker medians 为 2.96–3.02 ms；毫秒级绝对值仍受本机负载影响，架构比较必须保留 raw trials、worker spread 和复测环境。

## 必须理解的三个机制结论

### 1. 可训练参数少不等于训练便宜

zero-init adapter 只有 4,988 个可训练参数，但每步仍需执行 frozen FNO 和逐视角 3D set encoder。它的训练步约是 FNO 的 11.0 倍，推理约 2.0 倍。PEFT 的“少参数”主要减少 optimizer state 和需要更新的权重，不会自动消除前向、激活和输入分支成本。

### 2. FLOPs 估计必须带口径

`FLOPs-v1` 是 partial analytical estimate，由以下三部分组成：

1. Conv/Linear/显式 contraction：每个 real MAC 记 2 FLOPs；
2. FNO retained modes：每个 complex MAC 记 6 real FLOPs；
3. real FFT：按 `2.5 × N × log2(N)` / channel / transform 估计。

它不包含 normalization、activation、pooling、softmax、indexing、大多数逐元素运算和 optimizer arithmetic。因此它是**版本化的部分解析估计**，不是完整 forward FLOPs 或芯片硬件计数。ray-set 的实测训练延迟远高于 FLOPs-v1 已经说明：论文必须同时报告解析成本和真实 wall time。

### 3. 最低最终误差与最快达到目标不是同一件事

按每个三维场等权修正后，三种 FNO 协议到 epoch 240 都未达到 validation L2 ≤ 0.09。对更现实的 `≤0.10`：

- long-cosine：epoch 132，24.52 s/seed；
- validation 冠军：epoch 180，35.29 s/seed；
- restart-both：epoch 240，45.68 s/seed。

因此论文不能只报最终 endpoint；必须同时报告 time-to-target 和完整 Pareto frontier。

## 下一轮自研模型的公平合同

每个候选与 FNO、DeepONet、F-Adapter/static control 必须共同填写：

1. `60 / 120 / 180 / 240` epoch 的 validation prefix-best error；
2. 累计训练秒数、time-to-`0.12/0.10/0.09`；
3. 总参数、可训练参数、FLOPs-v1、推理 p50/p90；
4. 完整训练步 p50/p90、MPS/GPU memory；
5. dev2 的 mean、CI、p10、harm、最差 OOD 和 Q_audit；
6. 相同 geometry、相同输入、相同 ridge anchor、相同 loss 与 batch-order contract。

候选若只在一个 240-epoch endpoint 赢约 1%，但没有稳定 time-to-target、尾部或 OOD 优势，不支持架构贡献。

## 我现在应该学什么

1. 用 `v3e_compute_profiles.csv` 手算 adapter/FNO 的参数比、训练时间比和内存比。
2. 用 `v3e_fno_time_to_target.csv` 画横轴为秒、纵轴为 validation L2 的三策略曲线。
3. 对照 `run_v3e_compute_accounting.py`，解释一次 forward、一次 training step 和 end-to-end reconstruction 的成本边界。
4. 自己实现一个 Conv3D 与 Linear 的 MAC hook，并通过手算 shape 检查。
5. 向师兄口述：为什么 “4,988 trainable parameters” 不能证明 adapter 更轻量。

## 请师兄优先确认

1. 是否认可这套参数/FLOPs-v1/MPS wall-time/memory/time-to-target 作为开发期成本账本？
2. 下一轮是否要求 U-Net、DeepONet、FNO、F-Adapter 和自研模型全部跑相同 60/120/180/240 checkpoints？
3. 真实 BOST geometry 是否随 case、camera mask、calibration 或布局变化，足以支持 acquisition-conditioned 分支？
4. 组内更看重最低终点误差、固定时间误差，还是达到某个重构质量的最短时间？

## 证据入口

- 配置：`demo_t16_operator/configs/v3e_compute_accounting.json`
- 执行器：`demo_t16_operator/run_v3e_compute_accounting.py`
- 独立校验：`demo_t16_operator/validate_v3e_compute_results.py`
- 聚合成本：`demo_t16_operator/results/v3e_compute_accounting/v3e_compute_profiles.csv`
- 15 个 worker：`demo_t16_operator/results/v3e_compute_accounting/v3e_compute_trials.csv`
- FNO 固定 checkpoint：`demo_t16_operator/results/v3e_compute_accounting/v3e_fno_error_compute_checkpoints.csv`
- time-to-target：`demo_t16_operator/results/v3e_compute_accounting/v3e_fno_time_to_target.csv`
- 机器报告：`demo_t16_operator/results/v3e_compute_accounting/v3e_compute_report.json`

## 仍然不能声称

- 不能仅凭本页成本声称 DeepONet 比 FNO 更好；accuracy 结论必须引用后续 v3f matched development frontier。
- 不能声称 ray-set 或 adapter 是新算法；已有开发结果还是负对照。
- 不能声称 FLOPs-v1 是精确芯片操作数。
- 不能声称 synthetic 8×16×16 结果能迁移到真实 OERF BOST。
- geometry、独立 forward、真实样例、确认性 blind final 和 NeRIF 端到端成本仍未完成。
