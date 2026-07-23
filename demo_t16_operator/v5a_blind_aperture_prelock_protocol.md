# v5a 有限孔径盲标定：锁定集首开前预注册

状态：`PRELOCK_FROZEN`。本文件提交时，默认结果目录
`results/v5a_blind_aperture_calibration/` 尚不存在，独立锁定集尚未构造。

## 要检验的问题

在一个明确区分 `A_truth != A_reconstruction` 的弱偏折 BOST 合成问题中，先用
可见重建相机的残差在有限孔径候选算子库中做盲选择，再把有足够置信度的样本切换到
所选算子；它能否在未见反应场家族、未见真实孔径、未见机架几何和未见噪声水平上，
稳定优于等算子调用预算的针孔 FISTA？

这不是非线性光线追迹、真实 BOS 实验、NeRIF/TDBOST 复现或神经算子优越性证明。

## 冻结的开发集选择

- 最强等预算基线：`pinhole_fista_equal_calls`，60 次 forward/adjoint。
- 候选方法：`full_residual_hard`。
- 置信度阈值：`0.02182040736079216`。
- 开发集覆盖率：`10/30 = 33.33%`。
- 开发集平均相对 L2 增益：`5.342826843261719%`。
- 开发集 p10 增益：`0.0%`。
- 开发集恶化超过 1% 的比例：`1/30 = 3.33%`。
- 开发集针孔失配惩罚：`5.914941310882568%`。

开发集真值只用于选择基线、方法和阈值。部署时每个样本的接受/回退只依赖可观测
残差置信度。审计相机不进入重建、交叉视角探测、方法选择或噪声尺度估计。

## 等预算口径

- 5 个候选算子。
- 每个算子：2 步全视角探测 + 2 个交叉视角折各 2 步，共 6 次。
- 诊断合计 30 次 forward/adjoint；最终所选逐样本算子再运行 30 步 FISTA。
- 候选总计 60 次 forward/adjoint；等预算针孔 PBB/FISTA 各 60 次。
- 精确谱分解、候选矩阵生成和调度开销另行记账，因此只声称算子调用预算相同，
  不声称总墙钟时间或总 FLOPs 相同。

## 首开门槛

锁定集必须同时满足：

1. 针孔失配惩罚至少 `2%`；
2. 相对最强等预算基线的平均 L2 增益至少 `2%`；
3. p10 增益不低于 `0%`；
4. 恶化超过 `1%` 的比例不高于 `5%`；
5. 完全未参与重建的审计相机真算子重投影误差不得上升。

任一项失败，v5a 结论即为失败或不完整；不得依据锁定集结果修改本配置后仍称为同一次
首开。后续改进必须另立 `v5b` 并使用新的锁定证据。

## 冻结范围与哈希

- runner SHA-256：`c74dc6c9b4acce29d93a38b6cbd9bceceecf1ebffafd0f542baa6a2ff3482581`
- finite-aperture generator SHA-256：`86b7b2873e4035c03070c24c4659339677a04b620ff73d1b229141d6e2496a1d`
- config SHA-256：`2c87678d24394c53ab3b5136d312d8cca77c85d3e898406c124c778fe16169d5`
- 测试命令：
  `PYTHONPATH=. .venv/bin/pytest -q demo_t16_operator/test_finite_aperture_bost.py demo_t16_operator/test_v5a_blind_aperture_calibration.py`
- 冻结前结果：`8 passed`。

## 作废记录

- `v5a_blind_aperture_development/`：候选最终重建仅获 4 步，预算不公平，作废。
- `v5a_blind_aperture_development_v2/`：审计相机无噪声信号参与了噪声尺度估计，存在
  间接信息通道，作废。
- `v5a_blind_aperture_development_v3/`：修复上述问题后的选择集证据，只用于本预注册，
  不作为锁定集或论文结论。

## 首开命令

```bash
.venv/bin/python demo_t16_operator/run_v5a_blind_aperture_calibration.py \
  --output-dir demo_t16_operator/results/v5a_blind_aperture_calibration
```
