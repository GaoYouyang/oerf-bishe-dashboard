# R2-C2 全局 β 插值诊断：安全，但没有越过 field-gradient 冲突

> 日期：2026-07-21
>
> 证据级别：已打开 development set 的 post-open 机制诊断
>
> 判决：`VALID_POST_OPEN_DIAGNOSIS_NO_GO_INTERPOLATION`
>
> 独立复核：`VALID_INDEPENDENT_R2C2_POST_OPEN_NO_GO`，709 checks
>
> 突破状态：无算法突破、无真实 BOST、无 fresh validation、无泛化或论文成功

## 1. 这次到底问了什么

上一轮完整对角 PDHG 在同样的 `20F/20A^T` 下让 field relative-L2 平均改善
`0.1469%`，却让 gradient relative-L2 平均恶化 `0.8203%`。这轮没有继续扫正则权重，
而是问一个更窄的问题：

> 在严格安全的 scalar metric 和完整 diagonal metric 之间，是否存在一个全局标量
> `beta`，既保留一点 field 收益，又把 gradient 伤害消掉？

运行前固定：

- 同一组已打开的 6 个公开 PSU 几何 + 解析反应形态 case；
- 同一 Huber-gradient 目标，`lambda=1e-3`；
- 每条路径严格 `20F/20A^T`，另记 1 次 evaluator forward；
- `beta={0,.25,.5,.75,1}`；
- 只有 `.25/.5/.75` 是可选内点；
- 同时要求 mean field、mean gradient 为正，尾部不低于 `-2%`，残差比不超过 `1.05`；
- 通过也只能授权全新 seed 验证，不能授权论文结论。

## 2. 为什么这条插值仍然有证明

令加权数据与正则联合正规矩阵为

```text
H = sigma_A A^T A + sigma_G G_P^T G_P.
```

已有两个证书：

```text
diag(q_scalar P) - H >= 0,
diag(q_diagonal) - H >= 0.
```

对一个全局标量 `beta in [0,1]`，定义

```text
q_beta = (1-beta) q_scalar P + beta q_diagonal.
```

于是

```text
diag(q_beta) - H
= (1-beta)[diag(q_scalar P)-H] + beta[diag(q_diagonal)-H] >= 0.
```

再取 `tau_j=0.99/q_beta,j`，归一化联合算子的平方谱范数不超过 `0.99`。小矩阵测试对
5 个 beta 都显式构造 Schur 差并检查最小特征值；还检查了端点等价、MPS 转 CPU、输入不别名、
证书内容哈希和篡改前物理调用为零。

注意：证明只适用于每个 case 一个全局标量 beta。把它换成逐体素 `beta_j`，一般不能把矩阵差写成
两个 PSD 矩阵的标量凸组合，因此没有自动继承这份证明。

## 3. 结果

所有数字都相对独立运行的 scalar-PDHG reference：

| beta | mean field gain | mean gradient gain | worst field | worst gradient | residual ratio | 最大步长 / scalar |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 1.0000 | 1.000 |
| 0.25 | +0.0351% | -0.1879% | +0.0169% | -0.3010% | 0.9926 | 1.089 |
| 0.50 | +0.0713% | -0.3866% | +0.0343% | -0.6202% | 0.9850 | 1.195 |
| 0.75 | +0.1086% | -0.5971% | +0.0520% | -0.9591% | 0.9772 | 1.325 |
| 1.00 | +0.1469% | -0.8203% | +0.0702% | -1.3192% | 0.9693 | 1.486 |

调用账本：

- 36 条求解路径；
- `720F / 720A^T` solve calls；
- 36 次独立 evaluator forward；
- 30 个 q_beta 证书均为 `0F/0A^T` setup；
- beta=0 与 scalar reference 的最大 field 差为 `4.93e-8`，低于冻结的 `1e-6` 门。

按预写选择规则，最佳可选内点是 `beta=0.25`，但它仍在 6/6 case 上伤害 gradient，因而：

```text
eligible_gate_pass_count = 0
fresh_validation_authorized = false
```

## 4. 真正学到的不是“0.25 比较好”

独立 validator 逐项重算后确认，在这 5 个固定点上：

- mean field gain 随 beta 严格单调上升；
- mean gradient gain 随 beta 严格单调下降；
- residual ratio 随 beta 下降；
- 最大局部步长随 beta 增大。

这是一条非常整齐的有限预算权衡：越开放对角加速，数据拟合越快，field 的全局 L2 略好，但高频/梯度误差
越坏。它不是“beta 网格太粗”造成的偶然漏点；即使在 0 附近，梯度方向已经是负的。

更根本地说，预条件器改变的是到达同一优化目标的路径和速度，不改变目标本身。若迭代充分收敛，scalar 与
diagonal PDHG 应到达同一个解。当前 20 步代表图仍接近零场，也说明这个候选主要在比较欠收敛阶段的路径，
不能成为高质量重建方法的核心贡献。

## 5. 路线判决

### 正式关闭

- 不再把全局 beta 当作 fresh-validation 候选；
- 不训练网络预测 beta，因为当前没有一个通过双指标门的监督标签；
- 不把 `+0.035%` field 单指标微增益写成算法收益；
- 不继续用更密 beta 网格做结果驱动调参；
- 不把安全性证明误写成质量优越性证明。

### 下一条可执行候选：改变路径，而不是只改变步长

下一阶段转向 **H1-to-edge-preserving safeguarded path**，先做固定确定性版本：

1. 用冻结 H1 强基线产生稳定低频/体场解；
2. 在相同总 `F/A^T` 预算中分配少量 Huber/TV 或 edge-superiorization correction；
3. 每次局部修正后检查 active residual envelope，越界则回退到 H1；
4. 同时比较 field、gradient、front、active residual、held-out clean/B、逐 case 与逐 geometry 尾部；
5. 对手至少包括 CGLS、冻结 H1、scalar Huber、简单阻尼/线性插值和等成本 continuation；
6. 只有固定路径出现 H1 无法解释的 headroom，而且不同 case 的最优控制确实不同，才允许小模型读取
   geometry/noise/residual/Ritz 特征，输出一个有界强度或固定路径凸权重；
7. 任何 learner 必须允许拒答并精确回退 H1。

这条路线仍只是候选。没有何远哲师兄的真实 callable、曲/直光线残差层级、标定、数据 split 与组内强基线，
不能声称它解决真实 BOST 困境。

## 6. 复现与证据

- 冻结配置：`demo_t16_operator/configs/lgwo_a24_r2c2_beta_interpolation_diagnosis_v1.json`
- 证书构造：`demo_t16_operator/interpolated_pdhg_metric.py`
- 数学与失败关闭测试：`demo_t16_operator/test_interpolated_pdhg_metric.py`
- 运行器：`site_tools/run_lgwo_a24_r2c2_beta_interpolation_diagnosis.py`
- 独立验证器：`site_tools/validate_lgwo_a24_r2c2_beta_interpolation_diagnosis.py`
- 结果目录：`demo_t16_operator/results/lgwo_a24_r2c2_beta_interpolation_diagnosis_v1/`

```bash
.venv/bin/pytest -q \
  demo_t16_operator/test_interpolated_pdhg_metric.py \
  demo_t16_operator/test_diagonal_pdhg_metric.py \
  demo_t16_operator/test_interface_baselines.py

.venv/bin/python site_tools/run_lgwo_a24_r2c2_beta_interpolation_diagnosis.py
.venv/bin/python site_tools/validate_lgwo_a24_r2c2_beta_interpolation_diagnosis.py
```

## 7. 结论边界

可以说：一个具有显式 Schur 安全证明的全局 scalar-diagonal 插值，在已打开的 6 个公开 PSU 合成 case 上呈现
严格单调的 field-gradient 权衡，所有可选内点均未通过双指标门，因此该候选被关闭。

不能说：提出了新算法、改进了真实 BOST、优于 H1/DeepONet/FNO/NeRIF/TDBOST、获得未见 rig 泛化、完成论文
或出现突破。
