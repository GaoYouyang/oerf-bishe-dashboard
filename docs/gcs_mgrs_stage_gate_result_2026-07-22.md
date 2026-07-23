# MGRS 两阶段候选审计结果：Stage A NO-GO

日期：2026-07-22

证据等级：`E1_POSTOPEN_SYNTHETIC_CANDIDATE_AUDIT_ONLY`

正式状态：`MGRS_STAGE_A_NO_CANDIDATE_STAGE_B_SEALED`

## 一句话结论

多尺度梯度共识护栏确实让高频残差在新角度的连续投影上更稳定，但这种改善没有可靠转化为三维场误差下降；两个预先冻结的候选都没有通过 Stage A，因此四个形态留出单元没有运行。

这不是突破，也不是“差一点就算成功”。它把下一步问题从“怎样让投影拟合更好”收窄为“怎样限制投影零空间中的三维残差”。

## 1. 上游机制证据

先前的连续/离散审计固定 7 个未用于 post-hoc 探针的 family-noise 单元，并对同一已训练场分别使用 `FD(h)`、`FD(h/2)`、`FD(h/4)` 和 automatic derivative 渲染：

- 高频模型在 dense unseen angles 的 automatic renderer 下 7/7 比低频模型差；
- 高频减低频的 dense-AD relative-L2 中位数为 `+0.48161`；
- 高频模型自身的 automatic 减 `FD(h)` 中位数为 `+0.54144`；
- development GCS 与 dense-AD 损害的 Spearman 相关为 `0.82143`；
- 原 central-test 重放最大绝对差为 `0.0`。

因此，本轮不是凭肉眼猜测“高频可能过拟合”，而是在同一连续场上分离了离散训练渲染器与连续导数渲染器。

## 2. MGRS 做了什么

MGRS 暂指 `Multi-scale Gradient-consensus Residual Safeguard`：

1. 用 `[1,2,4]` Fourier 频率训练低频基座；
2. 冻结基座；
3. 添加最终输出严格为零的高频残差；
4. 同时优化 automatic、`FD(h)`、`FD(h/2)`、`FD(h/4)` 四个训练投影损失；
5. development checkpoint 必须四个 renderer 逐项都不劣于基座；
6. 归一化平均改善不足 0.5% 时恢复零残差，输出与基座精确相同。

这保证“未获准的残差不改变基座”，不保证“获准残差一定更接近真实三维场”。后者正是本轮被证伪的部分。

## 3. 冻结候选与判决

物理单元先对两个优化 seed 取中位数；seed 不是独立场样本。

| 候选 | Stage A 场改善单元 | 场 relative-L2 中位差 | 最大单元场损害 | dense-AD 中位差 | 判决 |
|---|---:|---:|---:|---:|---|
| `MGRS-56` | 0 / 3 | `+0.001859` | `+0.004316` | `-0.017156` | NO-GO |
| `MGRS-6816` | 2 / 3 | `-0.001082` | `+0.001372` | `-0.027056` | NO-GO |

冻结门要求至少 2/3 单元场改善、场误差中位差不高于 `-0.002`、单单元最大损害不高于 `+0.01`。`MGRS-6816` 只差在主效应幅度：它改善了 2/3 单元，也改善了 dense-AD，但 `-0.001082` 没达到预写门；不能在看见数字后把门改成 `-0.001`。

12 条 seed-level 候选路径中：

- `MGRS-56` 只有 2/6 残差通过 development 准入；
- `MGRS-6816` 有 4/6 通过；
- 未准入路径全部恢复为严格零残差；
- 基座对上游结果的重放最大绝对差为 `0.0`；
- 全部数值有限，结果包 checksum 通过；
- 运行耗时 `69.18 s`，设备为 Apple MPS。

## 4. 为什么投影改善没有稳定变成场改善

BOST 观测是折射率梯度沿射线的投影积分。不同三维扰动可能产生相近的有限视角投影，尤其在相机稀疏时。MGRS 的四个 renderer 都使用同一组 train/development 角度；它们能排除“只适配一个差分步长”的残差，却不能排除“在这些角度下几乎不可见、但改变三维场”的残差。

从线性化角度看，若残差 `delta n` 的投影满足 `A delta n` 很小，投影损失很难判断它是否有害。automatic 与多档 finite difference 改变了离散梯度算子，但没有凭空增加独立相机信息。因此，下一候选必须显式处理最小范数或结构先验，而不是继续加 renderer 数量。

## 5. 下一候选：先做经典最小 H1 残差，再谈学习

下一轮只在已经打开的 Stage A 单元上开发，不触碰 Stage B：

\[
\min_{\delta n}\; \frac{1}{4}\sum_{s\in\{AD,h,h/2,h/4\}}
\|R_s(n_0+\delta n)-y\|_{\Sigma^{-1}}^2
+ \lambda_0\|\delta n\|_2^2
+ \lambda_1\|\nabla\delta n\|_2^2.
\]

必须先比较：

1. 纯低频基座；
2. 当前 MGRS；
3. 经典 H1/Tikhonov residual；
4. TV/Huber residual；
5. 只有当学习式频带 gate 在同预算下超过 1--4，才进入神经算子版本。

这一步可能只得到更强经典基线。若是如此，应保留为毕业设计的可靠部分，不应硬写成新网络。

## 6. 需要何远哲师兄确认的五个问题

1. NeRIF/组内当前 renderer 在训练时主要使用 automatic、numerical，还是二者联合？中心差分步长由体素、ray interval 还是人工超参数决定？
2. 改变 ray sample 数或重建分辨率后，是否观察到高频细节、held-out view displacement 与三维场质量方向不一致？
3. 真实数据有没有一到两个完全不参与训练的相机角，且保留原始 displacement、mask、sigma/covariance？
4. 是否能提供同一场的不同采样密度或不同 optical forward（straight/curved、finite aperture）用于 renderer-transfer 审计？
5. 组内认为最重要的终点是 refractive-index field、density/temperature、held-out displacement，还是 PIV-BOST 补偿后的 velocity？它决定 H1/edge prior 是否有物理意义。

## 7. 复现

```bash
PYTHONPATH=. .venv/bin/python learning_labs/run_gcs_mgrs_stage_gate.py \
  --config demo_t16_operator/configs/gcs_mgrs_stage_gate_v1.json \
  --output-dir learning_labs/results/gcs_mgrs_stage_gate_v1

.venv/bin/python -m pytest -q \
  learning_labs/test_gcs_multiscale_residual.py \
  learning_labs/test_run_gcs_mgrs_stage_gate.py

cd learning_labs/results/gcs_mgrs_stage_gate_v1
sha256sum -c checksums.sha256
```

结果目录已经存在时 runner 会拒绝覆盖；正式重放应使用新的空目录。

## 8. 严格边界

当前允许：synthetic continuous/discrete aliasing 机制证据，以及一个 Stage A 失败的零初始化多尺度残差候选。

当前不允许：新算法、算子学习、优于 NeRIF/FNO/DeepONet、真实 BOST/OERF 重建、跨 rig 泛化、论文成功或突破。
