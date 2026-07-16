# T16 三维 BOST 神经算子：三随机种子因果消融审核 brief

更新：2026-07-10

## 一句话结论

当前 synthetic closure 支持继续研究 FNO，但不支持把结论写成“Residual FNO 在所有工况最好”。更准确的判断是：

- 在接近参数量下，Residual FNO 的场误差与 held-out 重投影均稳定优于 3D U-Net；
- BOST 重投影损失在五个测试域的平均方向均有利，但三种子区间仍宽；
- residual skip 在 IID / noise OOD 有利，却在 3-view / joint OOD 被 absolute-output FNO 反超；
- thin-front family OOD 仍约为 `0.67`，说明当前主要瓶颈已经不是换一个浅层架构，而是训练分布、几何和跨族先验。

因此最有研究价值的下一模型不是再堆一个热门架构，而是 **view-reliability-aware residual/absolute hybrid operator**：让模型根据视角数、几何覆盖和观测残差决定保留多少 physics lift，并继续用 held-out camera 验收。

## 实验契约

- 固定同一批 `168` 个 `8 x 16 x 16` synthetic volumes，不改变数据样本；
- 三个优化随机种子：`20260710 / 20260711 / 20260712`；
- 完整 12 次训练独立重跑后，排除 wall-clock 的配对指标 CSV 字节级一致；
- 每个模型 `30` epochs，同一 AdamW、batch、gradient/boundary 权重；
- 不确定性为三个优化种子均值的 Student-t 95% 区间，不是数据 bootstrap；
- 所有 FNO 都有 `43,363` 参数；匹配容量 U-Net 为 `49,111` 参数；
- absolute-output FNO 仍接收 calibrated physics lift，只拿掉 residual skip；它不是 raw projection-to-volume direct operator。

## 五个测试域的 field relative L2

数值为三个优化种子的均值；physics lift 是无训练的固定基线。

| 方法 | IID | 3-view OOD | noise OOD | joint OOD | family OOD |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics lift | 0.457 | 0.770 | 0.479 | 0.779 | 0.795 |
| residual FNO | **0.219** | 0.488 | **0.252** | 0.492 | **0.667** |
| absolute-output FNO | 0.237 | **0.457** | 0.277 | **0.454** | 0.670 |
| FNO without reprojection loss | 0.238 | 0.505 | 0.266 | 0.514 | 0.677 |
| parameter-matched U-Net | 0.300 | 0.612 | 0.331 | 0.615 | 0.741 |

IID held-out reprojection 均值为：physics lift `0.520`、residual FNO `0.251`、absolute FNO `0.268`、no-reprojection FNO `0.270`、matched U-Net `0.349`。

## 三条可以保留的研究判断

### 1. FNO 优势不是因为 U-Net 参数更少

匹配容量 U-Net 比 FNO 反而多约 `13.3%` 参数。Residual FNO 在五个测试域的 field 与 held-out 均值都更低，并且在 `3 seeds x 5 domains = 15` 个 field 单元和 15 个 held-out 单元中全部获胜。

这只支持“当前数据和预算下 spectral mixing 值得继续”，不证明任意 FNO 都优于任意 CNN。

### 2. 重投影损失方向一致，但证据强度还不够

去掉 reprojection loss 后，五个测试域的 field 均值全部上升；15 个 seed-domain 单元中，带 reprojection 的 residual FNO 有 13 个 field 单元更好。held-out 指标同样大多变差。

但是只有三个优化种子，部分 Student-t 区间跨零。下一轮应增加数据种子或训练种子，并扫描 `lambda_reprojection`，不能直接写成显著性结论。

### 3. residual skip 的收益取决于视角工作域

相对 absolute-output FNO：

- residual FNO 在 IID 与 noise OOD 的三个种子全部更好；
- absolute-output FNO 在 3-view 与 joint OOD 的三个种子全部更好；
- family OOD 基本打平，且两者绝对误差都很高。

这说明受损严重的 physics lift 可能通过固定 residual skip 把错误直接带入输出。下一步应测试：

1. fixed reliability gate：按 view coverage 降低 lift 权重；
2. learned gate：由 view count、geometry coverage、noise proxy、observed residual 预测 lift 权重；
3. dual head：同时输出 absolute volume 与 residual correction，再用 held-out consistency 选择或融合；
4. test-time refinement：用 operator 初始化 NeRIF，再用 BOST forward loss 修正少视角失败样本。

## 与 M3B / 4D BOST 的连接

当前 T16 是空间 inverse operator：

```text
multi-view observation + geometry -> 3D field
```

M3B 则审计 temporal low-rank prior 在 `rank x noise x views x dynamics` 工作域中的收益。它们的正确衔接顺序是：

1. 先稳定每一帧的 3D inverse operator，并输出 uncertainty / held-out residual；
2. 再把连续帧和条件元数据送入 temporal operator；
3. 用 M3B 的 rank 2/3/5/8 作为固定低秩 baseline；
4. 分别报告 field、held-out、mass trace、event amplitude 与 temporal-gradient；
5. 不用 M3B 的 temporal rank 直接指定 FNO Fourier modes。

本科可控的 4D 升级不是“直接训练完整 4D FNO”，而是比较：

- framewise 3D FNO；
- framewise FNO + rank-3 temporal projection；
- short-window temporal residual operator；
- operator initialization + NeRIF/TDBOST short refinement。

## 请何远哲师兄优先判断

1. 组里所说的算子学习更需要 projection-to-volume inverse operator，还是 time-window-to-next-volume evolution operator？
2. 少视角下应允许模型降低 physics-lift skip 权重吗，还是组内物理流程要求严格保留某个基线重建？
3. gate 的可靠度输入可否使用 view geometry、displacement residual、test camera 或标定不确定度？
4. 真数据无 field ground truth 时，held-out camera 是否能作为模型选择主指标？
5. 是否认可 operator 只负责快速初值，NeRIF/TDBOST forward optimization 负责最终物理一致性？
6. 第一份组内数据能否包含同一时刻的 train views、至少一个 test view 和相机几何？
7. 下一轮应优先做 reliability-gated 3D operator，还是直接进入短时间窗 temporal operator？

## 不能声称的内容

- 不是 NeRIF、PIV-BOST 或 TDBOST 复现；
- 没有 OERF 实验数据；
- 三个种子不足以支持强显著性措辞；
- absolute-output FNO 不是 raw-projection direct operator；
- 当前实验没有训练 4D neural operator；
- family OOD 仍失败，不能声称跨流态泛化。

## 可复跑入口

```bash
.venv/bin/python demo_t16_operator/run_ablations.py --device cpu
```

机器可读结果：

- `demo_t16_operator/results/ablations/ablation_report.json`
- `demo_t16_operator/results/ablations/ablation_runs.csv`
- `demo_t16_operator/results/ablations/ablation_summary.csv`
- `demo_t16_operator/results/ablations/ablation_paired_deltas.csv`
