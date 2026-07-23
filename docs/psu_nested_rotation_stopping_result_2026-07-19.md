# E72 PSU 嵌套旋转停止：正式 NO-GO、camera-tail 困境与下一算法门

## 一句话判决

**只用两个 rotation 之间的双向迁移选出的 early-stopping checkpoint，没有在第三个 rotation 上同时保护平均、最差相机与尾部指标。`16³` 和 `32³` 都是 `NO_GO`。**

机器状态：

```text
POST_HOC_PROGRAMMATICALLY_WITHHELD_CROSS_FIT_SCREEN_NO_GO_NO_FIELD_TRUTH
```

这不是算法胜利，也不是 field truth 失败。它是一个严格的反例：**inner projection transfer 上看似安全的停止步，不会自动外推成第三几何的 camera-tail 安全。**

## 1. 为什么先冻结

E72 在任何新单 rotation 轨迹开始前，就固定了：

- rotations `0° / 50° / 90°`；
- grids `16³ / 32³`；
- CGLS checkpoints `{0,1,2,3,4,6,8,12}`；
- primary baseline `k=4`；
- inner 双向角色、指标、容差、tie-break 和 fallback；
- outer 的 macro-improvement 与每个 group/camera/tail no-harm 门；
- 两级 cache hash、运行环境、逻辑调用预算和公开输出边界。

这一步的意义是：看完 outer 后不能临时换指标、改阈值或挑 checkpoint，再把同一份数据称为新验证。

### 冻结后的预检修订

初始提交 `5d47a80bef7091611dd4e424f8eb1f26b0ccb50c` 在正式 inner 启动时，被逐视角门拦下：`50°` 三台相机的 ray count 转录错了。错误三数总和恰好不变，所以只查总数会漏掉。两份 cache manifest 与 E71 均给出：

| view | rotation / camera | 正确 rays |
|---:|---|---:|
| 3 | `50° / C2` | 988,006 |
| 4 | `50° / C3` | 1,363,792 |
| 5 | `50° / C4` | 1,311,324 |

当时 shared adjoint preflight、CGLS 和 selection 都还没有开始；轨迹数为 `0`、已选 checkpoint 数为 `0`。修订只改三个元数据整数，所有数组、hash、指标和判决规则不变。

最终 protocol commit：

```text
78eb2d5f5a31b6bca68652828ca194900b4326ff
```

## 2. 六条唯一轨迹

唯一拟合键为：

```text
(train rotation, grid, cache-manifest hash, solver-config hash)
```

外层 fold 不能进入拟合键，因此不重复运行同一条轨迹：

```text
3 train rotations x 2 grids = 6 unique CGLS trajectories
```

每条轨迹只用一个 rotation 的三台相机拟合，先固定八个场快照，再加载非训练 rotation 观测并做完整九视角 forward。

| 证据 | 实测值 |
|---|---:|
| 私有 float64 checkpoints | 48 |
| 完整 artifact `A` 调用 | 122 |
| 完整 artifact `A^T` 调用 | 86 |
| outer join `A / A^T` | `0 / 0` |
| 总 wall time | 1,104.606 s |
| 进程最大 RSS | 7.110 GB |
| 中断 `.building` 附加调用上界 | `0 / 0` |
| 数值、隔离、调用、内存和隐私门 | 15 / 15 pass |

## 3. inner 真正选了什么

对一个 outer rotation，inner 只使用另外两个 rotation 的双向迁移。例如 outer `50°` 时：

```text
train 0°  -> validate 90°
train 90° -> validate 0°
```

一个非 `k=4` 候选必须在两个方向同时满足：

1. pooled/macro/worst relative-L2 no-harm；
2. group p95 no-harm；
3. 六个 target-camera relative-L2 和 p95 全部 no-harm；
4. 三项主风险的最大 ratio 严格小于 `1 - 1e-8`。

密封结果：

| outer | grid | selected | inner 最坏风险 / k4 | inner 平均 macro / k4 | 状态 |
|---:|---:|---:|---:|---:|---|
| 0° | 16³ | `k=4` | 1.000000 | 1.000000 | 无严格安全改善，回退 |
| 0° | 32³ | `k=4` | 1.000000 | 1.000000 | 无严格安全改善，回退 |
| 50° | 16³ | `k=3` | 0.998016749 | 0.996808334 | 严格安全改善 |
| 50° | 32³ | `k=2` | 0.998748518 | 0.998239369 | 严格安全改善 |
| 90° | 16³ | `k=4` | 1.000000 | 1.000000 | 无严格安全改善，回退 |
| 90° | 32³ | `k=4` | 1.000000 | 1.000000 | 无严格安全改善，回退 |

selection commit：

```text
08cb57ff282ac5ed81ad35fa1d8b3ca9de2cd583
```

selection SHA-256：

```text
bbce7570ab7c7be5946b2c25102985c03833bec4442d2f003e94c898a12e12ea
```

## 4. outer 为什么两个网格都失败

outer `0° / 90°` 都回退到 `k=4`，因此只有 outer `50°` 真正比较了新的 stopping decision。

| grid | outer | selected | macro `k4 - selected` | outer no-harm | 失败项 |
|---:|---:|---:|---:|---|---|
| 16³ | 0° | 4 | 0 | pass | 无，因为完全回退 |
| 16³ | 50° | 3 | +0.001253899 | **fail** | group p95；view 5 L2/p95 |
| 16³ | 90° | 4 | 0 | pass | 无，因为完全回退 |
| 32³ | 0° | 4 | 0 | pass | 无，因为完全回退 |
| 32³ | 50° | 2 | -0.006310027 | **fail** | pooled/macro/group p95；view 5 L2/p95 |
| 32³ | 90° | 4 | 0 | pass | 无，因为完全回退 |

两个 grid-level 判决：

| grid | 三 outer 等权 mean macro improvement | mean 门 | 所有 no-harm | 最终 |
|---:|---:|---|---|---|
| 16³ | +0.000417966 | pass | **fail** | `NO_GO` |
| 32³ | -0.002103342 | **fail** | **fail** | `NO_GO` |

### 16³：平均赢了，尾部仍然失败

outer `50°` 中，`k=3` 的 camera-macro 从 `0.961193995` 改善到 `0.959940096`。但：

| 指标 | k4 | selected k3 | 变化 |
|---|---:|---:|---:|
| group p95 | 2.088186845 | 2.089018184 | +0.0398% |
| view 5 relative-L2 | 0.893601874 | 0.908697823 | +1.6893% |
| view 5 p95 | 1.949612386 | 1.992301877 | +2.1896% |

所以不能用一个平均改善遮住单相机尾部伤害。

### 32³：平均和尾部同时失败

outer `50°` 中，`k=2` 的 camera-macro 从 `0.978850638` 退化到 `0.985160664`，同时：

- group p95 增加 `1.1220%`；
- view 5 relative-L2 增加 `3.0282%`；
- view 5 p95 增加 `3.8310%`。

## 5. 这个失败的物理含义

目前只能确定：

1. inner 的 `0° <-> 90°` 双向 projection transfer 没有预测好 outer `50°` 的 view 5/tail；
2. CGLS 的最佳可观测停止步依赖 rotation/camera geometry；
3. 一个 pooled 或 macro 标量不足以保护真实多相机 BOST 的尾部。

一个合理但尚未证明的推断是：view 5 对 support/null-space、有限光阑、投影覆盖或局部大密度梯度更敏感。它也可能包含更大的几何或位移提取误差。当前没有独立重复、噪声标定或 field truth，不能在这几个原因之间做强归因。

## 6. 经典对照说明了什么

| grid / outer | train residual minimum | sparse L-curve | outer oracle（不可部署） | sealed selection |
|---|---:|---:|---:|---:|
| 16³ / 0° | 12 | 3 | 1 | 4 |
| 16³ / 50° | 12 | 3 | 3 | 3 |
| 16³ / 90° | 12 | 3 | 3 | 4 |
| 32³ / 0° | 12 | 6 | 1 | 4 |
| 32³ / 50° | 12 | 4 | 3 | 2 |
| 32³ / 90° | 12 | 4 | 3 | 4 |

最小 train residual 在六个单元都选到最晚 `k=12`，正好展示了为什么训练一致性不能单独定义停止。sparse L-curve 和 outer oracle 只是诊断；当前没有有效 influence trace、grouped noise model 或独立噪声尺度，因此 GCV 与 discrepancy principle 继续禁用，不能假装比较过。

## 7. 对毕设选题的直接结论

### 现在关闭

- 把“nested rotation-aware stopping 已经成功”作为论文主贡献；
- 在同一 E71 outer 上继续调 threshold、指标或 checkpoint，再称为新验证；
- 用 macro 平均代替逐 camera/tail 安全门；
- 直接训一个 FNO/DeepONet 就宣称解决该困境。

### 现在保留

- E71 六条半收敛轨迹作为“停止确实是问题”的真实证据；
- E72 作为双向 rotation-transfer 无法保护第三几何尾部的反例；
- fixed `k=4`、train residual minimum、sparse L-curve、outer oracle 作为后续方法的固定对照；
- 逐 rotation、逐 camera、worst 与 tail no-harm 作为不可降低的验收门。

## 8. 下一代算法候选

### E73-0：先补强经典基线（必做）

在新的仿真 flow family 和后续真实 flow 上并列：

- H1/Tikhonov regularization path；
- TV/primal-dual reconstruction；
- 真正的 pyramid BOST：同步改变 field、background image、projection data 和 projection matrix；
- 有独立噪声尺度后的 discrepancy stopping。

如果一个强经典方法已能保护 camera 5/tail，就不应该把网络包装成必要创新。

### E73-A：GroupTail-CGLS（Mac 上可先做）

把组定义为：

```text
(flow instance, source rotation, target rotation, camera, residual-tail bin)
```

不再最小化平均 macro，而是在可校准的多 flow 数据上最小化 worst-group/CVaR 风险，近似并列时选更早 `k`。它的创新不能只是把损失函数换成 `max`；必须结合 BOST 的 camera geometry/support 与 fail-closed fallback。

最小判伪：新 flow 整组留出时，必须同时优于 fixed k4、L-curve、H1/TV 与 pyramid baseline，并且不伤害任何预声明 camera/tail。

### E73-B：Geo-Calibrated Tail Envelope（更贴近新算法）

用小模型或可解释回归预测“未见 camera 尾部误差上界”，输入只使用部署时可观测量：

- rotation/camera pose 与 ray coverage；
- finite-aperture 参数；
- 早期 residual curve 的斜率、曲率和分位数；
- support/null-space 的低成本谱摘要；
- 网格尺度和观测信号尺度。

上界不足以证明 no-harm 时，必须回退到 fixed/classical safe baseline。这一路线的主贡献可能是“geometry-conditioned risk certificate + abstention”，而不是又一个黑箱 `k` 分类器。

### E73-C：受证书约束的神经 correction（后做）

只在 E73-0/A/B 建立了可用风险合同后，再让 MgNO/CNO/FNO/DeepONet 类模型学 coarse reconstruction 上的 correction。网络输出先经物理 forward/data-consistency projection，再经 geometry-tail certificate；不通过就 abstain/fallback。

这才是“三维重建 + 算子学习”的结合点：网络不取代整个逆问题，只在经典解法无法安全表达的部分学可拒答 correction。

## 9. 下一步必须向何远哲师兄确认

1. 能否提供多个独立 flow condition/time block，而不是同一场的三个 rotation？
2. 是否有 CFD/合成 refractive-index field 与对应真实 camera geometry，用于 field relative-L2 的第一层判伪？
3. view 5 在真实系统中的 pose、aperture、可见体积和位移提取质量是否已有定量诊断？
4. 组内是否已有 H1/TV/pyramid BOST 的代码和标准参数，避免我做弱基线？
5. 后续论文更希望主贡献是 reconstruction quality、计算成本、camera-geometry robustness，还是 uncertainty/abstention？
6. 哪些数据可以公开、哪些只能本地评分？是否可以公开匿名聚合 metric 和预注册 protocol？

## 10. 与当前失败直接相关的一级来源

1. [He et al., NeRIF, Physics of Fluids 2025](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)：连续折射率场、BOST forward 和神经隐式表示的师兄主线。它说明“网络表示场”有直接价值，不证明 E72 的 cross-rotation stopping 或 camera-tail 泛化。
2. [Hu et al., A pyramid approach for BOST, Experiments in Fluids 2026](https://link.springer.com/article/10.1007/s00348-025-04153-3)：粗到细时同步 up/downsample field 与 background image，并逐层修正 projection data/matrix。因此它是 E73-0 必须比的强基线，普通 `resize(x16)` 不等价。
3. [Hucker & Reiß, Early stopping for conjugate gradients, Numerische Mathematik 2025](https://link.springer.com/article/10.1007/s00211-025-01469-4)：证明 early stopping 可作为 CG 逆问题正则化，并区分 prediction/reconstruction risk。其理论需要特定统计模型与噪声条件，不会自动覆盖 BOST 几何失配。
4. [Cai et al., Direct BOST using RBFs, Optics Express 2022](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100)：直接重建折射率并使用独立相机 ray-traced reprojection 验证，支持把留出 camera 作为独立评分面。重投影仍不是 field truth。
5. [Hu & Lei, Early stopping for iterative regularization, JMLR 2022](https://www.jmlr.org/beta/papers/v23/21-0983.html)：把 cross-validation stopping 与 Tikhonov path 联系起来；其 RKHS/凸损失结论不能保证第三 rotation 的 camera 5 tail。
6. [Sagawa et al., Distributionally Robust Neural Networks, ICLR 2020](https://iclr.cc/virtual/2020/poster/1491)：group DRO 在过参数网络中仍需更强正则化或 early stopping 才可以改善 worst-group generalization。它为 E73-A 提供损失视角，不是 BOST 物理保证。

## 11. 证据等级与不得越界

可以说：

> 在一个真实 PSU 流场的事后、程序化外层隔离 cross-fit 再分析中，双 rotation 之间的 early-stopping 选择未能在 `16³` 和 `32³` 上同时满足相对 fixed `k=4` 的留出投影 no-harm 条件。

不可以说：

- 这是人类未见的 final test；E71 outer 在 E72 设计前已存在并被看过；
- 某个 checkpoint 的三维场更接近真值；当前无 field truth；
- 有统计显著性、独立重复或跨 flow 泛化；
- nested stopping 、H1/TV、pyramid 或任何神经算子已经优越；
- camera 5 的原因已被确定为噪声、几何或 null space 之一。

## 12. 复现与文件

公开包独立验证：

```bash
.venv/bin/python -m site_tools.validate_psu_nested_rotation_stopping_public
```

图表重绘：

```bash
.venv/bin/python -m site_tools.plot_psu_nested_rotation_stopping
```

聚焦回归：

```bash
.venv/bin/python -m pytest -q \
  site_tools/test_validate_psu_nested_rotation_stopping_public.py \
  site_tools/test_plot_psu_nested_rotation_stopping.py \
  site_tools/test_psu_nested_rotation_selector.py \
  site_tools/test_nested_rotation_stopping_runner.py
```

文件入口：

- [冻结协议](psu_nested_rotation_stopping_protocol_2026-07-19.md)
- [inner 公开 summary](../demo_t16_operator/results/psu_nested_rotation_stopping_inner_public_v1/summary.json)
- [密封 selection](../demo_t16_operator/results/psu_nested_rotation_stopping_inner_public_v1/sealed_selection.json)
- [outer NO-GO summary](../demo_t16_operator/results/psu_nested_rotation_stopping_outer_public_v1/summary.json)
- [独立 validator](../site_tools/validate_psu_nested_rotation_stopping_public.py)
- [独立验证报告](psu_nested_rotation_stopping_validation_report_2026-07-19.json)
- [四联图 PNG](../assets/figures/psu_nested_rotation_stopping_result.png)
- [可缩放 SVG](../assets/figures/psu_nested_rotation_stopping_result.svg)
- [绘图脚本](../site_tools/plot_psu_nested_rotation_stopping.py)

私有 cache、逐射线观测、48 个场 checkpoint 与 private report 仅保留在本机 `private_library`，不进入 Git。
