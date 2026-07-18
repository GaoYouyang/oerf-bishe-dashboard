# N2-PVGR N4 曲线射线评估器收敛审计预注册

日期：2026-07-18
状态：只用于评估器审计；不授权算法、三维重建、真实数据、泛化或论文结论

## 1. 为什么必须先做 N4

N3 的 96 个物理条件已经完整运行，但 16 个 reference sentinel 没有通过：

- 10 个 smooth 条件的 H256 到 H512 曲线减直线匹配残差相对差异超过 `1e-2`；
- 6 个 wrinkled 条件的 H256 到 H512 曲线输出相对差异略高于 `1e-4`。

这些失败意味着高精度曲线射线结果本身尚未在相关条件上证明收敛。此时直接训练或比较 DeepONet、FNO、FFNO 或新模型，会把数值积分误差混入标签误差，无法区分“模型没学会”和“监督标签没收敛”。N4 因而只审计评估器，不测试任何候选算法。

## 2. 冻结样本

N4 使用 N3 全部且仅有的 16 个 sentinel 失败格，并为每格配置一个 matched control，共 32 格。每个对照满足：

1. 同一解析场族与同一 field seed；
2. 同一应力倍率；
3. 仅改变 orientation 或 aperture 中的一个因子；
4. 对照格在 N3 的两个 sentinel gate 均通过。

完整的 16 对 ID 在配置文件 `audit_pairs` 中逐项冻结。该选择是在看过 N3 后做出的机理诊断，不是独立测试集，也不得用于估计泛化误差。

## 3. 冻结数值路线

所有格均使用 CPU、float64、同一 field unit 的 256 条 common Sobol rays，并计算：

- H256：RK4 256 步曲线射线和同分辨率直线路径；
- H512：同上；
- H1024：同上；
- H2048：只在任一 H1024 预注册门失败时运行。

每级同时保存：

- detector-plane 曲线输出；
- 同分辨率 `curved - straight` 匹配残差；
- finite-ray fraction；
- domain/stencil 最小余量；
- 方向单位范数误差；
- 每条射线的 support crossing 与 frustum violation signature；
- wall time；
- 曲线、直线和总 logical scalar-grid point queries。

H 路线的 logical point-query 账本为每射线每步 35 次，直线路线为 7 次，合计 `42 * 256 * H`。wall time 只描述本机实现成本，不当作可移植复杂度结论。

## 4. 冻结 H1024 门

令 `d(a,b)` 为以较高 H 输出为分母的 relative-L2。H1024 必须同时满足：

- `d(high512, high1024) <= 2.5e-5`；
- `d(residual512, residual1024) <= 2.5e-3`；
- 两项相邻差异相对上一层差异的 contraction ratio 均不大于 `0.5`；
- 若上一层差异低于 `1e-12` 而不可评分，则当前差异也必须不大于 `1e-12`；
- finite-ray fraction 为 `1.0`；
- H1024 的 domain/stencil margin 不小于 `0`；
- 最大方向范数误差不大于 `1e-12`；
- H512 与 H1024 的 support-crossing signature 完全一致；
- H512 与 H1024 的 frustum-violation signature 完全一致，且 H1024 violation 数为 `0`；
- 重新计算的 H256-H512 两个指标与已冻结 N3 CSV 的绝对差不大于 `1e-12`。

`2.5e-5` 与 `2.5e-3` 分别是 N3 上限的四分之一。contraction gate 额外防止仅靠一个宽松绝对阈值把平台误差误认成收敛。这里不声称四阶渐近区，因为插值、路径积分和 support topology 可能限制观测阶。

## 5. 冻结 H2048 升级门

H2048 只能在 H1024 至少一门失败后运行。升级格必须同时满足：

- `d(high1024, high2048) <= 1.25e-5`；
- `d(residual1024, residual2048) <= 1.25e-3`；
- 两项 contraction ratio 相对 H512-H1024 均不大于 `0.5`；
- 与 H1024-H2048 端点对应的 finite、domain、stencil、direction 和 topology 门全部通过。

如果 H1024 全部通过，则获得统一 H1024 reference。若部分格升级且 H2048 全部通过，则只获得逐格 H1024/H2048 mixed reference。任一升级格仍失败，则受影响的 forward evaluator 继续保持未授权。

## 6. 机器判定与后续权限

只有 32 格都有有效的预注册 reference 时，机器才输出：

`EVALUATOR_CONVERGENCE_CLEARED_FOR_TINY_FIELD_JVP_VJP_GATE`

它只允许下一步运行一个小规模、可逐项核验的 field-JVP/VJP reconstruction gate。否则输出：

`FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED`

无论哪种结果，本实验都不授权：

- 把合成评估器收敛写成新算法成功；
- 声称优于 DeepONet、FNO、FFNO 或任何其他模型；
- 声称完成三维 BOST 重建；
- 声称真实实验、跨 rig、跨场族或跨噪声泛化；
- 打开 N3 保留的 `oblique_compression_sheet` 或 `shock_expansion_pair`。

## 7. 完整性与可恢复性

正式运行前，配置、runner、测试、校验器、图表逻辑、停止规则以及父级 N3 证据均须进入 protocol commit，并生成第二次 commit 的 attestation。attestation 生成时 formal output 和 formal work output 必须均不存在。

每个 `cell x H` 检查点独立原子写入，并保存：

- 预注册配置 SHA-256；
- 冻结 cell/H metadata；
- 输出和匹配残差的形状、有限性与数组 SHA-256。

恢复时任一项不一致均拒绝复用。最终目录先在 staging 中完整生成，再原子移动到 formal output。正式结果产生后，独立 validator 必须复算计数、门判定、CSV 行数和 manifest 哈希。

## 8. 预期解释，不是预注册结论

如果 smooth 的匹配残差随着 H 明显收缩，说明 N3 失败主要来自“残差分母小、绝对输出已较稳”的离散误差；如果 wrinkled 的输出差异不收缩或 topology signature 改变，则说明解析 wrinkled interface 与离散路径采样共同造成 evaluator floor。后者应先修正或限制监督标签定义，不能靠神经网络掩盖。
