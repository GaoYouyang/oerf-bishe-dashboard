# N5-D5：把合成导数证书接到真实 BOST 前向的最小接口桥

> 更新时间：2026-07-19  
> 当前判决：`SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION`  
> 正式协议提交：`ee792fdbf590c4194cdc120355e26e5b38aa2401`  
> 证据边界：只证明接口、调用账本和独立回放链工作；不证明真实 BOST、物理正确、完整导数、三维重建、算法优越、泛化或论文成立。

## 1. 这一步到底解决什么

D4c-v2 已在 explicit-matrix toy 上说明三件事：

1. `q^T(Jv) = v^T(J^Tq)` 只能检查 JVP 与 VJP 是否互为伴随，不能证明二者来自真实 forward；
2. direct residual 对自身做有限差分可以通过，却仍可能不等于 curved 减 straight；
3. diagnostic state 改变不能冒充真正影响 forward 控制流的 branch 改变。

因此下一步不能继续堆合成矩阵，也不能直接训练 DeepONet/FNO。N5-D5 把需要向实验室程序索取的内容缩成一个可执行接口：

- 4--16 条匿名 rays；
- 一个匿名 field 或 decoder parameter vector；
- 三条可区分路径 `curved`、`straight`、`direct_residual`；
- 每条路径两个 `Jv` 和一个 `J^Tq`；
- forward 自己返回的 branch state 与另行计算的 diagnostic state；
- 逐次累计的 API/primitive 成本账本。

这不是完整的 N2 数据合同。它是一个很窄的“插头标准”：师兄不需要先交整套火焰数据，只需把真实 renderer 包成 JSONL callable，便可先判断导数和路径语义是否值得进入重建。

## 2. 初学者先理解三个函数

设三维折射率场或网络参数写成向量 `x`，探测器上的观测写成向量：

```text
F_c(x) = curved-ray forward
F_s(x) = straight-ray forward
R(x)   = direct residual forward
```

如果代码语义正确，应在同一离散定义下满足：

```text
R(x) = F_c(x) - F_s(x)
J_R(x)v = J_c(x)v - J_s(x)v
J_R(x)^T q = J_c(x)^T q - J_s(x)^T q
```

- `Jv`：把一个很小的场扰动 `v` 送入 forward，问 detector 会朝哪个方向变化。它不需要显式生成巨大 Jacobian。
- `J^Tq`：把 detector 残差 `q` 反传到场空间，是梯度下降、PCGLS、NeRIF 和神经算子训练真正需要的量。
- 中心有限差分：实际调用 `F(x+hv)` 和 `F(x-hv)`，检查 `(F(x+hv)-F(x-hv))/(2h)` 是否接近接口报告的 `Jv`。

本协议固定消费所有 `h = 1e-5, 1e-4, 1e-3`，不准看完结果只挑最好看的步长。

## 3. 为什么正好是 53 次请求

每个正式 bundle 的调用预算在结果前冻结：

| 调用 | 数量 | 原因 |
|---|---:|---|
| `describe` | 2 | 检查同一进程内描述符可重复 |
| base forward repeat | 6 | 3 条路径各重复 2 次 |
| FD forward | 36 | 3 路径 × 2 tangents × 3 h × 正负两侧 |
| JVP | 6 | 3 路径 × 2 tangents |
| VJP | 3 | 3 路径 × 1 cotangent |
| **总请求** | **53** | `2 + 42 + 6 + 3` |

对一个真实实验室适配器，这个预算足够小，可以先在 4--16 rays 上运行；又足够严格，能看到接口是否偷偷切换路径、隐藏调用或漏报成本。

## 4. 正式合成参考包的结果

正式运行只使用提交后未改动的协议源码和配置。初版协议在 `a8d8849` 冻结；视觉/隐私审计发现公开结果会记录本机 Python 绝对路径，因此在任何公开结果入库前，用 `ee792fd` 增加命令脱敏与 5 条 validator 门并重新正式运行。独立 validator 不导入 runner、共享 wire helper 或 adapter；它重新生成探针、使用新 nonce 再启动 adapter，并重新计算全部指标。

| 项目 | 结果 | 解释 |
|---|---:|---|
| request / response | 53 / 53 | 完整 raw trace |
| forward / JVP / VJP | 42 / 6 / 3 | 与冻结账本一致 |
| independent checks | 1,370 | manifest、源码、公开进程记录、请求、回放、指标与结论 |
| 最大重复输出绝对差 | `0` | 合成参考可重复 |
| 最大三路径结构相对误差 | `1.2044e-15` | output/JVP/VJP 恒等式成立到浮点舍入量级 |
| 全部 h 的最大 FD 相对误差 | `2.0727e-8` | 不是只报 best-h |
| 最大伴随 normwise defect | `9.2827e-17` | 两组 tangent、三条路径 |
| 最小 bilinear signal/action | `3.7115e-2` | 本次探针不是低信号假通过 |
| actual branch change | `false` | 固定光滑合成路径 |

机器判决中的 `PASS` 只表示 synthetic protocol obligations met。所有科学授权字段仍为 `false`。

## 5. 原始事实与审计派生量分账

适配器只能报告原始事实：

- 每个 request 的 output；
- path/callable/implementation identity；
- forward 内部实际生成的 branch state；
- forward 后计算的 diagnostic state；
- 累计 runtime ledger；
- units、dtype、axis order 和 ray/component 数。

适配器不能自己宣布：

- FD 是否通过；
- adjoint 是否通过；
- structure 是否通过；
- branch 是否改变；
- 最终状态是什么；
- 能否进行重建或发表论文。

这些量全部由 runner 和独立 validator 从 raw trace 推导。这样可以避免“被测程序自己给自己打满分”。

## 6. branch state 与 diagnostic state 为什么要分开

`branch_state` 必须来自真正影响 forward 的控制流，例如：

- active ray count；
- dynamic sample count；
- occupancy pruning；
- ray termination；
- interpolation cell 或实际 hard mask 分支。

`diagnostic_state` 可以是输出幅值分箱、support 距离、frustum margin 等观察量。它们可以变化，却不一定改变离散程序。若把 diagnostic label 当 branch，有限差分会被过度拒绝；若 renderer 有 hard mask 却不报告，又可能把错误 autodiff 当成光滑导数。

## 7. 本轮专门防的假阳性

| 假阳性 | 为什么危险 | N5-D5 如何抓 |
|---|---|---|
| 同一个错矩阵生成 JVP/VJP | dot identity 仍可通过 | actual `F(x±hv)` |
| direct path 只是名字不同 | 三路径表面齐全，实际同路 | path/callable/semantic digest + output/JVP/VJP 唯一性 |
| 预存公共探针答案 | 对固定请求查表即可 | validator fresh nonce + 重建请求 + 重新启动进程 |
| 隐藏内部调用或缓存 | 速度比较与预算失真 | 每响应累计 ledger + 精确调用数 |
| wrapper 手填 branch | 标签变化不代表 forward 改变 | source-reviewed branch provenance |
| 单位、轴序或 geometry 错 | 数值门可能全过，物理量仍错 | descriptor/config 双边合同；真实阶段仍需师兄审阅 |
| 篡改结果后刷新 manifest | 只做文件哈希会被骗 | 独立回放重算 output、state、metrics、decision |

回归测试已覆盖 output、branch state、metric、decision 和 stored request 五类篡改，均 fail closed。

## 8. 现在你应该怎样读代码

按这个顺序，不要一上来啃 1,250 行 runner：

1. `data_templates/n5_d5_lab_interface.placeholder.json`：先看实验室要填哪些字段；
2. `demo_t16_operator/n5_d5_synthetic_reference_adapter.py`：只读 `_forward`、`_jvp`、`_vjp`；
3. 同文件的 `handle_request`：看 JSON 请求怎样变成一次真实调用；
4. `site_tools/run_n5_d5_minimum_interface_bridge.py` 的 `build_requests`：画出 53 次调用树；
5. 同文件的 `derive_metrics`：把公式和 raw response 对上；
6. `site_tools/validate_n5_d5_minimum_interface_bridge.py`：重点看它为什么不导入 runner；
7. 三份测试：先读篡改测试，再回头看实现。

学习验收不是“看完”。你应能独立回答：

- 为什么每条路径都要自己的 JVP/VJP？
- 为什么一个 cotangent 配两个 tangents 仍不能证明完整 Jacobian？
- 为什么 all-h FD 比 best-h 更可信？
- 为什么同一个 adapter 的独立重启仍不等于第二套物理 renderer？

## 9. 给何远哲师兄的最小请求，可直接发送

> 师兄，我已经把 D4c 的合成证书收缩成一个最小 JSONL 接口。暂时不需要整套实验数据，想先请您确认真实 NeRIF/BOST forward 能否匿名提供一个 4--16 rays 的小例子：一个 field 或 decoder parameter vector、curved/straight/direct-residual 三条 callable（若 direct 不存在也请明确）、两个 Jv、一个 J^Tq，以及实际 precision、units、axis order、sampling/interpolation/termination 规则。forward 如果有 hard mask、occupancy pruning、dynamic sample count 或 ray termination，希望能返回真正的 branch state；support/frustum 等只用于诊断的量单列。整个检查只做 42 次 forward、6 次 JVP、3 次 VJP，不公开 raw lab trace，也不会把通过写成重建或算法成功。请您优先帮我确认：真实 residual 是在同一 ray sample/integrand 层形成，还是先生成两张 detector map 再相减？

还需师兄回答四个决策问题：

1. 优先接 `field -> detector`，还是 `decoder parameters -> field -> detector`？
2. curved 与 straight 是否共享相同 sample points、interpolation query 和 aperture weights？
3. 组内实际使用 float32、float64 还是 mixed precision？可接受的 gradient-check noise floor 是多少？
4. 哪些标识、几何、单位和 trace 绝不能公开？真实 bundle 将只保存在 `private_library/`。

## 10. 收到接口后的 72 小时路线

### 第一天：只接线，不调阈值

- 复制 placeholder 到 `private_library/`；
- 填 anonymized context、units、axis order、dtype、path identity；
- adapter 只负责转换实验室 callable，不重写物理 forward；
- 跑 describe 与 base repeat，先确认 shape、determinism、cost ledger。

### 第二天：跑导数与三路径门

- 运行固定 53 请求；
- 所有 `h` 原样保存；
- 分别看 curved、straight、direct 的 FD；
- 再看 output/JVP/VJP 三路径 identity；
- actual branch 改变时 fail closed，不事后删样本。

### 第三天：由结果决定唯一下一步

- `FAIL_CONTRACT/COST/PATH_ALIAS`：修接口，不谈算法；
- `FAIL_BRANCH`：定位 hard visibility/termination，研究可微边界或局部稳定半径；
- `FAIL_STRUCTURE`：优先做 residual-native primitive；
- `FAIL_FD/ADJOINT`：检查 autodiff graph、precision、sample sharing；
- `LOW_SIGNAL_UNRESOLVED`：增加预注册 probe 或报告检测地板，不自动 pass；
- `LAB_INTERFACE_REPLAYED_NO_PHYSICS_AUTHORIZATION`：只解锁 decoder-chain 小门，仍不直接训练 FNO。

## 11. 与毕业设计和论文创新的关系

N5-D5 本身不是论文算法。它把后续创新分成可证伪的三条路：

1. **Residual-native differentiable BOST**：若真实 separate curved/straight 在 FD 中受相消污染，设计 sample-level paired residual 和混合尺度证书；
2. **Branch-aware differentiable ray tracing**：若 hard mask/termination 确实造成导数断裂，研究局部稳定半径、边界项或 reparameterization，而不是随手平滑；
3. **Certified operator reconstruction**：接口稳定后，把同一 JVP/VJP 接入 matrix-free PCGLS/unrolling，再让学习器预测受约束修正或安全预条件结构。

只有真实接口确定了哪一种物理困境实际存在，才选择论文主线。否则“新网络结构”很容易只在自造 toy 上胜出。

## 12. 一级来源阅读顺序

### 先读课题组与 BOST 主线

1. [He et al., Neural refractive index field, Physics of Fluids 37, 017143](https://doi.org/10.1063/5.0250899)：先提取测量模型、网络表示、loss、实验几何、重投影验证和计算成本；不要只看网络框图。
2. [Zheng et al., simultaneous PIV-BOST refractive-index compensation](https://doi.org/10.1007/s00348-025-04093-y)：看九视角火焰、折射率重建如何服务 PIV 误差补偿，以及论文报告的重投影检查。
3. [Grauer and Steinberg, unified BOST](https://doi.org/10.1007/s00348-020-2912-1)：理解为什么直接重建图像畸变可减少 optical-flow 中间误差和方程数。
4. [Raffel, Background-oriented schlieren techniques](https://doi.org/10.1007/s00348-015-1927-5)：补 BOS 光路、灵敏度、标定、误差源和 tomography 基础。

### 再读导数为什么可能失效

5. [Li et al., Differentiable Monte Carlo Ray Tracing through Edge Sampling](https://doi.org/10.1145/3272127.3275109)：看 visibility discontinuity 为什么需要边界采样；这是方法启发，不等于 BOST 已有相同不连续。
6. [Bangaru et al., Systematically Differentiating Parametric Discontinuities](https://doi.org/10.1145/3450626.3459775)：理解“先离散再 autodiff”为什么可能丢掉移动边界贡献。
7. [Higham, Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027)：重点读浮点误差、相消和稳定求和。
8. [Griewank and Walther, Evaluating Derivatives](https://doi.org/10.1137/1.9780898717761)：重点读 forward/reverse AD、Jacobian products 和 gradient checking。

## 13. 复现入口

```bash
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_n5_d5_synthetic_reference_adapter.py \
  site_tools/test_run_n5_d5_minimum_interface_bridge.py \
  site_tools/test_validate_n5_d5_minimum_interface_bridge.py

.venv/bin/python site_tools/run_n5_d5_minimum_interface_bridge.py \
  --config demo_t16_operator/configs/n5_d5_minimum_interface_bridge_preregistered_v1.json \
  --output /tmp/n5_d5_replay

.venv/bin/python site_tools/validate_n5_d5_minimum_interface_bridge.py \
  /tmp/n5_d5_replay
```

正式公开结果在 `demo_t16_operator/results/n5_d5_minimum_interface_bridge_synthetic_v1/`。真实实验室结果必须写入 `private_library/`，只把经过人工审核且合同允许的摘要发布到网页。
