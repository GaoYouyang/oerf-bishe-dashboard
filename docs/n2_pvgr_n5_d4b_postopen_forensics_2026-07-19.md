# N5-D4b 失败后取证：低信号伴随闭合与 support active-set

> 日期：2026-07-19  
> 性质：post-open 解释性审计，不是新预注册结果  
> 冻结历史判决：`D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`  
> 本页判决：`POSTOPEN_EXPLANATION_ONLY_NO_AUTHORIZATION`

## 先讲结论

这轮没有“把 D4b 救回来”，也没有放宽任何门槛。它回答了两个更具体的问题：

1. `p14` 的两个 dot-product failure **不是最终点积求和顺序造成的**。对保存的 `float64` JVP/VJP 数组做精确二进制有理数 contraction 后，相对缺陷仍为 `1.8416778571e-10` 与 `1.5343123146e-10`，高于冻结门 `1e-10`。
2. 6 个 topology failure 被定位成 9 个 `h=0.01` 扰动中的 **21 个 support 位翻转**；`h<=0.003` 在冻结网格上两侧都稳定。所有插值 cell hash 与 frustum sign hash 保持不变。
3. `p14` 的 curved 与 straight contraction 各约 `1.0866e-5`，两者相减后 residual signal 只有 `7.5114e-10`，缩小了约 **14,467 倍**；绝对缺陷却仍与 component map 同数量级。因此原相对指标被低信号抵消显著放大。
4. support 改变的 6 个 context 中，24/24 个 map 的 JVP/VJP/FD map gate 都通过；required-h 最大 FD 相对误差仍只有 `3.77e-7`，远低于 `1e-5`。这提示当前 support exact-match 门可能很保守，但它是看过结果后的线索，**不能据此删除历史门槛**。

最有价值的下一研究问题不是“换个阈值让它过”，而是：

> 对连续的曲光线 BOST forward，怎样设计既能识别真实不可微事件、又不会把平滑 support 等值面位移误判为程序分支改变的 fail-closed 导数证书？

这条问题与三维重建的可信伴随、NeRIF/神经场训练稳定性、后续算子学习 surrogate 的监督质量直接相关。

## 证据边界

本次工具只允许：

- 读取已经保存的 JVP、VJP、direction 与 cotangent；
- 用多种求和方式重算两侧 contraction；
- 对原来 6 个 topology-failing context 重放 field-program signature；
- 用冻结 hash 验证每次重放与原 `rows.json` 完全一致；
- 定位 support 位的 group、step、stage、ray 与 stencil offset。

明确没有调用：

- forward observable；
- 新 JVP/VJP；
- 新 FD sweep；
- decoder、三维重建、训练或真实数据。

源证据 SHA-256：

| 文件 | SHA-256 |
|---|---|
| formal `result.json` | `deebd2f53fae478c7f80fdd32acff2265f28ff2326b4aebf34056c830e502180` |
| formal `rows.json` | `5216b55a5b8f8f1f46084da4a40aeb157b18a2de9bdfec5d92086b6f7cca7111` |
| `derivative_arrays.npz` | `3b262207c2be5dd01ce2d8fa0c7a4b07a23b90e8ca49c9a50a72a37d5ff21ec3` |
| `frozen_inputs.npz` | `e40fc69356b7b86455ba51a11f094cb33b41de401df9d9fc0c146f7882e50d75` |

正式协议 commit 仍为 `cba4f285f03cc1bc791684d240d87d0c74cbd6cb`。

## 1. p14 contraction 到底坏在哪里

### 1.1 原门的定义

正式门计算：

\[
L=\langle Jv,w\rangle,
\qquad
R=\langle v,J^Tw\rangle,
\]

\[
e_{dot}=\frac{|L-R|}{\max(|L|,|R|,10^{-30})}.
\]

冻结阈值为 `e_dot <= 1e-10`。失败只出现在 `c26 d0`，即：

- cell：`wrinkled-s3163-orientation_22-wide__stress_1`；
- pair：`p14`；
- role：`n3_failure`；
- maps：`raw_curved_minus_straight`、`paired_neumaier_residual`。

### 1.2 多种重算都没有过门

| map | formal `torch.sum` | `math.fsum` | 精确二进制有理数 | 冻结门 |
|---|---:|---:|---:|---:|
| raw residual | `1.8416848775e-10` | `1.8416683590e-10` | `1.8416778571e-10` | `1e-10` |
| paired residual | `1.5343097539e-10` | `1.5343028712e-10` | `1.5343123146e-10` | `1e-10` |

这台 Mac 上 `np.longdouble` 的 mantissa 与 `float64` 同为 52 bits，因此 `longdouble` 不能提供更高精度。精确有理数方法直接把每个已保存的二进制浮点数转成 Fraction，再精确做乘法与求和；它仍不过门，所以“最后一次 reduction 顺序”不是充分解释。

### 1.3 真正明显的是 residual signal cancellation

| map | dot signal | 绝对缺陷 | 原相对缺陷 | normwise adjoint defect |
|---|---:|---:|---:|---:|
| curved detector | `1.08658e-5` | `2.19314e-19` | `2.01839e-14` | `1.85695e-16` |
| straight detector | `1.08665e-5` | `5.01666e-21` | `4.61661e-16` | `4.24722e-18` |
| raw residual | `7.51143e-10` | `1.38336e-19` | `1.84168e-10` | `5.20528e-13` |
| paired residual | `7.51143e-10` | `1.15249e-19` | `1.53431e-10` | `4.33655e-13` |

这里的 normwise 指标只作描述：

\[
e_{norm}=
\frac{|\langle Jv,w\rangle-\langle v,J^Tw\rangle|}
{\|Jv\|_2\|w\|_2+\|v\|_2\|J^Tw\|_2}.
\]

component dot signal 与 raw residual dot signal 的比值为 `14,466.67`，而 raw 的绝对缺陷仍是 curved map 绝对缺陷的约 `0.631`。因此：

- 这是一个强烈的低信号条件数线索；
- 不能简单说 AD 错了，也不能简单说它已经对了；
- 原 gate 对一般 map 合理，但对“两个近等量之差”的 residual map 会把很小的绝对不闭合放大；
- 新协议若要采用 mixed absolute/relative 或 normwise gate，必须先在独立 development population 上定规则，再对 fresh population 一次性检验，不能拿 `p14` 反向定阈值。

## 2. 21 个 support 位到底翻在哪里

### 2.1 support 位的严格语义

每个位是：

\[
b=\mathbf{1}(|f(x_{query})|\ge \tau_{support}).
\]

它按固定顺序记录：

- 16 个 RK4 step 的 `k1...k4`；
- curved path midpoints；
- straight path midpoints；
- 每个点的 `base,+x,+y,+z,-x,-y,-z` 七个 central-stencil query。

每个 signature 共 2,688 个 support bits。注意：当前 renderer 的 forward 是连续 smoothstep 场，`support_threshold` 主要用于安全证书/拓扑诊断，并不是 forward 中把 field 清零的 active mask。因而这里更准确的称呼是“协议定义的 support-set signature”，不能直接写成真实 forward 的 `if` 分支翻转。

### 2.2 逐 context 结果

| cell / direction | role | side at `h=0.01` | flips | `0→1` | `1→0` | 主要位置 |
|---|---|---|---:|---:|---:|---|
| `c06 d0` | N3 failure | plus | 4 | 4 | 0 | ray 0, step 14 k2/k3, curved/straight midpoint, `-y` |
| `c08 d0` | N3 failure | minus | 1 | 1 | 0 | ray 0, straight midpoint, `-y` |
| `c10 d0` | N3 failure | plus | 3 | 0 | 3 | ray 0, step 14 k2/k3 与 curved midpoint, `-x` |
| `c10 d0` | N3 failure | minus | 1 | 1 | 0 | ray 0, straight midpoint, `-y` |
| `c17 d0` | matched control | plus | 2 | 2 | 0 | ray 2, step 0 k4 / step 1 k1, `-y` |
| `c17 d0` | matched control | minus | 4 | 0 | 4 | ray 2, step 0 k4 / step 1 k1, `+z/-x` |
| `c19 d0` | matched control | plus | 2 | 0 | 2 | ray 2, step 0 k4 / step 1 k1, `-x` |
| `c19 d0` | matched control | minus | 2 | 2 | 0 | ray 2, step 0 k4 / step 1 k1, `-y` |
| `c19 d1` | matched control | plus | 2 | 2 | 0 | ray 2, step 0 k4 / step 1 k1, `-y` |

聚合后：

- 21 flips：12 个 `0→1`，9 个 `1→0`；
- 16/21 位于 RK4 stage，5/21 位于 midpoint；
- 12 个来自 matched controls，9 个来自 N3 failures；
- 只涉及 ray 0 与 ray 2；
- offset 主要为 `-y` 12 次、`-x` 7 次、`+z` 2 次；
- base support margin 的绝对值为 `1.05e-4...2.88e-4`，candidate 为 `2.51e-5...1.49e-4`，确实靠近阈值，但没有正当理由把它们事后删掉。

冻结 h-grid 只支持以下区间陈述：

\[
h_{stable}\ge 0.003,
\qquad
h_{change}\in(0.003,0.01].
\]

它不支持把某个值称为“精确临界半径”，更不支持称为真实火焰界面的物理跃迁。

## 3. support change 与 FD error 的关系

对每个 direction context，取四个 map 中最大的 required-h FD relative error：

| group | n | median | P90 | max |
|---|---:|---:|---:|---:|
| topology stable | 58 | `6.58e-8` | `1.70e-7` | `5.25e-7` |
| topology changed | 6 | `8.90e-8` | `2.68e-7` | `3.77e-7` |

可以诚实说：changed 组的中位数略高，但样本只有 6 个；两组最大值都低于 `1e-6`，更远低于 frozen required-h ceiling `1e-5`。而且 changed 组的 24/24 map gate 全部通过。

这给出一个可检验假设：

> 对当前连续 smoothstep renderer，exact support-bit equality 可能比局部可微性实际需要的条件更强；允许简单、横截的 support 等值面平滑位移，可能仍能保持可靠 JVP/VJP 与 FD 一致性。

但这是 post-open 假设，不是 D4b 改判依据。

## 4. 值得发展的算法方向

### 候选 A：mixed-scale adjoint certificate

目标不是换阈值，而是建立有误差模型的伴随闭合证书：

1. 保留原 dot-signal relative defect，暴露低信号问题；
2. 同时报 normwise bilinear defect；
3. 报 component-to-residual cancellation factor；
4. 给出基于浮点 backward error 与运算规模的 absolute envelope；
5. 在 fresh directions、fresh cotangents、fresh fields 上冻结双门或三门规则；
6. 任一非有限、结构关系破坏、FD 不一致仍 fail-closed。

论文价值只有在规则能预注册、跨 field/rig 泛化，并能预测真实 optimizer 中的梯度可靠性时成立。只让 `p14` 过门没有论文价值。

### 候选 B：transversality-aware support certificate

把 level set 写成：

\[
\phi(s,\epsilon)=|f_\epsilon(r_\epsilon(s))|-\tau.
\]

若 crossing 是 simple root，满足：

\[
|\partial_s\phi|\ge m>0,
\]

则隐函数定理允许 crossing 随场扰动平滑移动：

\[
\frac{ds_*}{d\epsilon}=-\frac{\partial_\epsilon\phi}{\partial_s\phi}.
\]

新的证书不要求所有采样 bit 完全不变，而要证明：

- crossing 数量与顺序不变；
- 每个 crossing 都是唯一、非 grazing 的 simple root；
- root interval 不跨越危险 cell/domain/frustum 事件；
- 若 interval Newton / Bernstein bound 无法证明，就 fail-closed；
- 若实验室真实 renderer 根本没有 mask/occupancy，support 只保留为解释性证书，不参与 forward derivative authorization。

这是更贴近物理的方向：反应前沿/密度界面可以平滑移动，不能因为采样点恰好跨过人为阈值就自动等同于 forward 不可微。

### 候选 C：证书进入三维重建与算子学习

最小闭环应当是：

1. 用可信 JVP/VJP 做 3D field reconstruction；
2. 每次 optimizer step 输出 gradient certificate 与 fail-closed reason；
3. 用可信 forward/adjoint 生成 operator-learning 训练对；
4. DeepONet/FNO/FFNO 只作 baseline；新模型可学习 residual/update operator，但必须继承物理 forward、view geometry 与不确定度；
5. 比较 field relative-L2、held-out-view reprojection、逐 rig tail、Schur/adjoint violation、A/A^T 调用与端到端成本；
6. 证书拒绝率与重建质量要一起报，不能只展示平均 PSNR。

这一阶段仍需要真实数据合同，不能从当前 synthetic derivative core 直接跳到“优于 FNO”。

## 5. 下一份预注册应怎样写

建议先做 `N5-D4c` development，再决定是否开 fresh audit：

1. **冻结目标**：区分 low-signal conditioning、真实 adjoint inconsistency、support simple-root motion 与真正拓扑事件。
2. **开发集**：与 D4b 32 cells 完全分开；D4b 只用于形成假设。
3. **方向设计**：每 cell 多个独立 `v,w`，覆盖高/低 dot signal，而不是只看两条方向。
4. **指标**：原 `e_dot`、normwise defect、absolute defect envelope、FD curve、component cancellation、support transversality margin。
5. **消融**：exact-bit gate、simple-root certificate、仅 cell/frustum gate；比较 false-safe 与 false-reject。
6. **fresh gate**：先冻结规则和阈值，再开未见 field/rig；不得从 fresh 结果反调。
7. **授权边界**：即使 D4c 通过，也只授权 derivative interface；decoder、3D reconstruction、real data、generalization 与 paper claim 仍需独立门。

## 6. 现在最需要问何远哲的 8 个问题

1. 真实 NeRIF/BOST renderer 是否存在随网络参数更新的 hard mask、occupancy pruning、ray termination 或 threshold branch？
2. support threshold 在真实代码里会改变 forward 值，还是只用于可视化/ROI/证书？
3. 优化目标主要用 raw curved-minus-straight residual，还是直接拟合 detector displacement？两者的典型数值尺度是多少？
4. 现有训练是否见过梯度爆炸、梯度消失、loss 抖动或 adjoint/FD 不一致？发生在哪些 view/field 区域？
5. 真实 field parameterization 是 voxel、MLP/NeRIF、tensor factorization 还是混合？网络参数到 RI field 的 decoder chain 能否单独做 JVP/VJP？
6. 一次真实 forward 与 backward 的显存、时间、ray count、view count、voxel/采样分辨率是多少？
7. 能否提供 1 个最小匿名 case：相机参数、1 个 field checkpoint、4 条 rays、forward output、loss cotangent，用于先闭合伴随接口？
8. 论文更看重“可信可微 renderer/证书”，还是“新 operator architecture/加速”？这决定先做候选 A/B，还是尽快接候选 C。

## 7. 可复现入口

```bash
cd /path/to/oerf-bishe-dashboard
.venv/bin/python site_tools/n5_d4b_postopen_forensics.py \
  --output /tmp/n5_d4b_postopen_forensics_replay
.venv/bin/python -m pytest -q \
  site_tools/test_n5_d4b_postopen_forensics.py
```

默认正式 output 是不可覆盖的；复现必须指定新目录。工具会验证所有冻结源文件在运行前后 hash 不变。

结果入口：

- `demo_t16_operator/results/n2_pvgr_n5_d4b_postopen_forensics_v1/result.json`
- `demo_t16_operator/results/n2_pvgr_n5_d4b_postopen_forensics_v1/n2_pvgr_n5_d4b_postopen_forensics.png`
- `site_tools/n5_d4b_postopen_forensics.py`
- `site_tools/test_n5_d4b_postopen_forensics.py`

## 最终边界

本轮已经把“不知道为什么失败”推进为两个明确机制：

- residual map 的低信号 cancellation；
- protocol-defined support set 在较大 `h` 上的有限位翻转。

它仍然不授权：

- D4b 改判；
- 32-cell derivative contract；
- decoder chain；
- 三维重建；
- 算子学习训练或优于 DeepONet/FNO/FFNO；
- 真实数据、泛化或论文结论。

下一步应把这两个机制变成**提前冻结、能在 fresh 数据上被证伪的证书设计**，而不是继续堆更多 post-open 解释。
