# N2-PVGR N5 cancellation-aware reference 路线草案

状态：研究设计草案；尚未预注册，尚未产生 N5 结果

## 研究问题

N4.1 的完整 output 在 32/32 格收敛，但两个 narrow-aperture controls 的 `curved - straight` 小残差 relative-L2 在 H2048 仍高于门。N5 不问“能不能调一个更松阈值”，而问：

1. 这是积分阶数尚未进入稳定渐近区，还是两个近似量相减造成的 reference floor？
2. 能否直接积分 paired residual，减少不必要的独立离散误差？
3. 该残差在真实 detector noise 下是否可辨认、值得成为 neural residual target？

## 四个候选 reference，不含神经网络

| ID | 方法 | 作用 | 主要风险 |
|---|---|---|---|
| R0 | raw `high_H - straight_H` | N4.1 原始基线 | 小量由两个独立积分相减 |
| R1 | H4096/H8192 raw refinement | 判断是否继续渐近收缩 | CPU 成本线性增加；仍保留相减结构 |
| R2 | direct paired-integrand quadrature | 在共享节点直接积曲/直 integrand 差 | 需要严格证明与原 observable 等价 |
| R3 | Richardson + compensated summation | 估计高阶极限与浮点敏感性 | 只有观测阶稳定时才合法，不得当真值 |

## 最小开发顺序

### D0：两失败格的尺度账本

交付：每 H 的 full output norm、residual norm、adjacent absolute difference、relative difference、Q95、domain/topology 与 wall/query cost。N4.1 已完成主体。

### D1：直接 paired residual 单元测试

从恒定场、线性折射率场和弱 Gaussian plume 开始。要求：零折射率梯度返回 machine zero；弱场极限与 finite difference 一致；反向/旋转对称性误差有明确上限。

### D2：H4096/H8192 mechanism probe

只开两个 failures 和两个 matched wide controls。先固定输出表和图，再运行。它是 post-N4 selected diagnosis，不产生泛化结论。

### D3：新鲜 reference gate

若 D1/D2 找到稳定参数化，另开未使用 field seeds 和 geometry cells，预注册后一次性验证。这里才决定哪种 reference 可进入 field-JVP/VJP。

### D4：实验 noise-floor gate

将 synthetic detector units 映射到实验 pixel/displacement units。用 flow-off repeats 估计每 ray/camera covariance，不用单个全局噪声标量掩盖相机差异。

## Go/No-Go

Go 进入 tiny field-JVP/VJP：

- 新 reference 在 fresh cells 的 full-output 与 matched-residual 门均通过；
- dot test 与多步长 centered FD 都闭合；
- H-P1 target 的能量与尾部稳定高于 evaluator floor；
- 至少一部分 target 高于真实 flow-off noise floor；
- reference 增加的端到端成本有完整查询/时间账本。

No-Go residual operator：

- H-P1 大多低于 evaluator 或实验 noise floor；
- paired quadrature 只在已见两格有效；
- reference 改善来自改变 observable 定义；
- field VJP 不能闭合；
- fallback 后成本接近 full high evaluator，失去学习意义。

## 与自有算法的连接

若 N5 通过，候选不是“再做一个 FNO”，而是：

`Picard-1 physical state + fidelity certificate + small residual operator + deterministic fallback`

模型只在 certificate 判断 residual 可辨认且处于训练支持域时输出校正；其余格回退。baseline 必须包含 P1、P2、full curved evaluator、DeepONet、FNO/FFNO，以及相同 forward/query budget 的传统插值或低秩回归。

如果 N5 证明残差低于实验噪声，正确贡献不是硬训练一个网络，而是证明在该弱偏折工作域中 Picard-1 已足够，研究重心转向 inverse field operator、真实噪声建模或 4D data assimilation。

## 必须向何远哲确认

1. 位移是 pixel、归一化 detector coordinate 还是物理长度？
2. 同一静态背景/flow-off 是否有 20 次以上 repeats？
3. 每条 ray/pixel 是否有 mask、confidence 或 optical-flow covariance？
4. 组内 forward 与 adjoint/VJP 的 observable 是 exit angle、path-integrated deflection 还是最终 pixel displacement？
5. 有限孔径 rays 如何采样，narrow/wide 对应的真实光圈范围是什么？
6. 允许用哪些 case 做开发，哪些必须封存为 blind audit？
