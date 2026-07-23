# GCS-Fourier 预实验：先验证“失配预警”有没有信息量

日期：2026-07-22  
状态：**FROZEN BEFORE FIRST EXECUTION**  
证据等级上限：**E1 post-open synthetic diagnostic only**

## 1. 这一步只回答一个小问题

我们暂时不声称发明了新算法。先问：当 Fourier 隐式场加入目标离散分辨率无法可靠承载的高频后，开发视角上“连续自动微分梯度”和“实际用于训练的中心差分梯度”之间的投影失配，能不能预警三维场或未见视角的退化？

如果不能，`GCS` 不再作为主 selector，后续 `CFH/NASH-Mix` 不能拿它当安全门；如果能，也只允许进入更大、重新预注册的实验，不能直接叫论文创新。

## 2. 为什么不是再抄一遍作者代码

当前公开 Neural Refractive Index Primitives 仓库没有根许可证，默认 validation/test 还复用了训练 pose。这里不复制其模型、数据或训练器，只实现一个 clean-room 机制实验：

```text
连续解析 morphology q(x)
  -> 解析 grad q + 96 点真值积分
  -> train / development / test 三组不重合角度
  -> 32 点 Fourier-field 逆 renderer
  -> 16 点 voxel CGLS 对照
```

teacher 和 inverse 的积分采样数、参数化与梯度原语不同，因此避免了最直接的同离散链 inverse crime；但它仍然是 normalized synthetic proxy，不是 CFD、密度、温度或实验火焰。

## 3. 冻结相机和统计单位

| split | 角度（度，平行束以 180 度为周期） | 用途 |
|---|---|---|
| train | 0, 30, 60, 90, 120, 150 | 优化参数 |
| development | 15, 105 | 选 checkpoint；计算 GCS |
| test | 45, 135 | 模型冻结后评估，不参与选择 |

每视角 8 x 8 条射线。四个 morphology family 与两个噪声层形成 `4 x 2 = 8` 个配对单位；每个单位内两个优化 seed 只取中位数，不能伪装成独立物理样本。

## 4. 三个对照

1. `CGLS-12`：16 cubed 体素有限差分 + 三线性采样的固定步数经典基线。
2. `Fourier-low`：频率 1、2、4；两层宽度 48 的 SiLU MLP。
3. `Fourier-high`：在相同宽度和深度上加入 8、16 频率。

两档 MLP 参数量并不相同，所以本实验不能回答“等参数谁更强”。它只制造并观察一个预声明的带宽风险。两者都乘二次 box window，使边界场和边界梯度为零，固定 BOS 加性常数零空间。

## 5. 为什么 GCS 的步长不是 1e-3

`1e-3` 中心差分只适合验证 autograd 实现是否正确；它接近连续导数，不代表部署 renderer 的可解析尺度。本实验的 operational step 固定为

```text
h = 2 / (16 - 1) = 0.13333333333333333
```

即目标 16 cubed 网格间距。主诊断量是 development rays 上

```text
GCS = || y_AD - y_FD(h) ||_2 / || y_FD(h) ||_2
```

它完全由模型、几何和开发射线计算，不读取真值场或 test。它可能只是频带大小的代理；本轮正是要证伪它是否有超出架构标签的实际预测信息。

## 6. 预声明结果与继续门

主结果：`field relative-L2(high) - field relative-L2(low)`。  
次结果：`test-view clean relative-L2(high) - low`。  
绝对差至少 `+0.05` 记作 harm，至少 `-0.05` 记作 help。

只有同时满足以下五项，才允许进入 16 场、3 seed 的正式预注册：

1. 8 个单位中至少 2 个 field harm；
2. 至少 2 个 field non-harm，保证有可区分对象；
3. high-model GCS 与 field delta 的 Spearman rho >= 0.5；
4. GCS 预测 field harm 的 AUROC >= 0.75；
5. field delta 与 test-view delta 的符号一致率 >= 75%。

样本只有 8 个配对单位，本轮不做确认性 p 值。阈值、频带、噪声、角度、代表图 cell 和训练预算均不得在看到结果后改写。

## 7. 无论结果多漂亮都禁止的措辞

- 禁止称为真实 BOST、OERF 火焰或密度重建；
- 禁止称为算子学习优于 DeepONet、FNO、NeRIF 或 Neural Refractive Index Primitives；
- 禁止把训练/开发重投影降低写成三维真值恢复；
- 禁止把 2 个优化 seed 写成独立样本；
- 禁止称为泛化、论文成功或突破。

冻结配置：[gcs_fourier_pretest_v1.json](../demo_t16_operator/configs/gcs_fourier_pretest_v1.json)  
核心实现：[gcs_fourier_bost_lab.py](../learning_labs/gcs_fourier_bost_lab.py)  
运行器：[run_gcs_fourier_pretest.py](../learning_labs/run_gcs_fourier_pretest.py)
