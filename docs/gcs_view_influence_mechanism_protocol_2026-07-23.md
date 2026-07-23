# View-Influence 候选：先用 28 路径 pilot 验证，再决定是否付出 912 路径成本

## 0. 先讲结论

当前没有新算法，也没有逐视角机制信号。现在完成的是一份可执行、可失败的 post-open 协议：用 exact leave-one-train-view 重训测试“逐相机影响”是否比总 L2/H1/Huber 多提供三维修正方向的信息。

协议验证状态：`VIEW_INFLUENCE_PROTOCOL_VALID_FULL_RUNNER_IMPLEMENTED`。此前两次工程启动中，第一次在训练前拦截边界网格错误；修复后的 pilot 状态为 `VIEW_INFLUENCE_PILOT_INVARIANTS_PASS_NOT_A_MECHANISM_RESULT`。full runner 加入后协议字节发生变化，因此启动 912 路径前还必须用当前协议再跑一次 28 路径 pilot。

28 条拟合路径的工程门已经通过：它不看真值标签、不选模型、不报 AUC，也不能写成科学结果。当前只授权实现并运行 912 路径的已开数据机制面板。

**突破监测：没有突破。** 当前新增的是可执行协议和工程 pilot，不是算法、重建、泛化或论文结果。

## 1. 为什么不是再算一个 reprojection error

上一轮已经看到：单个 holdout 投影变好，不保证三维 field 变好。原因是多个三维场可以在少数二维投影上非常接近。

逐视角影响问的是另一个问题。设全视角训练得到的 residual correction 为 $d$，拿掉第 $v$ 个训练相机、从同一随机初始化重新训练后得到 $d_{-v}$。我们比较：

1. 相对变化 $\lVert d_{-v}-d\rVert_2/\lVert d\rVert_2$；
2. 方向余弦 $\langle d_{-v},d\rangle/(\lVert d_{-v}\rVert_2\lVert d\rVert_2)$；
3. 沿全视角方向的有符号投影 $\langle d_{-v},d\rangle/\lVert d\rVert_2^2$；
4. correction norm ratio $\lVert d_{-v}\rVert_2/\lVert d\rVert_2$。

如果六个相机都支持相似的修正，移除任何一个后方向应相对稳定。如果某个相机独自拉动近零空间结构，拿掉它后 correction 可能缩小、旋转甚至翻转。这里比较的是重建方向对观测集合的敏感性，不是把一个二维误差冒充三维真值。

这个直觉仍可能失败。稳定的修正也可能稳定地错，不稳定也可能来自必要的稀有视角。因此 exact LOO 只是信息上界，不是安全证明。

## 2. exact 的严格含义

每个 full path 和 leave-one-view path 必须满足：

- base 与 residual 都从相同 seed 重新初始化；
- 只删除一个 train camera 的完整 ray block；
- 被删相机不能移动到 development、test 或 dense audit；
- development checkpoint、噪声 realization、central-step renderer 和回退规则保持不变；
- correction feature 在固定 $16^3$ 网格上从网络输出计算，不读取 analytic truth；
- full correction 为零或原修正未准入时，feature fail closed；
- 真值标签只能在全部 observable feature CSV 落盘后连接。

这比“把第 v 个相机的 residual 设为零再前向一次”贵得多，但它测的是训练解对视角集合的真实依赖。后者可以作为未来 JVP/VJP 近似，不能冒充 exact 上界。

## 3. 为什么先跑 28 条，而不是直接跑 912 条

完整面板的成本来自：

| 部分 | 计算 |
|---|---:|
| full replay | 12 phantom x 2 noise x 3 rig x 2 network repeat = 144 |
| exact LOO | 12 x 2 x 2 x (6+4+6) = 768 |
| 总计 | 912 条 base+residual fit path |

工程 pilot 只取已开的 smooth seed 3301、wrinkled seed 3401、两档噪声、nominal six-view 和一个 nuisance repeat：

- 4 条 full replay；
- 4 x 6 = 24 条 leave-one-view retraining；
- 共 28 条拟合路径。

选择这两个 seed 只是覆盖两种已开 family 并估算成本，不是因为它们的真值结果好。pilot 禁止做 mechanism selection。

## 4. pilot 必须通过什么

1. 四条 full replay 的 checkpoint、admission 和八个 observable scalar 与冻结源结果最大绝对差不超过 `1e-6`；
2. 恰好生成 4 full、24 LOO 和 4 influence rows；
3. 所有输出有限；
4. 四个 full correction 都非零且可计算 influence；
5. 代码从 committed tracked worktree 启动；
6. pilot summary 的所有科学 claim 必须仍为 false。

只要一项失败，状态就是 `VIEW_INFLUENCE_PILOT_INVARIANTS_FAIL_STOP`，不能启动完整机制面板。

## 5. 完整 post-open 面板怎样判“值得继续”

如果 pilot 通过，完整面板仍只用当前 12 个已经开过的 phantom，因此最多是 mechanism engineering。

预声明的 14 个 permutation-invariant observable features 输入固定 ridge logistic；除了 correction 变化/余弦/投影/norm ratio，还显式记录删视角后的 residual admission fraction。按 phantom seed 整组 leave-one-out，不调 lambda、不选 feature、不扫阈值。主信息门为：

- grouped out-of-fold ROC-AUC 至少 0.75；
- smooth 与 wrinkled 各自 ROC-AUC 至少 0.65；
- phantom-cluster bootstrap 95% 下界至少 0.50；
- source energy/holdout control、view-influence only 和二者组合必须用同一分组；组合模型的 grouped AUC 至少比 source control 高 0.05；
- source L2 gate、source ridge gate、view-count only 和 accept-all admitted 必须保留为负对照。

即便这些门全过，也只允许写一份“申请新 calibration phantom”的结果前协议。它不授权 safe gate、fresh audit、Stage B 或论文结论。

如果 grouped signal 不过，exact LOO 支线关闭，不训练 set encoder。这样能避免用更大的神经网络掩盖输入本身没有信息。

## 6. 从 exact LOO 到算子学习的正确顺序

| 层 | 方法 | 进入条件 | 当前状态 |
|---|---|---|---|
| V0 | energy/reprojection controls | 已完成 | NO-GO |
| V1 | exact leave-one-view | pilot 先过 | runner 已实现，未运行 |
| V2 | JVP/VJP influence approximation | V1 有信息且 exact 太贵 | 未授权 |
| V3 | grouped ridge | V1 feature 有稳定分离 | 未授权 |
| V4 | 小型 permutation-invariant set encoder | V3 留下非线性 headroom | 未授权 |
| V5 | 有界 operator correction + exact fallback | 新 calibration 与 final audit 均通过 | 未授权 |

这里真正可能形成论文算法的是 V2--V5：用少量 JVP/VJP 近似昂贵的 exact LOO，再用可变相机集合上的算子输出 correction bound 或 reject probability。V1 只是确认“信息是否存在”。

## 7. 需要师兄确认的接口

1. 真实 NeRIF/BOST 训练能否按 camera 返回 loss、residual 和 camera mask？
2. 删除一个 camera 后重训的单帧成本是多少？是否可 warm-start，还是必须同初始化 exact replay？
3. 对 neural field 参数或输出场能否计算 JVP/VJP？
4. 相机数量与 camera ordering 是否固定？是否存在失效 camera、遮挡、不同 ray density？
5. held-out camera 能否完全不进训练？
6. 最终物理终点能否是 PIV velocity compensation error，而不是只报 RI field 或 reprojection？

## 8. 复现入口

验证协议：

```bash
PYTHONPATH=. .venv/bin/python -m \
  learning_labs.validate_gcs_view_influence_protocol \
  --config demo_t16_operator/configs/gcs_view_influence_mechanism_v0.json \
  --root . \
  --output learning_labs/results/gcs_view_influence_protocol_v0/validation.json
```

在干净提交后运行 pilot：

```bash
PYTHONPATH=. .venv/bin/python -m \
  learning_labs.run_gcs_view_influence_pilot \
  --config demo_t16_operator/configs/gcs_view_influence_mechanism_v0.json \
  --root . \
  --output-dir learning_labs/results/gcs_view_influence_pilot_v0_run1
```

定向测试：

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  learning_labs/test_gcs_view_influence.py \
  learning_labs/test_validate_gcs_view_influence_protocol.py \
  learning_labs/test_run_gcs_view_influence_pilot.py
```

## 9. 当前禁止主张

不能声称逐视角信息有效、exact LOO 优于能量门、已经得到算子学习模型、JVP/VJP 近似可用、真实 BOST/PIV-BOST 验证、跨 rig 泛化、论文成功或突破。
