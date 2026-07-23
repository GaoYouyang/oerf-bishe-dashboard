# v3k-A 同场多布局反事实监督：部分恢复，但不足以支持全局几何调制

日期：2026-07-14
性质：8x16x16 synthetic development mechanism audit；不是 superiority、blind final 或真实 BOST 结论。

## 一句话结论

同一流场接受四种不同采集布局后，`correct geometry` 相对 `shuffled geometry` 的 validation 配对收益从 v3j 的近零/负值恢复为 **+0.0675% [95% field-cluster CI +0.0410%, +0.0952%]，3/3 seeds 正向**；但预注册的最小有意义收益是 +0.25%，CI 上界仍远低于门槛。正确 descriptor 已能令 correction 改变 8.1399%，最终场误差只改善 +0.0130% 且 CI 跨 0。因此：**v3j 确有一部分 one-field/one-geometry 数据不足，但当前“全局几何向量 -> channel modulation -> 低频残差”仍不是值得继续扩宽的机制。**

## 为什么不能只比较“一场一布局”和“一场四布局”

如果 M4 看过四倍训练行而 M1 只看一倍，那么 M4 变好可能只是优化步更多。v3k-A 因而锁定等曝光双臂：

| 训练臂 | 每个训练场 | 总训练行 | 独立训练场 | 16 个训练布局频次 |
| --- | ---: | ---: | ---: | ---: |
| M1-repeat | 1 个固定布局重复 4 次 | 640 | 160 | 每布局 40 行 |
| M4-counterfactual | 4 个不同布局各 1 次 | 640 | 160 | 每布局 40 行 |

两臂完全共享：

- 24 epochs、batch size、optimizer steps、loss 和 checkpoint 选择规则；
- 三个 model seeds、初始化、DataLoader 行顺序和 1,023 个可训练参数；
- 同一个冻结 240-epoch FNO、同一组 45,226 total parameters；
- 同一 source field 的 full-view noise；
- v3i 已冻结的 ray normalization；
- validation 的 40 个独立场 x 全部 4 个未见布局。

先在 `field x seed x method` 内平均四个布局，再在三个 seeds 间等权平均；bootstrap 只重采样 40 个 source fields。布局行、seed、voxel 都不冒充独立样本。

## 预注册门槛

支持“数据不足解释”必须同时满足：

1. M4 correct vs shuffled validation 平均收益至少 +0.25%，field-cluster CI 下界大于 0，3/3 seeds 正向；
2. M4 correct vs static 也满足同一门槛；
3. `(M4 correct-shuffled) - (M1 correct-shuffled)` 至少 +0.25%，CI 下界大于 0；
4. geometry-held-out development split 复现；
5. joint OOD 不出现超过 -0.25% 的平均伤害；
6. 同模型 descriptor swap 同时改变 correction，并带来至少 +0.25% 的场误差收益。

这些门槛在打开正式结果前写入配置。若关键 CI 上界均低于 +0.25%，停止当前全局调制的 width/rank/capacity search。

## 正式结果

### Validation 机制对照

| 对比 | 平均收益 | 95% field-cluster CI | 正向 seeds | 判定 |
| --- | ---: | ---: | ---: | --- |
| M4 correct vs shuffled | +0.0675% | [+0.0410%, +0.0952%] | 3/3 | 方向稳定，但低于 +0.25% |
| M4 correct vs static | +0.0801% | [+0.0528%, +0.1072%] | 3/3 | 方向稳定，但低于 +0.25% |
| M4-M1 correct-vs-shuffled interaction | +0.0349% | [+0.0126%, +0.0580%] | 2/3 | 多布局确有小作用，远低于门槛 |
| M4 correct vs shuffled，geometry OOD | +0.0684% | [-0.0088%, +0.1469%] | 2/3 | 未在 held-out geometry 稳定复现 |

绝对 validation field relative L2：

- Locked FNO：0.257826
- M1 correct：0.247256；M1 shuffled：0.247309
- M4 correct：0.247516；M4 shuffled：0.247620

不要把 M1/M4 的绝对误差直接当作因果对比。主问题是同一 arm 内 correct 是否胜 matched control，以及该收益是否因多布局监督而增加。

### 同模型 descriptor swap

| 训练臂 | embedding change L2 | modulation change L2 | correction change | 正确 descriptor 场收益 | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-repeat | 0.3010 | 0.2884 | 6.0904% | +0.0550% | [+0.0364%, +0.0743%] |
| M4-counterfactual | 0.2779 | 0.2442 | 8.1399% | +0.0130% | [-0.0138%, +0.0391%] |

v3j 的 correction swap 只有 0.1785%；v3k-A 提高到 8.1399%，说明反事实监督确实让 descriptor 更强地进入了 correction。可是 correction 的变化没有沿目标误差方向稳定下降，M4 的最终场收益接近零。这把故障位置从“信息进不去”进一步定位为“全局条件产生的空间修正不够针对观测射线与局部欠定方向”。

## 对研究路线的真实含义

### 已被证据支持

- one-field/one-geometry 配对会削弱几何机制学习；同场多布局能恢复一个小而稳定的 correct-vs-shuffled 信号。
- 全局 geometry embedding 并非完全无效；它会改变 modulation 和 correction。
- 1023 参数全局分支的 correction 变化与 field-error 改善严重脱钩。

### 仍不能声称

- 不能说 GC-SRO 优于 FNO、DeepONet、VIDON 或 NeRIF；
- 不能说“证明 v3j 只缺数据”；
- 不能把 +0.0675% 写成有论文价值的优势，预注册阈值是 +0.25%；
- 不能把三个 seeds 当统计样本量；独立单位仍是 source field；
- 不能外推到真实标定、曲线光路、PIV-BOST 或 4D BOST。

## v3k-B：空间 ray-set 分支，而不是继续加全局 hidden width

下一版只改变信息注入位置：让每个体素直接看到逐视角证据。

1. 对每个 active camera 构造体素级 token：`[normalized backprojection, sin(theta), cos(theta), mask, optional calibration/ray feature]`。
2. 用共享小网络编码每个 camera token，保持相机集合置换不变。
3. 由 ridge/base prediction、support 和坐标生成 voxel query。
4. 用 masked attention 或 Deep-Sets sum 在每个 voxel 聚合 camera evidence。
5. 聚合结果进入零初始化、support-limited correction；冻结同一个 FNO 起点。
6. 继续保留 M1-repeat/M4-counterfactual、correct/shuffled/static、相同参数与曝光量。

最小机制闸门仍先看 correct-vs-shuffled 和同模型 swap；只有空间 branch 通过后，才训练 matched variable-geometry FNO/VIDON strong baseline。若组内真实采集几何始终固定，直接停止 geometry 作为主创新，转向 `static operator -> NeRIF warm start / runtime-quality`。

## 初学者学习路径

### 第 1 层：先读懂这次失败

- 画出 `geometry descriptor -> embedding -> modulation -> correction -> field error` 五层链条；
- 手算为什么 640 `field-layout` 行仍只有 160 个独立 fields；
- 解释 M1-repeat 为什么能排除“只是多走了优化步”。

### 第 2 层：集合编码

- [Deep Sets，NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)：提取“逐元素共享编码 + permutation-invariant aggregation”的结构，不把论文应用任务照搬到 BOST。
- [VIDON，arXiv 2205.11404](https://arxiv.org/abs/2205.11404)：重点读 variable sensor number/location、permutation invariance 和 branch/trunk 角色；它是 v3k-B 必须公平面对的邻近方法，不等于本项目创新。

### 第 3 层：为什么 FiLM 在这里不够

- [FiLM，AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11671)：理解 feature-wise affine modulation。v3j/v3k-A 的负结果不是否定 FiLM，而是说明 BOST 的局部射线欠定性未必能靠一个全局向量修正。

### 第 4 层：几何与 operator 的接口

- [Geometry-Informed Neural Operator，OpenReview](https://openreview.net/pdf?id=86dXbqT5Ua)：关注 geometry encoder -> regular latent grid -> operator -> geometry decoder 的分工。GINO 处理复杂 PDE 几何，不可直接当作 BOST camera-set baseline；可借鉴的是“几何在局部映射中进入，而非只给全局通道缩放”。

## 可复现资产

- 配置：`demo_t16_operator/configs/v3k_a_counterfactual_supervision.json`
- lazy 数据层：`demo_t16_operator/counterfactual_geometry.py`
- 正式 runner：`demo_t16_operator/run_v3k_a_counterfactual_supervision.py`
- 独立 validator：`demo_t16_operator/validate_v3k_a_counterfactual_results.py`
- 单元测试：`demo_t16_operator/test_v3k_a_counterfactual_supervision.py`
- 公开结果：`demo_t16_operator/results/v3k_a_counterfactual_supervision/`
- 私有 checkpoints：`demo_t16_operator/results/v3k_a_counterfactual_supervision_work/`，已由 `.gitignore` 排除。

独立验证结果：24 training runs、576 history rows、20,160 sample rows、960 same-model swap rows、base checkpoint drift 0、全部 checksum 通过；superiority 与 blind final 继续关闭。
