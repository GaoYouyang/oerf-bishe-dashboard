# v3k-B 逐体素 ray-set：局部观测有用，但 ray-angle 配对机制有害

日期：2026-07-14

性质：`8x16x16` synthetic development mechanism pilot；不是 superiority、blind final、真实 BOST 或原创性结论。

## 一句话结论

逐体素 ray 数值相对 `geometry-only` 带来 **+1.4445% [95% field-cluster CI +0.5149%, +2.3801%]，3/3 seeds 正向**，说明 v3k-A 之后转向局部观测是对的；但把每条 ray 与正确 `sin(theta), cos(theta)` 配对，反而比“保留完全相同 ray 集合、只循环错配角度”差 **-0.1967% [-0.4023%, -0.0075%]**。同模型替换配对使 correction 改变 19.0683%，最终场误差却恶化 -0.3586% [-0.6490%, -0.1108%]，3/3 seeds 都负向。因此停止 attention width/head 搜索；下一轮改用与数据一致性下降方向对齐的 per-view adjoint residual，而不是继续强化标量角度 token。

## v3k-B 到底改了什么

v3k-A 的全局分支把整个相机集合压成一个向量，再统一调制空间通道。v3k-B 让每个 voxel 直接接收 9 个 camera tokens：

`[normalized backprojection, active mask, sin(theta), cos(theta)]`

每个 token 经过共享 MLP；由 `[frozen FNO prediction, ridge, support, view fraction, z, y, x]` 生成 voxel query，再用 masked attention 聚合 active cameras，最后进入零初始化、support-limited 的小型谱残差块。

工程合同：

- 915 个可训练参数，是 v3k-A 1,023 参数的 89.44%；
- 冻结同一个 240-epoch FNO，初始化输出逐元素等于 base；
- 联合置换 camera token 后输出不变；
- inactive camera 的任意值不能影响输出；
- audit camera 在正确与错配组件里都严格为零；
- M4 的 160 个训练场各看 4 个布局，共 640 行；四种方法共享 fields、rows、seeds、optimizer steps 和 checkpoint rule。

这套设计借鉴了 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) 的置换不变 attention、[VIDON](https://arxiv.org/abs/2205.11404) 的 variable-sensor weighted aggregation，以及 [GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) / [GNOT](https://proceedings.mlr.press/v202/hao23c.html) 的局部 geometry-query 交互。它们都是最近邻和强基线来源，不是本项目可以据为原创的模块。

## 一次被及时拦下的无效 control

最初版本曾把整个布局循环替换为另一个布局。第一种子的四个 runs 立即出现“错配组异常大幅更好”。检查后发现：错配布局会启用 base reconstruction 没见过的其他相机，相当于偷偷增加新观测，不再是同信息 control。

正式结果不包含这组无效 runs。修正版只在**同一个 active-camera 集合内部**循环重配 angle：

- mask 完全相同；
- 每条 ray 数值完全相同；
- angle 多重集完全相同；
- 相机数、噪声、source field、优化步完全相同；
- 唯一改变的是 `ray_i` 与 `theta_i` 的对应关系，且没有固定点。

这一步值得保留在论文方法部分：它说明功能对照必须守住信息集合，而不仅是“输入 shape 一样”。

## 正式 12-run 结果

### Validation 配对结果

| Correct ray-set 相对对照 | 平均收益 | 95% field-cluster CI | 正向 seeds | 判定 |
| --- | ---: | ---: | ---: | --- |
| Locked FNO | +10.9292% | [+8.1114%, +13.8952%] | 3/3 | 残差分支有开发集收益；不是 superiority |
| Pooled-static local evidence | +0.6994% | [+0.0544%, +1.3248%] | 2/3 | 均值/CI 正向，但未过 3/3 门槛 |
| Geometry-only local set | +1.4445% | [+0.5149%, +2.3801%] | 3/3 | raw local ray values 确实有用 |
| Shuffled ray-angle pairing | -0.1967% | [-0.4023%, -0.0075%] | 2/3 | 正确标量配对显著更差 |

`correct vs shuffled-pairing` 在 geometry-held-out `test_iid` 为 +0.1768% [-0.1164%, +0.4565%]，没有稳定复现；在 joint OOD 为 -0.2867% [-0.3988%, -0.1737%]，超过预注册 -0.25% 伤害线。

### 同模型 pairing swap

对每个 validation field-layout，固定训练好的 `correct_ray_set` 权重，只切换正确/错配 angle pairing：

- ray 输入变化：0.0000%，证明没有换观测；
- attention mean L1：0.002221；
- correction 相对变化：19.0683%；
- 正确 pairing 的场收益：-0.3586% [-0.6490%, -0.1108%]；
- 三个 seed 的场收益均为负。

因此不是“模型没看到 angle”。angle pairing 已明显改变 correction，只是改变方向与真实场误差下降方向相反。

## 为什么 scalar angle 可能误导

在当前线性 toy forward 中，局部 backprojection 的空间纹理已经携带了 view operator 的大量信息。额外加入一个全局标量角度，模型容易学到与训练布局相关、但不对应局部误差梯度的 shortcut。更关键的是，raw backprojection `A_i^T y_i` 描述“观测里有什么”，却不直接回答“当前 FNO 预测错在哪里”。

若数据项为

`L_data(x) = 1/2 sum_i ||A_i x - y_i||^2`

则真正指向误差下降的一阶信息是

`grad L_data(x0) = sum_i A_i^T(A_i x0 - y_i)`。

这正是 Landweber/gradient update 和 learned iterative reconstruction 的接口。v3k-C 不应再扩大 attention，而应先比较“纯数值 adjoint step”与“学习如何融合 per-view adjoint residual”。

## v3k-C 预备方向：Adjoint-Residual Ray-Set Operator

暂定工作名 `AR-RSO`，只用于组织实验，不代表原创性已经成立。

每视角 voxel token 改为：

`[A_i^T y_i, A_i^T(y_i - A_i x0), diag(A_i^T A_i), active mask]`

其中 `x0` 是冻结 FNO/ridge 起点。优先顺序：

1. 先实现零学习的 Landweber step，并在 validation 上选择单一 step size；
2. 再实现 per-view adjoint residual 的简单求和，证明 attention 不是必要条件；
3. 仅当数值方向有效时，学习 voxel-wise step/gate；
4. 再比较 Deep Sets/attention aggregation；
5. 最后才进入 matched variable-geometry FNO、VIDON、计算成本和真实 BOST forward。

必须加入的 controls：raw backprojection only、pooled adjoint residual、learned scalar step、same-parameter local CNN、数值 Landweber、correct/shuffled calibration、Q_audit、geometry OOD 和 joint OOD。

最近邻包括 [Learned Primal-Dual](https://arxiv.org/abs/1707.06474) 与 [Solving ill-posed inverse problems using iterative deep neural networks](https://arxiv.org/abs/1704.04058)。因此“把 adjoint residual 输入网络”本身不是创新。可能的论文空间只能来自 BOST 特定的 variable-camera ray set、曲线光路/标定表示、operator warm start 与严格的跨布局验证组合。

## 给基础薄弱本科生的学习任务

### 第一关：手推四行公式

从 `1/2 ||Ax-y||_2^2` 出发，手推梯度 `A^T(Ax-y)`；再写出 Landweber 更新 `x_(k+1)=x_k-tau A^T(Ax_k-y)`。能解释为什么 `A^T y` 和 `A^T(y-Ax0)` 角色不同才算过关。

### 第二关：做 2D 小实验

用一个 `32x32` phantom、少角度 Radon operator 比较 FBP、一次 Landweber、十次 Landweber；画 field error 与 held-out projection error。先不碰神经网络。

### 第三关：读代码时只追五个张量

在 v3k-B runner 中标出 `x0`、per-view ray token、attention weights、correction、final prediction 的 shape；手画 active-camera 维如何被聚合。

### 第四关：读两篇论文

- Learned Primal-Dual：只抽取 forward/adjoint 在每个 unrolled iteration 中的位置；
- VIDON：只抽取 sensor value/location 如何编码、如何保证 permutation invariance。

输出一页对照：`VIDON sensor aggregation`、`Learned Primal-Dual physics update`、`AR-RSO proposed interface` 分别解决什么，不解决什么。

## 可复现资产

- 配置：`demo_t16_operator/configs/v3k_b_voxel_ray_set_pilot.json`
- 模型：`demo_t16_operator/own_algorithm_models.py`
- ray-set 数据接口：`demo_t16_operator/counterfactual_geometry.py`
- runner：`demo_t16_operator/run_v3k_b_voxel_ray_set_pilot.py`
- 独立 validator：`demo_t16_operator/validate_v3k_b_voxel_ray_set_results.py`
- 工程测试：`demo_t16_operator/test_v3k_b_voxel_ray_set.py`
- 公开结果：`demo_t16_operator/results/v3k_b_voxel_ray_set_pilot/`
- 私有 checkpoints：`demo_t16_operator/results/v3k_b_voxel_ray_set_work/`，由 `.gitignore` 排除。

独立验证覆盖 12 training runs、288 history rows、10,080 sample rows、480 same-model swaps、168 个无固定点 pairing 映射、全部 checksum 和 base checkpoint drift 0。superiority、真实 geometry 与 blind final 继续关闭。
