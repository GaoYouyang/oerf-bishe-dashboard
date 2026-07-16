# v3i 变采集几何数据合同

> **当前判决**：328 个原始三维场已被确定性地分配到 28 种 K=6 相机布局，数据闸门通过，可以开始 GC-SRO 三种子功能实验；尚未训练、尚未比较优越性，也不包含真实 BOST 标定几何。

## 为什么不做 160 场 × 16 布局

全笛卡尔积会产生 2,560 个训练张量，不仅增加近 1 GB 内存压力，还会让同一个场在一个 batch 内重复出现，导致有效场多样性被高估。v3i 选择 **one field, one geometry** 功能协议：

- 每个原始场恰好出现一次，没有 source duplication。
- 每个 source split 内按 SHA-256 稳定排序，再循环均衡分配布局。
- 同一个场的全视角噪声由 `sample_seed + K` 一次生成，布局只决定哪些视角可见，不改变噪声实现。
- 60° `Q_audit` 在 mask、ridge、逐视角反投影和 sin/cos 通道中都严格为零。

## 分区与平衡性

| source split | 场数 | geometry partition | 布局数 | 每布局场数 | 几何熵 |
|---|---:|---|---:|---:|---:|
| train | 160 | train | 16 | 10 | 4.00 bit |
| val | 40 | validation | 4 | 10 | 2.00 bit |
| test_iid | 32 | geometry-OOD | 4 | 8 | 2.00 bit |
| test_noise_ood | 32 | geometry-OOD | 4 | 8 | 2.00 bit |
| test_family_ood | 32 | stress | 4 | 8 | 2.00 bit |
| test_joint_ood | 32 | stress | 4 | 8 | 2.00 bit |

训练与 validation 布局集完全不相交。geometry-OOD 与 stress 布局也不参与调参；但这仍是已查看的 synthetic development 体系，不是全项目新盲测集。

## 张量合同

- 样本数：`328`
- 输入：`[328, 42, 8, 16, 16]`
- 目标：`[328, 8, 16, 16]`
- 前 15 通道：ridge lift、support、K/9、9 个 camera-active mask、z/y/x。
- 后 27 通道：9 个逐视角反投影、9 个 active-sin、9 个 active-cos。
- 私有压缩 NPZ：`20.3 MiB`；网页只发布哈希、通道表、分配表和图，不发布数据张量。

## 下一轮五组功能对照

1. `locked_fno`：不训练，测量旧固定布局 FNO 在变几何上的直接迁移。
2. `static`：同一 1,023 参数频谱残差分支，只接收 K/9，测量“多一个 adapter”本身的收益。
3. `k_cardinality`：移除角度后的 mask-set；因 K 恒为 6，理论上应与 static 同信息。
4. `shuffled_geometry`：参数与 correct 完全相同，但向每个场提供确定性错误布局。
5. `correct_geometry`：唯一接收正确无序角度集的 GC-SRO。

首轮只跑 3 seeds 和短程学习曲线。通过条件不是“correct 的单一均值最低”，而是：在未见 validation 布局上同时胜 static 与 shuffled，三种子方向一致，且 geometry-OOD 不发生明显尾部恶化。

## 可复跑与审计

- 生成：`python demo_t16_operator/run_v3i_variable_geometry_dataset.py`
- 独立验证：`python demo_t16_operator/validate_v3i_variable_geometry_results.py`
- 源码测试：`pytest demo_t16_operator/test_v3i_variable_geometry_dataset.py`

## 请师兄确认

1. 功能实验先用 one-field/one-geometry 是否合理，还是应该为每个训练场在每个 epoch 在 16 个布局中重采样？
2. 真实 TDBOST/BOST 数据是否能从更多实体相机中下采样不同子集？
3. 真实标定除方位角外，还需加入哪些内参、外参或 ray-path 特征？
