# v3h GC-SRO 几何可辨识性闸门

> **一句话结论**：现有 K=6 直接算子数据的 328 个样本全部使用同一个相机布局，因此不能训练或宣称“学会采集几何”；28 布局协议已通过可辨识性闸门，GC-SRO v0 只通过工程闸门，尚未进行优越性训练。

## 今天真正证明了什么

- 全部合法布局：9 个候选角度中固定排除 60°，其余 8 中选 6，共 `C(8,6)=28` 种。
- 现有数据：唯一布局 `0°, 40°, 80°, 100°, 120°, 160°`，几何熵 `0 bit`，correct/shuffled/static 在样本间不可辨识。
- 28 布局在同一组 40 个 validation 场上的 ridge 平均场误差从 `0.33124` 到 `0.58209`，相对跨度 `64.95%`。
- 对每个场单独比较，最好与最差布局的误差跨度中位数 `71.53%`，p10/p90 为 `31.17% / 151.19%`。
- 算子非零条件数为 `17.56–59.82`，变异系数 `0.380`；最大角度缺口与误差 Spearman 相关 `0.537`。
- 旧固定布局在这 28 种中平均 ridge 误差排名 `1/28`。这只是已查看 validation 场上的数值观察，**不是“最优相机设计”结论**。

## 可反驳的 GC-SRO v0

GC-SRO（Geometry-Conditioned Spectral Residual Operator）保留已锁定 FNO 作为空间主干，用共享 MLP 编码无序的激活角度集合，再调制一个仅作用于低频模态的轻量残差分支。最后一层零初始化，所以第一次前向与冻结 FNO **逐点完全一致**。

| 闸门 | 结果 |
|---|---:|
| 初始输出与 FNO 最大差 | `0` |
| 三步优化后冻结 FNO 参数漂移 | `0` |
| 可训练 adapter / 总参数 | `1,023 / 45,226` |
| 28 布局几何 embedding 平均成对距离 | `0.01156` |
| 共同重排相机顺序的最大差 | `4.47e-8` |
| 无角度的 K-cardinality/static 对照 | 布局间距离约 `0` |

`mask-only` 在固定 K=6 且置换不变的集合编码中实际只看到基数，不能识别哪台相机激活。这是有用的负对照：如果 correct geometry 不能稳定赢它、static adapter 和 shuffled geometry，就应否定机制假设。

## 28 布局开发协议

| 分区 | 数量 | 用途 |
|---|---:|---|
| train | 16 | 学习变几何映射；旧固定布局强制放入此区 |
| validation | 4 | 选择学习率、残差幅度和停止点 |
| geometry-OOD | 4 | 检验未见布局泛化 |
| stress | 4 | 仅按最大角度缺口和算子条件数选择，未使用场误差 |

必做六组等数据/等训练预算对照：锁定 FNO、参数匹配 wider-FNO、static spectral adapter、K-cardinality control、shuffled-geometry GC-SRO、correct-geometry GC-SRO。首轮只允许 3 seeds 的小规模功能实验；在机制对照和 geometry-OOD 都通过以前，不开 blind final，不使用“优于 FNO/DeepONet”表述。

## 文献在本项目中的角色

- [Deep Sets (NeurIPS 2017)](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)：为相机集合的置换不变性提供理论起点。
- [Set Transformer (PMLR 2019)](https://proceedings.mlr.press/v97/lee19d.html)：当共享 MLP 不足以表达视角间相互作用时，作为预注册的二阶备选，不在看结果后临时加入。
- [VIDON (arXiv 2022)](https://arxiv.org/abs/2205.11404)：最近的变传感器数量/位置算子学习对照，需提取其传感器编码和泛化协议。
- [GINO (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)：用于理解任意物理域几何的神经算子；它编码的是 **physical-domain geometry**，不等同于本项目的 **acquisition geometry**。

## 请师兄审核的四个问题

1. 60° 查询视角在真实 BOST 系统中是否应始终排除，还是应改为多个 leave-one-view-out 协议？
2. 真实相机是否有俯仰角、内参和光线路径？如果有，二维方位角描述符只能作为功能原型。
3. 是否能提供多布局模拟/实验数据，或从现有更多相机数据中下采样出可变子集？
4. 作为本科毕设，第一篇成果应聚焦“变采集几何稳健重建”，还是“几何 OOD + 不确定性”？

## 可复跑入口

- 生成：`python demo_t16_operator/run_v3h_gc_sro_geometry_gate.py`
- 独立验证：`python demo_t16_operator/validate_v3h_gc_sro_geometry_results.py`
- 当前发布包只含 CSV/JSON/图和源码，不含 NPZ、checkpoint、VPN 论文或其他受限材料。
