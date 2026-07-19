# PSU rotation-40 多分辨率机制诊断：结果、审计与算法转向

## 一句话结论

**还不能把 32³ 的退化归因于“高频过拟合”或“CGLS 第四步走坏了”。**

把冻结的 16³ 场三线性重采样到 32³ 后，仅仅换成 32³ forward，global vector relative-L2
就从 `0.843263` 变为 `0.856804`，绝对差 `0.013541`，超过预先固定的 `0.01` 机制屏。
随后从 `U(x16)` 走向 `x32` 的冻结场修正在 camera 2 上与真实残差同向，在 camera 3/4 上反向。
因此机器输出：

```text
OPENED_BLOCK_FORWARD_GRID_CHANGE_MATERIAL_MECHANISM_UNRESOLVED
```

这里的 `MATERIAL` 只表示预先固定的数值屏被触发；不是统计、物理或工程显著性，也不是因果证明。
最稳妥的科学结论仍是：**forward 离散化与重建场差异都不可忽略，当前一个 rotation block 无法把两者
分干净。**

## 1. 为什么必须做这一步

E68 比较的是两个完整包：

- `16³ + zero-start CGLS4 + A16`；
- `32³ + zero-start CGLS4 + A32`。

它同时改变自由度、网格间距、有限差分、三线性插值和 Krylov 谱滤波轨迹。`32³` 在 support
views 上更好、rotation-40 上更差，只能证明迁移反转，不能说明是哪一部分造成。

本轮在结果前固定五个端点：

| ID | 计算 | global rel-L2 | 能看什么 |
|---|---|---:|---|
| `native_16` | `A16 x16` | 0.843263 | 原 16³ 端点复核 |
| `prolong_16_to_32` | `A32 U(x16)` | 0.856804 | 同一冻结粗场，只换 32³ 表示/forward |
| `restrict_32_to_16` | `A16 D(x32)` | 0.942126 | 32³ 场在端点对齐粗采样后的表现 |
| `roundtrip_32_via_16` | `A32 U(D(x32))` | 0.944557 | 去掉一部分细尺度后的 32³ forward 表现 |
| `native_32` | `A32 x32` | 0.959591 | 原 32³ 端点复核 |

`U/D` 是 `align_corners=True` 的**端点对齐三线性重采样**。16 与 32 的节点不是嵌套网格，
所以 `D` 不是质量正交 restriction，`U` 也不是可逆 prolongation；`DU != I`。因此后两行只能当
机制探针，不能当作算法基线或“低频/高频”的严格正交分解。

## 2. 同一粗场换 forward，也不是零影响

`A16 x16` 与 `A32 U(x16)` 的 relative-L2 差为：

```text
|0.8432631430 - 0.8568036990| = 0.0135405560
```

更直接的预测差异是：

```text
||A32 U(x16) - A16 x16|| / ||y||                = 0.1118379182
||A32 U(x16) - A16 x16|| / ||A16 x16||          = 0.2363811730
```

第二组数在看完结果后才作为独立审计量解释，没有进入预先固定的机器门。它提醒我们：只看两个
残差范数的差仍可能漏掉方向完全不同的预测。下一轮机器门必须同时约束预测差异，而不是只约束
relative-L2 gap。

## 3. 一个 pooled 标量掩盖了三台相机的冲突

定义：

\[
r=y-A_{32}U x_{16},\qquad
d=A_{32}(x_{32}-U x_{16}).
\]

`cos(r,d)` 为正，说明加入冻结细场修正会沿当前残差的正确方向移动；为负则反向。

| 范围 | `cos(r,d)` | 诊断 `alpha*` | 解释 |
|---|---:|---:|---|
| 拼接全部 rays | -0.052812 | -0.116261 | pooled 上继续加正向修正会变差 |
| camera 2 | +0.276931 | +0.388537 | 这一相机支持一部分修正 |
| camera 3 | -0.137083 | -0.365920 | 冻结修正方向相反 |
| camera 4 | -0.162454 | -0.380865 | 冻结修正方向相反 |

这里的 `alpha*` 是看过 rotation-40 后计算的解释量，**不是可部署参数**。不能选 `0.39` 给 camera
2，再在同一数据上声称算法成功。

这个冲突却给算法结构一个重要约束：未来的 correction gate 不能只看 pooled residual，也不能
默认一个全局标量同时适合所有相机。它至少要按 rotation group / camera geometry 检查 tail，并在
任一关键相机受损时回退 coarse 基线。

## 4. 固定 alpha 曲线告诉了我们什么

固定、不选优的路径：

\[
x(\alpha)=Ux_{16}+\alpha(x_{32}-Ux_{16}),
\quad \alpha\in\{0,0.25,0.5,0.75,1\}.
\]

| alpha | pooled | camera 2 | camera 3 | camera 4 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.856804 | 0.781147 | 0.881591 | 0.857711 |
| 0.25 | 0.867402 | 0.754549 | 0.896647 | 0.877224 |
| 0.50 | 0.888591 | 0.753157 | 0.918904 | 0.905597 |
| 0.75 | 0.919640 | 0.777108 | 0.947854 | 0.942030 |
| 1.00 | 0.959591 | 0.824194 | 0.982907 | 0.985630 |

`alpha=0.5` 还做了一次独立 direct forward，与线性端点组合的最大绝对差为
`1.67e-16`，远低于冻结的 `1e-10` 容差。

camera 2 在 `0.25-0.50` 改善，而 camera 3/4 单调变差。**这不是一个可发表的最优 alpha，
而是一个反例：单全局门可能天然掩盖几何分层冲突。**

## 5. 这一步排除了哪些便宜路线

1. **不能直接说“32³ 更细，所以更准”。**真实留出 rotation 已经反驳。
2. **不能直接说“高频过拟合”。**`U/D` 不是正交频带分解，且 forward 网格本身有明显变化。
3. **不能在 rotation-40 上拟合 amplitude 或 alpha。**它已经是开发数据，拟合后不能再当测试。
4. **不能只用 pooled gate。**camera 2 与 camera 3/4 的修正方向相反。
5. **不能马上训练 FNO/DeepONet。**尚未知道 correction 的可识别部分、停止步和几何条件。

## 6. 下一项真正有区分力的实验

### SUPPORT-LORO-MECH-1

只使用 support rotations `{0, 50, 90}`，每次整组留一 rotation，三台相机随 rotation 一起留出。
rotation-40 只做规则冻结后的已打开一致性回放；final rotations 继续封存。

固定：

- CGLS checkpoints `K={1,2,3,4,6,8,12}`；
- `U` 为端点对齐三线性 16³→32³；
- 32³ support cache 必须显式 `verify_hashes=True`；
- 质量加权 coarse 子空间投影 `P=U(U^T W U)^{-1}U^T W`，不再把普通 resize 当正交投影；
- 每个 rotation 等权，同时报告 equal-camera macro、worst camera、p95、调用数与时间；
- 选停止步只能在外层训练 rotations 内完成，不能逐相机选步。

建议的可加预测链：

\[
p_0=A_{16}x_{16}^{4},\quad
p_1=A_{32}Ux_{16}^{4},\quad
p_2=A_{32}x_U^{4},\quad
p_3=A_{32}Px_{32}^{4},\quad
p_4=A_{32}x_{32}^{4}.
\]

随后用平方误差 Shapley 分解给各段归因。三折不能同向复现就输出
`ROTATION40_RUN_SPECIFIC_OR_UNRESOLVED`，不挑最好看的 fold。

## 7. 现在值得保留的算法骨架

工作名仍可用 `RTG-MRC`，但结构要收紧为：

\[
x_f^{cand}=Ux_c+
g_{safe}(\mathcal G,r,\text{camera tails})\,
P_{DC}\,\delta_\theta(Ux_c,r,\mathcal G,h),
\qquad r=y-A_hUx_c.
\]

- `x_c`：经典 coarse CGLS/H1/TV 基线；
- `delta_theta`：MgNO/CNO/轻量坐标网络预测的细尺度修正；
- `P_DC`：数据一致或近零空间投影；
- `g_safe`：rotation-group 与 camera-tail 门；任何关键门恶化就回退 `Ux_c`；
- 网络必须和无网络的 early stopping、H1/TV、pyramid/coarse-to-fine 基线比较。

这仍只是**待证伪候选**。fail-closed 回退是我们的工程设计主张，现有文献只支持多尺度表示、
aliasing 风险和 data consistency 的必要性，不能替我们证明它安全或更准。

## 8. 最相关的一级来源阅读顺序

1. [He et al., Neural Refractive Index Field, Physics of Fluids 2025](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)：先学 BOST forward、坐标隐式场和 projection consistency；不要把逐场景优化误称为神经算子泛化。
2. [Hu et al., A pyramid approach for BOST, Experiments in Fluids 2026](https://link.springer.com/article/10.1007/s00348-025-04153-3)：最直接的物理同步 coarse-to-fine 对照；场、背景图、warp 与投影矩阵需要共同跨尺度更新。
3. [Bartolucci et al., Representation Equivalent Neural Operators, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html)：理解不同离散表示中的 operator aliasing；不能反推本次失败一定由 aliasing 造成。
4. [He, Liu & Xu, MgNO, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/eb3c8135137c8a60425a0320869ad87e-Abstract-Conference.html)：学习 V-cycle、restriction/prolongation 与 multigrid operator parameterization；其证据来自 PDE，不是 BOST。
5. [Boink et al., Data-consistent neural networks for nonlinear inverse problems](https://www.aimsciences.org/article/doi/10.3934/ipi.2022037?viewType=HTML)：学习 data-consistent/near-null-space correction；数据一致不等于场正确。

## 9. 证据与复现入口

- [冻结开发协议](psu_rotation40_multiresolution_diagnosis_protocol_2026-07-19.md)
- [独立结果审计与冻结术语勘误](psu_rotation40_multiresolution_diagnosis_independent_audit_2026-07-19.md)
- [post-open 运行环境](psu_rotation40_multiresolution_diagnosis_environment_2026-07-19.json)
- [机器 summary](../demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1/summary.json)
- [汇总 CSV](../demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1/comparison_rows.csv)
- [可视化 PNG](../demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1/diagnostic.png)
- [公开包独立 validator](../site_tools/validate_psu_rotation40_multiresolution_diagnosis_public.py)
- protocol commit：`48e32d780e66926dfb493f28e0375fb83c4057f9`

## 10. 仍然不能声称什么

- 没有 volumetric truth，不能说哪个三维场更准；
- 一个 rotation block 不能给统计泛化；三台相机不是三个独立实验重复；
- 不能证明细网格普遍过拟合、高频一定是噪声或 CGLS4 是唯一原因；
- 不能把 camera 2 的 post-open `alpha*` 变成算法结果；
- 没有训练神经算子，没有击败 FNO/DeepONet/MgNO，也没有论文级优越性结论。
