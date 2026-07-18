# N2 给何远哲师兄的一页确认单

日期：2026-07-18
当前状态：`N2_WAITING_FOR_LAB_INPUT`，资料齐备度 0/7；尚未运行或训练 N2 算法。

## 想请师兄只做五个选择

| 决策 | 可选项 | 为什么现在必须确认 |
|---|---|---|
| 主要真实失配 | finite aperture / ray bending / calibration drift / displacement extraction / discretization | 五者需要的额外数据、forward fidelity 与主终点不同 |
| 最小接口 | sparse/matrix-free `A/Aᵀ`，或非线性 `F` 的 `JVP/VJP` / reprojection only | 决定能否做严格 adjoint 与同预算求解器 |
| 永久留出单位 | camera/view / session / geometry / condition | 防止同一 run 随机拆帧造成泄漏 |
| 真实主终点 | held-out reprojection / front / PIV velocity / temporal stability | 无独立真值时不报 field-L2 |
| 使用权限 | 本机保存 / 本机训练 / 组会 / 毕业论文文字 / 图 / 公开派生指标 | 原始数据默认不进 Git |

## 最小接口样例包

- 2 个 reconstruction views + 1 个永久 audit view；这只用于验证 loader/接口，不能支撑三维重建或论文统计；
- reference/flow-on 原图或双分量 displacement；
- mask、confidence/相关峰、单位和 u/v 顺序；
- camera/ray、标定 ID/version 与重投影误差；
- case/run/session/geometry/condition ID；
- reconstruction volume 的 shape、米制边界、轴顺序和 support；
- 当前 forward 的版本、normalization 和线性 `A/Aᵀ` 或非线性 `F` 的 `JVP/VJP` 接口；
- 每个固定条件的 flow-off repeats、逐文件 manifest 与使用权限；
- 若要研究背景依赖误差，另需多个独立背景或受控背景位移，不能用同背景 repeats 替代。

## 我拿到后 48 小时交付

1. 私有 loader、checksum、shape/dtype/单位与坐标审计；
2. 每 view 的 reference/flow-on/displacement/mask/confidence 可视化；
3. 线性时比较 `dot(Ax,r)` 与 `dot(x,Aᵀr)`；非线性时审计 `JVP/VJP` 与 finite difference；
4. matched-call CGLS/Landweber/Tikhonov 基线；
5. 只使用 reconstruction/validation，audit 保持 sealed；
6. 一份脱敏 readiness 报告，不含路径、case ID 或原始数据。

## 请师兄注意的边界

- 现在的 50 flow-off repeats 是本项目工程采集门，不声称为普适样本量；
- held-out reprojection 是独立观测一致性，不是唯一 3D 真值；
- 若只有一个真实场，不能把相机组合数写成独立样本量；
- 没有多 f-number/focus 就不做 finite-aperture 泛化主张；
- 没有 `A/Aᵀ` 或 `JVP/VJP` 仍可做 loader/benchmark，但不写 solver/operator 论文主张。

完整技术合同：`docs/oerf_n2_physical_mismatch_data_contract_2026-07-18.md`。
