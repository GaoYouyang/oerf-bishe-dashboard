# 12 周推进验收板

用途：把三个月预研变成每周可检查的产物。每周至少留下一个图、一个文件、一个问题。

## 使用规则

- 每周只追一个主目标。
- 周末必须有可展示产物。
- 没有图、没有代码、没有问题清单，算本周没有真正推进。
- 和何远哲沟通后，把“师兄反馈”写进会后纪要。

## 12 周总表

| 周 | 主目标 | 必交产物 | 可展示图 | 要问何远哲 |
| --- | --- | --- | --- | --- |
| 1 | 建立 BOST 物理链条 | 一页变量图、术语表、读 NeRIF intro | `T-rho-n-displacement` 链条图 | 组里真实数据的变量和单位怎么定义 |
| 2 | 2D phantom 与 forward model | `phantom.py` 和 `forward_bos.py` 草版 | 2D 折射率场和位移场 | 简化小角度模型是否足够第一阶段用 |
| 3 | 2D baseline / neural toy | 2D baseline 或 MLP 拟合代码 | ground truth / reconstruction / error | baseline 应该优先复现哪篇 |
| 4 | 3D synthetic BOST | 3D phantom 和多视角位移 | 中心切片和多视角位移图 | OERF 九视角几何能否抽象成这个接口 |
| 5 | NeRIF 简化版 | 坐标 MLP + 训练脚本 | 训练曲线和重构切片 | 是否需要预测 gradient head |
| 6 | 指标与自动报告 | `metrics.py`、`visualize.py`、自动保存结果 | L2/CC/重投影误差表 | 真实数据主要看哪些指标 |
| 7 | 视角数扫描 | 3/5/7/9 视角实验 | view-count curve | 组内最常用视角数是多少 |
| 8 | 噪声与编码扫描 | noise / encoding 消融 | noise curve、encoding comparison | 噪声量级应如何模拟 |
| 9 | 真实/开源数据接口 | data loader 和目录规范 | 数据样例可视化 | 能否给一小份 BOST 或 PIV-BOST 数据 |
| 10 | 选择升级方向 | PIV toy 或 4D toy 二选一 | 补偿误差图或时序平滑图 | 师兄更需要 PIV、4D 还是工具 |
| 11 | 开题 memo | 6-8 页 memo 和 PPT 初稿 | 总流程图、核心结果图 | 开题题目是否需要收窄 |
| 12 | 汇报彩排 | PPT、demo、数据请求清单 | 三张核心图 | 下一阶段周会节奏和交付标准 |

## 当前已提前完成的材料

- M0 2D BOST / coordinate-field toy demo：`demo_m0/`。
- M1 3D-stack sparse-view BOST toy demo：`demo_m1/`。
- M2 noise-view-capacity robustness scan：`demo_m2/`。
- M3A PIV-BOST velocity compensation toy：`demo_m3a/`。
- M3B 4D BOST low-rank temporal toy：`demo_m3b/`。
- 开题可用图：`demo_m0/results/m0_summary.png`、`demo_m1/results/m1_volume_summary.png`、`demo_m2/results/m2_improvement_heatmap.png`、`demo_m2/results/m2_capacity_scan.png`、`demo_m3a/results/m3a_compensation_summary.png`、`demo_m3b/results/m3b_rank_scan.png`。
- 因此前 8 周的代码闭环已经有第一版，后续重点应从“能不能跑”转为“物理几何是否更接近 OERF、指标是否更可信、真实数据接口是否能接上”。

## 每周记录模板

周次：

日期：

本周主目标：

完成的文件：

生成的图：

读过的论文/章节：

遇到的问题：

下周要问何远哲：

师兄反馈：

下一周动作：

## 红线提醒

- 不要连续两周只读论文没有图。
- 不要在没跑通 synthetic demo 前承诺真实 4D BOST。
- 不要把“学流体力学”写成开放式任务，每周只补和变量链条有关的概念。
- 不要等师兄给数据才开始，synthetic pipeline 必须先跑。

## 12 周后应具备的材料

- 一页开题提案。
- 10 页开题 PPT。
- 一个 synthetic BOST/NeRIF demo。
- 一张鲁棒性曲线。
- 一个数据请求清单。
- 一个真实数据接口草案。
- 一个明确题目：保底主线 + 冲刺副线。
