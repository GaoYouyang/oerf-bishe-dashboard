# T16 同重建预算红队结果：给何远哲的审核简报

更新：2026-07-11

当前判决：**既有 QC-SNCO checkpoint 路径在受控同重建预算推理审计中失败；不继续扩模型。该结果仍是 exploratory pilot，不是训练匹配后的最终算法判决。**

## 这次真正比较了什么

对每个总重建预算 `K ∈ {4,6,8}`：

1. `K-1` 个 support views 产生初始 physics lift、Residual/Absolute experts 与 support-fit；
2. 一个 `Q_fit` 视角分别用于：
   - 直接加入 `S∪Q` 重建；
   - 给 learned support-null direction 求 clipped scalar alpha；
   - 做不学习的 query-null ridge update；
3. 60° 相机在任何 query 选择之前锁定为 `Q_audit`，只做最终重投影审计；
4. fixed query 固定为 80°；random、max-gap、adaptive-energy 只作次级分析；
5. 88 个独立体场上运行 3 个模型种子，先在每个体场内汇总种子，再让五个来源 split 等权做 20,000 次分层 paired bootstrap。

注意：`K` 是**重建视角预算**。因为另有一台 `Q_audit`，实际采集/安装视角数是 `K+1`。

## Fixed query 主结果

| 重建预算 | `S∪Q direct` 相对 S | learned correction 相对 S | learned 相对 direct | 95% cluster CI 下界 | p10 | >1% 伤害率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K=4 | +15.25% | +1.12% | **-20.17%** | -24.32% | -54.54% | 88.0% |
| K=6 | +10.51% | +1.37% | **-13.07%** | -17.10% | -46.29% | 72.0% |
| K=8 | +12.69% | +1.18% | **-15.11%** | -18.16% | -41.17% | 86.2% |

结论不是“query 没用”，而是：**query 作为直接重建信息远比作为弱 learned-null direction 的幅度标定更有价值。**

## 不学习数值更新的结果

- K=4 fixed 时，numeric query-null update 相对 S 改善约 `+5.75%`，但仍比 `S∪Q direct` 低 `-13.89%`；
- K=6 fixed 时相对 S 为 `-1.01%`；
- K=8 fixed 时相对 S 为 `-9.72%`；
- 当前 `lambda=0.001` 与 cap 不是在 validation 上调出的最终强基线，因此只能说“这个固定数值配置未能胜出”，不能说数值更新路线被证明无效。

## 唯一值得保留的几何线索

K=6、max-gap query 下：

- learned correction 相对 `S∪Q direct` 的均值为 `+0.56%`；
- 95% cluster CI 下界为 `-0.28%`；
- p10 为 `-5.90%`；
- 仍有 `28.7%` 体场比 direct 差超过 1%。

因此它不是正结果，但提示：**相机几何/VOI 可能比继续堆 correction 网络更值得研究。**

## 为什么还不能写“算法最终失败”

1. base/corrector checkpoint 来自旧 view-dropout 训练分布，没有在本轮固定 `S/Q_fit/Q_audit` mask 上重新训练；
2. 当前 88 个体场已参与 v1-v2c 多轮开发，只能作为 development audit；
3. forward 仍是逐 z 的 `8×16×16` 线性 synthetic stack，不是真实 cone-ray / ray-bending BOST；
4. random query 只跑了一个 policy seed；
5. numerical update 的 ridge/cap 尚未在独立 validation 上锁定。

所以机器报告状态是：`PILOT_ONLY_CURRENT_CHECKPOINT_PATH_FAILS`。

## 我建议师兄批准的研究转向

### A. 当前主任务：camera-budgeted direct inverse operator

在训练 mask 匹配的条件下，比较传统重建、matched U-Net/FNO、geometry-aware view encoder 与 NeRIF。核心问题改成：

> 在固定实验视角预算和真实 ray geometry 下，如何把每个相机的位移场与几何信息最高效地编码进三维逆算子？

### B. 最贴何远哲的升级：operator warm-start NeRIF

比较 random、SIRT/FBP 和 operator 三种 NeRIF 初始化的：

- 总 wall time；
- 最终 held-out reprojection；
- 失败率；
- 同等最终误差所需迭代数。

恢复论文升级的最低条件：wall time 降至少 20% 或失败率降至少 30%，最终误差非劣不超过 1%。

### C. QC-SNCO 的处理

保留为本科论文中的机制性负结果：

> support-null consistency 可以做到数值精确，但并不保证 learned correction direction 接近真实三维误差方向；额外相机直接进入重建通常更有效。

不再把它作为当前新模型主线。

## 希望师兄判断四件事

1. 是否认可把主线从 QC-SNCO 转为 direct inverse operator + operator warm-start NeRIF？
2. 组内能否提供 NeRIF forward/loss、最小 displacement + geometry + mask 数据包？
3. 实际相机布局是否固定，未来样本间是否会变化？
4. 真实无三维真值时，最认可的 audit 是 held-out camera、front location、密度积分还是 PIV 补偿误差？

## 复现入口

- 可视化判决页：`operator_fair_budget_dashboard.html`
- 运行脚本：`demo_t16_operator/run_fair_camera_budget.py`
- 配置：`demo_t16_operator/configs/fair_camera_budget.json`
- 完整性校验：`demo_t16_operator/validate_fair_camera_budget_results.py`
- 机器报告：`demo_t16_operator/results/fair_camera_budget/fair_camera_budget_report.json`
- 逐体场-种子数据：`demo_t16_operator/results/fair_camera_budget/fair_camera_budget_samples.csv`
- 体场级聚类数据：`demo_t16_operator/results/fair_camera_budget/fair_camera_budget_clusters.csv`
