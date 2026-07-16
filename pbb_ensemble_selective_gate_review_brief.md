# PBB ensemble selective gate：师兄审阅简报

## 先给结论

这四次实验仍然只是**小规模、预设线性弱偏折 curved/cone generator**上的独立门控证据，不是对真实非线性折射、CFD、OpenBOST/OERF、FNO/FFNO、DeepONet 或 Learned Primal-Dual 的结论。四个 `report.json` 的 claim status 都是 `FAILED_OR_INCOMPLETE`；因此目前**不能声称 superiority**。最多可以说：v2/v3 在当前合成生成器和指定 `projected_bb` 基线下显示出有限的平均误差改善趋势；v4 的 selective 机制在给定门槛下没有找到可行阈值。

## v1-v4 锁定数字

下表的 gain 定义为相对 L2 误差相对 `projected_bb` 的下降百分比；正数才表示 candidate 误差更低。`p10` 是每个 seed 的最小 p10 gain，`harm` 是误差恶化超过 1% 的比例上界。v1-v3 直接取各自 `report.json` 的 `independent_select` / `independent_lock` aggregates；v4 的 candidate 是全量回退后的结果，取 `summary.csv`。

| 版本 | 独立划分 | candidate mean L2 | projected_bb mean L2 | mean gain | p10 gain | harm | acceptance / coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 base correction | select | 0.5086893439 | 0.4390458763 | -18.9068965912% | -32.3714141846% | 1.0000000000 | 0.0000000000 |
| v1 base correction | lock | 0.4987358749 | 0.4306121171 | -17.9029254913% | -31.4239807129% | 1.0000000000 | 0.0000000000 |
| v2 PBB base | select | 0.4383450945 | 0.4398469329 | 0.1870397503% | -1.7483505011% | 0.2500000000 | 1.0000000000 |
| v2 PBB base | lock | 0.4179967244 | 0.4179792702 | -0.1023394763% | -2.8188378811% | 0.1944444444 | 1.0000000000 |
| v3 saturation guard | select | 0.3769307435 | 0.3791571558 | 0.5247963763% | -0.6157947183% | 0.0714285714 | 0.9880952438 |
| v3 saturation guard | lock | 0.4421897531 | 0.4439255595 | 0.2809306482% | -0.8312377334% | 0.0750000000 | 0.9916666740 |
| v4 ensemble selective | select | 0.4461926222 | 0.4461926222 | 0.0000000000% | 0.0000000000% | 0.0000000000 | 0.0000000000 |
| v4 ensemble selective | lock | 0.5018178225 | 0.5018178225 | 0.0000000000% | 0.0000000000% | 0.0000000000 | 0.0000000000 |

补充对照：v1/v2/v3 的 validation `projected_bb` mean gain 分别为 `-25.4143028259%`、`2.368151028951%`、`2.084106763204%`；对应 p10 为 `-37.8196067810%`、`1.2673127651%`、`0.1754220873%`，对应 harm 为 `1.0000000000`、`0.0000000000`、`0.0000000000`。这些 validation 数字不能替代独立 select/lock。

## 为什么 v4 必须按“全部回退”解读

- v4 训练了 **5 个头**，seed 为 `121, 122, 123, 124, 125`，每头 `5989` 个参数，总参数量 `29945`。
- 五头共享同一条物理 forward/adjoint 轨迹：`candidate_shared_forward=4`、`candidate_shared_adjoint=4`，`correction_head_passes=5`，`physical_trajectory_repeated_per_head=false`。所以不能把五头误解成五次独立物理模拟；它们是在共享 PBB 方向/轨迹上产生 correction disagreement，用于 uncertainty。
- Lipschitz 相关量采用 `exact_small_matrix`，不是把每个头的物理算子重复做一遍 power iteration；实际账本为 `lipschitz_power_iterations_precomputation=0`、`exact_spectral_decompositions=526`，并有 `certificate_violation_rate=0`。
- v4 的 family-held-out lock 是独立于阈值校准的两层划分：训练/validation family 为 `gaussian, flame`；select 为 `expanding_kernel, jet_shear, interacting_fronts, shock_cell`；最终 lock 只用未参与 select 的 `helical_plume, stratified_ignition`。select/lock 的 operator hash 不相同，且 `select_lock_equal=false`。因此 lock 更接近真正的 family-held-out 检验，但它没有被用来选阈值。
- selective gate 的要求是 coverage 至少 `0.2`、p10 gain 至少 `0%`、harm 不超过 `5%`。`threshold_calibration.csv` 的 41 个候选阈值全部 `feasible=False`；commit 明确记录 `threshold=-1.0`、`coverage=0.0`、`selection_reason=no_feasible_select_threshold_abstain_all`。于是 v4 的 select 和 lock 都逐样本回退到 `projected_bb`，表中的零 gain 是回退恒等式，不是 ensemble superiority 或等价性能证明。

## 当前可以、不能说什么

可以说：证书检查在已跑的 synthetic linear setting 中为零违规；v2/v3 的 PBB-based correction 相对 `projected_bb` 有小幅平均改善，但 p10/harm gate 仍未完整通过；v4 发现当前 uncertainty 与性能约束之间没有可行的 coverage-operating point。

不能说：ensemble 优于 PBB、v3 已经稳定优于基线、方法可泛化到真实 OERF/CFD、五头提供了五条独立物理证据，或 v4 的零 harm 代表模型安全。尤其不能把 fixed-PG/FISTA 上的 gain 当成对选定 `projected_bb` 基线的 superiority。

## 请师兄下一步提供或确认

1. **数据**：真实观测/重建数据的格式、单位、标定方式、ground truth 或可接受 proxy；是否有 train/validation/test 的时间或样本隔离规则，以及允许的样本量。
2. **几何**：真实射线/视角/相机位姿、体素或网格尺寸、边界条件、路径采样规则、是否存在非线性折射或遮挡；最好提供一份可复现的 operator 或矩阵样例。
3. **噪声**：噪声是独立、相关、异方差还是 camera-specific；实际噪声水平和标定文件；是否需要把 gain drift、missing views、outlier 纳入 lock。
4. **基线**：最终认可的 baseline 清单和实现版本，尤其是 `projected_bb` 的步长/停止准则，以及 fixed-PG、FISTA、传统反演方法或已有 OERF baseline 的统一预算、初始化和 forward/adjoint 计数。
5. **验收口径**：mean、p10、harm、coverage、certificate 的门槛是否仍为当前数值；是否要求在真实 family-held-out 数据上预注册阈值，并由师兄指定不可改动的 lock 集。

## 证据位置

- `demo_t16_operator/results/base_correction_independent_gate/{report.json,summary.csv}`
- `demo_t16_operator/results/pbb_base_correction_independent_gate/{report.json,summary.csv}`
- `demo_t16_operator/results/pbb_saturation_guard_independent_gate/{report.json,summary.csv}`
- `demo_t16_operator/results/pbb_ensemble_selective_gate/{report.json,summary.csv,threshold_calibration.csv,selection_commit.json}`
