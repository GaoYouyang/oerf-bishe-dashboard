# v283：投影吻合仍不代表三维梯度正确

2026-09-05。完整采样的固定 TSVD 参考通过 8/12 分层：七、九相机全部通过，五相机四个时刻全部失败。五相机投影相对误差低于 5.2e-7，但全梯度 p90 仍为 1.258–1.273，高于 0.75 门。

## 做了什么

在已独立验证的有限背景像素位移接口上，使用每相机完整 24×24 射线，而不是此前 8×8 的光学检查子集。未知量为实际进入光学仿真的支撑后三维场：边界为零，保留全部 5,880 个内部节点，不去均值、不用旧 DCT 降维。

唯一参考是结果前固定相对截断 1e-6 的最小范数 TSVD。正式与独立实现分别重建几何、输入和求解，再分别评分；所有预测先封存，之后才读评分真值。它不拟合模型参数，不用真值选截断。Zero、观测线搜索 BP、未改 CGLS K16 和几何 Jacobi PCGLS K16 均为便宜对照，四者各通过 0/12 分层。

## 独立结果

18/18 独立检查通过；场、算子和指标最大差分别为 2.96e-10、2.82e-11、1.94e-11。正式判决为 `FAIL_FIXED_NODAL_TSVD_REFERENCE_V283`，不是数值复算失效。

每层 117 个模型—标定文件组合，higher 分位数。四指标均同时检查 p90 和 worst；表中仅压缩显示 p90，完整双门见脱敏汇总。

| 相机 / Cameras | t | Field p90 | Full gradient p90 | Interior gradient p90 | Observation p90 | Gate |
|---:|---:|---:|---:|---:|---:|:---|
| 5 | 0.0 | 0.506245 | 1.25863 | 1.00796 | 3.02501e-07 | FAIL |
| 5 | 0.25 | 0.526044 | 1.27276 | 1.01571 | 3.69591e-07 | FAIL |
| 5 | 0.75 | 0.502514 | 1.26716 | 0.997408 | 3.25391e-07 | FAIL |
| 5 | 1.0 | 0.500955 | 1.25803 | 0.997621 | 3.25892e-07 | FAIL |
| 7 | 0.0 | 1.80435e-11 | 3.55703e-11 | 3.46387e-11 | 1.80328e-14 | PASS |
| 7 | 0.25 | 1.79357e-11 | 3.53822e-11 | 3.41605e-11 | 1.83437e-14 | PASS |
| 7 | 0.75 | 1.7596e-11 | 3.60185e-11 | 3.48622e-11 | 1.79787e-14 | PASS |
| 7 | 1.0 | 1.80785e-11 | 3.58483e-11 | 3.4828e-11 | 1.81249e-14 | PASS |
| 9 | 0.0 | 1.56989e-12 | 2.55422e-12 | 2.50418e-12 | 1.9453e-14 | PASS |
| 9 | 0.25 | 1.47647e-12 | 2.34578e-12 | 2.29205e-12 | 1.96996e-14 | PASS |
| 9 | 0.75 | 1.46753e-12 | 2.50737e-12 | 2.45485e-12 | 1.93691e-14 | PASS |
| 9 | 1.0 | 1.5428e-12 | 2.48289e-12 | 2.43798e-12 | 1.94666e-14 | PASS |

五相机四个时刻的 field p90 为 0.501–0.526，全梯度 p90 为 1.258–1.273，内部梯度 p90 为 0.997–1.016；相应门为 0.5、0.75、0.75。虽然其最坏投影误差只有约 5.19e-7，三维梯度并没有恢复正确。七相机最坏场/全梯度误差约 0.0453/0.0954；九相机约 6.32e-12/1.01e-11，但不能把它们事后挑出来当作完整目标成功。

固定截断下，五/七/九相机保留的谱秩分别为 4483–5117、5859–5880、5880。这里说的是截断后的数值秩，不是精确数学零空间维数，也不证明其他先验或估计器不可能成功。

## 边界与成本

独立复算有效，固定参考方法整体失败。1,404 个单元使用同一离散模型生成和反演干净虚拟数据；不是实测 BOST、跨模型学习或加速。十三份标定仅有十一种不同几何，不能视为十三个独立实验。 生成和反演使用同一网格、相机和线性算子，是有利的理想条件；没有验证噪声、标定误差或有限幅度非线性光线。物理目标含固定边界支撑窗，不能套用旧实验分数。

每套实现有 33 次稠密分解，六个完全重复设置复用结果。缓存因子后的 TSVD 逻辑算子调用为 0A+0A^T，但仍要稠密乘法，并且几何分解与存储不能免费。对照的逻辑账为 Zero 0/0、BP 1/1、CGLS16 与 PCGLS16 16/16；另有生成、探针和评分重放开销。没有实测方法间 fresh wall/RSS 优势。

关闭此固定截断参考，不调阈值、截断或迭代深度追通过。下一步先用事先选定的小诊断解释五相机的场与梯度歧义，再判断是否值得扩大；完整研究目标不缩减。 所有算法突破、论文成功、资源加速、外部泛化、曲线光线与真实 BOST 标志仍为 false。

# v283: fitting projections does not guarantee the 3D gradient

The full-sampling fixed TSVD reference passes 8/12 strata: all seven/nine-camera strata pass, while all four five-camera times fail. Five-camera relative projection error stays below 5.2e-7, but full-gradient p90 is 1.258–1.273 versus the 0.75 limit.

## Experiment and independent evidence

The independently qualified finite-background pixel interface now uses all 24×24 rays per camera, not the earlier 8×8 optical-check subset. The unknown is the actual supported optical field: zero boundary and all 5,880 interior nodes, without demeaning or old DCT reduction.

The only primary is a minimum-norm TSVD with a prospectively fixed relative cutoff of 1e-6. Formal and independent implementations rebuild geometry, inputs and solutions separately. All predictions seal before truth scoring; no parameters are trained and no truth selects the cutoff. Zero, observation-line-search BP, unchanged CGLS K16 and geometry-Jacobi PCGLS K16 each pass 0/12 strata.

All 18 independent checks pass, with maximum field/operator/metric differences 2.96e-10/2.82e-11/1.94e-11. The scientific verdict is `FAIL_FIXED_NODAL_TSVD_REFERENCE_V283`, not invalid recomputation. Each table row aggregates 117 model/calibration-file combinations using higher quantiles. All four metrics must pass both p90 and worst; the compact table shows p90 only, with full tails in the public summary.

Five-camera field/full-gradient/interior-gradient p90 ranges are 0.501–0.526/1.258–1.273/0.997–1.016 against limits 0.5/0.75/0.75. Its worst projection error is only about 5.19e-7, so an excellent image fit does not establish correct 3D gradients. Seven-camera worst field/full-gradient errors are about 0.0453/0.0954; nine-camera errors about 6.32e-12/1.01e-11. Selecting only these successful strata would not pass the complete objective.

Retained spectral ranks at the fixed cutoff are 4483–5117/5859–5880/5880 for five/seven/nine cameras. These are thresholded numerical ranks, not exact nullspace dimensions or a proof that other priors/estimators cannot succeed.

## Limits, cost and next step

Independent recomputation is valid, but the fixed reference fails overall. The 1,404 cells use the same discretization for clean virtual generation and inversion; this is not measured BOST, cross-model learning or acceleration. Thirteen calibration files contain only eleven distinct geometries, not thirteen independent experiments. Generation and inversion share grid, cameras and linear operator, a favorable ideal condition. Noise, calibration mismatch and finite-amplitude nonlinear rays are untested. The target includes the fixed support window, so old scores do not transfer.

Each implementation performs 33 dense decompositions and reuses six exactly duplicated setups. Cached TSVD has logical 0A+0A^T but still requires dense factor products, geometry setup and storage. Control logical accounts are Zero0/0, BP1/1 and CGLS16/PCGLS16 16/16, with generation, probes and scoring replays accounted separately. There is no comparative fresh-process wall or whole-pipeline RSS benefit.

Close this fixed-cutoff reference without threshold, cutoff or iteration-depth rescue. Next use a small prospectively chosen diagnostic to examine five-camera field/gradient ambiguity before expansion; the full research objective remains unchanged. Algorithm, paper, resource, external, curved-ray and real-BOST success flags remain false.
