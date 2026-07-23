# JACRU N1.9 contrast/global-K：候选对分支关闭

日期：2026-07-18

机器状态：`N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED`

证据等级：`E1_SYNTHETIC_OPENED_DEVELOPMENT_HYPOTHESIS_DESIGN_ONLY`

## 1. 一句话结论

两个预先冻结、同为 rank-6 与 `25F/24A^T` 的 camera-contrast/global-K 表示均未通过联合门。
Residual-Contrast 通过 `16/17` 个重建门，但阻尼后的额外 headroom 只有
`51.408% < 60%`；Damping-Contrast 通过 `15/17`，同时未达 exact retention
`64.042% < 70%` 和 extra-headroom `36.864% < 60%`。因此不授权新 split、不训练
learner，并关闭在同一已打开 development 上继续堆这类 rank-6 basis 的路线。

## 2. 运行前冻结与完整性

- 设计、配置、实现和测试先提交为 `52490e5c33f93245e5b287b8bbd41fe7b630c082`，完整结果随后运行。
- runner 在 case preparation 前核对固定 source path，以及 T0、N1.5--N1.8 config/result、
  fixture、operator、teacher 和复用 Python module/runner 共 16 项 SHA-256。
- 决定性运行要求 N1.9 config、freeze、model、runner 与 tests 已提交且相对 HEAD 无修改。
- N1.7、N1.8、N1.9 的 development case ID 与 geometry digest 完全相同：6 个 geometry
  clusters、12 个 paired fields；两个 family 共享 geometry，不算 12 个独立实验。
- 结果包 11 个受校验文件全部通过 SHA-256；provenance 指向冻结提交 `52490e5`。
- 没有打开新 geometry、fresh、OOD、final 或真实 BOST；没有 learner，也没有 finite-K truth search。

## 3. 两个且仅两个候选

记 `d` 为部署可见 component damping，`r` 为 warm residual，`C1/C2` 为三相机 Helmert
contrast，`K=A P A^T`。冻结候选为：

```text
Residual-Contrast Global-K6 = orth(d, r, C1 r, C2 r, Kd, Kr)
Damping-Contrast Global-K6  = orth(d, r, C1 d, C2 d, Kd, Kr)
```

两者都用 12 步 warm CGLS、1 次 warm projection、`2F/2A^T` basis setup 和 10 步 refine，
总计 `25F/24A^T`，无 high-order 调用。每个 case 都达到精确 rank 6；最大正交缺陷分别约为
`6.66e-16` 与 `4.44e-16`。

## 4. 完整结果

| 冻结表示 | field vs CGLS-24 | H1 vs CGLS-24 | field vs damping | exact retention | extra-headroom | P A^T gain | 重建门 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Residual-Contrast Global-K6 | **+6.207%** | **+10.672%** | **+2.672%** | **72.917%** | **51.408%** | 28.112% | **16/17** |
| Damping-Contrast Global-K6 | +5.452% | +8.768% | +1.888% | 64.042% | 36.864% | **35.787%** | 15/17 |

两者相对 CGLS-24 的最坏逐 case field gain 均为正，分别为 `+1.811%` 与 `+1.834%`；最坏
geometry gain 分别为 `+3.265%` 与 `+3.125%`，相对 CGLS-24 或 damping 的 `>1%` harm rate
均为 0。这些是稳定性信号，但不能覆盖冻结门失败。

### 4.1 明确失败项

- Residual-Contrast：只失败 `extra_headroom_retention = 51.408% < 60%`。
- Damping-Contrast：失败 `exact_gain_retention = 64.042% < 70%` 和
  `extra_headroom_retention = 36.864% < 60%`。
- 两者 support-adjoint gain 都小于 50% 的 forward-correction 机制线；由于重建门未全过，
  最终角色均为 `REPRESENTATION_NO_GO`，而不是“差一点授权”。

## 5. 真正值得保留的物理信号

Residual 与 Damping 的差异不是简单的总分胜负：

- Residual 在 `12/12` 个 case 的 H1 都更低；
- 对 `single_interface`，Residual 在 `6/6` 个 case 的 field error 更低；相对 damping 的
  family mean field gain 为 `+4.946%`，Damping 为 `+3.348%`；
- 对 `smooth_no_interface`，Damping 在 `6/6` 个 case 略好，但差异很小；两者 family gain
  分别为 `+0.399%` 与 `+0.428%`；
- Damping 在 `8/12` 个 case 的 data residual 更低，并有更高的 support-adjoint gain
  `35.787% vs 28.112%`。

因此最准确的解释是：在当前 surrogate 中，把 camera contrast 乘在 residual 上更偏向界面与 H1
恢复；乘在 damping 上更偏向 measurement/adjoint 一致性。二者尚未在同一物理模型里统一。
这个观察可以产生新的研究问题，但不能据此在旧 development 上设计 selector、改 basis 或声称
Residual 算法胜出。

## 6. 成本门怎样读

两候选都通过了预先冻结的本机配对成本门：

| 表示 | median solver-path ratio | P90 solver-path ratio | 门 |
|---|---:|---:|---:|
| Residual-Contrast | 1.077 | 1.118 | 通过 `<=1.25 / <=1.50` |
| Damping-Contrast | 1.066 | 1.168 | 通过 `<=1.25 / <=1.50` |

这里的时间只包含 shared warm、basis setup 与 refinement，不包含 evaluator oracle 系数投影；
每 case 只有一次、约 7.6--10.4 ms 的本机测量。它最多是实现路径的筛查，不是可部署方法的
end-to-end runtime，也不支持跨硬件速度结论。未来若计时参与正向授权，必须重复测量并报告区间。

当前候选没有 covariance head 或 majorizer，所以 Schur 门是
`NOT_APPLICABLE_NO_COVARIANCE_OR_MAJORIZER`，不是“零违反”。

## 7. 关闭了什么，没有关闭什么

正式关闭：在这个已打开、三相机 synthetic development 上，对这两个预冻结 rank-6
camera-contrast/global-K 复合表示继续调半径、堆 basis 或接 learner。

没有关闭：所有 camera-aware 或 global-K 方法、其他 rank、真实 mask/ROI 加权、新 geometry/session
上的不同物理假设，以及真实 BOST 的 measurement mismatch 建模。当前 Helmert contrast 还要求每台
相机有效 ray 数相等；真实不均匀 ROI 必须使用加权内积重新构造，不能直接平移。

## 8. 下一主线 N2：物理 mismatch 合同 + 独立 holdout

下一步不再问“还能拼哪六个向量”，而问：

> 标定、mask、ROI 不均衡、位移提取或视线模型误差，怎样同时影响界面恢复与 held-out
> measurement consistency？

在拿到真实数据合同前，只完成接口与预注册，不生成成功主张：

1. 固定真实输入：camera pose/intrinsics、pixel/ray coordinates、mask/confidence、标定版本、参考图、
   时间/session ID、单位与 support；
2. 确认 `A/A^T`、JVP/VJP 或可审计重投影接口，并先过 dot-product/finite-difference tests；
3. 按 geometry/session/camera 留出不可回看的 holdout，不能随机切 field；
4. synthetic 有 truth 时同时报告 field/H1 与 held-out reprojection；真实无 truth 时以 held-out camera
   reprojection、稳定性、物理一致性和成本为主终点；
5. 新模型只读部署可见量，若生成 response basis 或系数，必须把 generator 时间、内存和失败回退
   计入端到端成本；
6. DeepONet/FNO/NeRIF 只作匹配预算和数据合同下的基线，不因模型名称自动获得创新性。

## 9. 给何远哲师兄确认的问题

1. 真实 BOST/TDBOST 数据能否提供 camera ID、内外参、像素坐标、有效 mask/confidence 与标定版本？
2. 组内当前主要 forward mismatch 是 finite aperture、ray bending、标定漂移、位移提取，还是离散误差？
3. 是否存在可调用的 `A/A^T`、JVP/VJP 或 held-out-view reprojection 接口？
4. 能否永久留出一台 camera、一个 session 或一个工况，直到方法和阈值冻结后再打开？
5. 真实项目最认可的主终点是什么：密度/折射率 field、界面位置、held-out image、PIV compensation，
   还是时间分辨稳定性？
6. 能否给一个 NeRIF/TDBOST 的典型失败 case，并说明失败属于数据、标定、优化还是表示能力？

## 10. 可复现入口

- [运行前冻结](jacru_n1_9_global_contrast_freeze_2026-07-18.md)
- [机器摘要](../demo_t16_operator/results/jacru_n1_9_global_contrast_postopen_full1/summary.json)
- [逐 case 指标](../demo_t16_operator/results/jacru_n1_9_global_contrast_postopen_full1/case_metrics.csv)
- [basis 诊断](../demo_t16_operator/results/jacru_n1_9_global_contrast_postopen_full1/basis_diagnostics.csv)
- [SHA-256 清单](../demo_t16_operator/results/jacru_n1_9_global_contrast_postopen_full1/checksums.sha256)

当前允许的论文表述是：在已打开 synthetic development 上，Residual-Contrast 比
Damping-Contrast 更偏向界面/H1 恢复，而 Damping-Contrast 更偏向 measurement/adjoint 一致性；
两个固定表示均未通过预先冻结的联合门，因此该候选对不进入新 split 或 learner。不能写算法成功、
fresh 泛化、真实 BOST 优越性或优于 DeepONet/FNO/NeRIF/TDBOST。
