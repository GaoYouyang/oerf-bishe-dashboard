# v116 Global Phase Transport: no usable rigid-translation signal

> **Evidence status:** `PASS_INDEPENDENT_RECOMPUTATION_PHASE_TRANSPORT_V116`  
> **Scientific decision:** `FAIL_NO_GLOBAL_INTEGER_TRANSLATION_SIGNAL`  
> **Boundary:** This is an input-level diagnostic on already-open, known-geometry, noise-free straight-ray development bundles. It is not a reconstruction, resource, external-generalization, or real-BOST result.

## 中文摘要

在 v112.4 关闭当前 CNN/FNO 延伸后，v116 检验了一个物理上不同、无需训练的解释：相邻时刻的 cheap factor-BP 是否包含可用的**全局三维平移**信号。

诊断只读取部署时可见的相邻 BP 与已知支撑；不读取真值、不调用 scorer。它通过 FFT 相位相关估计把前一时刻整体移到当前时刻的整数位移，再检查该平移是否提高两帧的一致性。

完整已开封 roster 覆盖五条 PoolFire 轨迹、三套九视角几何、六张 held-out 坐标图和十个相邻帧对，共 `900` 个 deployment-only pair。结果：

- `900/900` 对可辨识；
- 非零全局整数位移为 `0/900`；
- 超过 `1e-12` 容差的一致性改善为 `0/900`；
- 独立实现逐对重算，shift 最大差为 `0`，一致性改善最大差为 `0`。

因此，当前输入中没有证据支持把一个**全局刚性平移**用作反应流 warm start 的物理机制。这关闭的是最简 global-translation 支线；它不排除局部、非刚性或需要额外测量合同的输运表示。

`algorithm_breakthrough=false` · `paper_success=false` · `real_bost=false`

---

# v116 Global Phase Transport: no usable rigid-translation signal

After v112.4 closed the current CNN/FNO continuation, v116 tested a physically distinct, training-free explanation: whether adjacent cheap factor backprojections contain a usable **global 3-D translation**.

The diagnostic consumes only deployment-visible adjacent BP fields and known support. It estimates an integer translation using FFT phase correlation and measures whether shifting the previous field improves agreement with the current field. No truth field or scorer is loaded.

Across 900 already-open development pairs spanning five PoolFire trajectories, three known nine-view geometries, six held-out coordinate maps, and ten consecutive frame pairs, all pairs were identifiable but none had a nonzero integer shift or a coherence improvement above `1e-12`. An independent implementation reproduced every shift and improvement exactly.

This closes the simplest global-rigid-translation mechanism for the present proxy inputs. It does not rule out local or nonrigid transport, and it provides no reconstruction, resource, external, or real-BOST claim.
