# N2-PVGR N5-D3 32 格自适应残差参考包预注册

日期：2026-07-18

## 问题收缩

N4.1 在 32 格中授权了 23 个 H1024 参考和 7 个 H2048 参考，留下两个 narrow control。D1 排除了浮点累加顺序是误差平台主因，D2 证明这两格在 H8192 的 selected synthetic 尾部通过了预注册门。D3 只做一件事：

> 不重跑 forward、不重选样本、不改阈值，把 `23×H1024 + 7×H2048 + 2×H8192` 的残差数组封装为一个逐格可追溯的参考包。

## 为什么必须叫“混合残差参考包”

- N4.1 的 30 个已授权数组是 `matched_residual_uv`，其路由是 curved/straight 分别累加后相减；
- D2 的两个 H8192 尾部数组使用 `paired_neumaier`；
- D1 只在 p04/p05 四格上验证过 raw 与 paired 路由等价，覆盖率是 `4/32`，不是 `32/32`。

因此，D3 的最佳可能机器判决是 `D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY`。`uniform_paired_reference` 必须保持 `false`，不允许用语言包装掉方法覆盖缺口。

## 冻结的映射

1. 严格复用 N4.1 `result.json` 中 32 格的原始顺序；
2. `final_cellwise_reference_authorized=true` 的格必须使用其 `final_reference_step_count`；
3. 仅两个 N4.1 未授权格可以转向 D2 H8192 `paired_neumaier`；
4. 任何缺失、重复、替换、路径或字节漂移都 fail closed。

## 证据与数组合同

每格必须保存：

- cell / pair / role / field / seed / orientation / aperture / stress；
- source path、source file SHA-256、source record SHA-256；
- 对 H8192 tail，额外绑定 D2 config / attestation / runner / protocol commit，并保存来自父 N4 格的 canonical identity hash；D2 level 原行只含 `cell_id/pair_id/role/step_count` 的限制必须显式披露；
- step count、reference method、observable ID、units；
- `(256, 2)` float64 残差数组、`source_array_sha256_legacy`、`pack_array_sha256_float64_le_c_order`、L2 norm、finite 状态；
- source logical point queries 与可用的 domain/stencil/direction diagnostics。

全包额外保存 cell-order hash、stacked-array hash、source checkpoint Merkle root 与 manifest。D3 组装本身的 field-point query 成本必须为零。

## 结果前冻结的门

| 门 | 要求 |
|---|---:|
| 格数 / 唯一格 | `32 / 32` |
| 步数分配 | `23 / 7 / 2` |
| 每格 shape | `(256, 2)` |
| finite fraction | `1.0` |
| D1 路由等价最大 relative-L2 | `<=5e-12` |
| D2 父判决 | `D2_SELECTED_TAIL_RESOLVED_AT_H8192` |
| N4 checkpoint 数量 | `105` |
| N4 H2048 checkpoint 数量 | `9` |
| N4 checkpoint Merkle root | 必须与 recovery attestation 相同 |
| 所有 parent/result/manifest/validation hash | 必须相同 |

## 独立校验要求

validator 不得导入 D3 runner，必须独立重算：

- 32 格映射和 `23/7/2` 计数；
- N4 全 checkpoint inventory Merkle root；
- 每个 source file / record / legacy source-array hash，以及从同一 decoded array 独立重算的 float64-le C-order pack hash；
- 每格与 stacked float64-le C-order array hash；
- source query ledger、pack-level gates、result 摘要和 manifest；
- 图像存在且非空白。

## 允许与禁止

D3 通过只表明：当前 32 格 synthetic residual 有一个字节级可追溯、但方法混合的参考包。它不授权 fresh reference、uniform paired reference、field JVP/VJP、三维重建、神经算子训练、真实数据、泛化或论文结论。
