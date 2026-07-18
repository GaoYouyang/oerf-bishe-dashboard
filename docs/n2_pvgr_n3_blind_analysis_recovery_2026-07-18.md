# N2-PVGR-N3 盲态分析代码恢复协议

> 日期：2026-07-18
>
> 状态：`BLIND_RECOVERY_PROTOCOL_NO_NUMERICAL_CHECKPOINT_INSPECTION`

## 1. 已观察到的事实

原预注册提交为 `676c23d1962c93f17e8c2d0c0a81332146268790`，独立 attestation 提交为
`8a684e6`。正式 runner 完成并哈希写入 `96/96` 个逐格 checkpoint，又完成计时阶段，但在生成
dominance 汇总时退出：

```text
KeyError: 'logical_scalar_grid_point_queries'
```

错误发生在正式结果目录创建、machine decision、blocked bootstrap 输出、CSV、图和 manifest 发布
之前。发现错误后只查看了 traceback、checkpoint 文件数量/文件名和已提交源代码中的数据结构定义；
没有解析任何 checkpoint JSON，没有读取方法误差、门槛通过数、时间数字或胜负。

## 2. 根因

预注册 runner 假定每种方法的查询账本都有统一字段：

```text
logical_scalar_grid_point_queries
```

冻结的 OCBH、teacher 与 H128 确实使用该字段。冻结的 Picard 数据类则使用：

```text
total_field_point_queries
```

`PicardQueryAccounting.as_dict()` 保留该名称。两者在本项目中具有同一计量单位：为得到该方法返回
输出而执行的“一个坐标上的一次标量网格求值”。错误属于汇总 schema mismatch，不涉及 forward
数值、阈值、随机种子或物理格。

## 3. 唯一允许的修复

恢复层只允许：

1. OCBH、teacher、H128 继续读取 `logical_scalar_grid_point_queries`；
2. Picard-1/2 在 canonical 字段不存在时读取 `total_field_point_queries`；
3. 若两个字段同时存在但值不同，立即失败；
4. 若 canonical 与该方法已冻结 alias 都不存在，立即失败；
5. 导出的 query CSV 同时保留原字段、canonical 字段和来源字段名；
6. 在 result、summary 与 manifest 中记录恢复协议、attestation 和 checkpoint Merkle root。

恢复层不得修改原 runner。它通过窄包装临时替换两个 query-ratio helper、query CSV flatten helper，
以及只负责追加恢复元数据的 JSON/summary writer；退出后恢复原函数。

## 4. 明确禁止的变化

- 不改任何 residual、reference、tail、validity、wall-time 或 query threshold；
- 不改 8 个 field units、32 个 geometry cases、96 个 conditions、256 rays 或 128/256/512 steps；
- 不换 seed、不删格、不按结果早停；
- 不改 family-blocked bootstrap seed、20,000 repeats、CI 或三个 machine decisions；
- 不重算 96 个 physical cells；
- 不打开 reserved families、真实数据、三维重建、神经算子优越或论文 claim；
- 不把这次分析修复描述成算法改进。

第一次失败没有发布可使用的 timing 结果，因此恢复运行会完整重跑四个 geometry packages 的预注册
交错计时；checkpoint 内的 96 个物理格只能按原 attestation SHA 和 cell metadata 原样复用。

## 5. 盲态封存

本协议的配置、wrapper、tests、本文、attestation builder，以及原 config/runner/attestation 先组成
第一个恢复提交。随后 builder：

1. 验证正式结果目录仍不存在；
2. 只按字节读取 `cells/*.json` 计算 SHA-256，不调用 JSON parser；
3. 要求恰好 96 个文件；
4. 以“相对路径 + 文件 SHA-256”生成 checkpoint Merkle root；
5. 记录所有恢复文件在第一个恢复提交中的 SHA-256；
6. 生成第二个提交中的 recovery attestation。

只有第二个提交完成、所有哈希匹配、checkpoint Merkle root 未改变，wrapper 才可第一次解析数值。

## 6. 证据解释

这套修复能证明的是：分析 schema 错误在看数值前被限定为一个预先声明的等义字段映射，96 个输入
checkpoint 在修复前后字节不变，原统计门槛与判决逻辑不变。

它不能把 N3 变成完全无瑕的初始预注册。论文或网页必须同时披露原始 crash 和盲态恢复，不能只展示
恢复后的整洁结果。即使恢复后某方法通过所有门，也仍只是两个 synthetic development families、
每 family `n=4` 的结果，不授权真实数据、三维重建、泛化或投稿结论。

## 7. 可复现入口

- recovery config：`demo_t16_operator/configs/n2_pvgr_n3_blind_analysis_recovery_v1.json`
- recovery wrapper：`demo_t16_operator/run_n2_pvgr_n3_blind_analysis_recovery.py`
- tests：`demo_t16_operator/test_run_n2_pvgr_n3_blind_analysis_recovery.py`
- attestation builder：`site_tools/create_n2_pvgr_n3_blind_analysis_recovery_attestation.py`
- original protocol：`docs/n2_pvgr_n3_96cell_preregistration_2026-07-18.md`

本文件不含任何 checkpoint 数值或方法胜负。
