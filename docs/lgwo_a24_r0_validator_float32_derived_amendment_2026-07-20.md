# LGWO-A24 R0 独立验证器修正 VA2

修正 ID：`R0-VA2-FLOAT32-DERIVED-TOLERANCE`  
日期：2026-07-20  
前置修正：`R0-VA1-FLOAT32-PROVENANCE-TOLERANCE`  
正式结果仍为 commit `9021ff9` 生成的原 sealed package。

## 触发原因

VA1 后，输入 tensor hashes 已通过到第一条 trajectory arithmetic 检查。validator 随后比较：

```text
recorded torch.float32 residual/noise ratio = 35.90143585205078
ratio recomputed from CSV decimal values    = 35.90143517671522
```

相对差约 `1.9e-8`。前者是在 torch float32 中做除法后写入 CSV；后者是从两个 CSV 十进制值用 Python float64 再除。原统一 `rtol=2e-9, atol=2e-10` 再次错误地把这类跨表示重算要求成近乎 float64 位级一致。

## 唯一新增修正

以下两项 **float32 派生一致性量** 使用与 VA1 相同的：

```text
rtol = 1e-6
atol = 1e-8
```

1. `weighted_residual_to_sensor_noise_ratio` 与 `weighted_residual_norm / expected_weighted_sensor_noise_norm`；
2. `path_projection_consistency_relative_difference` 与 CSV 中两个 residual norm 的重算值。

## 不变项

- 不改变任何原始 tensor hash 或输入重建；
- 不改变 field/gradient/F1/residual 数值；
- 不改变 validation checkpoint 或八种 selector；
- 不改变 paired gain、harm、bootstrap、gate 或状态；
- 不改变正式 result package、checksums 或 source commit；
- 不把 NO-GO 改写成正结果。

validation report 必须同时记录 VA1、VA2、修正提交、amended validator hash 和两个修正文档 hash。
