# V5P-V6A 发布复现说明

这份发布包服务于一个很窄的目的：让第三方核对 V5P-V6A 的 synthetic evidence chain，而不是把 development NO-GO 包装成论文成功。

## 两层校验

1. 每个结果目录的 `checksums.sha256` 锁定该阶段的 CSV、JSON、NPZ 等核心产物。
2. `demo_t16_operator/results/v5p_v6a_release_checksums.sha256` 锁定网页引用的全部报告、图、配置、runner、validator 和三份 V5P replay 权重。

运行：

```bash
.venv/bin/python demo_t16_operator/validate_v5p_v6a_release.py
```

`PASS_INTERNAL_CONSISTENCY_ONLY` 只表示文件、provenance、V5P replay hash 和 V6A 聚合数值相互一致。它不表示 V6A 通过 scientific gate；V6A 的结论仍是 development NO-GO。

## 三份权重为什么进入公开仓库

V5P、V5Q、V5R 依赖 `sf_rio_adjoint_only` 的 3101/3102/3103 三份自生成 synthetic checkpoint。它们不含实验数据、论文 PDF、账号信息或 VPN 内容，只用于重放冻结预测。其 SHA-256 同时记录在 V5P report、各后续 provenance 和顶层发布清单中。

## MPS 边界

V5Y、V5Z、V6A 的存档参考运行使用本机 MPS。report 记录 Python、NumPy、Torch、系统与 device；发布 validator 对存档结果做容差重算，不声称跨设备或跨 Torch 版本逐字节一致。要做新一轮 fresh gate，应另存 CPU 小规模 reference 或固定服务器镜像，不能覆盖这次存档。

## 公开边界

本发布包不含受限论文、PDF、私有论文库、VPN 链接、凭据、cookie、绝对用户路径或真实实验数据。
